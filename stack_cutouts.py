import argparse
import glob
import logging
import multiprocessing
import os  # noqa: E402
import queue
import threading
import time
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import (
    Manager,  # noqa: E402
)

import h5py
import numpy as np
import pandas as pd
import torch
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from tqdm import tqdm

from logging_setup import setup_logger

setup_logger(
    log_dir='./logs',
    name='inference_test',
    logging_level=logging.INFO,
)
logger = logging.getLogger()

import psutil  # noqa: E402
from vos import Client  # noqa: E402

from download import download_worker  # noqa: E402
from kd_tree import build_tree  # noqa: E402
from make_rbg import preprocess_cutout  # noqa: E402
from postprocess import (  # noqa: E402
    load_segmap,
    make_cutouts,
    match_coordinates_across_bands,
    read_band_data,
    save_to_h5,
)
from shutdown import GracefulKiller  # noqa: E402
from tile_cutter import tile_finder  # noqa: E402
from track_progress import (  # noqa: E402
    get_progress_summary,
    get_unprocessed_jobs,
    init_cutouts_db,
    update_cutout_info,
)
from utils import (  # noqa: E402
    TileAvailability,
    extract_tile_numbers,
    get_dwarf_tile_list,
    load_available_tiles,
    open_fits,
    tile_str,
    update_available_tiles,
)
from zoobot_utils import ZooBot_lightning, get_dwarf_predictions, load_model  # noqa: E402

warnings.filterwarnings('ignore', message="'datfix' made the change", append=True)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='invalid value encountered in log10'
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='divide by zero encountered in log10'
)

client = Client()

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .

# define the band directory containing
# information on the different
# photometric bands in the
# survey and their file systems

band_dictionary = {
    'cfis-u': {
        'name': 'CFIS',
        'band': 'u',
        'vos': 'vos:cfis/tiles_DR5/',
        'suffix': '.u.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'whigs-g': {
        'name': 'calexp-CFIS',
        'band': 'g',
        'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme/',
        'suffix': '.fits',
        'delimiter': '_',
        'fits_ext': 1,
        'zfill': 0,
        'zp': 27.0,
    },
    'cfis_lsb-r': {
        'name': 'CFIS_LSB',
        'band': 'r',
        'vos': 'vos:cfis/tiles_LSB_DR5/',
        'suffix': '.r.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'ps-i': {
        'name': 'PS-DR3',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR3/tiles/',
        'suffix': '.i.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'wishes-z': {
        'name': 'WISHES',
        'band': 'z',
        'vos': 'vos:cfis/wishes_1/coadd/',
        'suffix': '.z.fits',
        'delimiter': '.',
        'fits_ext': 1,
        'zfill': 0,
        'zp': 27.0,
    },
    'ps-z': {
        'name': 'PSS.DR4',
        'band': 'ps-z',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
        'suffix': '.z.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
}

# define the bands to consider
considered_bands = ['whigs-g', 'cfis_lsb-r', 'ps-i']
# create a dictionary with the bands to consider
band_dict_incl = {key: band_dictionary.get(key) for key in considered_bands}

### pipeline options ###

# list of bands for which detections should be matched and cutouts combined
fuse_bands = ['whigs-g', 'cfis_lsb-r', 'ps-i']
# process all available bands?
process_all_available = True
# combine cutouts?
combine_cutouts = False
# aggregate cutouts to larger files?
aggregate_cutouts = False
# retrieve from the VOSpace and update the currently available tiles; takes some time to run
update_tiles = False
# build kd tree with updated tiles otherwise use the already saved tree
if update_tiles:
    build_new_kdtree = True
else:
    build_new_kdtree = False
# return the number of available tiles that are available in at least 5, 4, 3, 2, 1 bands
at_least_key = False
# show stats on currently available tiles, remember to update
show_tile_statistics = True
# define the minimum number of bands that should be available for a tile
band_constraint = 3
# print per tile availability
print_per_tile_availability = False
# how to treat the segmentation mask
segmentation_mode = None  # 'concatenate', 'multiply', None
# process only tiles with known dwarfs
process_only_known_dwarfs = False
# cutout objects?
cutout_objects = True
# accumulate all lsb cutouts to a single file?
accumulate_lsb_to_h5 = False
# cutout size
cutout_size = 256
# use original resolution?
use_full_resolution = True
# process only group tiles?
process_groups_only = False
# apply trained model on the data?
run_inference = True
# maximum on-sky separation to match detections across different bands in arcsec
maximum_match_separation = 10.0
# number of negative examples (nearest non-dwarf neighbors) for each positive example (dwarf)
negatives_per_positive = 5

### Multiprocessing constants
NUM_CORES = psutil.cpu_count(logical=False)  # Number of physical cores
PREFETCH_FACTOR = 3  # Number of prefetched tiles per core
### Inference
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
model_name = 'epoch43_train_loss_0.0187_valid_loss_0.0285.ckpt'

### paths ###
platform = 'CANFAR'  #'CANFAR' #'Narval'
if platform == 'CANFAR':
    root_dir_main = '/arc/home/heestersnick/dwarforge'
    root_dir_data = '/arc/projects/unions'
    unions_detection_directory = os.path.join(
        root_dir_data, 'catalogues/unions/GAaP_photometry/UNIONS2000'
    )
    redshift_class_catalog = os.path.join(
        root_dir_data, 'catalogues/redshifts/redshifts-2024-05-07.parquet'
    )
    download_directory = os.path.join(root_dir_data, 'ssl/data/raw/tiles/dwarforge')
    cutout_directory = os.path.join(root_dir_main, 'cutouts')
    os.makedirs(cutout_directory, exist_ok=True)
elif platform == 'LOCAL':
    root_dir_main = '/home/nick/astro/DwarForge'
    root_dir_data = '/home/nick/astro/DwarForge/data'
    download_directory = '/media/nick/Passport/UNIONS'
else:  # assume compute canada for now
    root_dir_main = '/home/heesters/projects/def-sfabbro/heesters/github/TileSlicer'
    root_dir_data_ashley = '/home/heesters/projects/def-sfabbro/a4ferrei/data'
    root_dir_data = '/home/heesters/projects/def-sfabbro/heesters/data/unions'
    unions_detection_directory = os.path.join(root_dir_data, 'catalogs/GAaP/UNIONS2000')
    redshift_class_catalog = os.path.join(
        root_dir_data, 'catalogs/labels/redshifts/redshifts-2024-05-07.parquet'
    )
    download_directory = os.path.join(root_dir_data, 'tiles')
    os.makedirs(download_directory, exist_ok=True)
    cutout_directory = os.path.join(root_dir_data, 'cutouts')
    os.makedirs(cutout_directory, exist_ok=True)

# paths
# define the root directory
main_directory = root_dir_main
data_directory = root_dir_data
table_directory = os.path.join(main_directory, 'tables')
os.makedirs(table_directory, exist_ok=True)
# define the path to the catalog containing known lenses
lens_catalog = os.path.join(table_directory, 'known_lenses.parquet')
# define the path to the master catalog that accumulates information about the cut out objects
catalog_master = os.path.join(table_directory, 'cutout_cat_master.parquet')
# define the path to the catalog containing known dwarf galaxies
dwarf_catalog = os.path.join(table_directory, 'all_known_dwarfs_v3_processed.csv')
# g,r,i tiles that are near massive galaxies
tiles_in_groups = os.path.join(table_directory, 'tiles_gri_group.csv')
# define path to file containing the processed h5 files
processed_file = os.path.join(table_directory, 'processed.txt')
# define catalog file
# catalog_file = 'all_known_dwarfs.csv'
# catalog_script = pd.read_csv(os.path.join(table_directory, catalog_file))
# define the keys for ra, dec, and id in the catalog
ra_key_script, dec_key_script, id_key_script = 'ra', 'dec', 'ID'
# define where the information about the currently available tiles should be saved
tile_info_directory = os.path.join(main_directory, 'tile_info/')
os.makedirs(tile_info_directory, exist_ok=True)
# define where figures should be saved
figure_directory = os.path.join(main_directory, 'figures/')
os.makedirs(figure_directory, exist_ok=True)
# define where the logs should be saved
log_directory = os.path.join(main_directory, 'logs/')
os.makedirs(log_directory, exist_ok=True)
# define location where to store the aggregated h5 files
aggregate_h5_directory = os.path.join(download_directory, 'combined_cutouts')
# define where the databases should be saved
database_directory = os.path.join(main_directory, 'databases')
os.makedirs(database_directory, exist_ok=True)
# define file path where to save the lsb cutouts
accumulated_lsb_path = os.path.join(cutout_directory, 'lsb_gri.h5')
# model directory
model_dir = os.path.join(main_directory, 'models')
os.makedirs(model_dir, exist_ok=True)
path_to_model = os.path.join(model_dir, model_name)


def query_availability(update, in_dict, at_least_key, show_stats, build_kdtree, tile_info_dir):
    """
    Gather information on the currently available tiles.

    Args:
        update (bool): update the available tiles
        in_dict (dict): band dictionary
        at_least_key (bool): print the number of tiles in at least (not exactly) 5, 4, ... bands
        show_stats (bool): show stats on the currently available tiles
        build_kdtree (bool): build a kd tree from the currently available tiles
        tile_info_dir (str): path to save the tile information

    Returns:
        TileAvailability: availability of the tiles
    """
    # update information on the currently available tiles
    if update:
        update_available_tiles(tile_info_dir, in_dict)
    # extract the tile numbers from the available tiles
    all_bands = extract_tile_numbers(load_available_tiles(tile_info_dir, in_dict), in_dict)
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict, at_least_key)
    # build the kd tree
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats()
    return availability, all_bands


def import_coordinates(coordinates, ra_key_default, dec_key_default, id_key_default):
    """
    Process coordinates provided from the command line.

    Args:
        coordinates (nested list): ra, dec coordinates
        ra_key_default (str): default right ascention key
        dec_key_default (str): default declination key
        id_key_default (str): default ID key

    Raises:
        ValueError: error if the number of coordinates is not even

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    coordinates = coordinates[0]
    if (len(coordinates) == 0) or len(coordinates) % 2 != 0:
        raise ValueError('Provide even number of coordinates.')

    ras, decs, ids = (
        coordinates[::2],
        coordinates[1::2],
        list(np.arange(1, len(coordinates) // 2 + 1)),
    )
    ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
    df_coordinates = pd.DataFrame({id_key: ids, ra_key: ras, dec_key: decs})

    formatted_coordinates = ' '.join([f'({ra}, {dec})' for ra, dec in zip(ras, decs)])
    logging.info(f'Coordinates received from the command line: {formatted_coordinates}')
    catalog = df_coordinates
    coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')
    return catalog, coord_c


def import_dataframe(
    dataframe_path, ra_key, dec_key, id_key, ra_key_default, dec_key_default, id_key_default
):
    """
    Process a DataFrame provided from the command line.

    Args:
        dataframe_path (str): path to the DataFrame
        ra_key (str): right ascention key
        dec_key (str): declination key
        id_key (str): ID key
        ra_key_default (str): default right ascention key
        dec_key_default (str): default declination key
        id_key_default (str): default ID key

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    logging.info('Dataframe received from command line.')
    catalog = pd.read_csv(dataframe_path)

    if ra_key is None or dec_key is None or id_key is None:
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default

    if (
        ra_key not in catalog.columns
        or dec_key not in catalog.columns
        or id_key not in catalog.columns
    ):
        logging.error(
            'One or more keys not found in the DataFrame. Please provide the correct keys '
            'for right ascention, declination and object ID \n'
            'if they are not equal to the default keys: ra, dec, ID.'
        )
        return None, None

    coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')

    return catalog, coord_c


def import_tiles(tiles, availability, band_constr):
    """
    Process tiles provided from the command line.

    Args:
        tiles (nested list): tile numbers
        availability (TileAvailability): instance of the TileAvailability class
        band_constr (int): minimum number of bands that should be available

    Raises:
        ValueError: provide two three digit numbers for each tile

    Returns:
        list: list of tiles that are available in r and at least two other bands
    """
    tiles = tiles[0]
    if (len(tiles) == 0) or len(tiles) % 2 != 0:
        raise ValueError('Provide two three digit numbers for each tile.')

    tile_list = [tuple(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    logging.info(f'Tiles received from command line: {tiles}')

    return [
        tile
        for tile in tile_list
        if 'r' in availability.get_availability(tile)[0]
        and len(availability.get_availability(tile)[1]) >= band_constr
    ]


def input_to_tile_list(
    availability,
    band_constr,
    coordinates=None,
    dataframe_path=None,
    tiles=None,
    ra_key=None,
    dec_key=None,
    id_key=None,
    tile_info_dir=None,
    ra_key_default='ra',
    dec_key_default='dec',
    id_key_default='ID',
):
    """
    Process the input to get a list of tiles that are available in r and at least two other bands.

    Args:
        availability (TileAvailability): instance of the TileAvailability class
        band_constr (int): minimum number of bands that should be available
        coordinates (nested list, optional): coordinates from the command line. Defaults to None.
        dataframe_path (str, optional): path to dataframe. Defaults to None.
        tiles (nested list, optional): tiles from the command line. Defaults to None.
        ra_key (str, optional): right ascention key. Defaults to None.
        dec_key (str_, optional): declination key. Defaults to None.
        id_key (str, optional): ID key. Defaults to None.
        tile_info_dir (str, optional): path to save the tile information. Defaults to None.
        ra_key_default (str, optional): default right ascention key. Defaults to 'ra'.
        dec_key_default (str, optional): default declination key. Defaults to 'dec'.
        id_key_default (str, optional): default ID key. Defaults to 'ID'.

    Returns:
        list: list of tiles that are available in r and at least two other bands
        catalog (dataframe): updated catalog with tile information
    """

    if coordinates is not None:
        catalog, coord_c = import_coordinates(
            coordinates, ra_key_default, dec_key_default, id_key_default
        )
    elif dataframe_path is not None:
        catalog, coord_c = import_dataframe(
            dataframe_path, ra_key, dec_key, id_key, ra_key_default, dec_key_default, id_key_default
        )
    elif tiles is not None:
        return import_tiles(tiles, availability, band_constr), None, None
    else:
        logging.info('No coordinates or DataFrame provided. Processing all available tiles..')
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        return None, None, None

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        availability, catalog, coord_c, tile_info_dir, band_constr
    )

    return unique_tiles, tiles_x_bands, catalog


def combine_h5_files(source_dir, destination_dir, objects_per_file=1000):
    """
    Combine individual tile H5 files into larger files with a specified number of objects per file.

    Args:
    source_dir (str): Directory containing the individual tile H5 files.
    destination_dir (str): Directory where the combined H5 files will be stored.
    objects_per_file (int): Number of objects to store in each combined file (default: 1000).

    Returns:
    None
    """
    os.makedirs(destination_dir, exist_ok=True)

    combined_data = {
        'images': [],
        'ra': [],
        'dec': [],
        'tile': [],
        'known_id': [],
        'mto_id': [],
        'label': [],
        'zspec': [],
    }
    file_counter = 1
    object_counter = 0

    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc='Processing files'):
            if file.endswith('_matched_cutouts.h5'):
                file_path = os.path.join(root, file)

                with h5py.File(file_path, 'r') as f:
                    num_objects = f['images'].shape[0]
                    for key in combined_data.keys():
                        combined_data[key].extend(f[key][:])

                    object_counter += num_objects

                if object_counter >= objects_per_file:
                    # Write combined data to a new file
                    output_file = os.path.join(destination_dir, f'combined_{file_counter:04d}.h5')
                    with h5py.File(output_file, 'w') as f_out:
                        for key, value in combined_data.items():
                            if key == 'known_id':
                                dt = h5py.special_dtype(vlen=str)
                                f_out.create_dataset(key, data=value, dtype=dt)
                            else:
                                f_out.create_dataset(key, data=value)

                        # Store band information
                        f_out.create_dataset(
                            'band_names',
                            data=np.array(['whigs-g', 'cfis_lsb-r', 'ps-i'], dtype='S'),
                        )

                    # Reset combined_data and counters
                    combined_data = {key: [] for key in combined_data}
                    object_counter = 0
                    file_counter += 1

    # Write any remaining data
    if object_counter > 0:
        output_file = os.path.join(destination_dir, f'combined_{file_counter:04d}.h5')
        with h5py.File(output_file, 'w') as f_out:
            for key, value in combined_data.items():
                if key == 'known_id':
                    dt = h5py.special_dtype(vlen=str)
                    f_out.create_dataset(key, data=value, dtype=dt)
                else:
                    f_out.create_dataset(key, data=value)

            # Store band information
            f_out.create_dataset(
                'band_names', data=np.array(['whigs-g', 'cfis_lsb-r', 'ps-i'], dtype='S')
            )

    logger.info(f'Completed. Created {file_counter} combined files.')


def make_cutouts_for_band(data_path, tile, cut_size, seg_mode):
    segmap = load_segmap(data_path)
    binned_data, binned_header = open_fits(data_path, fits_ext=0)
    path, extension = os.path.splitext(data_path)
    det_pattern = f'{path}*_det_params.parquet'
    det_path = glob.glob(det_pattern)
    mto_det = pd.read_parquet(det_path)
    cutouts, cutouts_seg = make_cutouts(
        binned_data,
        tile_str=tile_str(tile),
        df=mto_det,
        segmap=segmap,
        cutout_size=cut_size,
        seg_mode=seg_mode,
    )
    cutout_path = f'{path}_cutouts_single.h5'
    save_to_h5(
        stacked_cutout=cutouts,
        stacked_cutout_seg=cutouts_seg,
        object_df=mto_det,
        tile_numbers=tile,
        save_path=cutout_path,
        seg_mode=seg_mode,
    )


def save_cutouts_to_h5(
    tile,
    output_path,
    cutouts,
    segmaps,
    ras,
    decs,
    known_ids,
    labels,
    zspecs,
    band_names,
    seg_mode,
    unique_id,
    zoobot_pred,
):
    try:
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(output_path, 'w', libver='latest') as f:
            f.create_dataset('images', data=cutouts.astype(np.float32))
            if seg_mode == 'concatenate':
                f.create_dataset('segmaps', data=segmaps.astype(np.float32))
            f.create_dataset('ra', data=ras.astype(np.float32))
            f.create_dataset('dec', data=decs.astype(np.float32))
            f.create_dataset('tile', data=np.asarray(tile), dtype=np.int32)
            f.create_dataset('known_id', data=np.asarray(known_ids, dtype='S'), dtype=dt)
            f.create_dataset('label', data=labels.astype(np.float32))
            f.create_dataset('zspec', data=zspecs.astype(np.float32))
            f.create_dataset('band_names', data=np.array(band_names, dtype='S'))
            f.create_dataset('unique_id', data=unique_id.astype(np.int32))
            if zoobot_pred is not None:
                f.create_dataset('zoobot_pred', data=zoobot_pred.astype(np.float32))
        logger.debug(f'Created matched cutouts file: {output_path}')
    except Exception as e:
        logger.error(f'Error saving to H5 file {output_path}: {e}', exc_info=True)


def initialize_lsb_file(output_path, band_names, size):
    """Initialize the LSB accumulation file with empty datasets that can be extended."""
    if os.path.exists(output_path):
        try:
            with h5py.File(output_path, 'r') as f:
                # Verify file structure and dimensions
                if (
                    'images' in f
                    and 'band_names' in f
                    and len(f['band_names']) == len(band_names)
                    and f['images'].shape[2:] == (size, size)
                ):
                    logger.info(
                        f'Using existing LSB accumulation file: {output_path} '
                        f'with {f["images"].shape[0]} existing objects'
                    )
                    return
                else:
                    logger.warning(
                        f'Existing LSB file {output_path} has incorrect structure. Creating new file.'
                    )
        except Exception as e:
            logger.error(f'Error reading existing LSB file {output_path}: {e}. Creating new file.')

    # Create new file if we get here
    logger.info(f'Creating new LSB accumulation file: {output_path}')
    with h5py.File(output_path, 'w', libver='latest') as f:
        # Create empty datasets with maxshape=None for the first dimension
        f.create_dataset(
            'images',
            shape=(0, len(band_names), size, size),
            maxshape=(None, len(band_names), size, size),
            dtype=np.float32,
        )
        f.create_dataset(
            'segmaps',
            shape=(0, len(band_names), size, size),
            maxshape=(None, len(band_names), size, size),
            dtype=np.float32,
        )
        f.create_dataset('ra', shape=(0,), maxshape=(None,), dtype=np.float32)
        f.create_dataset('dec', shape=(0,), maxshape=(None,), dtype=np.float32)
        f.create_dataset('tile', shape=(0, 2), maxshape=(None, 2), dtype=np.int32)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('known_id', shape=(0,), maxshape=(None,), dtype=dt)
        f.create_dataset('label', shape=(0,), maxshape=(None,), dtype=np.float32)
        f.create_dataset('zspec', shape=(0,), maxshape=(None,), dtype=np.float32)
        f.create_dataset('band_names', data=np.array(band_names, dtype='S'))


def append_lsb_data(
    lsb_file_path, cutouts, segmaps, ras, decs, tile, known_ids, labels, zspecs, file_lock
):
    """
    Append LSB galaxy data to the accumulation file in a thread-safe way.
    Only appends data where labels == 1 (LSB galaxies only) and the object doesn't already exist.
    """
    # Get mask for LSB galaxies only
    lsb_mask = labels == 1
    if not np.any(lsb_mask):
        return

    with file_lock:
        with h5py.File(lsb_file_path, 'a', libver='latest') as f:
            # Get existing known_ids
            existing_ids = set([x.decode('utf-8') for x in f['known_id'][:]])

            # Create mask for new objects
            new_objects_mask = np.array([kid not in existing_ids for kid in known_ids[lsb_mask]])

            if not np.any(new_objects_mask):
                logger.info(f'No new LSB galaxies to add from tile {tile}')
                return

            # Get data to append (only new objects)
            cutouts_to_add = cutouts[lsb_mask][new_objects_mask]
            segmaps_to_add = segmaps[lsb_mask][new_objects_mask] if segmaps is not None else None
            ras_to_add = ras[lsb_mask][new_objects_mask]
            decs_to_add = decs[lsb_mask][new_objects_mask]
            known_ids_to_add = known_ids[lsb_mask][new_objects_mask]
            labels_to_add = labels[lsb_mask][new_objects_mask]
            zspecs_to_add = zspecs[lsb_mask][new_objects_mask]
            tiles_to_add = np.tile(tile, (np.sum(new_objects_mask), 1))

            # Get current and new sizes
            current_size = f['images'].shape[0]
            new_size = current_size + len(cutouts_to_add)

            # Resize all datasets
            f['images'].resize(new_size, axis=0)
            if segmaps_to_add is not None:
                f['segmaps'].resize(new_size, axis=0)
            f['ra'].resize(new_size, axis=0)
            f['dec'].resize(new_size, axis=0)
            f['tile'].resize(new_size, axis=0)
            f['known_id'].resize(new_size, axis=0)
            f['label'].resize(new_size, axis=0)
            f['zspec'].resize(new_size, axis=0)

            # Add new data
            f['images'][current_size:new_size] = cutouts_to_add
            if segmaps_to_add is not None:
                f['segmaps'][current_size:new_size] = segmaps_to_add
            f['ra'][current_size:new_size] = ras_to_add
            f['dec'][current_size:new_size] = decs_to_add
            f['tile'][current_size:new_size] = tiles_to_add
            f['known_id'][current_size:new_size] = known_ids_to_add
            f['label'][current_size:new_size] = labels_to_add
            f['zspec'][current_size:new_size] = zspecs_to_add

            logger.info(
                f'Added {len(cutouts_to_add)} new LSB galaxies from tile {tile} to accumulated file '
                f'(skipped {np.sum(~new_objects_mask)} existing objects)'
            )


def process_tile(
    tile,
    parent_dir,
    in_dict,
    band_names,
    cut_size,
    seg_mode,
    max_sep,
    n_neighbors=10,
    use_full_res=False,
    accumulate_lsb=False,
    lsb_file_path=None,
    file_lock=None,
    inference=False,
    model=None,
):
    try:
        logger.info(f'Matching and combining detections in tile {tile_str(tile)}')
        num_objects = 0
        tile_dir = f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
        if use_full_res:
            output_file = f'{tile_dir}_matched_cutouts_full_res_final.h5'
        else:
            output_file = f'{tile_dir}_matched_cutouts.h5'
        out_dir = os.path.join(parent_dir, tile_dir, 'gri')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, output_file)

        # Read data and coordinates for all bands
        band_data = {}
        valid_bands = []

        for band in band_names:
            try:
                start_read = time.time()
                data, header, segmap, ra, dec, df = read_band_data(
                    parent_dir, tile_dir, tile, band, in_dict, seg_mode, use_full_res
                )
                logger.debug(
                    f'Reading band data for tile {tile} took {time.time() - start_read:.2f} seconds.'
                )
                if data is not None and len(ra) > 0:
                    band_data[band] = {
                        'data': data,
                        'header': header,
                        'segmap': segmap,
                        'df': df,
                        'ra': ra,
                        'dec': dec,
                    }
                    valid_bands.append(band)
            except Exception as e:
                logger.warning(f'Failed to read band {band} for tile {tile}: {e}')
                continue

        # Check if we have enough valid bands to proceed
        if len(valid_bands) < 2:
            logger.warning(f'Insufficient valid bands ({len(valid_bands)}) for tile {tile}')
            return 0

        # Match objects across bands
        start_match = time.time()

        matched_df = match_coordinates_across_bands(band_data=band_data, max_sep=max_sep)

        if len(matched_df) == 0:
            logger.info(f'No matched objects found in tile {tile}')
            return 0

        final_ras, final_decs, labels, known_ids, zspecs, unique_id = (
            matched_df['ra'].values,
            matched_df['dec'].values,
            matched_df['lsb'].values,
            matched_df['ID_known'].values,
            matched_df['zspec'].values,
            matched_df['unique_id'].values,
        )
        logger.debug(f'Matching for tile {tile} took {time.time() - start_match:.2f} seconds.')

        # Create cutouts for matched objects
        num_objects = len(matched_df)
        logger.info(f'There are {num_objects} matched detections in tile {tile}.')

        if num_objects < n_neighbors:
            logger.info(f'Tile {tile}: num matches is smaller than {n_neighbors}.')

        final_cutouts = np.zeros(
            (num_objects, len(band_names), cut_size, cut_size), dtype=np.float32
        )
        final_segmaps = (
            np.zeros((num_objects, len(band_names), cut_size, cut_size), dtype=np.float32)
            if seg_mode is not None
            else None
        )
        try:
            for i, band in enumerate(band_names):
                if band not in valid_bands:
                    continue
                data = band_data[band]['data']
                header = band_data[band]['header']
                segmap = band_data[band]['segmap']

                cutouts, cutouts_seg = make_cutouts(
                    data=data,
                    header=header,
                    tile_str=tile_str(tile),
                    ra=final_ras,
                    dec=final_decs,
                    segmap=segmap,
                    cutout_size=cutout_size,
                    seg_mode=seg_mode,
                )

                final_cutouts[:, i, :, :] = cutouts
                if seg_mode is not None:
                    final_segmaps[:, i, :, :] = cutouts_seg
        except Exception as e:
            logger.error(f'Error in cutout creation: {e}.')
            return 0

        if inference:
            start_rgb = time.time()
            # make cutouts rgb-ready for inference
            cutouts_rgb = np.zeros_like(final_cutouts, dtype=np.float32)
            for i, cutout in enumerate(final_cutouts):
                cutouts_rgb[i] = preprocess_cutout(cutout, mode='vis')
            # make sure there are no nan values in the cutouts
            cutouts_rgb = np.nan_to_num(cutouts_rgb, nan=0.0)
            logger.debug(
                f'Creating RGB images for tile {tile_dir} took {time.time() - start_rgb:.2f} seconds.'
            )
            start_inf = time.time()
            # run inference
            zoobot_predictions = get_dwarf_predictions(
                model=model,
                data=cutouts_rgb,
                batch_size=cutouts_rgb.shape[0],
                dtype=DTYPE,
                device=DEVICE,
            )
            logger.debug(
                f'Inference on {cutouts_rgb.shape[0]} objects took {time.time() - start_inf:.2f} seconds.'
            )
            # add predictions to combined detection df
            matched_df['zoobot_pred'] = zoobot_predictions
        else:
            zoobot_predictions = None

        # Process LSB objects
        lsb_indices = np.where(labels == 1)[0]  # Assuming 1 indicates LSB objects
        if len(lsb_indices) > 0:
            logger.info(f'Found {len(lsb_indices)} LSB object(s) in tile {tile}.')
            # Create LSB mask
            lsb_mask = np.zeros_like(labels, dtype=bool)
            lsb_mask[lsb_indices] = True
            # Convert RA and Dec to Cartesian coordinates
            coords = SkyCoord(ra=final_ras, dec=final_decs, unit='deg')
            cartesian = coords.cartesian.xyz.value.T
            # Take minimum between n_neighbors+1 and the total number of matched objects to avoid errors
            total_points = len(cartesian)
            k = min(n_neighbors + 1, total_points)
            # Find nearest neighbors for LSB objects
            tree = cKDTree(cartesian)
            _, neighbor_indices = tree.query(cartesian[lsb_indices], k=k)

            # Create mask for neighbor labels
            neighbor_mask = np.zeros_like(labels, dtype=bool)
            for idx_list in neighbor_indices:
                neighbor_mask[idx_list[1:]] = True  # Skip first index (the LSB object itself)
            # Remove LSB objects from neighbors
            neighbor_mask = neighbor_mask & ~lsb_mask
            # Update labels:
            # - Keep 1 for LSB objects
            # - Set 0 for nearest neighbors
            # - Leave others as NaN
            updated_labels = np.full_like(labels, np.nan, dtype=np.float32)
            # Apply masks
            updated_labels[lsb_mask] = 1  # LSB objects
            updated_labels[neighbor_mask] = 0  # Nearest neighbors

            logger.debug(
                f'Tile: {tile}, updated labels: {np.sum(updated_labels == 1)} LSB objects, '
                f'{np.sum(updated_labels == 0)} nearest neighbors, '
                f'{np.sum(np.isnan(updated_labels))} unknown objects, '
                f'lsb indices: {lsb_indices}'
            )
        else:
            logger.debug(f'No LSB objects found in tile {tile}.')
            updated_labels = labels

        # Save all cutouts with updated labels
        save_cutouts_to_h5(
            tile=tile,
            output_path=output_path,
            cutouts=final_cutouts,
            segmaps=final_segmaps,
            ras=final_ras,
            decs=final_decs,
            known_ids=known_ids,
            labels=updated_labels,
            zspecs=zspecs,
            band_names=band_names,
            seg_mode=seg_mode,
            unique_id=unique_id,
            zoobot_pred=zoobot_predictions,
        )
        # save cross-matched detection dataframe
        matched_df.to_parquet(
            os.path.join(out_dir, f'{tile_dir}_matched_detections.parquet'), index=False
        )

        if accumulate_lsb and lsb_file_path and file_lock:
            append_lsb_data(
                lsb_file_path=lsb_file_path,
                cutouts=final_cutouts,
                segmaps=final_segmaps,
                ras=final_ras,
                decs=final_decs,
                tile=tile,
                known_ids=known_ids,
                labels=updated_labels,
                zspecs=zspecs,
                file_lock=file_lock,
            )

        return num_objects
    except Exception as e:
        logger.error(
            f'An error occurred during cutout combination of tile {tile}, num matches: {num_objects}: {e}'
        )


def fuse_cutouts_parallel(
    parent_dir,
    tiles,
    in_dict,
    band_names=['whigs-g', 'cfis_lsb-r', 'ps-i'],
    num_processes=None,
    cut_objects=False,
    cut_size=256,
    seg_mode='concatenate',
    max_sep=15.0,
    n_neighbors=10,
):
    logger.info(f'Starting to fuse cutouts for {len(tiles)} tiles in the bands: {band_names}')

    num_matches = []
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks and create a dictionary to track futures
        future_to_tile = {
            executor.submit(
                process_tile,
                tile,
                parent_dir,
                in_dict,
                band_names,
                cut_size,
                seg_mode,
                max_sep,
                n_neighbors,
            ): tile
            for tile in tiles
        }

        # Process completed futures with a progress bar
        for future in tqdm(as_completed(future_to_tile), total=len(tiles)):
            tile = future_to_tile[future]
            try:
                result = future.result()  # This will raise an exception if the task failed
                if result is not None:
                    num_matches.append(result)
            except Exception as e:
                logger.error(f'Error processing tile {tile}: {e}')

    # Calculate and print the mean
    if num_matches:
        mean_matches = np.mean(num_matches)
        logger.info(f'Mean matches across all processed tiles: {mean_matches}')
    else:
        logger.warning('No results were collected, unable to calculate mean.')


def process_worker(
    process_queue,
    database,
    band_dict,
    required_bands,
    download_dir,
    db_lock,
    all_downloads_complete,
    shutdown_flag,
    queue_lock,
    processed_in_current_run,
    cut_size=256,
    seg_mode='concatenate',
    max_sep=15.0,
    n_neighbors=10,
    accumulate_lsb=False,
    lsb_h5_path=None,
    lsb_h5_lock=None,
    inference=False,
    model_state_dict=None,
    hparams=None,
):
    """
    Worker that processes downloaded full resolution data.

    Args:
        process_queue: Queue containing (tile, paths_dict) tuples
        database: SQLite database path
        band_dict: Dictionary with band information
        required_bands: Set of bands that must be present for processing
        download_dir: Directory containing downloaded files
        db_lock: Lock for database access
        all_downloads_complete: Event indicating all downloads are finished
        shutdown_flag: Flag to signal shutdown
        queue_lock: Lock for queue access
        processed_in_current_run: Dict tracking processed tiles per band
        cut_size: Size of cutouts in pixels
        seg_mode: Segmentation mode
        max_sep: Maximum separation for matching
        n_neighbors: Number of neighbors for training samples
        accumulate_lsb: Accumulate all lsb cutouts to a single h5 file
        lsb_h5_path: Path to lsb h5 file
        lsb_h5_lock: Multiprocessing lock to avoid race conditions
        model_state_dict: model state dictionary
        hparams: model hyperparameters
    """
    worker_id = os.getpid()
    logger.debug(f'Processing worker {worker_id} started')

    if inference:
        model = ZooBot_lightning(**hparams)
        model.load_state_dict(model_state_dict)
        model.freeze()
        model.eval()
        model = model.to(DEVICE)
    else:
        model = None

    while not (
        shutdown_flag.is_set() or (all_downloads_complete.is_set() and process_queue.empty())
    ):
        try:
            # Get data from queue with timeout
            try:
                tile, paths_dict = process_queue.get(timeout=1)
                logger.info(f'Processing worker {worker_id} received tile {tile}')
            except queue.Empty:
                continue

            if tile is None:  # Check for sentinel
                logger.info(f'Process worker {worker_id} received sentinel, exiting')
                break

            # Verify we have all required bands
            if not all(band in paths_dict for band in required_bands):
                logger.error(
                    f'Missing required bands for tile {tile}. Have: {set(paths_dict.keys())}, Need: {required_bands}'
                )
                continue

            # Initialize processing status
            cutout_info = {
                'tile': tile,
                'start_time': time.time(),
                'status': 'processing',
            }

            try:
                # Update status for all bands
                for band in required_bands:
                    cutout_info['band'] = band
                    update_cutout_info(database, cutout_info, db_lock)

                logger.info(f'Processing tile {tile}..')

                process_start = time.time()
                result = process_tile(
                    tile=tile,
                    parent_dir=download_dir,
                    in_dict=band_dict,
                    band_names=required_bands,
                    cut_size=cut_size,
                    seg_mode=seg_mode,
                    max_sep=max_sep,
                    n_neighbors=n_neighbors,
                    use_full_res=True,
                    accumulate_lsb=accumulate_lsb,
                    lsb_file_path=lsb_h5_path,
                    file_lock=lsb_h5_lock,
                    inference=inference,
                    model=model,
                )
                logger.debug(
                    f'Finished processing tile {tile} in {time.time() - process_start:.2f}s.'
                )

                # Update status for all bands
                if result is not None:
                    cutout_info.update(
                        {'status': 'completed', 'cutout_count': result, 'end_time': time.time()}
                    )
                else:
                    cutout_info.update(
                        {
                            'status': 'failed',
                            'error_message': 'No cutouts created',
                            'end_time': time.time(),
                        }
                    )

            except Exception as e:
                cutout_info.update(
                    {'status': 'failed', 'error_message': str(e), 'end_time': time.time()}
                )
                logger.error(f'Error processing tile {tile}: {str(e)}')

            finally:
                # Update final status for all bands
                for band in required_bands:
                    cutout_info['band'] = band
                    update_cutout_info(database, cutout_info, db_lock)

                # Clean up downloaded files
                for band, paths in paths_dict.items():
                    try:
                        if os.path.exists(paths['final_path']):
                            os.remove(paths['final_path'])
                            logger.debug(f'Cleaned up file for tile {tile} band {band}')
                    except Exception as cleanup_error:
                        logger.error(
                            f'Error during cleanup of tile {tile} band {band}: {cleanup_error}'
                        )

                # Update processed count
                with queue_lock:
                    for band in required_bands:
                        processed_in_current_run[band] += 1

                process_queue.task_done()

        except Exception as e:
            logger.error(f'Unexpected error in process worker {worker_id}: {str(e)}')
            if shutdown_flag.is_set():
                break

    logger.info(f'Process worker {worker_id} exiting')


def main(
    update,
    band_dict,
    download_dir,
    at_least,
    show_tile_stats,
    build_kdtree,
    tile_info_dir,
    band_constr,
    coordinates,
    dataframe_path,
    tiles,
    ra_key,
    ra_key_default,
    dec_key,
    dec_key_default,
    id_key,
    id_key_default,
    bands_to_combine,
    num_processes,
    comb_cutouts,
    aggr_cutouts,
    aggr_dir,
    dwarfs_only,
    seg_mode,
    dwarf_cat,
    cut_objects,
    cut_size,
    max_sep,
    n_neighbors,
    database,
    catalog_path,
    process_all_avail,
    use_full_res=False,
    accumulate_lsb=False,
    lsb_h5_path=None,
    process_groups=False,
    group_tiles=None,
    inference=False,
    model_path=None,
):
    try:
        if inference:
            model_state_dict, hparams = load_model(model_path)
        else:
            model_state_dict, hparams = None, None

        # query availability of the tiles
        availability, all_tiles = query_availability(
            update, band_dict, at_least, show_tile_stats, build_kdtree, tile_info_dir
        )

        # read the input catalog
        try:
            input_catalog = pd.read_csv(catalog_path)
        except FileNotFoundError:
            logger.error(f'File not found: {catalog_path}')
            raise FileNotFoundError

        _, tiles_x_bands, _ = input_to_tile_list(
            availability,
            band_constr,
            coordinates,
            dataframe_path,
            tiles,
            ra_key,
            dec_key,
            id_key,
            tile_info_dir,
            ra_key_default,
            dec_key_default,
            id_key_default,
        )

        if tiles_x_bands is not None:
            selected_all_tiles = [
                [tile for tile in band_tiles if tile in tiles_x_bands] for band_tiles in all_tiles
            ]
            availability = TileAvailability(selected_all_tiles, band_dict, at_least_key)
        try:
            if use_full_res:
                # Initialize the database for progress tracking
                init_cutouts_db(database)
                # Initialize shutdown manager
                killer = GracefulKiller()
                # Initialize job queue and result queue
                manager = Manager()
                shutdown_flag = manager.Event()
                download_queue = manager.Queue()
                process_queue = manager.Queue(maxsize=num_processes * PREFETCH_FACTOR)
                db_lock = manager.Lock()
                queue_lock = manager.Lock()
                downloaded_bands = manager.dict()

                if accumulate_lsb:
                    lsb_h5_lock = manager.Lock()
                    initialize_lsb_file(lsb_h5_path, bands_to_combine, cut_size)
                else:
                    lsb_h5_lock = None

                # dictionary to keep track of processed tiles per band in current run
                processed_in_current_run = manager.dict({band: 0 for band in band_dict.keys()})

                unprocessed_jobs = get_unprocessed_jobs(
                    database=database,
                    tile_availability=availability,
                    dwarf_df=input_catalog,
                    in_dict=band_dict,
                    process_band=bands_to_combine,
                    process_all_bands=process_all_avail,
                    only_known_dwarfs=dwarfs_only,
                    process_type='cutouts',
                    process_groups=process_groups,
                    group_tiles=group_tiles,
                )

                unprocessed_jobs_at_start = {band: 0 for band in band_dict.keys()}

                for job in unprocessed_jobs:
                    logger.info(f'Job: {job}')
                    unprocessed_jobs_at_start[job[1]] += 1
                    download_queue.put(job)

                logger.info(f'Number of unprocessed jobs: {unprocessed_jobs_at_start}')

                # Create an event to signal when all downloads are complete
                all_downloads_complete = multiprocessing.Event()
                # Set number of download threads
                num_download_threads = min(PREFETCH_FACTOR * num_processes, len(unprocessed_jobs))
                logger.info(f'Using {num_download_threads} download threads.')

                # Start download threads
                download_threads = []
                for _ in range(num_download_threads):
                    t = threading.Thread(
                        target=download_worker,
                        args=(
                            database,
                            download_queue,
                            process_queue,
                            set(bands_to_combine),
                            band_dictionary,
                            download_dir,
                            db_lock,
                            shutdown_flag,
                            queue_lock,
                            processed_in_current_run,
                            downloaded_bands,
                        ),
                    )
                    t.daemon = True
                    t.start()
                    download_threads.append(t)

                # Start processing workers
                processes = []
                for _ in range(num_processes):
                    p = multiprocessing.Process(
                        target=process_worker,
                        args=(
                            process_queue,
                            database,
                            band_dict,
                            bands_to_combine,
                            download_dir,
                            db_lock,
                            all_downloads_complete,
                            shutdown_flag,
                            queue_lock,
                            processed_in_current_run,
                            cut_size,
                            seg_mode,
                            max_sep,
                            n_neighbors,
                            accumulate_lsb,
                            lsb_h5_path,
                            lsb_h5_lock,
                            inference,
                            model_state_dict,
                            hparams,
                        ),
                    )
                    p.start()
                    processes.append(p)
                    # process_ids.append(p.pid)

                all_jobs_completed = False
                while not killer.kill_now and not shutdown_flag.is_set():
                    progress_results = get_progress_summary(
                        database,
                        availability,
                        bands_to_combine,
                        unprocessed_jobs_at_start,
                        processed_in_current_run,
                        process_type='cutouts',
                    )

                    # Collect all log messages
                    log_messages = []
                    for band in bands_to_combine:
                        stats = progress_results[band]
                        log_messages.append(f'\nProgress for band {band}:')
                        log_messages.append(
                            f'  Overall: {stats["total_completed"]}/{stats["total_available"]} completed, {stats["total_failed"]} failed, {stats["download_failed"]} download failed, {stats["mostly_zeros"]} mostly_zeros'
                        )
                        log_messages.append(
                            f'  Current run: {stats["current_run_processed"]} processed, {stats["in_progress"]} in progress, {stats["downloaded"]} downloaded, {stats["downloading"]} downloading, {stats["remaining_in_run"]} remaining'
                        )

                    # Log all messages together
                    logger.info('\n'.join(log_messages))

                    if all(
                        progress_results[band]['in_progress'] == 0
                        and progress_results[band]['remaining_in_run'] == 0
                        for band in bands_to_combine
                    ):
                        if not all_jobs_completed:
                            logger.info('All jobs completed. Initiating shutdown.')
                            all_jobs_completed = True
                            all_downloads_complete.set()

                            # Add sentinel values to signal the end of processing
                            for _ in range(num_processes):
                                process_queue.put((None, None))

                        # Check if all worker processes have exited
                        if all(not p.is_alive() for p in processes):
                            logger.info('All worker processes have exited. Ending main loop.')
                            break

                    time.sleep(10)  # Check every 10 seconds

            else:
                if dwarfs_only:
                    tiles_to_process = get_dwarf_tile_list(
                        dwarf_cat, in_dict=band_dict, bands=bands_to_combine
                    )
                else:
                    tiles_to_process = availability.get_tiles_for_bands(bands_to_combine)

                # tiles_to_process = [
                #     (np.int64(216), np.int64(306)),
                #     (np.int64(220), np.int64(253)),
                #     (np.int64(225), np.int64(270)),
                #     (np.int64(232), np.int64(250)),
                #     (np.int64(250), np.int64(269)),
                #     (np.int64(314), np.int64(248)),
                #     (np.int64(317), np.int64(280)),
                #     (np.int64(323), np.int64(244)),
                #     (np.int64(328), np.int64(244)),
                #     (np.int64(375), np.int64(267)),
                #     (np.int64(389), np.int64(269)),
                #     (np.int64(391), np.int64(256)),
                #     (np.int64(428), np.int64(249)),
                #     (np.int64(439), np.int64(243)),
                #     (np.int64(337), np.int64(243)),
                # ]

                if comb_cutouts:
                    fuse_cutouts_parallel(
                        download_dir,
                        tiles_to_process,
                        band_dict,
                        band_names=bands_to_combine,
                        num_processes=num_processes,
                        cut_objects=cut_objects,
                        cut_size=cut_size,
                        seg_mode=seg_mode,
                        max_sep=max_sep,
                        n_neighbors=n_neighbors,
                    )

        except Exception as e:
            logger.error(f'There was an error getting the tile numbers: {e}.')

        if aggr_cutouts:
            combine_h5_files(download_dir, aggr_dir, objects_per_file=1000)

    except Exception as e:
        logger.error(f'An error occurred in the main process: {str(e)}')
    finally:
        if use_full_res:
            # Cleanup for full resolution mode
            shutdown_flag.set()
            all_downloads_complete.set()

            # Clean up download threads
            for _ in range(num_download_threads):
                download_queue.put((None, None))
            for t in download_threads:
                t.join(timeout=10)

            # Clean up processing workers
            for p in processes:
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning(
                        f'Process {p.pid} did not terminate gracefully. Forcing termination.'
                    )
                    p.terminate()
                    p.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    print('Starting script...')
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        action='append',
        metavar=('ra', 'dec'),
        help='list of pairs of coordinates to make cutouts from',
    )
    parser.add_argument('--dataframe', type=str, help='path to a CSV file containing the DataFrame')
    parser.add_argument('--ra_key', type=str, help='right ascension key in the DataFrame')
    parser.add_argument('--dec_key', type=str, help='declination key in the DataFrame')
    parser.add_argument('--id_key', type=str, help='id key in the DataFrame')
    parser.add_argument(
        '--tiles',
        type=int,
        nargs='+',
        action='append',
        metavar=('tile'),
        help='list of tiles to make cutouts from',
    )
    parser.add_argument(
        '--processing_cores',
        type=int,
        default=15,
        help='Number of cores to use for processing (default: 15)',
    )
    parser.add_argument(
        '--database',
        type=str,
        default='progress.db',
        help='Database file to keep track of progress (default: cutout_gen.db)',
    )

    args = parser.parse_args()

    # define the arguments for the main function

    arg_dict_main = {
        'update': update_tiles,
        'band_dict': band_dict_incl,
        'at_least': at_least_key,
        'show_tile_stats': show_tile_statistics,
        'build_kdtree': build_new_kdtree,
        'tile_info_dir': tile_info_directory,
        'coordinates': args.coordinates,
        'dataframe_path': args.dataframe,
        'tiles': args.tiles,
        'ra_key': args.ra_key,
        'ra_key_default': ra_key_script,
        'dec_key': args.dec_key,
        'dec_key_default': dec_key_script,
        'id_key': args.id_key,
        'id_key_default': id_key_script,
        'band_constr': band_constraint,
        'download_dir': download_directory,
        'bands_to_combine': fuse_bands,
        'num_processes': args.processing_cores,
        'comb_cutouts': combine_cutouts,
        'aggr_cutouts': aggregate_cutouts,
        'aggr_dir': aggregate_h5_directory,
        'dwarfs_only': process_only_known_dwarfs,
        'seg_mode': segmentation_mode,
        'dwarf_cat': dwarf_catalog,
        'cut_objects': cutout_objects,
        'cut_size': cutout_size,
        'max_sep': maximum_match_separation,
        'n_neighbors': negatives_per_positive,
        'database': os.path.join(database_directory, args.database),
        'catalog_path': dwarf_catalog,
        'process_all_avail': process_all_available,
        'use_full_res': use_full_resolution,
        'accumulate_lsb': accumulate_lsb_to_h5,
        'lsb_h5_path': accumulated_lsb_path,
        'process_groups': process_groups_only,
        'group_tiles': tiles_in_groups,
        'inference': run_inference,
        'model_path': path_to_model,
    }

    start = time.time()
    main(**arg_dict_main)
    end = time.time()
    elapsed = end - start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = (
        np.float32(elapsed_string.split(':')[0]),
        np.float32(elapsed_string.split(':')[1]),
        np.float32(elapsed_string.split(':')[2]),
    )
    logger.info(
        f'Done! Execution took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds.'
    )
