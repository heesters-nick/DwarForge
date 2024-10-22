import argparse
import glob
import logging
import multiprocessing
import os  # noqa: E402
import time
import traceback
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from tqdm import tqdm

from logging_setup import setup_logger

setup_logger(
    log_dir='./logs',
    name='fuse_cutouts_dwarfs',
    logging_level=logging.INFO,
)
logger = logging.getLogger()

import psutil  # noqa: E402
from vos import Client  # noqa: E402

from kd_tree import build_tree  # noqa: E402
from postprocess import (  # noqa: E402
    load_segmap,
    make_cutouts,
    match_coordinates_across_bands,
    read_band_data,
    save_to_h5,
)
from tile_cutter import tile_finder  # noqa: E402
from utils import (  # noqa: E402
    TileAvailability,
    extract_tile_numbers,
    get_dwarf_tile_list,
    load_available_tiles,
    open_fits,
    tile_str,
    update_available_tiles,
)

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
        'suffix': '.u',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'whigs-g': {
        'name': 'calexp-CFIS',
        'band': 'g',
        'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme/',
        'suffix': '',
        'delimiter': '_',
        'fits_ext': 1,
        'zfill': 0,
        'zp': 27.0,
    },
    'cfis_lsb-r': {
        'name': 'CFIS_LSB',
        'band': 'r',
        'vos': 'vos:cfis/tiles_LSB_DR5/',
        'suffix': '.r',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'ps-i': {
        'name': 'PS-DR3',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR3/tiles/',
        'suffix': '.i',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'wishes-z': {
        'name': 'WISHES',
        'band': 'z',
        'vos': 'vos:cfis/wishes_1/coadd/',
        'suffix': '.z',
        'delimiter': '.',
        'fits_ext': 1,
        'zfill': 0,
        'zp': 27.0,
    },
    'ps-z': {
        'name': 'PSS.DR4',
        'band': 'ps-z',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
        'suffix': '.z',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
}

# define the bands to consider
considered_bands = ['cfis-u', 'whigs-g', 'cfis_lsb-r', 'ps-i', 'wishes-z']
# create a dictionary with the bands to consider
band_dict_incl = {key: band_dictionary.get(key) for key in considered_bands}

### pipeline options ###

# list of bands for which detections should be matched and cutouts combined
fuse_bands = ['whigs-g', 'cfis_lsb-r', 'ps-i']
# combine cutouts?
combine_cutouts = True
# download fused cutouts?
download_cutouts = False
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
band_constraint = 1
# print per tile availability
print_per_tile_availability = False
# how to treat the segmentation mask
segmentation_mode = None  # 'concatenate', 'multiply', None
# process only tiles with known dwarfs
process_only_known_dwarfs = True
# cutout objects?
cutout_objects = True
# cutout size
cutout_size = 64
# maximum on-sky separation to match detections across different bands in arcsec
maximum_match_separation = 10.0
# number of negative examples (nearest non-dwarf neighbors) for each positive example (dwarf)
negatives_per_positive = 20

### Multiprocessing constants
NUM_CORES = psutil.cpu_count(logical=False)  # Number of physical cores
PREFETCH_FACTOR = 2  # Number of prefetched tiles per core

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
# define directory where cutouts should be saved locally
local_download_directory = '/home/nick/astro/DwarForge/data'
# local_download_directory = '/media/nick/Passport/UNIONS'
# define location where to store the aggregated h5 files
aggregate_h5_directory = os.path.join(download_directory, 'combined_cutouts')


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
    tile, output_path, cutouts, segmaps, ras, decs, known_ids, labels, zspecs, band_names, seg_mode
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
        logger.debug(f'Created matched cutouts file: {output_path}')
    except Exception as e:
        logger.error(f'Error saving to H5 file {output_path}: {e}', exc_info=True)


def process_tile(
    tile, parent_dir, in_dict, band_names, cut_size, seg_mode, max_sep, n_neighbors=10
):
    try:
        logger.debug(f'Matching and combining detections in tile {tile_str(tile)}')
        tile_dir = f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
        output_file = f'{tile_dir}_matched_cutouts.h5'
        output_file_train = f'{tile_dir}_training_cutouts.h5'
        out_dir = os.path.join(parent_dir, tile_dir, 'gri')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, output_file)
        output_path_train = os.path.join(out_dir, output_file_train)

        # Read data and coordinates for all bands
        band_data = {}
        for band in band_names:
            try:
                data, header, segmap, ra, dec, df = read_band_data(
                    parent_dir, tile_dir, tile, band, in_dict, seg_mode
                )
            except Exception as e:
                logger.error(f'Tile {tile}, error reading band data: {e}')
                return
            band_data[band] = {
                'data': data,
                'header': header,
                'segmap': segmap,
                'df': df,
                'ra': ra,
                'dec': dec,
            }
        try:
            # Match objects across bands
            multi_band_objects, object_indices, final_ras, final_decs, known_ids, labels, zspecs = (
                match_coordinates_across_bands(band_data, max_sep, tile)
            )
        except Exception as e:
            logger.error(f'Tile: {tile}: error in match_coordinates_across_bands: {e}.')
            logger.error(f'Tile {tile}: Unhandled error in match_coordinates_across_bands: {e}')
            logger.error(f'Tile {tile}: Error type: {type(e).__name__}')
            logger.error(f'Tile {tile}: Error args: {e.args}')
            logger.error(f'Tile {tile}: Traceback: {traceback.format_exc()}')
            raise

        # Create cutouts for matched objects
        num_objects = len(multi_band_objects)
        if num_objects < n_neighbors:
            logger.info(f'Tile {tile}: num matches is smaller than {n_neighbors}.')
        logger.debug(f'There are {num_objects} matched detections in tile {tile}.')
        final_cutouts = np.zeros(
            (num_objects, len(band_names), cut_size, cut_size), dtype=np.float32
        )
        final_segmaps = (
            np.zeros((num_objects, len(band_names), cut_size, cut_size), dtype=np.float32)
            if seg_mode is not None
            else None
        )

        for i, band in enumerate(band_names):
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

        save_cutouts_to_h5(
            tile=tile,
            output_path=output_path,
            cutouts=final_cutouts,
            segmaps=final_segmaps,
            ras=final_ras,
            decs=final_decs,
            known_ids=known_ids,
            labels=labels,
            zspecs=zspecs,
            band_names=band_names,
            seg_mode=seg_mode,
        )

        # Process LSB objects
        lsb_indices = np.where(labels == 1)[0]  # Assuming 1 indicates LSB objects
        if len(lsb_indices) > 0:
            logger.debug(f'Found {len(lsb_indices)} LSB object(s) in tile {tile}.')

            # Convert RA and Dec to Cartesian coordinates
            coords = SkyCoord(ra=final_ras, dec=final_decs, unit='deg')
            cartesian = coords.cartesian.xyz.value.T
            # Take minimum between n_neighbors+1 and the total number of matched objects to avoid errors
            total_points = len(cartesian)
            k = min(n_neighbors + 1, total_points)
            # Find nearest neighbors for LSB objects
            tree = cKDTree(cartesian)
            _, neighbor_indices = tree.query(cartesian[lsb_indices], k=k)

            # Flatten and remove duplicates
            all_indices = np.unique(neighbor_indices.flatten())

            # Create cutouts for LSB objects and their neighbors
            train_cutouts = final_cutouts[all_indices]
            train_segmaps = final_segmaps[all_indices] if final_segmaps is not None else None
            train_ras = final_ras[all_indices]
            train_decs = final_decs[all_indices]
            train_known_ids = known_ids[all_indices]
            train_labels = labels[all_indices]
            train_zspecs = zspecs[all_indices]

            # Save LSB cutouts
            save_cutouts_to_h5(
                tile=tile,
                output_path=output_path_train,
                cutouts=train_cutouts,
                segmaps=train_segmaps,
                ras=train_ras,
                decs=train_decs,
                known_ids=train_known_ids,
                labels=train_labels,
                zspecs=train_zspecs,
                band_names=band_names,
                seg_mode=seg_mode,
            )
            logger.debug(
                f'Saved {len(all_indices)} LSB objects and neighbors to {output_file_train}'
            )
        else:
            logger.debug(f'No LSB objects found in tile {tile}.')
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
    parent_dir_destination=None,
    download_file=False,
    num_processes=None,
    cut_objects=False,
    cut_size=64,
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
    local_download_dir,
    dl_cutouts,
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
):
    try:
        # query availability of the tiles
        availability, all_tiles = query_availability(
            update, band_dict, at_least, show_tile_stats, build_kdtree, tile_info_dir
        )

        # get the list of tiles for which r and at least two more bands are available
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
        except Exception as e:
            logger.error(f'There was an error getting the tile numbers: {e}.')

        # with open('tiles_gri.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for item in tiles_to_process:
        #         writer.writerow([item])

        logger.info(
            f'There are {len(tiles_to_process)} bands that are available in {bands_to_combine}'
        )

        if comb_cutouts:
            fuse_cutouts_parallel(
                download_dir,
                tiles_to_process,
                band_dict,
                band_names=bands_to_combine,
                parent_dir_destination=local_download_dir,
                download_file=dl_cutouts,
                num_processes=num_processes,
                cut_objects=cut_objects,
                cut_size=cut_size,
                seg_mode=seg_mode,
                max_sep=max_sep,
                n_neighbors=n_neighbors,
            )
        if aggr_cutouts:
            combine_h5_files(download_dir, aggr_dir, objects_per_file=1000)

    except Exception as e:
        logger.error(f'An error occurred in the main process: {str(e)}')


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
        default=16,
        help='Number of cores to use for processing (default: 15)',
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
        'local_download_dir': local_download_directory,
        'dl_cutouts': download_cutouts,
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
