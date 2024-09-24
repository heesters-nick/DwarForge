import argparse
import logging
import multiprocessing
import os  # noqa: E402
import subprocess
import time
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from logging_setup import setup_logger

logger = setup_logger(
    log_dir='./logs',
    name='fuse_cutouts',
    logging_level=logging.INFO,
)
logger = logging.getLogger()

import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from vos import Client  # noqa: E402

from kd_tree import build_tree  # noqa: E402
from postprocess import match_coordinates  # noqa: E402
from tile_cutter import read_h5, tile_finder  # noqa: E402
from utils import (  # noqa: E402
    TileAvailability,
    extract_tile_numbers,
    load_available_tiles,
    transform_path,
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
        'name': 'PSS.DR4',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
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
# download fused cutouts?
download_cutouts = True
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
dwarf_catalog = os.path.join(table_directory, 'all_known_dwarfs_v2_processed.csv')
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


def process_tile(tile, parent_dir, in_dict, band_names, parent_dir_destination, download_file):
    try:
        all_band_data = {}
        r_band_data = {}

        tile_dir = f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}'
        # set outpath
        output_file = f'{tile_dir}_matched_cutouts.h5'
        out_dir = os.path.join(parent_dir, tile_dir, 'gri')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, output_file)

        # skip if it already exists
        if not os.path.isfile(output_path):
            # Read data for all bands
            for band in band_names:
                zfill = in_dict[band]['zfill']
                file_prefix = in_dict[band]['name']
                delimiter = in_dict[band]['delimiter']
                suffix = in_dict[band]['suffix']
                num1, num2 = str(tile[0]).zfill(zfill), str(tile[1]).zfill(zfill)
                cutout_file = f'{file_prefix}{delimiter}{num1}{delimiter}{num2}{suffix}_cutouts.h5'
                cutout_path = os.path.join(parent_dir, tile_dir, band, cutout_file)
                cutout_dict = read_h5(cutout_path)

                if band == 'cfis_lsb-r':
                    r_band_data = cutout_dict.copy()
                all_band_data[band] = {
                    'cutouts': cutout_dict['images'],
                    'ra': cutout_dict['ra'],
                    'dec': cutout_dict['dec'],
                }

            # Use r-band as reference
            r_band = 'cfis_lsb-r'
            r_band_coords = SkyCoord(
                all_band_data[r_band]['ra'], all_band_data[r_band]['dec'], unit=u.deg
            )

            # Match cutouts for each non-r band to r-band
            matched_indices = {r_band: np.arange(len(all_band_data[r_band]['ra']))}
            for band in band_names:
                if band != r_band:
                    target_coords = SkyCoord(
                        all_band_data[band]['ra'], all_band_data[band]['dec'], unit=u.deg
                    )
                    matched_r_indices, matched_target_indices = match_coordinates(
                        r_band_coords, target_coords
                    )
                    matched_indices[band] = (matched_r_indices, matched_target_indices)

            # Find common matches across all bands
            common_r_indices = matched_indices[r_band]
            for band in band_names:
                if band != r_band:
                    common_r_indices = np.intersect1d(common_r_indices, matched_indices[band][0])

            # Update matched indices for all bands based on common matches
            final_indices = {r_band: common_r_indices}
            for band in band_names:
                if band != r_band:
                    mask = np.isin(matched_indices[band][0], common_r_indices)
                    final_indices[band] = matched_indices[band][1][mask]

            num_matched = len(common_r_indices)
            logger.info(f'Tile {tile}: number of matched cutouts: {num_matched}')

            # Create the final array with shape (num_cutouts, num_bands, cutout_size, cutout_size)
            cutout_size = all_band_data[r_band]['cutouts'].shape[1:]
            final_cutouts = np.zeros((num_matched, len(band_names), *cutout_size), dtype=np.float32)

            # Populate the final_cutouts array
            for i, band in enumerate(band_names):
                final_cutouts[:, i] = all_band_data[band]['cutouts'][final_indices[band]]

            # write fused cutouts to new h5 file
            dt = h5py.special_dtype(vlen=str)
            with h5py.File(output_path, 'w', libver='latest') as f:
                # Store the matched cutouts
                f.create_dataset('images', data=final_cutouts.astype(np.float32))
                # Store r-band ra and dec
                f.create_dataset('ra', data=all_band_data[r_band]['ra'][final_indices[r_band]])
                f.create_dataset('dec', data=all_band_data[r_band]['dec'][final_indices[r_band]])
                f.create_dataset('tile', data=r_band_data['tile'], dtype=np.int32)
                f.create_dataset(
                    'known_id', data=r_band_data['known_id'][final_indices[r_band]], dtype=dt
                )
                f.create_dataset('mto_id', data=r_band_data['mto_id'][final_indices[r_band]])
                f.create_dataset('label', data=r_band_data['label'][final_indices[r_band]])
                f.create_dataset('zspec', data=r_band_data['zspec'][final_indices[r_band]])
                # Store band information
                f.create_dataset('band_names', data=np.array(band_names, dtype='S'))

            logger.debug(f'Created matched cutouts file: {output_path}')

        if download_file:
            name, ext = os.path.splitext(output_file)
            # dir_destination = os.path.join(parent_dir_destination, tile_dir)
            # os.makedirs(dir_destination, exist_ok=True)
            temp_path = os.path.join(parent_dir_destination, name + '_temp' + ext)
            final_path = os.path.join(parent_dir_destination, output_file)

            if os.path.exists(final_path):
                logger.info(f'File {output_file} was already downloaded. Skipping to next tile.')
                return

            logger.info(f'Downloading {output_file}...')

            output_path = transform_path(output_path)
            result = subprocess.run(
                f'vcp -v {output_path} {temp_path}',
                shell=True,
                stderr=subprocess.PIPE,
                text=True,
            )

            result.check_returncode()

            os.rename(temp_path, final_path)

            logger.info(f'Successfully downloaded {output_file} to {parent_dir_destination}.')
        else:
            return

    except Exception as e:
        logger.error(f'Error processing tile {tile}: {e}', exc_info=True)
        raise  # Re-raise the exception to be caught by the parallel executor


def fuse_cutouts_parallel(
    parent_dir,
    tiles,
    in_dict,
    band_names=['whigs-g', 'cfis_lsb-r', 'ps-i'],
    parent_dir_destination=None,
    download_file=False,
    num_processes=None,
):
    logger.info(f'Starting to fuse cutouts for {len(tiles)} tiles in the bands: {band_names}')

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
                parent_dir_destination,
                download_file,
            ): tile
            for tile in tiles
        }

        # Process completed futures with a progress bar
        for future in tqdm(as_completed(future_to_tile), total=len(tiles)):
            tile = future_to_tile[future]
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as e:
                logger.error(f'Error processing tile {tile}: {e}')


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

        tiles_to_process = availability.get_tiles_for_bands(bands_to_combine)

        logger.info(
            f'There are {len(tiles_to_process)} bands that are available in {bands_to_combine}'
        )

        fuse_cutouts_parallel(
            download_dir,
            tiles_to_process,
            band_dict,
            band_names=bands_to_combine,
            parent_dir_destination=local_download_dir,
            download_file=dl_cutouts,
            num_processes=num_processes,
        )

    except Exception as e:
        logger.error(f'An error occurred in the main process: {str(e)}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

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
        help='Number of cores to use for processing (default: 3)',
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
