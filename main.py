import argparse
import logging
import multiprocessing
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from multiprocessing import Manager

from logging_stuff import setup_logging

setup_logging(
    log_dir='./logs',
    script_name=__file__,
    logging_level=logging.INFO,
)
logger = logging.getLogger()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from vos import Client  # noqa: E402

from detect import param_phot, run_mto, run_mto_old  # noqa: E402
from kd_tree import build_tree  # noqa: E402
from postprocess import add_labels, match_detections_with_catalogs  # noqa: E402
from preprocess import prep_tile  # noqa: E402
from tile_cutter import (  # noqa: E402
    download_tile_all_bands,
    download_tile_one_band,
    make_cutouts_all_bands,
    save_to_h5,
    tile_band_specs,
    tile_finder,
)
from track_progress import get_unprocessed_jobs, init_db, update_progress  # noqa: E402
from utils import (  # noqa: E402
    TileAvailability,
    delete_file,
    extract_tile_numbers,
    load_available_tiles,
    update_available_tiles,
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
        'name': 'PSS.DR4',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
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
considered_bands = ['cfis-u', 'whigs-g', 'cfis_lsb-r', 'ps-i', 'wishes-z']
# create a dictionary with the bands to consider
band_dict_incl = {key: band_dictionary.get(key) for key in considered_bands}

### pipeline options ###

create_cutouts = True
# plot cutouts?
plot = True
# Plot a random cutout from one of the tiles after execution else plot all cutouts
plot_random_cutout = False
# Show plot
show_plot = False
# Save plot
save_plot = True
# define the band that should be used to detect objects
anchor_band = 'cfis_lsb-r'
# process all available tiles
process_all_available = False
# define the square cutout size in pixels
cutout_size = 224
# minimum surface brightness to select objects
mu_limit = 19.1
# minimum effective radius to select objects
reff_limit = 1.6


# retrieve from the VOSpace and update the currently available tiles; takes some time to run
update_tiles = True
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

### paths ###
platform = 'CANFAR'  #'CANFAR' #'Narval'
if platform == 'CANFAR':
    root_dir_main = '/arc/home/heestersnick/tileslicer'
    root_dir_data = '/arc/projects/unions'
    unions_detection_directory = os.path.join(
        root_dir_data, 'catalogues/unions/GAaP_photometry/UNIONS2000'
    )
    redshift_class_catalog = os.path.join(
        root_dir_data, 'catalogues/redshifts/redshifts-2024-05-07.parquet'
    )
    download_directory = os.path.join(root_dir_data, 'ssl/data/raw/tiles/tiles2024')
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
dwarf_catalog = os.path.join(table_directory, 'all_known_dwarfs_processed.csv')
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


def process_tiles_parallel(
    avail,
    in_dict,
    tile_nums_list,
    download_dir,
    cutout_dir,
    table_dir,
    model_dir,
    mto_path,
    band,
    with_download,
    with_detection,
    with_cat_match,
    with_cutouts,
    size,
    zp,
    mu_lim,
    reff_lim,
    id_key,
    ra_key,
    dec_key,
    in_cat,
    with_streak_mask,
):
    """
    Process tiles in parallel.

    Args:
        avail (TileAvailability): instance of TileAvailability
        in_dict (dictionary): band dictionary
        tile_nums_list (list): list of tile numbers
        download_dir (str): download directory
        cutout_dir (str): cutout directory
        table_dir (str): table directory
        model_dir (str): model directory
        mto_path (str): path to mto.py
        band (str): the band we use for object detection
        with_download (bool): download the tiles
        with_detection (bool): detect objects in the tiles
        with_cat_match (bool): match detected objects with known objects
        with_cutouts (bool): produce cutouts of the detected objects
        size (int): square size of the cutouts in pixels
        zp (float): photometric zero point of the band we use for detection
        mu_lim (float): minimum surface brightness for object selection
        reff_lim (float): minimum effective radius for object selection
        id_key (str): ID key in the dataframe
        ra_key (str): right ascension key in the dataframe
        dec_key (str): declination key in the dataframe
        in_cat (dataframe): input catalog
        with_streak_mask (bool): mask streaks
    """
    with ThreadPoolExecutor() as executor:
        # Create a list of futures for concurrent downloads
        futures = []
        for tile_nums in tile_nums_list:
            future = executor.submit(
                complete_tile_processing_workflow,
                avail,
                in_dict,
                tile_nums,
                download_dir,
                cutout_dir,
                table_dir,
                model_dir,
                mto_path,
                band,
                with_download,
                with_detection,
                with_cat_match,
                with_cutouts,
                size,
                zp,
                mu_lim,
                reff_lim,
                id_key,
                ra_key,
                dec_key,
                in_cat,
                with_streak_mask,
            )
            futures.append(future)

        # Wait for all downloads to complete
        for future in futures:
            future.result()


def complete_tile_processing_workflow(
    avail,
    in_dict,
    tile_nums,
    download_dir,
    cutout_dir,
    table_dir,
    model_dir,
    mto_path,
    band,
    with_download,
    with_detection,
    with_cat_match,
    with_cutouts,
    size,
    zp,
    mu_lim,
    reff_lim,
    id_key,
    ra_key,
    dec_key,
    in_cat,
    with_streak_mask,
):
    """
    Process a single tile.

    Args:
        avail (TileAvailability): instance of TileAvailability
        in_dict (dictionary): band dictionary
        tile_nums (tuple): tile numbers
        download_dir (str): download directory
        cutout_dir (str): cutout directory
        table_dir (str): table directory
        model_dir (str): model directory
        mto_path (str): path to mto.py
        band (str): the band we use for object detection
        with_download (bool): download the tiles
        with_detection (bool): detect objects in the tiles
        with_cat_match (bool): match detected objects with known objects
        with_cutouts (bool): produce cutouts of the detected objects
        size (int): square size of the cutouts in pixels
        zp (float): photometric zero point of the band we use for detection
        mu_lim (float): minimum surface brightness for object selection
        reff_lim (float): minimum effective radius for object selection
        id_key (str): ID key in the dataframe
        ra_key (str): right ascension key in the dataframe
        dec_key (str): declination key in the dataframe
        in_cat (dataframe): input catalog
        with_streak_mask (bool): mask streaks
    """
    vos_dir = in_dict[band]['vos']
    prefix = in_dict[band]['name']
    suffix = in_dict[band]['suffix']
    delimiter = in_dict[band]['delimiter']
    fits_ext = in_dict[band]['fits_ext']
    zfill = in_dict[band]['zfill']
    zp = in_dict[band]['zp']
    tile_dir = download_dir + f'{str(tile_nums[0]).zfill(3)}_{str(tile_nums[1]).zfill(3)}'
    os.makedirs(tile_dir, exist_ok=True)
    tile_fitsfilename = f'{prefix}{delimiter}{str(tile_nums[0]).zfill(zfill)}{delimiter}{str(tile_nums[1]).zfill(zfill)}{suffix}'
    temp_name = '.'.join(tile_fitsfilename.split('.')[:-1]) + '_temp.fits'
    temp_path = os.path.join(tile_dir, temp_name)
    final_path = os.path.join(tile_dir, tile_fitsfilename)
    vos_path = os.path.join(vos_dir, tile_fitsfilename)
    obj_to_cut = None
    avail_bands = ''.join(avail.get_availability(tile_nums)[0])

    if with_download:
        download_tile_one_band(tile_nums, tile_fitsfilename, final_path, temp_path, vos_path, band)
    else:
        logging.info('Skipping download.')

    if with_detection:
        bkg, header = prep_tile(final_path, fits_ext, zp)
        params_mto = run_mto_old(
            tile_nums, final_path, model_dir, mto_path, band, mu_lim, reff_lim, bkg, zp, header
        )

        if with_cat_match:
            if np.count_nonzero(params_mto['label'].values == 1) > 0:  # type: ignore
                obj_to_cut = match_detections_with_catalogs(tile_nums, params_mto, table_dir)
            else:
                logging.info('No detections found. Skipping catalog matching.')

        else:
            logging.info('Skipping catalog matching.')
            obj_to_cut = params_mto.loc[params_mto['label'] == 1].reset_index(drop=True)  # type: ignore

    if with_cutouts:
        if in_cat is not None:
            obj_to_cut = in_cat
        if obj_to_cut is not None:
            if download_tile_all_bands(avail, tile_nums, in_dict, download_dir):
                logging.info('Successfully downloaded the remaining bands.')
                cutout = make_cutouts_all_bands(
                    avail, tile_nums, obj_to_cut, download_dir, in_dict, size
                )
                save_to_h5(
                    cutout,
                    tile_nums,
                    obj_to_cut[id_key].values,
                    obj_to_cut[ra_key].values,
                    obj_to_cut[dec_key].values,
                    size,
                    avail_bands,
                    cutout_dir,
                )
        else:
            logging.info('Skipping download and cutout, no objects to cut out.')
    else:
        logging.info('No tiles were harmed while running this script.')


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


def process_tile_for_band(job_queue, in_dict, input_catalog, download_dir):
    while True:
        try:
            tile, band = job_queue.get(block=False)
        except queue.Empty:
            logger.error('Queue is empty.')
            break

        tile_fitsfilename, final_path, temp_path, vos_path, fits_ext, zp = tile_band_specs(
            tile, in_dict, band, download_dir
        )
        # Download the tile in the given band
        if download_tile_one_band(tile, tile_fitsfilename, final_path, temp_path, vos_path, band):
            # Preprocess (bin)
            prepped_path, prepped_header = prep_tile(final_path, fits_ext, zp, bin_size=4)

            # Run detection
            param_path = run_mto(
                file_path=prepped_path,
                band=band,
                with_segmap=False,
                move_factor=0.39,
                min_distance=0.0,
            )
            # Calculate photometric parameters
            mto_det, mto_all = param_phot(
                param_path, header=prepped_header, zp=30, mu_min=21.0, reff_min=1.4
            )

            # save filtered MTO detections
            mto_det.to_parquet(os.path.splitext(param_path)[0] + '.parquet', index=False)

            # Match detections with catalog
            if input_catalog is not None:
                mto_det = add_labels(
                    tile, band, det_df=mto_det, det_df_full=mto_all, dwarfs_df=input_catalog
                )

            # delete raw data
            delete_file(param_path)
            delete_file(prepped_path)
            delete_file(final_path)

            # track processed tiles
            update_progress(tile, band)


def main(
    update,
    band_dict,
    at_least,
    show_tile_stats,
    build_kdtree,
    tile_info_dir,
    coordinates,
    dataframe_path,
    tiles,
    ra_key,
    ra_key_default,
    dec_key,
    dec_key_default,
    id_key,
    id_key_default,
    band_constr,
    download_dir,
    anch_band,
    process_all_avail,
    catalog_path,
):
    init_db()
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
        availability = TileAvailability(selected_all_tiles, band_dict_incl, at_least_key)

    # Initialize job queue and result queue
    manager = Manager()
    job_queue = manager.Queue()

    unprocessed_jobs = get_unprocessed_jobs(
        availability, process_band=anch_band, process_all_bands=process_all_avail
    )
    for job in unprocessed_jobs:
        job_queue.put(job)

    # Determine the number of processes to use
    num_processes = os.cpu_count()
    assert num_processes is not None
    logger.info(f'Using {num_processes} CPUs.')
    # Start worker processes
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(
            target=process_tile_for_band,
            args=(job_queue, band_dict_incl, input_catalog, download_dir),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == '__main__':
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
        'anch_band': anchor_band,
        'process_all_avail': process_all_available,
        'catalog_path': dwarf_catalog,
    }

    start = time.time()
    main(**arg_dict_main)
    end = time.time()
    elapsed = end - start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = (
        elapsed_string.split(':')[0],
        elapsed_string.split(':')[1],
        elapsed_string.split(':')[2],
    )
    print(f'Done! Execution took {hours} hours, {minutes} minutes, and {seconds} seconds.')
