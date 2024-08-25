import argparse
import gc
import logging
import multiprocessing
import os  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import timedelta  # noqa: E402
from multiprocessing import (
    Manager,  # noqa: E402
)

from logging_setup import setup_logger

logger = setup_logger(
    log_dir='./logs',
    name='dwarforge',
    logging_level=logging.INFO,
)
logger = logging.getLogger()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from vos import Client  # noqa: E402

from detect import param_phot, run_mto  # noqa: E402
from kd_tree import build_tree  # noqa: E402
from postprocess import add_labels  # noqa: E402
from preprocess import prep_tile  # noqa: E402
from shutdown import GracefulKiller, shutdown_worker  # noqa: E402
from tile_cutter import (  # noqa: E402
    download_tile_one_band,
    tile_band_specs,
    tile_finder,
)
from track_progress import (  # noqa: E402
    MemoryTracker,
    generate_summary_report,
    get_progress_summary,
    get_unprocessed_jobs,
    init_db,
    report_progress_and_memory,
    update_tile_info,
)
from utils import (  # noqa: E402
    TileAvailability,
    delete_file,
    extract_tile_numbers,
    load_available_tiles,
    tile_str,
    update_available_tiles,
)
from warning_manager import clear_warnings, get_warnings  # noqa: E402

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

create_cutouts = False
# plot cutouts?
plot = False
# Plot a random cutout from one of the tiles after execution else plot all cutouts
plot_random_cutout = False
# Show plot
show_plot = False
# Save plot
save_plot = True
# define the band that should be used to detect objects
anchor_band = 'whigs-g'
# process all available tiles
process_all_available = False
# define the square cutout size in pixels
cutout_size = 224
# minimum surface brightness to select objects
mu_limit = 19.1
# minimum effective radius to select objects
reff_limit = 1.6


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


def download_worker(
    database, download_queue, ready_queue, in_dict, download_dir, db_lock, shutdown_flag, queue_lock
):
    worker_id = threading.get_ident()
    while not shutdown_flag.is_set():
        try:
            tile, band = download_queue.get(timeout=1)
            if tile is None:  # Sentinel value to indicate end of downloads
                logger.info(f'Download worker {worker_id} received sentinel, exiting')
                break

            tile_info = {
                'tile': tile,
                'band': band,
                'start_time': time.time(),
                'status': 'downloading',
            }
            update_tile_info(database, tile_info, db_lock)

            try:
                tile_fitsfilename, final_path, temp_path, vos_path, fits_ext, zp = tile_band_specs(
                    tile, in_dict, band, download_dir
                )
                success = download_tile_one_band(
                    tile, tile_fitsfilename, final_path, temp_path, vos_path, band
                )
                if success:
                    ready_queue.put((tile, band, final_path, fits_ext, zp))
                    tile_info['status'] = 'ready_for_processing'
                else:
                    tile_info['status'] = 'download_failed'
            except Exception as e:
                tile_info['status'] = 'download_failed'
                tile_info['error_message'] = str(e)
            finally:
                tile_info['end_time'] = time.time()
                update_tile_info(database, tile_info, db_lock)
                download_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f'Unexpected error in download worker {worker_id}: {str(e)}')
            if shutdown_flag.is_set():
                break

    logger.info(f'Download worker {worker_id} exiting')


def process_tile_for_band(
    process_queue,
    in_dict,
    input_catalog,
    download_dir,
    db_lock,
    all_downloads_complete,
    shutdown_flag,
    queue_lock,
    processed_in_current_run,
    database,
):
    while (
        not (all_downloads_complete.is_set() and process_queue.empty())
        and not shutdown_flag.is_set()
    ):
        try:
            tile, band, final_path, fits_ext, zp = process_queue.get(
                timeout=120
            )  # 2 minute timeout
            if tile is None:  # Sentinel value
                logger.info('Received sentinel, exiting.')
                break
        except queue.Empty:
            if all_downloads_complete.is_set() and process_queue.empty():
                logger.info('Found empty queue and all downloads complete, exiting.')
                break
            logger.debug('Process queue is empty. Continuing to wait.')
            continue

        tile_info = {
            'tile': tile,
            'band': band,
            'start_time': time.time(),
            'status': 'processing',
            'error_message': None,
            'detection_count': 0,
            'known_dwarfs_count': 0,
            'matched_dwarfs_count': 0,
            'unmatched_dwarfs_count': 0,
        }

        update_tile_info(database, tile_info, db_lock)
        logger.info(f'Started processing tile {tile_str(tile)}, band {band}.')

        try:
            # Preprocess (bin)
            prepped_path, prepped_header = prep_tile(final_path, fits_ext, zp, bin_size=4)

            # Run detection
            param_path = run_mto(
                file_path=prepped_path,
                band=band,
                with_segmap=True,
                move_factor=0.39,
                min_distance=0.0,
            )
            # Calculate photometric parameters
            mto_det, mto_all = param_phot(
                param_path, header=prepped_header, zp=zp, mu_min=21.0, reff_min=1.4
            )

            # save filtered MTO detections
            mto_det.to_parquet(os.path.splitext(param_path)[0] + '.parquet', index=False)
            mto_all.to_csv(param_path, index=False)

            # Match detections with catalog
            if input_catalog is not None:
                clear_warnings()
                mto_det, matching_stats = add_labels(
                    tile,
                    band,
                    det_df=mto_det,
                    det_df_full=mto_all,
                    dwarfs_df=input_catalog,
                    header=prepped_header,
                )
                tile_info.update(matching_stats)

                match_warnings = get_warnings()
                if match_warnings:
                    tile_info['status'] = 'failed'
                    tile_info['error_message'] = '\n\t'.join(match_warnings)

            tile_info['detection_count'] = len(mto_det)

            # delete raw data
            delete_file(final_path)

            if tile_info['status'] != 'failed':
                tile_info['status'] = 'completed'
                logger.info(f'Successfully processed tile {tile_str(tile)}, band {band}.')
                # delete rebinned data + full MTO parameter file only when processing finished without errors
                delete_file(param_path)
                delete_file(prepped_path)

            else:
                logger.warning(
                    f'Warning was raised while matching to catalog. Revisit tile {tile_str(tile)}, band {band}.'
                )

        except Exception as e:
            tile_info['status'] = 'failed'
            tile_info['error_message'] = str(e)
            logger.error(
                f'Error processing tile {tile_str(tile)}, band {band}: {str(e)}',
                exc_info=True,
            )
        finally:
            tile_info['end_time'] = time.time()

            update_tile_info(database, tile_info, db_lock)

            with queue_lock:
                processed_in_current_run[band] += 1

            logger.info(
                f'Finished processing of tile {tile_str(tile)}, band {band}. '
                f'Status: {tile_info["status"]}, '
                f'Detections: {tile_info["detection_count"]}.'
            )

            # garbage collection to keep memory consumption low
            gc.collect()

            # Mark the task as done in the queue
            process_queue.task_done()

        if shutdown_flag.is_set():
            logger.info('Received shutdown signal. Exiting.')
            break

    logger.info('Exiting. All tasks completed or shutdown requested.')


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
    num_processing_cores,
    mem_track_inter,
    mem_aggr_period,
    database,
):
    # Initialize the database for progress tracking
    init_db(database)

    # Initialize shutdown manager
    killer = GracefulKiller()
    # Initialize job queue and result queue
    manager = Manager()
    shutdown_flag = manager.Event()
    download_queue = manager.Queue()
    process_queue = manager.Queue(maxsize=num_processing_cores * PREFETCH_FACTOR)
    db_lock = manager.Lock()
    queue_lock = manager.Lock()
    process_ids = manager.list()

    # dictionary to keep track of processed tiles per band in current run
    processed_in_current_run = manager.dict({band: 0 for band in band_dict.keys()})

    # Add main process ID
    process_ids.append(os.getpid())

    # Initialize MemoryTracker
    memory_tracker = MemoryTracker(
        process_ids, interval=mem_track_inter, aggregation_period=mem_aggr_period
    )
    memory_tracker.start()

    try:
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
            availability = TileAvailability(selected_all_tiles, band_dict, at_least_key)

        unprocessed_jobs = get_unprocessed_jobs(
            database=database,
            tile_availability=availability,
            process_band=anch_band,
            process_all_bands=process_all_avail,
        )
        unprocessed_jobs_at_start = {band: 0 for band in band_dict.keys()}

        for job in unprocessed_jobs:
            logger.debug(f'Job: {job}')
            unprocessed_jobs_at_start[job[1]] += 1
            download_queue.put(job)

        logger.info(f'Number of unprocessed jobs: {unprocessed_jobs_at_start}')

        # Create an event to signal when all downloads are complete
        all_downloads_complete = multiprocessing.Event()

        # Start download threads
        num_download_threads = min(num_processing_cores, len(unprocessed_jobs))
        download_threads = []
        for _ in range(num_download_threads):
            t = threading.Thread(
                target=download_worker,
                args=(
                    database,
                    download_queue,
                    process_queue,
                    band_dict,
                    download_dir,
                    db_lock,
                    shutdown_flag,
                    queue_lock,
                ),
            )
            t.daemon = True
            t.start()
            download_threads.append(t)

        if process_all_avail:
            bands_to_report = band_dict.keys()
        else:
            bands_to_report = [anch_band]

        # Start progress reporting thread
        progress_thread = threading.Thread(
            target=report_progress_and_memory,
            args=(
                database,
                availability,
                bands_to_report,
                unprocessed_jobs_at_start,
                processed_in_current_run,
                process_ids,
                shutdown_flag,
            ),
            daemon=True,
        )
        progress_thread.start()

        logger.info(f'Using {num_processing_cores} CPUs for processing and 1 for downloading.')
        logger.info(
            f'There are {psutil.cpu_count(logical=False)} physical cores available in total.'
        )

        # Start worker processes
        processes = []
        for _ in range(num_processing_cores):
            p = multiprocessing.Process(
                target=process_tile_for_band,
                args=(
                    process_queue,
                    band_dict,
                    input_catalog,
                    download_dir,
                    db_lock,
                    all_downloads_complete,
                    shutdown_flag,
                    queue_lock,
                    processed_in_current_run,
                    database,
                ),
            )
            p.start()
            processes.append(p)
            process_ids.append(p.pid)  # Add worker process ID

        all_jobs_completed = False
        while not killer.kill_now and not shutdown_flag.is_set():
            progress_results = get_progress_summary(
                database,
                availability,
                bands_to_report,
                unprocessed_jobs_at_start,
                processed_in_current_run,
            )

            # Collect all log messages
            log_messages = []
            for band in bands_to_report:
                stats = progress_results[band]
                log_messages.append(f'\nProgress for band {band}:')
                log_messages.append(
                    f"  Overall: {stats['total_completed']}/{stats['total_available']} completed, {stats['total_failed']} failed"
                )
                log_messages.append(
                    f"  Current run: {stats['current_run_processed']} processed, {stats['in_progress']} in progress, {stats['remaining_in_run']} remaining"
                )

            # Log all messages together
            logger.info('\n'.join(log_messages))

            if all(
                progress_results[band]['in_progress'] == 0
                and progress_results[band]['remaining_in_run'] == 0
                for band in bands_to_report
            ):
                if not all_jobs_completed:
                    logger.info('All jobs completed. Initiating shutdown.')
                    all_jobs_completed = True
                    all_downloads_complete.set()

                    # Add sentinel values to signal the end of processing
                    for _ in range(num_processing_cores):
                        process_queue.put((None, None, None, None, None))

                # Check if all worker processes have exited
                if all(not p.is_alive() for p in processes):
                    logger.info('All worker processes have exited. Ending main loop.')
                    break

            time.sleep(10)  # Check every 10 seconds

    except Exception as e:
        logger.error(f'An error occurred in the main process: {str(e)}')

    finally:
        logger.info('Initiating graceful shutdown...')
        shutdown_flag.set()  # Signal all processes to shut down

        # Signal that all downloads are complete
        all_downloads_complete.set()

        # Add sentinel values to signal the end of downloads
        for _ in range(num_download_threads):
            download_queue.put((None, None))

        for t in download_threads:
            t.join(timeout=40)

        # Start a shutdown worker to handle remaining items in the process queue
        shutdown_thread = threading.Thread(
            target=shutdown_worker, args=(process_queue, db_lock, all_downloads_complete)
        )
        shutdown_thread.start()

        # Stop worker processes
        for p in processes:
            p.join(timeout=120)  # Give each process 60 seconds to finish

        # If any processes are still running, terminate them
        for p in processes:
            if p.is_alive():
                logger.warning(
                    f'Process {p.pid} did not terminate gracefully. Forcing termination.'
                )
                p.terminate()
                p.join()

        # Wait for the shutdown worker to finish
        shutdown_thread.join(timeout=60)

        # Stop the progress thread
        progress_thread.join(timeout=5)  # Give the thread 5 second to finish

        # Stop the memory tracker
        memory_tracker.stop()

        # Get and log the memory usage statistics
        peak_memory, mean_memory, std_memory, runtime_hours = memory_tracker.get_memory_stats()

        if peak_memory is not None:
            logger.info(f'Total runtime: {runtime_hours:.2f} hours')
            logger.info(f'Peak total memory usage: {peak_memory:.2f} GB')
            logger.info(f'Mean total memory usage: {mean_memory:.2f} GB')
            logger.info(f'Standard deviation of total memory usage: {std_memory:.2f} GB')
        else:
            logger.warning('No memory usage data was collected.')

        # Generate and log summary report
        summary_report = generate_summary_report(
            database,
            availability,
            bands_to_report,
            unprocessed_jobs_at_start,
            processed_in_current_run,
        )
        logger.info(summary_report)


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
    parser.add_argument(
        '--memory_tracking_interval',
        type=int,
        default=60,
        help='Interval in seconds between memory usage measurements (default: 60)',
    )
    parser.add_argument(
        '--memory_aggregation_period',
        type=int,
        default=3600,
        help='Period in seconds for aggregating memory statistics (default: 3600)',
    )
    parser.add_argument(
        '--database',
        type=str,
        default='progress.db',
        help='Database file to keep track of progress (default: progress.db)',
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
        'num_processing_cores': args.processing_cores,
        'mem_track_inter': args.memory_tracking_interval,
        'mem_aggr_period': args.memory_aggregation_period,
        'database': args.database,
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
