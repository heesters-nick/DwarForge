import gc
import logging
import multiprocessing
import os
import queue
import threading
import time
import warnings
from datetime import timedelta
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from multiprocessing.queues import JoinableQueue
from multiprocessing.synchronize import Event, Lock
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from astropy.coordinates import SkyCoord
from dwarforge.config import Inputs, ensure_runtime_dirs, load_settings, settings_to_jsonable
from dwarforge.detection_utils import param_phot, run_mto
from dwarforge.kd_tree import build_tree
from dwarforge.logging_setup import setup_logger
from dwarforge.postprocess import add_labels, make_cutouts, save_to_h5
from dwarforge.preprocess import prep_tile
from dwarforge.shutdown import GracefulKiller, shutdown_worker
from dwarforge.tile_cutter import (
    download_tile_one_band,
    tile_band_specs,
    tile_finder,
)
from dwarforge.track_progress import (
    MemoryTracker,
    generate_summary_report,
    get_progress_summary,
    get_unprocessed_jobs,
    init_db,
    report_progress_and_memory,
    update_tile_info,
)
from dwarforge.utils import (
    TileAvailability,
    delete_file,
    extract_tile_numbers,
    is_mostly_zeros,
    load_available_tiles,
    open_fits,
    purge_previous_run,
    tile_str,
    update_available_tiles,
)
from dwarforge.warning_manager import clear_warnings, get_warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', message="'datfix' made the change", append=True)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='invalid value encountered in log10'
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='divide by zero encountered in log10'
)

# To work with the vos client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .


def query_availability(
    update: bool, in_dict: dict, show_stats: bool, build_kdtree: bool, tile_info_dir: Path
) -> tuple[TileAvailability, list[list[tuple[int, int]]]]:
    """
    Gather information on the currently available tiles.

    Args:
        update: update the available tiles
        in_dict: band dictionary
        show_stats: show stats on the currently available tiles
        build_kdtree: build a kd tree from the currently available tiles
        tile_info_dir: path to save the tile information

    Returns:
        A pair with the availability object and the band-by-tile listings.
    """
    # update information on the currently available tiles
    if update:
        update_available_tiles(tile_info_dir, in_dict)
    # extract the tile numbers from the available tiles
    all_bands = extract_tile_numbers(load_available_tiles(tile_info_dir, in_dict), in_dict)
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict)
    # build the kd tree
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats()
    return availability, all_bands


def import_coordinates(
    coordinates: list[tuple[float, float]],
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame, SkyCoord]:
    """
    Process coordinates provided in the config file.

    Args:
        coordinates: ra, dec coordinates
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Raises:
        ValueError: error if the number of coordinates is not even

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    catalog = pd.DataFrame(coordinates, columns=[ra_key, dec_key], dtype=np.float32)
    # assign IDs to the coordinates
    catalog[id_key] = pd.RangeIndex(start=1, stop=len(catalog) + 1, step=1)
    logger.info('Coordinates received from config: %s', coordinates)
    coord_c = SkyCoord(
        ra=catalog[ra_key].to_numpy(), dec=catalog[dec_key].to_numpy(), unit='deg', frame='icrs'
    )
    return catalog, coord_c


def import_dataframe(
    dataframe_path: Path,
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame | None, SkyCoord | None]:
    """
    Process a DataFrame provided in the config file.

    Args:
        dataframe_path: path to the DataFrame
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    logging.info('Dataframe read from config file.')
    catalog = pd.read_csv(dataframe_path)

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

    coord_c = SkyCoord(
        catalog[ra_key].to_numpy(), catalog[dec_key].to_numpy(), unit='deg', frame='icrs'
    )

    return catalog, coord_c


def import_tiles(
    tiles: list[tuple[int, int]], availability: TileAvailability, band_constr: int
) -> list[tuple[int, int]]:
    """
    Process tiles provided in the config file.

    Args:
        tiles: tile numbers
        availability: instance of the TileAvailability class
        band_constr: minimum number of bands that should be available

    Raises:
        ValueError: provide two three digit numbers for each tile

    Returns:
        list: list of tiles that are available in r and at least two other bands
    """
    logging.info(f'Tiles read from config file: {tiles}')

    return [tile for tile in tiles if len(availability.get_availability(tile)[1]) >= band_constr]


def input_to_tile_list(
    availability: TileAvailability,
    band_constr: int,
    inputs: Inputs,
    tile_info_dir: Path,
    ra_key_default: str = 'ra',
    dec_key_default: str = 'dec',
    id_key_default: str = 'ID',
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None, pd.DataFrame | None]:
    """
    Process the input to get a list of tiles that are available in r and at least two other bands.

    Args:
        availability: instance of the TileAvailability class
        band_constr: minimum number of bands that should be available
        inputs: input dictionary with coordinates, a dataframe, or tiles
        tile_info_dir: path to tile information.
        ra_key_default: default right ascention key. Defaults to 'ra'.
        dec_key_default: default declination key. Defaults to 'dec'.
        id_key_default: default ID key. Defaults to 'ID'.

    Returns:
        list: list of tiles that are available in r and at least two other bands
        catalog (dataframe): updated catalog with tile information
    """
    source = inputs.source
    if source == 'coordinates':
        catalog, coord_c = import_coordinates(
            inputs.coordinates, ra_key_default, dec_key_default, id_key_default
        )
    elif source == 'dataframe':
        catalog, coord_c = import_dataframe(
            inputs.dataframe.path,
            inputs.dataframe.columns.ra,
            inputs.dataframe.columns.dec,
            inputs.dataframe.columns.id,
        )
    elif source == 'tiles':
        return None, import_tiles(inputs.tiles, availability, band_constr), None
    else:
        logging.info(
            'No coordinates, DataFrame or tiles provided. Processing all available tiles..'
        )
        return None, None, None

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        availability, catalog, coord_c, tile_info_dir, band_constr
    )

    return unique_tiles, tiles_x_bands, catalog


def download_worker(
    database: Path,
    download_queue: JoinableQueue,
    ready_queue: JoinableQueue,
    in_dict: dict,
    download_dir: Path,
    db_lock: Lock,
    shutdown_flag: Event,
    queue_lock: Lock,
    processed_in_current_run: DictProxy,
    full_res: bool,
) -> None:
    """Download tiles for assigned bands and enqueue them for processing.

    Args:
        database: Progress database path.
        download_queue: Queue of pending downloads.
        ready_queue: Queue for completed downloads ready for processing.
        in_dict: Band configuration.
        download_dir: Base download directory.
        db_lock: Inter-process lock for DB updates.
        shutdown_flag: Event to request termination.
        queue_lock: Lock for per-band counters.
        processed_in_current_run: Shared per-band counters for this run.
        full_res: Whether to download full-resolution tiles.
    """
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
                specs = tile_band_specs(tile, in_dict, band, download_dir)
                # uncomment to delete folder contents from previous runs
                # delete_folder_contents(os.path.dirname(specs['final_path']))
                # download the tile
                success = download_tile_one_band(
                    tile_numbers=tile,
                    tile_fitsname=specs['fitsfilename'],
                    final_path=specs['final_path'],
                    final_path_binned=specs['final_path_binned'],
                    temp_path=specs['temp_path'],
                    vos_path=specs['vos_path'],
                    band=band,
                    full_res=full_res,
                )
                if success:
                    ready_queue.put(
                        (
                            tile,
                            band,
                            specs['final_path'],
                            specs['final_path_binned'],
                            specs['fits_ext'],
                            specs['zp'],
                        )
                    )
                    tile_info['status'] = 'ready_for_processing'
                else:
                    tile_info['status'] = 'download_failed'
                    with queue_lock:
                        processed_in_current_run[band] += 1
            except Exception as e:
                tile_info['status'] = 'download_failed'
                tile_info['error_message'] = str(e)
                logger.exception(
                    f'Download worker error for tile {tile} band {band} (vos={specs["vos_path"]}): {e}'
                )
                with queue_lock:
                    processed_in_current_run[band] += 1
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
    process_queue: JoinableQueue,
    input_catalog: pd.DataFrame,
    db_lock: Lock,
    all_downloads_complete: Event,
    shutdown_flag: Event,
    queue_lock: Lock,
    processed_in_current_run: DictProxy,
    database: Path,
    cut_objects: bool,
    cut_size: int,
    mto_path: Path,
    mto_with_segmap: bool,
    mto_move_factor: int,
    mto_min_distance: int,
    re_lim: float,
    mu_lim: float,
    z_class_cat: Path,
    seg_mode: str,
) -> None:
    """Consume ready tiles, run preprocessing and detection, then persist outputs.

    Args:
        process_queue: Queue of tiles ready for processing.
        input_catalog: Catalog used for matching detections.
        db_lock: Inter-process lock for DB writes.
        all_downloads_complete: Event set when downloads finish.
        shutdown_flag: Event to request termination.
        queue_lock: Lock for per-band counters.
        processed_in_current_run: Shared per-band counters for this run.
        database: Progress database path.
        cut_objects: Whether to create cutouts for detections.
        cut_size: Cutout edge length in pixels.
        mto_path: Path to the MTO script.
        mto_with_segmap: Whether to also produce a segmentation map.
        mto_move_factor: Movement factor for MTO.
        mto_min_distance: Minimum distance for MTO.
        re_lim: Effective-radius threshold for photometry selection.
        mu_lim: Surface-brightness threshold for photometry selection.
        z_class_cat: Redshift classification catalog.
        seg_mode: Cutout segmentation mode.
    """
    while (
        not (all_downloads_complete.is_set() and process_queue.empty())
        and not shutdown_flag.is_set()
    ):
        try:
            tile, band, final_path, final_path_binned, fits_ext, zp = process_queue.get(
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
            if is_mostly_zeros(
                file_path=final_path,
                file_path_binned=final_path_binned,
                fits_ext=fits_ext,
                band=band,
            ):
                tile_info['status'] = 'skipped_mostly_zeros'
                logger.info(
                    f'Skipped tile {tile_str(tile)}, band {band} as it contains mostly zeros.'
                )
                tile_info['detection_count'] = 0
                param_path, seg_path, prepped_path = None, None, None
                delete_file(final_path)
            else:
                # Preprocess
                binned_data, prepped_data, prepped_path, prepped_header = prep_tile(
                    tile, final_path, final_path_binned, fits_ext, zp, band, bin_size=4
                )

                # Run detection
                param_path, seg_path = run_mto(
                    file_path=prepped_path,
                    band=band,
                    mto_path=mto_path,
                    with_segmap=mto_with_segmap,
                    move_factor=mto_move_factor,
                    min_distance=mto_min_distance,
                )
                # Calculate photometric parameters
                mto_det, mto_all = param_phot(
                    param_path,
                    header=prepped_header,
                    zp=zp,
                    mu_lim=mu_lim,
                    re_lim=re_lim,
                    band=band,
                )

                # Match detections with catalog
                if input_catalog is not None:
                    clear_warnings()
                    mto_det, matching_stats = add_labels(
                        tile,
                        band,
                        det_df=mto_det,
                        det_df_full=mto_all,
                        dwarfs_df=input_catalog,
                        fits_path=prepped_path,
                        fits_ext=fits_ext,
                        header=prepped_header,
                        z_class_cat=z_class_cat,
                    )
                    tile_info.update(matching_stats)

                    match_warnings = get_warnings()
                    if match_warnings:
                        tile_info['status'] = 'failed'
                        tile_info['error_message'] = '\n\t'.join(match_warnings)

                if mto_det is not None and mto_all is not None:
                    # save filtered MTO detections
                    mto_det.to_parquet(param_path.with_suffix('.parquet'), index=False)
                    mto_all.to_csv(param_path, index=False)

                    tile_info['detection_count'] = len(mto_det)
                else:
                    tile_info['detection_count'] = 0

                # make cutouts
                if cut_objects:
                    segmap, header_seg = open_fits(seg_path, fits_ext=0)
                    cutouts, cutouts_seg = make_cutouts(
                        binned_data,
                        header=prepped_header,
                        tile_str=tile_str(tile),
                        df=mto_det,
                        segmap=segmap,
                        cutout_size=cut_size,
                        seg_mode=seg_mode,
                    )
                    cutout_path = final_path.with_stem(
                        final_path.stem + '_cutouts_single'
                    ).with_suffix('.h5')
                    save_to_h5(
                        stacked_cutout=cutouts,
                        stacked_cutout_seg=cutouts_seg,
                        object_df=mto_det,
                        tile_numbers=tile,
                        save_path=cutout_path,
                        seg_mode=seg_mode,
                    )

                if tile_info['status'] != 'failed':
                    tile_info['status'] = 'completed'
                    logger.info(f'Successfully processed tile {tile_str(tile)}, band {band}.')
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

            if 'final_path' in locals() and final_path is not None and os.path.isfile(final_path):
                # delete raw data
                delete_file(final_path)
            if 'param_path' in locals() and param_path is not None and os.path.isfile(param_path):
                # delete full MTO parameter file
                delete_file(param_path)
            if 'seg_path' in locals() and seg_path is not None and os.path.isfile(seg_path):
                # delete MTO segmentation map
                delete_file(seg_path)
            if (
                'prepped_path' in locals()
                and prepped_path is not None
                and os.path.isfile(prepped_path)
            ):
                # delete preprocessed data
                delete_file(prepped_path)
            # garbage collection to keep memory consumption low
            gc.collect()

            # Mark the task as done in the queue
            process_queue.task_done()

        if shutdown_flag.is_set():
            logger.info('Received shutdown signal. Exiting.')
            break

    logger.info('Exiting. All tasks completed or shutdown requested.')


def main() -> None:
    """Orchestrate downloads, processing, reporting, and graceful shutdown."""
    cfg = load_settings('configs/default.yaml')
    # remove previous run's database and logs if resume = false
    purge_previous_run(cfg)
    setup_logger(
        log_dir=cfg.paths.log_directory,
        name=cfg.logging.name,
        logging_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        force=True,
    )
    logger.info('Config and logging initialized.')

    cfg_dict = settings_to_jsonable(cfg)

    # Print settings in human readable format
    cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False)
    logger.info(f'Resolved config (YAML):\n{cfg_yaml}')

    # filter considered bands from the full band dictionary
    band_dict = {k: cfg.bands[k].model_dump(mode='python') for k in cfg.runtime.considered_bands}
    # make sure necessary directories exist
    ensure_runtime_dirs(cfg=cfg)

    # Define frequently used variables
    database = cfg.paths.progress_db_path
    tile_info_dir = cfg.paths.tile_info_directory
    download_dir = cfg.paths.download_directory
    dwarf_catalog = cfg.catalog.dwarf
    anch_band = cfg.runtime.anchor_band
    num_cores = cfg.runtime.num_cores
    prefetch_factor = cfg.runtime.prefetch_factor
    process_all_avail = cfg.runtime.process_all_available

    # Initialize the database for progress tracking
    init_db(database)
    # Initialize shutdown manager
    killer = GracefulKiller()
    # Initialize job queue and result queue
    manager = Manager()
    shutdown_flag = manager.Event()
    download_queue = manager.Queue()
    process_queue = manager.Queue(maxsize=num_cores * prefetch_factor)
    db_lock = manager.Lock()
    queue_lock = manager.Lock()
    process_ids = manager.list()
    # Create an event to signal when all downloads are complete
    all_downloads_complete = multiprocessing.Event()
    # Initialize variables
    num_download_threads = 0
    download_threads = []
    processes = []
    progress_thread = threading.Thread(target=lambda: None, daemon=True)
    availability = None
    bands_to_report = []
    unprocessed_jobs_at_start = {}

    # dictionary to keep track of processed tiles per band in current run
    processed_in_current_run = manager.dict({band: 0 for band in band_dict.keys()})

    # Add main process ID
    process_ids.append(os.getpid())

    # Initialize MemoryTracker
    memory_tracker = MemoryTracker(
        process_ids,
        interval=cfg.monitoring.memory.interval_s,
        aggregation_period=cfg.monitoring.memory.aggregation_s,
    )
    memory_tracker.start()

    try:
        # query availability of the tiles
        availability, all_tiles = query_availability(
            cfg.tiles.update_tiles,
            band_dict,
            cfg.tiles.show_tile_statistics,
            cfg.tiles.build_new_kdtree,
            tile_info_dir,
        )

        # read the input catalog
        try:
            input_catalog = pd.read_csv(dwarf_catalog.path)
        except FileNotFoundError:
            logger.error(f'File not found: {dwarf_catalog.path}')
            raise FileNotFoundError

        # get the list of tiles for which r and at least two more bands are available
        _, tiles_x_bands, _ = input_to_tile_list(
            availability,
            cfg.tiles.band_constraint,
            cfg.inputs,
            tile_info_dir,
            dwarf_catalog.columns.ra,
            dwarf_catalog.columns.dec,
            dwarf_catalog.columns.id,
        )

        if tiles_x_bands is not None:
            tiles_set = set(tiles_x_bands)  # Convert list to set for faster lookup
            selected_all_tiles = [
                [tile for tile in band_tiles if tile in tiles_set] for band_tiles in all_tiles
            ]
            availability = TileAvailability(selected_all_tiles, band_dict)

        unprocessed_jobs = get_unprocessed_jobs(
            database=database,
            tile_availability=availability,
            dwarf_df=input_catalog,
            in_dict=band_dict,
            process_band=anch_band,
            process_all_bands=process_all_avail,
            only_known_dwarfs=cfg.runtime.process_only_known_dwarfs,
        )

        unprocessed_jobs_at_start = {band: 0 for band in band_dict.keys()}

        for job in unprocessed_jobs:
            logger.debug(f'Job: {job}')
            unprocessed_jobs_at_start[job[1]] += 1
            download_queue.put(job)

        logger.info(f'Number of unprocessed jobs: {unprocessed_jobs_at_start}')

        # Start download threads
        num_download_threads = min(num_cores, len(unprocessed_jobs)) * prefetch_factor
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
                    processed_in_current_run,
                    cfg.runtime.use_full_resolution,
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
        n_processes = min(num_cores, len(unprocessed_jobs))
        logger.info(f'Using {n_processes} CPUs for processing and 1 for downloading.')
        logger.info(
            f'There are {psutil.cpu_count(logical=False)} physical cores available in total.'
        )

        # Start worker processes
        processes = []
        for _ in range(n_processes):
            p = multiprocessing.Process(
                target=process_tile_for_band,
                args=(
                    process_queue,
                    input_catalog,
                    db_lock,
                    all_downloads_complete,
                    shutdown_flag,
                    queue_lock,
                    processed_in_current_run,
                    database,
                    cfg.cutouts.create,
                    cfg.cutouts.size_px,
                    cfg.detection.mto.script_path,
                    cfg.detection.mto.with_segmap,
                    cfg.detection.mto.move_factor,
                    cfg.detection.mto.min_distance,
                    cfg.detection.re_limit,
                    cfg.detection.mu_limit,
                    cfg.paths.redshift_class_catalog,
                    cfg.cutouts.segmentation_mode,
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
                    f"  Overall: {stats['total_completed']}/{stats['total_available']} completed, {stats['total_failed']} failed, {stats['download_failed']} download failed, {stats['mostly_zeros']} mostly_zeros"
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
                    for _ in range(num_cores):
                        process_queue.put((None, None, None, None, None, None))

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

        shutdown_thread = threading.Thread(
            target=shutdown_worker,
            args=(database, process_queue, db_lock, all_downloads_complete),
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
    multiprocessing.set_start_method('spawn', force=True)

    start = time.time()
    main()
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
