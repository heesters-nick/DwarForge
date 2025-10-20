import glob
import logging
import multiprocessing
import os
import queue
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import Event, JoinableQueue, Lock, Manager
from multiprocessing.synchronize import Event as EventT
from multiprocessing.synchronize import Lock as LockT
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from dwarforge.config import ensure_runtime_dirs, load_settings, settings_to_jsonable
from dwarforge.download import download_worker
from dwarforge.import_utils import input_to_tile_list, query_availability
from dwarforge.logging_setup import setup_logger
from dwarforge.postprocess import (
    load_segmap,
    make_cutouts,
    match_coordinates_across_bands,
    read_band_data,
    save_to_h5,
)
from dwarforge.shutdown import GracefulKiller
from dwarforge.track_progress import (
    get_progress_summary,
    get_unprocessed_jobs,
    init_cutouts_db,
    update_cutout_info,
)
from dwarforge.utils import (
    TileAvailability,
    get_dwarf_tile_list,
    open_fits,
    purge_previous_run,
    tile_str,
)
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings('ignore', message="'datfix' made the change", append=True)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='invalid value encountered in log10'
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='divide by zero encountered in log10'
)

logger = logging.getLogger(__name__)

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .

# define the band directory containing
# information on the different
# photometric bands in the
# survey and their file systems


def make_cutouts_for_band(
    data_path: Path, tile: tuple[int, int], cut_size: int, seg_mode: str | None
):
    segmap = load_segmap(data_path)
    binned_data, binned_header = open_fits(data_path, fits_ext=0)
    path, extension = os.path.splitext(data_path)
    det_pattern = f'{path}*_det_params.parquet'
    det_path = glob.glob(det_pattern)
    mto_det = pd.read_parquet(det_path[0])
    cutouts, cutouts_seg = make_cutouts(
        binned_data,
        header=binned_header,
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
    zoobot_pred=None,
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
                    and len(f['band_names']) == len(band_names)  # type: ignore
                    and f['images'].shape[2:] == (size, size)  # type: ignore
                ):
                    logger.info(
                        f'Using existing LSB accumulation file: {output_path} '
                        f'with {f["images"].shape[0]} existing objects'  # type: ignore
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
            existing_ids = set([x.decode('utf-8') for x in f['known_id'][:]])  # type: ignore

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
            current_size = f['images'].shape[0]  # type: ignore
            new_size = current_size + len(cutouts_to_add)

            # Resize all datasets
            f['images'].resize(new_size, axis=0)  # type: ignore
            if segmaps_to_add is not None:
                f['segmaps'].resize(new_size, axis=0)  # type: ignore
            f['ra'].resize(new_size, axis=0)  # type: ignore
            f['dec'].resize(new_size, axis=0)  # type: ignore
            f['tile'].resize(new_size, axis=0)  # type: ignore
            f['known_id'].resize(new_size, axis=0)  # type: ignore
            f['label'].resize(new_size, axis=0)  # type: ignore
            f['zspec'].resize(new_size, axis=0)  # type: ignore

            # Add new data
            f['images'][current_size:new_size] = cutouts_to_add  # type: ignore
            if segmaps_to_add is not None:
                f['segmaps'][current_size:new_size] = segmaps_to_add  # type: ignore
            f['ra'][current_size:new_size] = ras_to_add  # type: ignore
            f['dec'][current_size:new_size] = decs_to_add  # type: ignore
            f['tile'][current_size:new_size] = tiles_to_add  # type: ignore
            f['known_id'][current_size:new_size] = known_ids_to_add  # type: ignore
            f['label'][current_size:new_size] = labels_to_add  # type: ignore
            f['zspec'][current_size:new_size] = zspecs_to_add  # type: ignore

            logger.info(
                f'Added {len(cutouts_to_add)} new LSB galaxies from tile {tile} to accumulated file '
                f'(skipped {np.sum(~new_objects_mask)} existing objects)'
            )


def process_tile(
    tile: tuple[int, int],
    parent_dir: Path,
    in_dict: dict,
    band_names: list[str],
    cut_size: int,
    seg_mode: str | None,
    max_sep: float,
    n_neighbors: int,
    use_full_res: bool,
    accumulate_lsb: bool,
    lsb_file_path: Path,
    file_lock: LockT | None,
):
    try:
        logger.info(f'Matching and combining detections in tile {tile_str(tile)}')
        num_objects = 0
        tile_dir = Path(f'{str(tile[0]).zfill(3)}_{str(tile[1]).zfill(3)}')
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
            matched_df['ra'].to_numpy(),
            matched_df['dec'].to_numpy(),
            matched_df['lsb'].to_numpy(),
            matched_df['ID_known'].to_numpy(),
            matched_df['zspec'].to_numpy(),
            matched_df['unique_id'].to_numpy(),
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
                    cutout_size=cut_size,
                    seg_mode=seg_mode,
                )

                final_cutouts[:, i, :, :] = cutouts
                if seg_mode is not None:
                    final_segmaps[:, i, :, :] = cutouts_seg  # type: ignore
        except Exception as e:
            logger.error(f'Error in cutout creation: {e}.')
            return 0

        # Process LSB objects
        lsb_indices = np.where(labels == 1)[0]  # Assuming 1 indicates LSB objects
        if len(lsb_indices) > 0:
            logger.info(f'Found {len(lsb_indices)} LSB object(s) in tile {tile}.')
            # Create LSB mask
            lsb_mask = np.zeros_like(labels, dtype=bool)
            lsb_mask[lsb_indices] = True
            # Convert RA and Dec to Cartesian coordinates
            coords = SkyCoord(ra=final_ras, dec=final_decs, unit='deg')
            cartesian = coords.cartesian.xyz.value.T  # type: ignore
            # Take minimum between n_neighbors+1 and the total number of matched objects to avoid errors
            total_points = len(cartesian)
            k = min(n_neighbors + 1, total_points)
            # Find nearest neighbors for LSB objects
            tree = cKDTree(cartesian)  # type: ignore
            _, neighbor_indices = tree.query(cartesian[lsb_indices], k=k)  # type: ignore

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
    parent_dir: Path,
    tiles: list[tuple[int, int]],
    in_dict: dict,
    band_names: list[str],
    num_processes: int,
    cut_size: int,
    seg_mode: str | None,
    max_sep: float,
    n_neighbors: int,
    use_full_res: bool,
    accumulate_lsb: bool,
    lsb_file_path: Path,
    file_lock: LockT | None,
):
    logger.info(f'Starting to fuse cutouts for {len(tiles)} tiles in the bands: {band_names}')

    num_matches = []
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks and create a dictionary to track futures
        future_to_tile = {
            executor.submit(
                process_tile,
                tile=tile,
                parent_dir=parent_dir,
                in_dict=in_dict,
                band_names=band_names,
                cut_size=cut_size,
                seg_mode=seg_mode,
                max_sep=max_sep,
                n_neighbors=n_neighbors,
                use_full_res=use_full_res,
                accumulate_lsb=accumulate_lsb,
                lsb_file_path=lsb_file_path,
                file_lock=file_lock,
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
    process_queue: JoinableQueue,
    database: Path,
    band_dict: dict,
    required_bands: list[str],
    download_dir: Path,
    db_lock: LockT,
    all_downloads_complete: EventT,
    shutdown_flag: EventT,
    queue_lock: LockT,
    processed_in_current_run: dict,
    use_full_res: bool,
    cut_size: int,
    seg_mode: str,
    max_sep: float,
    n_neighbors: int,
    accumulate_lsb: bool,
    lsb_h5_path: Path,
    lsb_h5_lock: LockT,
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
        use_full_res: Whether to use full resolution data
        cut_size: Size of cutouts in pixels
        seg_mode: Segmentation mode
        max_sep: Maximum separation for matching
        n_neighbors: Number of neighbors for training samples
        accumulate_lsb: Accumulate all lsb cutouts to a single h5 file
        lsb_h5_path: Path to lsb h5 file
        lsb_h5_lock: Multiprocessing lock to avoid race conditions
    """
    worker_id = os.getpid()
    logger.debug(f'Processing worker {worker_id} started')

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
                    use_full_res=use_full_res,
                    accumulate_lsb=accumulate_lsb,
                    lsb_file_path=lsb_h5_path,
                    file_lock=lsb_h5_lock,
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
                            # os.remove(paths['final_path'])
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


def main() -> None:
    try:
        cfg = load_settings('configs/combination_config.yaml')
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
        band_dict = {
            k: cfg.bands[k].model_dump(mode='python') for k in cfg.runtime.considered_bands
        }
        # make sure necessary directories exist
        ensure_runtime_dirs(cfg=cfg)

        # Define frequently used variables
        database = cfg.paths.progress_db_path
        tile_info_dir = cfg.paths.tile_info_directory
        download_dir = cfg.paths.download_directory
        dwarf_catalog = cfg.catalog.dwarf
        use_full_res = cfg.runtime.use_full_resolution
        num_cores = cfg.runtime.num_cores
        prefetch_factor = cfg.runtime.prefetch_factor
        process_all_avail = cfg.runtime.process_all_available
        bands_to_combine = cfg.combination.bands_to_combine
        cut_size = cfg.cutouts.size_px
        dwarfs_only = cfg.runtime.process_only_known_dwarfs
        accumulate_lsb = cfg.combination.accumulate_lsb_to_h5
        lsb_h5_path = cfg.combination.lsb_cutout_path
        seg_mode = cfg.cutouts.segmentation_mode
        seg_mode = (
            None if cfg.cutouts.segmentation_mode == 'none' else cfg.cutouts.segmentation_mode
        )
        max_sep = cfg.combination.max_match_sep_arcsec
        n_neighbors = cfg.combination.negatives_per_positive

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

        lsb_h5_lock = Lock()

        try:
            if use_full_res:
                # Initialize the database for progress tracking
                init_cutouts_db(database)
                # Initialize shutdown manager
                killer = GracefulKiller()
                # Initialize job queue and result queue
                shutdown_flag = Event()
                download_queue = JoinableQueue()
                process_queue = JoinableQueue(maxsize=num_cores * prefetch_factor)
                db_lock = Lock()
                queue_lock = Lock()
                manager = Manager()
                downloaded_bands = manager.dict()

                if accumulate_lsb:
                    initialize_lsb_file(
                        output_path=cfg.combination.lsb_cutout_path,
                        band_names=bands_to_combine,
                        size=cut_size,
                    )
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
                    process_groups=cfg.combination.process_groups_only,
                    group_tiles=cfg.combination.group_tiles_csv,
                )

                print(unprocessed_jobs)

                unprocessed_jobs_at_start = {band: 0 for band in band_dict.keys()}

                for job in unprocessed_jobs:
                    logger.info(f'Job: {job}')
                    unprocessed_jobs_at_start[job[1]] += 1
                    download_queue.put(job)

                logger.info(f'Number of unprocessed jobs: {unprocessed_jobs_at_start}')

                # Create an event to signal when all downloads are complete
                all_downloads_complete = manager.Event()
                # Set number of download threads
                num_download_threads = min(num_cores * prefetch_factor, len(unprocessed_jobs))
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
                            bands_to_combine,
                            band_dict,
                            download_dir,
                            db_lock,
                            shutdown_flag,
                            queue_lock,
                            processed_in_current_run,
                            downloaded_bands,
                            use_full_res,
                        ),
                    )
                    t.daemon = True
                    t.start()
                    download_threads.append(t)

                # Start processing workers
                processes = []
                for _ in range(num_cores):
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
                            use_full_res,
                            cut_size,
                            seg_mode,
                            max_sep,
                            n_neighbors,
                            accumulate_lsb,
                            lsb_h5_path,
                            lsb_h5_lock,
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
                            for _ in range(num_cores):
                                process_queue.put((None, None))

                        # Check if all worker processes have exited
                        if all(not p.is_alive() for p in processes):
                            logger.info('All worker processes have exited. Ending main loop.')
                            break

                    time.sleep(10)  # Check every 10 seconds

            else:
                if dwarfs_only:
                    tiles_to_process = get_dwarf_tile_list(
                        dwarf_cat=cfg.catalog.dwarf.path, in_dict=band_dict, bands=bands_to_combine
                    )
                else:
                    tiles_to_process = availability.get_tiles_for_bands(bands_to_combine)

                if cfg.combination.combine_cutouts:
                    fuse_cutouts_parallel(
                        parent_dir=download_dir,
                        tiles=tiles_to_process,
                        in_dict=band_dict,
                        band_names=bands_to_combine,
                        num_processes=num_cores,
                        cut_size=cut_size,
                        seg_mode=seg_mode,
                        max_sep=max_sep,
                        n_neighbors=n_neighbors,
                        use_full_res=use_full_res,
                        accumulate_lsb=accumulate_lsb,
                        lsb_file_path=lsb_h5_path,
                        file_lock=lsb_h5_lock,
                    )

        except Exception as e:
            logger.error(f'There was an error getting the tile numbers: {e}.')

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
