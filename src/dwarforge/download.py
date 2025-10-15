import logging
import queue
import threading
import time
from multiprocessing import JoinableQueue
from multiprocessing.synchronize import Event, Lock
from pathlib import Path

from dwarforge.tile_cutter import download_tile_one_band, tile_band_specs
from dwarforge.track_progress import update_cutout_info
from dwarforge.utils import tile_str

logger = logging.getLogger(__name__)

QUEUE_TIMEOUT = 1  # seconds


def download_worker(
    database: Path,
    download_queue: JoinableQueue,
    process_queue: JoinableQueue,
    required_bands: list[str],
    band_dictionary: dict,
    download_dir: Path,
    db_lock: Lock,
    shutdown_flag: Event,
    queue_lock: Lock,
    processed_in_current_run: dict,
    downloaded_bands: dict,
    full_res: bool,
):
    """Worker that downloads data and tracks completion per tile"""
    worker_id = threading.get_ident()
    logger.debug(f'Download worker {worker_id} started')

    while not shutdown_flag.is_set():
        try:
            tile, band = download_queue.get(timeout=1)

            if tile is None:
                logger.info(f'Download worker {worker_id} received sentinel, exiting')
                break

            cutout_info = {
                'tile': tile,
                'band': band,
                'start_time': time.time(),
                'status': 'downloading',
            }
            update_cutout_info(database, cutout_info, db_lock)

            try:
                paths = tile_band_specs(
                    tile=tile, in_dict=band_dictionary, band=band, download_dir=download_dir
                )

                success = download_tile_one_band(
                    tile_numbers=tile,
                    tile_fitsname=paths['fitsfilename'],
                    final_path=paths['final_path'],
                    final_path_binned=paths['final_path_binned'],
                    temp_path=paths['temp_path'],
                    vos_path=paths['vos_path'],
                    band=band,
                    full_res=full_res,
                )

                if success:
                    cutout_info['status'] = 'downloaded'

                    # Get current state for this tile
                    tile_string = tile_str(tile)
                    if tile_string not in downloaded_bands:
                        downloaded_bands[tile_string] = {'bands': set(), 'paths': {}}

                    current_state = downloaded_bands[tile_string]
                    current_state['bands'].add(band)
                    current_state['paths'][band] = paths
                    downloaded_bands[tile_string] = current_state  # Update shared dict

                    logger.debug(f'Tile {tile} has bands: {downloaded_bands[tile_string]["bands"]}')

                    # Check if all required bands are present
                    if downloaded_bands[tile_string]['bands'] == set(required_bands):
                        logger.info(
                            f'Tile {tile} downloaded in all bands. Queueing for processing.'
                        )
                        process_queue.put((tile, downloaded_bands[tile_string]['paths']))
                        del downloaded_bands[tile_string]
                        cutout_info['status'] = 'ready_for_processing'
                else:
                    cutout_info.update(
                        {
                            'status': 'download_failed',
                            'error_message': 'Download failed',
                            'end_time': time.time(),
                        }
                    )
                    with queue_lock:
                        processed_in_current_run[band] += 1

            except Exception as e:
                cutout_info.update(
                    {
                        'status': 'download_failed',
                        'error_message': str(e),
                        'end_time': time.time(),
                    }
                )
                logger.error(f'Error downloading tile {tile} band {band}: {str(e)}')
                with queue_lock:
                    processed_in_current_run[band] += 1

            finally:
                cutout_info['end_time'] = time.time()
                update_cutout_info(database, cutout_info, db_lock)
                download_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f'Unexpected error in download worker {worker_id}: {str(e)}')
            if shutdown_flag.is_set():
                break

    logger.info(f'Download worker {worker_id} exiting')
