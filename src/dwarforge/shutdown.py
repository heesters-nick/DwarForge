import logging
import queue
import signal

logger = logging.getLogger(__name__)

from dwarforge.track_progress import update_tile_info  # noqa: E402


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def shutdown_worker(database, process_queue, db_lock, all_downloads_complete):
    logger.info('Shutdown worker started')
    while not process_queue.empty():
        try:
            tile, band, final_path, final_path_binned, fits_ext, zp = process_queue.get(timeout=1)
            # Update the database to mark this job as interrupted
            update_tile_info(
                database,
                {
                    'tile': tile,
                    'band': band,
                    'status': 'interrupted',
                    'error_message': 'Job interrupted during shutdown',
                },
                db_lock,
            )
        except queue.Empty:
            break
    all_downloads_complete.set()
    logger.info('Shutdown worker finished')
