import ctypes
import sqlite3
import threading
import time
from collections import deque
from multiprocessing import Lock, Queue, Value

import numpy as np
import psutil

from logging_setup import get_logger

logger = get_logger()

from sqlite_utils import retry_on_db_locked  # noqa: E402
from utils import ensure_list, extract_tile_numbers_from_job, get_dwarf_tile_list  # noqa: E402


def init_db(database):
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS processed_tiles
                 (tile_id TEXT,
                  band TEXT,
                  start_time REAL,
                  end_time REAL,
                  status TEXT,
                  error_message TEXT,
                  detection_count INTEGER,
                  known_dwarfs_count INTEGER,
                  matched_dwarfs_count INTEGER,
                  unmatched_dwarfs_count INTEGER,
                  PRIMARY KEY (tile_id, band))""")
    conn.commit()
    conn.close()


# def get_unprocessed_jobs(
#     database,
#     tile_availability,
#     dwarf_df,
#     in_dict,
#     process_band=None,
#     process_all_bands=False,
#     only_known_dwarfs=False,
# ):
#     conn = sqlite3.connect(database)
#     c = conn.cursor()

#     unprocessed = []
#     if process_band:
#         # Get tiles available in the specified band
#         tiles_to_process = tile_availability.band_tiles(process_band)
#     else:
#         tiles_to_process = tile_availability.unique_tiles

#     if only_known_dwarfs:
#         tiles_with_dwarfs = get_dwarf_tile_list(dwarf_df, in_dict, bands=ensure_list(process_band))
#         tiles_to_process = [tile for tile in tiles_to_process if tile in tiles_with_dwarfs]
#         logger.info(f'Tiles with dwarfs in {process_band}: {len(tiles_to_process)}')

#     # test_tiles = [
#     #     (np.int64(0), np.int64(227)),
#     #     (np.int64(0), np.int64(228)),
#     #     (np.int64(0), np.int64(229)),
#     #     (np.int64(0), np.int64(230)),
#     #     (np.int64(0), np.int64(231)),
#     #     (np.int64(269), np.int64(268)),
#     #     (np.int64(263), np.int64(267)),
#     #     (np.int64(267), np.int64(258)),
#     # ]

#     # test_tiles = [
#     #     (np.int64(45), np.int64(237)),
#     #     (np.int64(46), np.int64(237)),
#     #     (np.int64(46), np.int64(238)),
#     #     (np.int64(53), np.int64(338)),
#     #     (np.int64(80), np.int64(317)),
#     #     (np.int64(90), np.int64(325)),
#     #     (np.int64(93), np.int64(323)),
#     #     (np.int64(94), np.int64(331)),
#     #     (np.int64(95), np.int64(321)),
#     #     (np.int64(98), np.int64(320)),
#     #     (np.int64(118), np.int64(296)),
#     #     (np.int64(134), np.int64(295)),
#     #     (np.int64(153), np.int64(311)),
#     #     (np.int64(160), np.int64(324)),
#     #     (np.int64(175), np.int64(282)),
#     # ]

#     # print(test_tiles)
#     for tile in tiles_to_process:
#         available_bands, _ = tile_availability.get_availability(tile)
#         for band in available_bands:
#             if process_band and not process_all_bands and band != process_band:
#                 continue

#             # if only_known_dwarfs:
#             #     # If processing only known dwarfs, add all tiles regardless of database status
#             #     unprocessed.append((tile, band))
#             # else:
#             c.execute(
#                 """
#                 SELECT status
#                 FROM processed_tiles
#                 WHERE tile_id = ? AND band = ?
#             """,
#                 (str(tile), band),
#             )
#             result = c.fetchone()
#             if result is None or result[0] != 'completed':
#                 unprocessed.append((tile, band))

#     conn.close()
#     return unprocessed


@retry_on_db_locked()
def get_unprocessed_jobs(
    database,
    tile_availability,
    dwarf_df,
    in_dict,
    process_band=None,
    process_all_bands=False,
    only_known_dwarfs=False,
):
    conn = sqlite3.connect(database)

    try:
        c = conn.cursor()

        unprocessed = []
        if process_band:
            # Get tiles available in the specified band
            tiles_to_process = tile_availability.band_tiles(process_band)
        else:
            tiles_to_process = tile_availability.unique_tiles

        if only_known_dwarfs:
            tiles_with_dwarfs = get_dwarf_tile_list(
                dwarf_df, in_dict, bands=ensure_list(process_band)
            )
            tiles_to_process = [tile for tile in tiles_to_process if tile in tiles_with_dwarfs]
            logger.info(f'Tiles with dwarfs in {process_band}: {len(tiles_to_process)}')

        # test_tiles = [
        #     (np.int64(0), np.int64(227)),
        #     (np.int64(0), np.int64(228)),
        #     (np.int64(0), np.int64(229)),
        #     (np.int64(0), np.int64(230)),
        #     (np.int64(0), np.int64(231)),
        #     (np.int64(269), np.int64(268)),
        #     (np.int64(263), np.int64(267)),
        #     (np.int64(267), np.int64(258)),
        # ]

        # test_tiles = [
        #     (np.int64(45), np.int64(237)),
        #     (np.int64(46), np.int64(237)),
        #     (np.int64(46), np.int64(238)),
        #     (np.int64(53), np.int64(338)),
        #     (np.int64(80), np.int64(317)),
        #     (np.int64(90), np.int64(325)),
        #     (np.int64(93), np.int64(323)),
        #     (np.int64(94), np.int64(331)),
        #     (np.int64(95), np.int64(321)),
        #     (np.int64(98), np.int64(320)),
        #     (np.int64(118), np.int64(296)),
        #     (np.int64(134), np.int64(295)),
        #     (np.int64(153), np.int64(311)),
        #     (np.int64(160), np.int64(324)),
        #     (np.int64(175), np.int64(282)),
        # ]

        # test_tiles = [(np.int64(243), np.int64(290)), (np.int64(175), np.int64(282))]
        test_tiles = None
        if 'test_tiles' in locals() and test_tiles is not None and len(test_tiles) != 0:
            tiles_to_process = test_tiles

        # print(test_tiles)
        for tile in tiles_to_process:
            available_bands, _ = tile_availability.get_availability(tile)
            for band in available_bands:
                if process_band and not process_all_bands and band != process_band:
                    continue

                # if only_known_dwarfs:
                #     # If processing only known dwarfs, add all tiles regardless of database status
                #     unprocessed.append((tile, band))
                # else:
                c.execute(
                    """
                    SELECT status
                    FROM processed_tiles
                    WHERE tile_id = ? AND band = ?
                """,
                    (str(tile), band),
                )
                result = c.fetchone()
                if result is None or result[0] != 'completed':
                    unprocessed.append((tile, band))
    finally:
        conn.close()
    return unprocessed


# def update_tile_info(database, tile_info, db_lock):
#     with db_lock:
#         conn = sqlite3.connect(database)
#         c = conn.cursor()

#         fields = ['tile_id', 'band', 'status']
#         values = [str(tile_info['tile']), tile_info['band'], tile_info['status']]

#         for field in [
#             'start_time',
#             'end_time',
#             'error_message',
#             'detection_count',
#             'known_dwarfs_count',
#             'matched_dwarfs_count',
#             'unmatched_dwarfs_count',
#         ]:
#             if field in tile_info:
#                 fields.append(field)
#                 values.append(tile_info[field])

#         placeholders = ', '.join(['?' for _ in fields])
#         fields_str = ', '.join(fields)

#         query = f"""
#             INSERT OR REPLACE INTO processed_tiles
#             ({fields_str})
#             VALUES ({placeholders})
#         """

#         c.execute(query, values)
#         conn.commit()
#         conn.close()


@retry_on_db_locked()
def update_tile_info(database, tile_info, db_lock):
    with db_lock:
        conn = sqlite3.connect(database)

        try:
            c = conn.cursor()

            fields = ['tile_id', 'band', 'status']
            values = [str(tile_info['tile']), tile_info['band'], tile_info['status']]

            for field in [
                'start_time',
                'end_time',
                'error_message',
                'detection_count',
                'known_dwarfs_count',
                'matched_dwarfs_count',
                'unmatched_dwarfs_count',
            ]:
                if field in tile_info:
                    fields.append(field)
                    values.append(tile_info[field])

            placeholders = ', '.join(['?' for _ in fields])
            fields_str = ', '.join(fields)

            query = f"""
                INSERT OR REPLACE INTO processed_tiles
                ({fields_str})
                VALUES ({placeholders})
            """

            c.execute(query, values)
            conn.commit()
        finally:
            conn.close()


# @retry_on_db_locked()
# def update_tile_info(connection_pool, tile_info, db_lock):
#     with db_lock:
#         fields = ['tile_id', 'band', 'status']
#         values = [str(tile_info['tile']), tile_info['band'], tile_info['status']]

#         for field in [
#             'start_time',
#             'end_time',
#             'error_message',
#             'detection_count',
#             'known_dwarfs_count',
#             'matched_dwarfs_count',
#             'unmatched_dwarfs_count',
#         ]:
#             if field in tile_info:
#                 fields.append(field)
#                 values.append(tile_info[field])

#         placeholders = ', '.join(['?' for _ in fields])
#         fields_str = ', '.join(fields)

#         query = f"""
#             INSERT OR REPLACE INTO processed_tiles
#             ({fields_str})
#             VALUES ({placeholders})
#         """

#         connection_pool.execute(query, values)


# def get_progress_summary(
#     database, tile_availability, bands, unprocessed_jobs_at_start, processed_in_current_run
# ):
#     conn = sqlite3.connect(database)
#     c = conn.cursor()

#     results = {}
#     for band in bands:
#         # Get total available tiles for the band
#         total_available = len(tile_availability.band_tiles(band))

#         # Get overall completed, failed, and in-progress counts
#         c.execute(
#             """
#             SELECT
#                 SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
#                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
#                 SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as in_progress,
#                 SUM(CASE WHEN status = 'download_failed' THEN 1 ELSE 0 END) as download_failed
#             FROM processed_tiles
#             WHERE band = ?
#         """,
#             (band,),
#         )

#         result = c.fetchone()
#         total_completed, total_failed, in_progress, download_failed = (
#             result[0] or 0,
#             result[1] or 0,
#             result[2] or 0,
#             result[3] or 0,
#         )

#         # Get processed count for the current run
#         current_run_processed = processed_in_current_run[band]

#         # Calculate remaining jobs in the current run
#         remaining_in_run = max(0, unprocessed_jobs_at_start[band] - current_run_processed)

#         results[band] = {
#             'total_available': total_available,
#             'total_completed': total_completed,
#             'total_failed': total_failed,
#             'in_progress': in_progress,
#             'download_failed': download_failed,
#             'current_run_processed': current_run_processed,
#             'remaining_in_run': remaining_in_run,
#         }

#     conn.close()
#     return results


@retry_on_db_locked()
def get_progress_summary(
    database, tile_availability, bands, unprocessed_jobs_at_start, processed_in_current_run
):
    conn = sqlite3.connect(database)
    results = {}

    try:
        c = conn.cursor()

        for band in bands:
            # Get total available tiles for the band
            total_available = len(tile_availability.band_tiles(band))

            # Get overall completed, failed, and in-progress counts
            c.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as in_progress,
                    SUM(CASE WHEN status = 'download_failed' THEN 1 ELSE 0 END) as download_failed
                FROM processed_tiles
                WHERE band = ?
            """,
                (band,),
            )

            result = c.fetchone()
            total_completed, total_failed, in_progress, download_failed = (
                result[0] or 0,
                result[1] or 0,
                result[2] or 0,
                result[3] or 0,
            )

            # Get processed count for the current run
            current_run_processed = processed_in_current_run[band]

            # Calculate remaining jobs in the current run
            remaining_in_run = max(0, unprocessed_jobs_at_start[band] - current_run_processed)

            results[band] = {
                'total_available': total_available,
                'total_completed': total_completed,
                'total_failed': total_failed,
                'in_progress': in_progress,
                'download_failed': download_failed,
                'current_run_processed': current_run_processed,
                'remaining_in_run': remaining_in_run,
            }
    finally:
        conn.close()

    return results


# @retry_on_db_locked()
# def get_progress_summary(
#     connection_pool, tile_availability, bands, unprocessed_jobs_at_start, processed_in_current_run
# ):
#     results = {}
#     for band in bands:
#         # Get total available tiles for the band
#         total_available = len(tile_availability.band_tiles(band))

#         # Get overall completed, failed, and in-progress counts
#         result = connection_pool.execute(
#             """
#             SELECT
#                 SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
#                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
#                 SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as in_progress,
#                 SUM(CASE WHEN status = 'download_failed' THEN 1 ELSE 0 END) as download_failed
#             FROM processed_tiles
#             WHERE band = ?
#         """,
#             (band,),
#         )

#         result = result[0] if result else (0, 0, 0, 0)
#         total_completed, total_failed, in_progress, download_failed = (
#             result[0] or 0,
#             result[1] or 0,
#             result[2] or 0,
#             result[3] or 0,
#         )

#         # Get processed count for the current run
#         current_run_processed = processed_in_current_run[band]

#         # Calculate remaining jobs in the current run
#         remaining_in_run = max(0, unprocessed_jobs_at_start[band] - current_run_processed)

#         results[band] = {
#             'total_available': total_available,
#             'total_completed': total_completed,
#             'total_failed': total_failed,
#             'in_progress': in_progress,
#             'download_failed': download_failed,
#             'current_run_processed': current_run_processed,
#             'remaining_in_run': remaining_in_run,
#         }

#     return results


class MemoryTracker:
    def __init__(self, process_ids, interval=60, aggregation_period=3600):
        self.process_ids = process_ids
        self.interval = interval  # Seconds between readings
        self.aggregation_period = (
            aggregation_period  # Seconds for each aggregation bucket (e.g., 1 hour)
        )
        self.current_bucket = deque(maxlen=self.aggregation_period // self.interval)
        self.aggregated_stats = []
        self.stop_flag = threading.Event()
        self.thread = None
        self.peak_memory = 0
        self.start_time = time.time()

    def start(self):
        self.thread = threading.Thread(target=self._track_memory)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=self.interval * 2)
        self._aggregate_current_bucket()  # Ensure the last bucket is aggregated

    def _track_memory(self):
        while not self.stop_flag.is_set():
            current_memory = self._get_current_memory_usage()
            self.current_bucket.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)

            if len(self.current_bucket) == self.current_bucket.maxlen:
                self._aggregate_current_bucket()

            time.sleep(self.interval)

    def _aggregate_current_bucket(self):
        if self.current_bucket:
            bucket_peak = max(self.current_bucket)
            bucket_mean = np.mean(self.current_bucket)
            bucket_std = np.std(self.current_bucket)
            self.aggregated_stats.append((bucket_peak, bucket_mean, bucket_std))
            self.current_bucket.clear()

    def _get_current_memory_usage(self):
        total_memory = 0
        for pid in self.process_ids:
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    total_memory += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return total_memory / (1024 * 1024 * 1024)  # Convert to GB

    def get_memory_stats(self):
        self._aggregate_current_bucket()  # Aggregate any remaining data
        if not self.aggregated_stats:
            return None, None, None, None

        overall_peak = self.peak_memory
        mean_of_means = np.mean([stats[1] for stats in self.aggregated_stats])
        overall_std = np.sqrt(np.mean([stats[2] ** 2 for stats in self.aggregated_stats]))
        runtime_hours = (time.time() - self.start_time) / 3600

        return overall_peak, mean_of_means, overall_std, runtime_hours


def check_time_recordings(database):
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute("""
        SELECT tile_id, band, status, start_time, end_time
        FROM processed_tiles
        WHERE status = 'completed'
        LIMIT 10
    """)
    results = c.fetchall()
    conn.close()

    print('Sample of completed jobs:')
    for row in results:
        tile_id, band, status, start_time, end_time = row
        print(f'Tile: {tile_id}, Band: {band}, Start: {start_time}, End: {end_time}')

    if not results:
        print('No completed jobs found.')

    return bool(results)


def debug_avg_processing_time(database):
    conn = sqlite3.connect(database)
    c = conn.cursor()

    c.execute(
        """
        SELECT tile_id, band, start_time, end_time, (end_time - start_time) / 60.0 as duration_minutes
        FROM processed_tiles 
        WHERE status = 'completed' AND start_time IS NOT NULL AND end_time IS NOT NULL
        LIMIT 5
        """
    )
    results = c.fetchall()

    print('Debug: Sample processing times')
    for row in results:
        print(f'Tile: {row[0]}, Band: {row[1]}, Duration: {row[4]:.2f} minutes')

    conn.close()


class MonitoredQueue:
    def __init__(self, maxsize=0):
        self.queue = Queue(maxsize)
        self.size = Value(ctypes.c_int, 0)
        self.size_lock = Lock()

    def put(self, *args, **kwargs):
        self.queue.put(*args, **kwargs)
        with self.size_lock:
            self.size.value += 1

    def get(self, *args, **kwargs):
        item = self.queue.get(*args, **kwargs)
        with self.size_lock:
            self.size.value -= 1
        return item

    def qsize(self):
        with self.size_lock:
            return self.size.value

    def empty(self):
        return self.qsize() == 0

    def full(self):
        return self.qsize() == self.queue._maxsize

    def task_done(self):
        self.queue.task_done()

    def join(self):
        self.queue.join()


def periodic_queue_check(download_queue, process_queue, shutdown_flag):
    while not shutdown_flag.is_set():
        download_size = download_queue.qsize()
        process_size = process_queue.qsize()
        logger.info(f'Queue sizes - Download: {download_size}, Process: {process_size}')
        time.sleep(60)  # Check every 60 seconds


# def generate_summary_report(
#     database,
#     tile_availability,
#     bands_to_report,
#     unprocessed_jobs_at_start,
#     processed_in_current_run,
# ):
#     progress_results = get_progress_summary(
#         database,
#         tile_availability,
#         bands_to_report,
#         unprocessed_jobs_at_start,
#         processed_in_current_run,
#     )

#     conn = sqlite3.connect(database)
#     c = conn.cursor()

#     total_jobs = sum(unprocessed_jobs_at_start.values())
#     total_completed = sum(progress_results[band]['total_completed'] for band in bands_to_report)
#     total_failed = sum(progress_results[band]['total_failed'] for band in bands_to_report)
#     total_in_progress = sum(progress_results[band]['in_progress'] for band in bands_to_report)
#     total_download_failed = sum(
#         progress_results[band]['download_failed'] for band in bands_to_report
#     )
#     total_remaining = sum(progress_results[band]['remaining_in_run'] for band in bands_to_report)

#     c.execute(
#         """
#         SELECT AVG(end_time - start_time) / 60.0
#         FROM processed_tiles
#         WHERE status = 'completed' AND start_time IS NOT NULL AND end_time IS NOT NULL
#         """
#     )
#     avg_processing_time = c.fetchone()[0]

#     if avg_processing_time is None:
#         avg_time_str = 'No completed jobs yet'
#     else:
#         avg_time_str = f'{avg_processing_time:.2f} minutes'

#     c.execute("SELECT tile_id, band, error_message FROM processed_tiles WHERE status = 'failed'")
#     failed_jobs = c.fetchall()

#     conn.close()

#     report = f"""
#     Processing Summary:
#     -------------------
#     Total Jobs: {total_jobs}
#     Completed: {total_completed}
#     Failed: {total_failed}
#     Download Failed: {total_download_failed}
#     In Progress: {total_in_progress}
#     Remaining: {total_remaining}
#     Average Processing Time: {avg_time_str}

#     Per-Band Summary:
#     -----------------
#     """

#     for band in bands_to_report:
#         stats = progress_results[band]
#         report += f"""
#     Band {band}:
#         Total Available: {stats['total_available']}
#         Completed: {stats['total_completed']}
#         Failed: {stats['total_failed']}
#         Download Failed: {stats['download_failed']}
#         In Progress: {stats['in_progress']}
#         Remaining in Run: {stats['remaining_in_run']}
#         """

#     report += """
#     Failed Jobs:
#     ------------
#     """
#     for job in failed_jobs:
#         report += f'\n\tTile: {extract_tile_numbers_from_job(job[0])}, Band: {job[1]}\n\tError(s):\n\t{job[2]}\n\t'

#     return report


def generate_summary_report(
    database,
    tile_availability,
    bands_to_report,
    unprocessed_jobs_at_start,
    processed_in_current_run,
):
    progress_results = get_progress_summary(
        database,
        tile_availability,
        bands_to_report,
        unprocessed_jobs_at_start,
        processed_in_current_run,
    )

    conn = sqlite3.connect(database)

    try:
        c = conn.cursor()

        total_jobs = sum(unprocessed_jobs_at_start.values())
        total_completed = sum(progress_results[band]['total_completed'] for band in bands_to_report)
        total_failed = sum(progress_results[band]['total_failed'] for band in bands_to_report)
        total_in_progress = sum(progress_results[band]['in_progress'] for band in bands_to_report)
        total_download_failed = sum(
            progress_results[band]['download_failed'] for band in bands_to_report
        )
        total_remaining = sum(
            progress_results[band]['remaining_in_run'] for band in bands_to_report
        )

        c.execute(
            """
            SELECT AVG(end_time - start_time) / 60.0
            FROM processed_tiles
            WHERE status = 'completed' AND start_time IS NOT NULL AND end_time IS NOT NULL
            """
        )
        avg_processing_time = c.fetchone()[0]

        if avg_processing_time is None:
            avg_time_str = 'No completed jobs yet'
        else:
            avg_time_str = f'{avg_processing_time:.2f} minutes'

        c.execute(
            "SELECT tile_id, band, error_message FROM processed_tiles WHERE status = 'failed'"
        )
        failed_jobs = c.fetchall()
    finally:
        conn.close()

    report = f"""
    Processing Summary:
    -------------------
    Total Jobs: {total_jobs}
    Completed: {total_completed}
    Failed: {total_failed}
    Download Failed: {total_download_failed}
    In Progress: {total_in_progress}
    Remaining: {total_remaining}
    Average Processing Time: {avg_time_str}

    Per-Band Summary:
    -----------------
    """

    for band in bands_to_report:
        stats = progress_results[band]
        report += f"""
    Band {band}:
        Total Available: {stats['total_available']}
        Completed: {stats['total_completed']}
        Failed: {stats['total_failed']}
        Download Failed: {stats['download_failed']}
        In Progress: {stats['in_progress']}
        Remaining in Run: {stats['remaining_in_run']}
        """

    report += """
    Failed Jobs:
    ------------
    """
    for job in failed_jobs:
        report += f'\n\tTile: {extract_tile_numbers_from_job(job[0])}, Band: {job[1]}\n\tError(s):\n\t{job[2]}\n\t'

    return report


# @retry_on_db_locked()
# def generate_summary_report(
#     connection_pool,
#     tile_availability,
#     bands_to_report,
#     unprocessed_jobs_at_start,
#     processed_in_current_run,
# ):
#     progress_results = get_progress_summary(
#         tile_availability,
#         bands_to_report,
#         unprocessed_jobs_at_start,
#         processed_in_current_run,
#     )

#     total_jobs = sum(unprocessed_jobs_at_start.values())
#     total_completed = sum(progress_results[band]['total_completed'] for band in bands_to_report)
#     total_failed = sum(progress_results[band]['total_failed'] for band in bands_to_report)
#     total_in_progress = sum(progress_results[band]['in_progress'] for band in bands_to_report)
#     total_download_failed = sum(
#         progress_results[band]['download_failed'] for band in bands_to_report
#     )
#     total_remaining = sum(progress_results[band]['remaining_in_run'] for band in bands_to_report)

#     avg_processing_time_result = connection_pool.execute(
#         """
#         SELECT AVG(end_time - start_time) / 60.0
#         FROM processed_tiles
#         WHERE status = 'completed' AND start_time IS NOT NULL AND end_time IS NOT NULL
#         """
#     )
#     avg_processing_time = avg_processing_time_result[0][0] if avg_processing_time_result else None

#     if avg_processing_time is None:
#         avg_time_str = 'No completed jobs yet'
#     else:
#         avg_time_str = f'{avg_processing_time:.2f} minutes'

#     failed_jobs = connection_pool.execute(
#         "SELECT tile_id, band, error_message FROM processed_tiles WHERE status = 'failed'"
#     )

#     report = f"""
#     Processing Summary:
#     -------------------
#     Total Jobs: {total_jobs}
#     Completed: {total_completed}
#     Failed: {total_failed}
#     Download Failed: {total_download_failed}
#     In Progress: {total_in_progress}
#     Remaining: {total_remaining}
#     Average Processing Time: {avg_time_str}

#     Per-Band Summary:
#     -----------------
#     """

#     for band in bands_to_report:
#         stats = progress_results[band]
#         report += f"""
#     Band {band}:
#         Total Available: {stats['total_available']}
#         Completed: {stats['total_completed']}
#         Failed: {stats['total_failed']}
#         Download Failed: {stats['download_failed']}
#         In Progress: {stats['in_progress']}
#         Remaining in Run: {stats['remaining_in_run']}
#         """

#     report += """
#     Failed Jobs:
#     ------------
#     """
#     for job in failed_jobs:
#         report += f'\n\tTile: {extract_tile_numbers_from_job(job[0])}, Band: {job[1]}\n\tError(s):\n\t{job[2]}\n\t'

#     return report


def report_progress_and_memory(
    database,
    tile_availability,
    bands_to_report,
    unprocessed_jobs_at_start,
    processed_in_current_run,
    process_ids,
    shutdown_flag,
):
    while not shutdown_flag.is_set():
        try:
            progress_results = get_progress_summary(
                database,
                tile_availability,
                bands_to_report,
                unprocessed_jobs_at_start,
                processed_in_current_run,
            )
            memory_usage = get_total_memory_usage(process_ids)

            # Collect all log messages
            log_messages = []
            for band in bands_to_report:
                stats = progress_results[band]
                log_messages.append(f'\nProgress for band {band}:')
                log_messages.append(
                    f"  Overall: {stats['total_completed']}/{stats['total_available']} completed, {stats['total_failed']} failed, {stats['download_failed']} download failed"
                )
                log_messages.append(
                    f"  Current run: {stats['current_run_processed']} processed, {stats['in_progress']} in progress, {stats['remaining_in_run']} remaining"
                )

            log_messages.append(f'\nTotal memory usage across all processes: {memory_usage:.2f} GB')

            # Log all messages together
            logger.info('\n'.join(log_messages))

            # Clean up process_ids list
            process_ids[:] = [pid for pid in process_ids if psutil.pid_exists(pid)]

            for _ in range(60):  # Check shutdown flag every second for 60 seconds
                if shutdown_flag.is_set():
                    break
                time.sleep(1)
        except Exception as e:
            logger.error(f'Error in progress reporting: {str(e)}')

    logger.info('Progress reporting thread exiting.')


def get_total_memory_usage(process_ids):
    total_memory = 0
    for pid in process_ids:
        try:
            process = psutil.Process(pid)
            total_memory += process.memory_info().rss
        except psutil.NoSuchProcess:
            logger.warning(
                f'Process with PID {pid} no longer exists. Skipping in memory calculation.'
            )
        except psutil.AccessDenied:
            logger.warning(
                f'Access denied to process with PID {pid}. Skipping in memory calculation.'
            )
        except Exception as e:
            logger.error(f'Error getting memory info for process {pid}: {str(e)}')
    return total_memory / (1024 * 1024 * 1024)  # Convert to GB


def bytes_to_gb(bytes_value):
    return bytes_value / (1024 * 1024 * 1024)
