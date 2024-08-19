import ctypes
import logging
import sqlite3
import threading
import time
from collections import deque
from multiprocessing import Lock, Queue, Value

import numpy as np
import psutil


def init_db():
    conn = sqlite3.connect('progress.db')
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


def get_unprocessed_jobs(tile_availability, process_band=None, process_all_bands=False):
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()

    unprocessed = []
    if process_band:
        # Get tiles available in the specified band
        tiles_to_process = tile_availability.band_tiles(process_band)
    else:
        tiles_to_process = tile_availability.unique_tiles

    for tile in tiles_to_process[:16]:
        available_bands, _ = tile_availability.get_availability(tile)
        for band in available_bands:
            if process_band and not process_all_bands and band != process_band:
                continue
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

    conn.close()
    return unprocessed


# def update_progress(tile, band, status, db_lock, error_message=None):
#     with db_lock:
#         conn = sqlite3.connect('progress.db')
#         c = conn.cursor()
#         now = datetime.now().isoformat()  # type: ignore
#         if status == 'started':
#             c.execute(
#                 """
#                 INSERT OR REPLACE INTO processed_tiles (tile_id, band, start_time, status)
#                 VALUES (?, ?, ?, ?)
#             """,
#                 (str(tile), band, now, status),
#             )
#         elif status in ['completed', 'failed']:
#             c.execute(
#                 """
#                 UPDATE processed_tiles
#                 SET end_time = ?, status = ?, error_message = ?
#                 WHERE tile_id = ? AND band = ?
#             """,
#                 (now, status, error_message, str(tile), band),
#             )
#         conn.commit()
#         conn.close()


def update_tile_info(tile_info, db_lock):
    with db_lock:
        conn = sqlite3.connect('progress.db')
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
        conn.close()


def get_progress_summary():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as in_progress
        FROM processed_tiles
    """)
    result = c.fetchone()
    conn.close()
    return result


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


def check_time_recordings():
    conn = sqlite3.connect('progress.db')
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


def debug_avg_processing_time():
    conn = sqlite3.connect('progress.db')
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
        logger = logging.getLogger()
        logger.info(f'Queue sizes - Download: {download_size}, Process: {process_size}')
        time.sleep(60)  # Check every 60 seconds
