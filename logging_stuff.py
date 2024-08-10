import logging
import os
import sqlite3
import time
from logging.handlers import RotatingFileHandler

import psutil

from track_progress import get_progress_summary


def setup_logging(log_dir, script_name, logging_level=logging.INFO):
    """
    Set up a custom logger for a given script

    Args:
        log_dir (str): directory where logs should be saved
        script_name (str): script name
        logging_level (int): logging level (e.g. logging.INFO, logging.DEBUG)
    """
    log_filename = os.path.join(
        log_dir, f'{os.path.splitext(os.path.basename(script_name))[0]}.log'
    )

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - Process %(process)d - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = RotatingFileHandler(
        log_filename,
        mode='w',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,  # 10 MB per file, keep 5 backups
    )

    file_handler.setFormatter(file_formatter)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging_level,
        handlers=[file_handler, console_handler],
    )


def generate_summary_report():
    total, completed, failed, in_progress = get_progress_summary()

    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute(
        "SELECT AVG(julianday(end_time) - julianday(start_time)) * 24 * 60 FROM processed_tiles WHERE status = 'completed'"
    )
    avg_processing_time = c.fetchone()[0]

    c.execute("SELECT tile_id, band, error_message FROM processed_tiles WHERE status = 'failed'")
    failed_jobs = c.fetchall()

    conn.close()

    report = f"""
    Processing Summary:
    -------------------
    Total Jobs: {total}
    Completed: {completed}
    Failed: {failed}
    In Progress: {in_progress}
    Average Processing Time: {avg_processing_time:.2f} minutes

    Failed Jobs:
    ------------
    """
    for job in failed_jobs:
        report += f'Tile: {job[0]}, Band: {job[1]}, Error: {job[2]}\n'

    return report


def report_progress_and_memory(process_ids, interval=300):  # Report every 5 minutes
    logger = logging.getLogger()
    while True:
        total, completed, failed, in_progress = get_progress_summary()
        memory_usage = get_total_memory_usage(process_ids)
        logger.info(
            f'Progress: {completed}/{total} completed, {failed} failed, {in_progress} in progress'
        )
        logger.info(f'Total memory usage across all processes: {memory_usage:.2f} GB')
        time.sleep(interval)


def get_total_memory_usage(process_ids):
    total_memory = 0
    for pid in process_ids:
        try:
            process = psutil.Process(pid)
            total_memory += process.memory_info().rss
        except psutil.NoSuchProcess:
            # Process might have ended
            pass
    return bytes_to_gb(total_memory)


def bytes_to_gb(bytes_value):
    return bytes_value / (1024 * 1024 * 1024)
