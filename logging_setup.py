import logging
import os
import sqlite3
import time
from logging.handlers import RotatingFileHandler

import psutil

from track_progress import get_progress_summary


def setup_logger(log_dir, name, logging_level=logging.INFO):
    """
    Set up a custom logger for a given script

    Args:
        log_dir (str): directory where logs should be saved
        name (str): logger name
        logging_level (int): logging level (e.g. logging.INFO, logging.DEBUG)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'{os.path.splitext(os.path.basename(name))[0]}.log')

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - ID %(process)d - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Filter redundant logging messages to decrease clutter
    log_filter = LoggingFilter()

    # Set up file handler
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,  # keep 5 backups
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(log_filter)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(log_filter)

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
    Average Processing Time: {avg_time_str} minutes

    Failed Jobs:
    ------------
    """
    for job in failed_jobs:
        report += f'Tile: {job[0]}, Band: {job[1]}, Error: {job[2]}\n'

    return report


def report_progress_and_memory(process_ids, interval=120):  # Report every 5 minutes
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
    total_memory = sum(psutil.Process(pid).memory_info().rss for pid in process_ids)
    return total_memory / (1024 * 1024 * 1024)


def bytes_to_gb(bytes_value):
    return bytes_value / (1024 * 1024 * 1024)


class LoggingFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.msg.startswith('Using config file')
            and 'default-vos-config' in record.msg
            and record.levelno == logging.INFO
        )
