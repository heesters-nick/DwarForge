import logging
import os

from concurrent_log_handler import ConcurrentRotatingFileHandler


def get_logger():
    """
    Get the configured logger

    Returns:
        logging.Logger: configured logger
    """
    return logging.getLogger()


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
    file_handler = ConcurrentRotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=1000,  # keep all log files
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


class LoggingFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.msg.startswith('Using config file')
            and 'default-vos-config' in record.msg
            and record.levelno == logging.INFO
        )
