import logging
import os


def setup_logging(log_dir, script_name, logging_level):
    """
    Set up a custom logger for a given script

    Args:
        log_dir (str): directory where logs should be saved
        script_name (str): script name
    """
    log_filename = os.path.join(
        log_dir, f'{os.path.splitext(os.path.basename(script_name))[0]}.log'
    )
    # handler = ConcurrentRotatingFileHandler(log_filename, mode='w', maxBytes=10 * 1024 * 1024, backupCount=5)
    # logging.FileHandler(log_filename, mode='w')
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()],
    )
