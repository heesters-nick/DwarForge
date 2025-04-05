import gc
import logging
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from logging_setup import setup_logger
from make_rbg import preprocess_cutout
from zoobot_utils import ensemble_predict, load_models

# Default configuration
DEFAULT_CONFIG = {
    'base_dir': None,
    'model_paths': [],
    'num_workers': None,
    'inference_batch_size': 64,
    'process_batch_size': 100,
    'log_dir': './logs',
    'log_name': 'inference',
    'log_level': 'INFO',
    'resume': False,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_logging(config):
    # Use your existing setup_logger function
    setup_logger(
        log_dir=config['log_dir'],
        name=config['log_name'],
        logging_level=getattr(logging, config['log_level']),
    )
    return logging.getLogger()


def load_config(config_path):
    """
    Load configuration from YAML file

    Args:
        config_path (str): Path to YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()

    # Load configuration from file
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            # Update default config with file values
            config.update(file_config)
    else:
        raise FileNotFoundError(f'Configuration file not found: {config_path}')

    # Validate essential parameters
    if not config['base_dir']:
        raise ValueError('base_dir must be specified in the configuration file')
    if not config['model_paths']:
        raise ValueError('model_paths must be specified in the configuration file')

    return config


def find_valid_tiles(base_dir):
    """
    Scans the base directory to find all tiles that have the required h5 file.

    Args:
        base_dir (str): Path to the base directory containing tile subdirectories

    Returns:
        list: List of valid tile paths that have the required h5 file
    """
    logger.info(f'Scanning base directory: {base_dir}')
    valid_tiles = []

    # Get all subdirectories in the base directory
    tile_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    logger.info(f'Found {len(tile_dirs)} potential tile directories')

    # Check for the required h5 file in each directory
    for tile_number in tqdm(tile_dirs, desc='Validating tiles'):
        h5_path = os.path.join(
            base_dir,
            tile_number,
            'gri',
            f'{tile_number}_matched_cutouts_full_res_final.h5',
        )
        parquet_path = os.path.join(
            base_dir, tile_number, 'gri', f'{tile_number}_matched_detections.parquet'
        )

        if os.path.exists(h5_path) and os.path.exists(parquet_path):
            valid_tiles.append(
                {
                    'tile_number': tile_number,
                    'h5_path': h5_path,
                    'parquet_path': parquet_path,
                }
            )

    logger.info(f'Found {len(valid_tiles)} valid tiles with both h5 and parquet files')
    return valid_tiles


def process_cutouts(cutouts, process_batch_size):
    """
    Preprocesses a batch of cutout images.

    Args:
        cutouts (np.ndarray): Array of cutout images
        process_batch_size (int): Number of images to process at once

    Returns:
        list: List of preprocessed images
    """
    all_preprocessed = []

    # Process images in batches to manage memory
    for i in range(0, len(cutouts), process_batch_size):
        batch_cutouts = cutouts[i : i + process_batch_size]
        batch_preprocessed = []

        # Preprocess each image
        for cutout in batch_cutouts:
            rgb_image = preprocess_cutout(cutout)
            rgb_image_clean = np.nan_to_num(rgb_image, nan=0.0, posinf=0.0, neginf=0.0)
            batch_preprocessed.append(rgb_image_clean)

        all_preprocessed.extend(batch_preprocessed)

    return all_preprocessed


def worker_process(tile_batch, model_paths, process_batch_size, inference_batch_size):
    """
    Worker process function that processes a batch of tiles.
    This loads models once per worker for efficiency.

    Args:
        tile_batch (list): Batch of tile dictionaries to process
        model_paths (list): List of paths to model checkpoints
        process_batch_size (int): Number of images to preprocess in a batch
        inference_batch_size (int): Batch size for inference

    Returns:
        list: List of tiles that failed processing
    """
    # Load models once per worker
    models = load_models(model_paths)
    if not models:
        logger.error('No models loaded in worker, skipping all tiles')
        return tile_batch  # Return all tiles as failed

    failed_tiles = []

    for tile_info in tile_batch:
        tile_number = tile_info['tile_number']
        h5_path = tile_info['h5_path']
        parquet_path = tile_info['parquet_path']

        logger.info(f'Processing tile {tile_number}')

        try:
            # === SAFER H5 FILE HANDLING ===
            # First, read data from the h5 file
            with h5py.File(h5_path, 'r') as h5_file:
                # Get image dataset
                if 'images' not in h5_file:
                    logger.error(f'No image dataset found in {h5_path}, skipping')
                    failed_tiles.append(tile_info)
                    continue

                # Read all necessary data into memory
                cutouts = h5_file['images'][:]  # type: ignore
                object_ids = h5_file['unique_id'][:]  # type: ignore

            # Preprocess all cutouts
            preprocessed_images = process_cutouts(cutouts, process_batch_size=process_batch_size)

            # Get ensemble predictions
            all_predictions = ensemble_predict(
                models, preprocessed_images, batch_size=inference_batch_size
            )

            # === WRITE BACK TO H5 FILE SAFELY ===
            # Create a temporary file path
            h5_temp_path = f'{h5_path}.temp'

            # Copy the original file to the temporary path first
            import shutil

            shutil.copy2(h5_path, h5_temp_path)

            # Modify the temporary file
            with h5py.File(h5_temp_path, 'r+') as h5_temp:
                # Remove existing dataset if it exists
                if 'zoobot_pred_v2' in h5_temp:
                    del h5_temp['zoobot_pred_v2']

                # Create the new dataset
                h5_temp.create_dataset('zoobot_pred_v2', data=all_predictions.astype(np.float32))

                # Flush to ensure data is written to disk
                h5_temp.flush()

            # Atomic rename operation (safer than direct overwrite)
            os.replace(h5_temp_path, h5_path)

            # === SAFER PARQUET FILE HANDLING ===
            # Read the parquet file
            df = pd.read_parquet(parquet_path)

            # Create a mapping from object_id to prediction
            prediction_map = dict(zip(object_ids, all_predictions))  # type: ignore

            # Add predictions to dataframe
            df['zoobot_pred_v2'] = df['unique_id'].map(prediction_map)

            # Write to a temporary file first
            parquet_temp_path = f'{parquet_path}.temp'
            df.to_parquet(parquet_temp_path, index=False)

            # Atomic rename to replace the original
            os.replace(parquet_temp_path, parquet_path)

            logger.info(f'Successfully processed tile {tile_number}')

        except Exception as e:
            logger.error(f'Error processing tile {tile_number}: {e}')
            # Clean up any temp files that might have been created
            try:
                if os.path.exists(f'{h5_path}.temp'):
                    os.remove(f'{h5_path}.temp')
                if os.path.exists(f'{parquet_path}.temp'):
                    os.remove(f'{parquet_path}.temp')
            except Exception:
                pass  # Ignore errors in cleanup

            failed_tiles.append(tile_info)

    # Clean up to avoid memory leaks
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return failed_tiles


def distribute_work(valid_tiles, config):
    """
    Distribute work across multiple processes for parallel processing.
    Groups tiles into batches for each worker to process.

    Args:
        valid_tiles (list): List of valid tile dictionaries
        config (dict): configuration

    Returns:
        list: List of tiles that failed processing
    """
    model_paths = config['model_paths']
    num_workers = config['num_workers']
    process_batch_size = config['process_batch_size']
    inference_batch_size = config['inference_batch_size']
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f'Starting parallel processing with {num_workers} workers')

    # Split tiles into batches for workers
    tile_batches = []
    batch_size = len(valid_tiles) // num_workers
    if batch_size == 0:
        batch_size = 1

    for i in range(0, len(valid_tiles), batch_size):
        end = min(i + batch_size, len(valid_tiles))
        tile_batches.append(valid_tiles[i:end])

    logger.info(f'Split {len(valid_tiles)} tiles into {len(tile_batches)} batches')

    # Create a partial function with model_paths already specified
    worker_func = partial(
        worker_process,
        model_paths=model_paths,
        process_batch_size=process_batch_size,
        inference_batch_size=inference_batch_size,
    )

    # Keep track of failed tiles
    all_failed_tiles = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = [executor.submit(worker_func, batch) for batch in tile_batches]

        # Process as completed
        for future in tqdm(futures, total=len(futures), desc='Processing batches'):
            try:
                batch_failed_tiles = future.result()
                all_failed_tiles.extend(batch_failed_tiles)
            except Exception as e:
                logger.error(f'Exception in worker process: {e}')

    return all_failed_tiles


def main(config):
    """
    Main function to run the inference pipeline.

    Args:
        config (dict): Configuration dictionary
    """
    logger = logging.getLogger()
    logger.info('Starting inference pipeline')
    logger.info(f'Using device: {DEVICE}')
    logger.info(f'Model paths: {config["model_paths"]}')

    # Find valid tiles
    valid_tiles = find_valid_tiles(config['base_dir'])

    if not valid_tiles:
        logger.error('No valid tiles found. Exiting.')
        return

    logger.info(f'Found {len(valid_tiles)} valid tiles to process')

    # Distribute work across workers
    failed_tiles = distribute_work(valid_tiles, config)

    # Report results
    successful_tiles = len(valid_tiles) - len(failed_tiles)
    logger.info(
        f'Inference completed. Processed {successful_tiles}/{len(valid_tiles)} tiles successfully.'
    )

    if failed_tiles:
        logger.warning(f'{len(failed_tiles)} tiles failed processing:')
        for tile in failed_tiles[:10]:  # Show only first 10 failed tiles in log
            logger.warning(f'  - {tile["tile_number"]}')

        if len(failed_tiles) > 10:
            logger.warning(f'  ... and {len(failed_tiles) - 10} more')

        # Save failed tiles to file for later reprocessing
        with open('failed_tiles.txt', 'w') as f:
            for tile in failed_tiles:
                f.write(f'{tile["tile_number"]}\n')
        logger.info('Failed tiles list saved to failed_tiles.txt')

    # Create a summary report
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'inference_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write('Inference Summary\n')
        f.write('================\n')
        f.write(f'Date: {pd.Timestamp.now()}\n')
        f.write(f'Base directory: {config["base_dir"]}\n')
        f.write(f'Number of workers: {config["num_workers"]}\n')
        f.write(f'Number of models: {len(config["model_paths"])}\n')
        f.write(f'Total tiles found: {len(valid_tiles)}\n')
        f.write(f'Successfully processed: {successful_tiles}\n')
        f.write(f'Failed: {len(failed_tiles)}\n')
        f.write(f'Success rate: {successful_tiles / len(valid_tiles) * 100:.2f}%\n')

    logger.info(f'Created inference summary report: {summary_file}')


def resume_from_failed(config):
    """
    Resume processing from previously failed tiles

    Args:
        config (dict): Configuration dictionary
    """
    logger = logging.getLogger()

    if not os.path.exists('failed_tiles.txt'):
        logger.error('No failed_tiles.txt file found. Cannot resume.')
        return

    logger.info('Resuming from failed tiles')
    with open('failed_tiles.txt', 'r') as f:
        failed_tile_numbers = [line.strip() for line in f.readlines()]

    logger.info(f'Found {len(failed_tile_numbers)} failed tiles to retry')

    # Find all valid tiles
    all_valid_tiles = find_valid_tiles(config['base_dir'])

    # Filter to only the failed tiles
    valid_tiles = [tile for tile in all_valid_tiles if tile['tile_number'] in failed_tile_numbers]

    logger.info(f'Retrying {len(valid_tiles)} failed tiles')

    # Distribute work
    failed_tiles = distribute_work(valid_tiles, config)

    # Report results
    successful_tiles = len(valid_tiles) - len(failed_tiles)
    logger.info(
        f'Retry completed. Processed {successful_tiles}/{len(valid_tiles)} previously failed tiles successfully.'
    )

    if failed_tiles:
        logger.warning(f'{len(failed_tiles)} tiles still failed processing')
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        retry_file = f'failed_tiles_retry_{timestamp}.txt'
        with open(retry_file, 'w') as f:
            for tile in failed_tiles:
                f.write(f'{tile["tile_number"]}\n')
        logger.info(f'Remaining failed tiles saved to {retry_file}')


def run_test(config):
    """
    Run a test on the first 10 valid tiles without modifying original files.

    Args:
        config (dict): Configuration dictionary
    """
    logger = logging.getLogger()
    logger.info('Starting TEST MODE inference on first 10 tiles')
    logger.info(f'Using device: {DEVICE}')

    # Create test output directory
    test_output_dir = 'test_output'
    os.makedirs(test_output_dir, exist_ok=True)
    logger.info(f'Created test output directory: {test_output_dir}')

    # Find valid tiles
    all_valid_tiles = find_valid_tiles(config['base_dir'])

    if not all_valid_tiles:
        logger.error('No valid tiles found. Exiting test.')
        return

    # Take only the first 10 tiles for testing
    test_tiles = all_valid_tiles[10:20]
    logger.info(f'Selected {len(test_tiles)} tiles for testing')

    # Process test tiles sequentially (no parallelization for easier debugging)
    failed_tiles = []
    successful_tiles = []

    # Load models once
    logger.info('Loading models for test...')
    models = load_models(config['model_paths'])

    for tile_info in tqdm(test_tiles, desc='Testing tiles'):
        tile_number = tile_info['tile_number']
        h5_path = tile_info['h5_path']
        parquet_path = tile_info['parquet_path']

        # Create test output paths
        test_h5_path = os.path.join(test_output_dir, f'{tile_number}_test.h5')
        test_parquet_path = os.path.join(test_output_dir, f'{tile_number}_test.parquet')

        logger.info(f'Testing tile {tile_number}')
        logger.info(f'Original H5: {h5_path} -> Test H5: {test_h5_path}')

        try:
            # First copy the entire original H5 file to preserve all datasets
            if os.path.exists(h5_path):
                start_copy = time.time()
                shutil.copy2(h5_path, test_h5_path)
                logger.info(
                    f'Copied original H5 file to {test_h5_path} in {(time.time() - start_copy):.2f} seconds.'
                )
            else:
                logger.error(f'Original H5 file not found: {h5_path}')
                failed_tiles.append(tile_info)
                continue

            # Now read data from the original file
            with h5py.File(h5_path, 'r') as original_h5:
                # Check for required datasets
                if 'images' not in original_h5:
                    logger.error(f'No images dataset found in {h5_path}, skipping')
                    failed_tiles.append(tile_info)
                    continue

                # Read necessary data for inference
                cutouts = original_h5['images'][:]  # type: ignore
                object_ids = original_h5['unique_id'][:]  # type: ignore

            # Process the data
            start_process = time.time()
            preprocessed_images = process_cutouts(
                cutouts, process_batch_size=config['process_batch_size']
            )
            logger.info(f'Processed cutouts in {(time.time() - start_process):.2f} seconds.')

            # Get predictions
            start_pred = time.time()
            all_predictions = ensemble_predict(
                models, preprocessed_images, config['inference_batch_size']
            )
            logger.info(f'Predictions done in {(time.time() - start_pred):.2f} seconds.')

            # Update the test H5 file with predictions
            with h5py.File(test_h5_path, 'r+') as test_h5:
                # Remove existing predictions dataset if it exists
                if 'zoobot_pred_v2' in test_h5:
                    del test_h5['zoobot_pred_v2']

                # Add the predictions dataset
                test_h5.create_dataset('zoobot_pred_v2', data=all_predictions.astype(np.float32))

            # Handle the parquet file
            # Copy the original parquet file
            shutil.copy2(parquet_path, test_parquet_path)

            # Update the test parquet file with predictions
            df = pd.read_parquet(test_parquet_path)

            # Create a mapping from object_id to prediction
            prediction_map = dict(zip(object_ids, all_predictions))  # type: ignore

            # Add predictions to dataframe
            df['zoobot_pred_v2'] = df['unique_id'].map(prediction_map)

            # Save updated test parquet file
            df.to_parquet(test_parquet_path, index=False)

            logger.info(f'Successfully processed test tile {tile_number}')
            successful_tiles.append(tile_info)

        except Exception as e:
            logger.error(f'Error processing test tile {tile_number}: {e}')
            failed_tiles.append(tile_info)

    # Clean up
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate test report
    logger.info(
        f'Test completed. Successfully processed {len(successful_tiles)}/{len(test_tiles)} test tiles.'
    )

    if failed_tiles:
        logger.warning(f'{len(failed_tiles)} test tiles failed processing:')
        for tile in failed_tiles:
            logger.warning(f'  - {tile["tile_number"]}')

    # Create a test summary report
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'test_summary_{timestamp}.txt'
    with open(os.path.join(test_output_dir, summary_file), 'w') as f:
        f.write('Test Inference Summary\n')
        f.write('=====================\n')
        f.write(f'Date: {pd.Timestamp.now()}\n')
        f.write(f'Base directory: {config["base_dir"]}\n')
        f.write(f'Number of models: {len(config["model_paths"])}\n')
        f.write(f'Total test tiles: {len(test_tiles)}\n')
        f.write(f'Successfully processed: {len(successful_tiles)}\n')
        f.write(f'Failed: {len(failed_tiles)}\n')
        f.write(f'Success rate: {len(successful_tiles) / len(test_tiles) * 100:.2f}%\n\n')

        f.write('Successful tiles:\n')
        for tile in successful_tiles:
            f.write(f'  - {tile["tile_number"]}\n')

    logger.info(f'Created test summary report: {os.path.join(test_output_dir, summary_file)}')
    logger.info(f'Test output files are in: {test_output_dir}')


if __name__ == '__main__':
    # Hardcoded configuration file path
    CONFIG_FILE = 'inference_config.yaml'

    # Load configuration
    config = load_config(CONFIG_FILE)

    # Initialize logging using your existing setup_logger function
    logger = initialize_logging(config)
    logger.info(f'Loaded configuration from {CONFIG_FILE}')

    # Check if we're in test mode
    if config.get('test_mode', False):
        run_test(config)
    # Otherwise run normal processing or resume
    elif config['resume']:
        resume_from_failed(config)
    else:
        main(config)
