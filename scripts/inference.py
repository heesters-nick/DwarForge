import copy
import gc
import logging
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Event, Process, Queue
from queue import Empty

import h5py
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from tqdm import tqdm

from dwarforge.inference_utils import (
    cpu_preprocess_worker,
    cpu_write_worker,
    gpu_worker,
    process_cutouts,
    read_tile_data,
)
from dwarforge.logging_setup import setup_logger
from dwarforge.zoobot_utils import ensemble_predict, load_models

DEFAULT_CONFIG = {
    # General settings
    'base_dir': None,
    'model_paths': [],
    'log_dir': './logs',
    'log_name': 'inference_missing',
    'log_level': 'INFO',
    'resume': False,
    'test_mode': False,
    'test_size': 10,  # Number of tiles to test on
    'test_dir': './test_results',
    # Processing mode selection
    'processing_mode': 'cpu',  # 'cpu' or 'gpu'
    # CPU mode specific settings
    'cpu': {
        'num_workers': None,  # If None, will use CPU count - 1
        'inference_batch_size': 150,
        'process_batch_size': 150,
    },
    # GPU mode specific settings
    'gpu': {
        'preprocessing_workers': 4,
        'writer_workers': 2,
        'batch_size': 128,  # Typically larger for GPU
        'process_batch_size': 150,
        'queue_size': 50,
    },
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_logging(config):
    setup_logger(
        log_dir=config['log_dir'],
        name=config['log_name'],
        logging_level=getattr(logging, config['log_level'].upper()),
    )
    return logging.getLogger(config['log_name'])


def log_memory_usage(location_tag=''):
    """
    Log current process memory usage with an identifying tag.

    Args:
        location_tag (str): Identifier for where memory is being measured
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Convert to MB for readability
    rss_mb = memory_info.rss / (1024 * 1024)
    vms_mb = memory_info.vms / (1024 * 1024)

    logger.debug(f'Memory usage at {location_tag}: RSS: {rss_mb:.2f} MB, VMS: {vms_mb:.2f} MB')


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
        with open(config_path) as f:
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


def find_valid_tiles(base_dir, input_tiles):
    """
    Scans the base directory to find all tiles that have the required h5 file.

    Args:
        base_dir (str): Path to the base directory containing tile subdirectories
        input_tiles (str): Path to file containing tiles to be processed. Will be processed if not None.

    Returns:
        list: List of valid tile paths that have the required h5 file
    """
    logger.info(f'Scanning base directory: {base_dir}')
    valid_tiles = []

    # Get all subdirectories in the base directory
    if input_tiles is not None:
        tile_dirs = read_tile_data(input_tiles)
    else:
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
    load_start = time.time()
    log_memory_usage('before_model_load')
    models = load_models(model_paths)
    log_memory_usage('after_model_load')
    logger.info(f'Models loaded in {(time.time() - load_start):.2f} seconds.')
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
            # First, read data from the h5 file
            with h5py.File(h5_path, 'r') as h5_file:
                # Get image dataset
                if 'images' not in h5_file or 'unique_id' not in h5_file:
                    logger.error(f'No image dataset found in {h5_path}, skipping')
                    failed_tiles.append(tile_info)
                    continue

                # Read all necessary data into memory
                cutouts = h5_file['images'][:]  # type: ignore
                object_ids = h5_file['unique_id'][:]  # type: ignore

            # Preprocess all cutouts
            prep_start = time.time()
            log_memory_usage('before_preprocessing')
            preprocessed_images = process_cutouts(cutouts, process_batch_size=process_batch_size)
            log_memory_usage('after_preprocessing')
            logger.info(f'Tile prepped in {(time.time() - prep_start):.2f} seconds.')

            predict_start = time.time()
            # Get ensemble predictions
            log_memory_usage('before_inference')
            all_predictions = ensemble_predict(
                models, preprocessed_images, batch_size=inference_batch_size, device=DEVICE
            )
            log_memory_usage('after_inference')
            logger.info(
                f'Ensemble prediction finished in {(time.time() - predict_start):.2f} seconds.'
            )

            # Create a temporary file path
            h5_temp_path = f'{h5_path}.temp'

            # Copy the original file to the temporary path first
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

            # Read the parquet file
            df = pd.read_parquet(parquet_path)

            # Create a mapping from object_id to prediction
            prediction_map = dict(zip(object_ids, all_predictions, strict=False))  # type: ignore

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


def run_cpu_processing(valid_tiles, config):
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
    num_workers = config['cpu']['num_workers']
    process_batch_size = config['cpu']['process_batch_size']
    inference_batch_size = config['cpu']['inference_batch_size']
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


def worker_process_shared(
    tile_batch, shared_models, process_batch_size, inference_batch_size, device
):
    """
    Worker process function that uses pre-loaded, shared models.

    Args:
        tile_batch (list): Batch of tile dictionaries to process.
        shared_models (list): List of PyTorch models residing in shared memory.
        process_batch_size (int): Number of images to preprocess in a batch.
        inference_batch_size (int): Batch size for inference.
        device (torch.device): The device to run inference on (CPU for this function).

    Returns:
        list: List of tiles that failed processing.
    """
    worker_pid = os.getpid()
    log_memory_usage(f'Worker {worker_pid} start')
    logger.info(f'Worker {worker_pid} starting processing {len(tile_batch)} tiles.')

    # Ensure models are in evaluation mode
    for model in shared_models:
        model.eval()

    failed_tiles = []

    for tile_info in tile_batch:
        tile_number = tile_info['tile_number']
        h5_path = tile_info['h5_path']
        h5_temp_path = f'{h5_path}.temp'
        parquet_path = tile_info['parquet_path']
        parquet_temp_path = f'{parquet_path}.temp'

        logger.debug(f'Worker {worker_pid} processing tile {tile_number}')

        try:
            # Read data from the h5 file
            with h5py.File(h5_path, 'r') as h5_file:
                if 'images' not in h5_file or 'unique_id' not in h5_file:
                    logger.error(
                        f'Worker {worker_pid}: No image dataset found in {h5_path}, skipping'
                    )
                    failed_tiles.append(tile_info)
                    continue
                cutouts = h5_file['images'][:]  # type: ignore
                object_ids = h5_file['unique_id'][:]  # type: ignore

            if len(cutouts) == 0:  # type: ignore
                logger.warning(
                    f'Worker {worker_pid}: No images found in tile {tile_number}, skipping'
                )
                continue  # Skip empty tiles

            # Preprocess images
            prep_start = time.time()
            log_memory_usage(f'Worker {worker_pid} before_preprocessing T{tile_number}')
            preprocessed_images = process_cutouts(cutouts, process_batch_size=process_batch_size)
            log_memory_usage(f'Worker {worker_pid} after_preprocessing T{tile_number}')
            logger.debug(
                f'Worker {worker_pid}: Tile {tile_number} prepped in {(time.time() - prep_start):.2f}s.'
            )

            # Use shared models for inference
            predict_start = time.time()
            log_memory_usage(f'Worker {worker_pid} before_inference T{tile_number}')
            all_predictions = ensemble_predict(
                shared_models, preprocessed_images, batch_size=inference_batch_size, device=device
            )
            log_memory_usage(f'Worker {worker_pid} after_inference T{tile_number}')
            logger.debug(
                f'Worker {worker_pid}: Tile {tile_number} ensemble prediction finished in {(time.time() - predict_start):.2f}s.'
            )

            # Update H5 file with predictions
            shutil.copy2(h5_path, h5_temp_path)
            with h5py.File(h5_temp_path, 'r+') as h5_temp:
                if 'zoobot_pred_v2' in h5_temp:
                    del h5_temp['zoobot_pred_v2']
                h5_temp.create_dataset('zoobot_pred_v2', data=all_predictions.astype(np.float32))
                h5_temp.flush()
            os.replace(h5_temp_path, h5_path)

            # Update Parquet file with predictions
            df = pd.read_parquet(parquet_path)
            prediction_map = dict(zip(object_ids, all_predictions, strict=False))  # type: ignore
            df['zoobot_pred_v2'] = df['unique_id'].map(prediction_map)
            df.to_parquet(parquet_temp_path, index=False)
            os.replace(parquet_temp_path, parquet_path)

            logger.debug(f'Worker {worker_pid}: Successfully processed tile {tile_number}')

        except Exception as e:
            logger.error(
                f'Worker {worker_pid}: Error processing tile {tile_number}: {e}\n{traceback.format_exc()}'
            )
            # Clean up temp files if they exist
            if os.path.exists(h5_temp_path):
                try:
                    os.remove(h5_temp_path)
                    logger.debug(f'Removed temp file: {h5_temp_path}')
                except OSError as cleanup_error:
                    logger.warning(f'Failed to remove {h5_temp_path}: {cleanup_error}')

            if os.path.exists(parquet_temp_path):
                try:
                    os.remove(parquet_temp_path)
                    logger.debug(f'Removed temp file: {parquet_temp_path}')
                except OSError as cleanup_error:
                    logger.warning(f'Failed to remove {parquet_temp_path}: {cleanup_error}')

            failed_tiles.append(tile_info)

    # Collect garbage but don't delete shared models
    gc.collect()
    log_memory_usage(f'Worker {worker_pid} finish')
    logger.info(f'Worker {worker_pid} finished processing batch. Failures: {len(failed_tiles)}')
    return failed_tiles


def run_cpu_processing_shared(valid_tiles, config):
    """
    Distribute work across multiple processes for parallel processing using shared models.

    Args:
        valid_tiles (list): List of valid tile dictionaries.
        config (dict): Configuration dictionary.

    Returns:
        list: List of tiles that failed processing.
    """
    model_paths = config['model_paths']
    num_workers = config['cpu']['num_workers']
    process_batch_size = config['cpu']['process_batch_size']
    inference_batch_size = config['cpu']['inference_batch_size']

    # Ensure 'fork' start method for effective shared memory
    # try:
    #     if multiprocessing.get_start_method(allow_none=True) is None:
    #         multiprocessing.set_start_method('fork')
    #     elif multiprocessing.get_start_method() != 'fork':
    #         logger.warning(
    #             f"Multiprocessing start method is '{multiprocessing.get_start_method()}', not 'fork'. Shared memory may not be efficient."
    #         )
    # except Exception as e:
    #     logger.warning(f"Could not set multiprocessing start method to 'fork': {e}")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Load models in parent process
    logger.info('Loading models in the main process...')
    log_memory_usage('Parent before model load')
    load_start_time = time.time()
    models = load_models(model_paths)
    if not models:
        logger.error('Failed to load models in the main process. Aborting.')
        return valid_tiles  # Return all as failed

    logger.info(f'Models loaded in {(time.time() - load_start_time):.2f} seconds.')
    log_memory_usage('Parent after model load')

    # Move model parameters to shared memory
    logger.info('Moving model parameters and buffers to shared memory...')
    share_start_time = time.time()
    for model in models:
        model.eval()  # Ensure models are in eval mode
        # Share parameters and buffers
        for param in model.parameters():
            param.share_memory_()
        for buf in model.buffers():
            buf.share_memory_()
    logger.info(f'Models moved to shared memory in {(time.time() - share_start_time):.2f} seconds.')
    log_memory_usage('Parent after model sharing')

    logger.info(f'Starting parallel processing with {num_workers} workers using shared models.')

    # Distribute tiles to workers optimally
    tile_batches = []
    if num_workers > 0 and len(valid_tiles) > 0:
        # Calculate batch size to distribute tiles evenly
        base_batch_size = len(valid_tiles) // num_workers
        remainder = len(valid_tiles) % num_workers
        current_pos = 0

        for i in range(num_workers):
            # Add extra tile to early workers if there's a remainder
            size = base_batch_size + (1 if i < remainder else 0)
            if size > 0:
                batch = valid_tiles[current_pos : current_pos + size]
                if batch:  # Ensure batch is not empty after slicing
                    tile_batches.append(batch)
                current_pos += size
    elif len(valid_tiles) > 0:  # Handle case for num_workers=0 or 1
        tile_batches.append(valid_tiles)

    # Sanity check for empty batches
    if not tile_batches and len(valid_tiles) > 0:
        logger.warning(
            'Tile batching resulted in no batches, but there are valid tiles. Check logic.'
        )
        # Fallback: create one large batch if needed
        tile_batches = [valid_tiles]
    elif not tile_batches and len(valid_tiles) == 0:
        logger.info('No valid tiles to process.')
        return []  # No failed tiles if none were processed

    logger.info(
        f'Split {len(valid_tiles)} tiles into {len(tile_batches)} batches for {num_workers} workers.'
    )

    # Create a partial function with shared_models and other fixed arguments
    worker_func = partial(
        worker_process_shared,
        shared_models=models,  # Pass the list of shared models
        process_batch_size=process_batch_size,
        inference_batch_size=inference_batch_size,
        device=torch.device('cpu'),  # Explicitly pass CPU device
    )

    all_failed_tiles = []
    processing_start_time = time.time()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_func, batch) for batch in tile_batches]

        # Process results as they complete
        for future in tqdm(futures, total=len(futures), desc='Processing batches'):
            try:
                batch_failed_tiles = future.result()
                if batch_failed_tiles:
                    all_failed_tiles.extend(batch_failed_tiles)
            except Exception as e:
                logger.error(
                    f'Exception returned from worker process future: {e}\n{traceback.format_exc()}'
                )

    processing_time = time.time() - processing_start_time
    logger.info(f'All workers completed in {processing_time:.2f} seconds')

    # Calculate statistics
    successful_tiles = len(valid_tiles) - len(all_failed_tiles)
    if successful_tiles > 0:
        tiles_per_second = successful_tiles / processing_time
        seconds_per_tile = processing_time / successful_tiles
        logger.info(f'Processing rate: {tiles_per_second:.2f} tiles/second')
        logger.info(f'Average time per tile: {seconds_per_tile:.2f} seconds')

    return all_failed_tiles


def run_gpu_processing(valid_tiles, config):
    """
    Distributes work using a GPU-accelerated approach.

    Args:
        valid_tiles (list): List of valid tile dictionaries
        config (dict): Configuration dictionary

    Returns:
        list: List of tiles that failed processing
    """
    logger.info('Starting GPU-accelerated processing')

    # Verify GPU is available
    if not torch.cuda.is_available():
        raise ValueError('GPU processing mode selected but no GPU is available')

    # Get configuration parameters
    preprocessing_workers = config['gpu'].get('preprocessing_workers', 4)
    writer_workers = config['gpu'].get('writer_workers', 2)
    queue_size = config['gpu'].get('queue_size', 50)

    logger.info(
        f'Starting GPU-accelerated processing with {preprocessing_workers} preprocessors and {writer_workers} writers'
    )
    logger.info(f'Processing {len(valid_tiles)} tiles using GPU: {torch.cuda.get_device_name(0)}')

    # Create queues for communication
    tile_queue = Queue(maxsize=queue_size)
    gpu_task_queue = Queue(maxsize=queue_size)
    results_queue = Queue(maxsize=queue_size)
    error_queue = Queue()
    completion_queue = Queue()  # New queue for tracking completed tiles

    # Create event for signaling completion
    done_event = Event()

    # Start GPU worker
    gpu_process = Process(
        target=gpu_worker,
        args=(gpu_task_queue, results_queue, error_queue, done_event, config),
        name='GPU-Worker',
    )
    gpu_process.daemon = True
    gpu_process.start()

    # Start preprocessor workers
    preprocessors = []
    for i in range(preprocessing_workers):
        p = Process(
            target=cpu_preprocess_worker,
            args=(tile_queue, gpu_task_queue, error_queue, done_event, config),
            name=f'Preprocessor-{i + 1}',
        )
        p.daemon = True
        p.start()
        preprocessors.append(p)

    # Start writer workers
    writers = []
    for i in range(writer_workers):
        w = Process(
            target=cpu_write_worker,
            args=(
                results_queue,
                error_queue,
                completion_queue,
                done_event,
                config,
            ),  # Pass completion queue
            name=f'Writer-{i + 1}',
        )
        w.daemon = True
        w.start()
        writers.append(w)

    # Failed tiles tracking
    failed_tiles = []
    completed_tiles = set()  # Track completed tiles by ID

    try:
        # Fill the tile queue with work
        for tile in valid_tiles:
            tile_queue.put(tile)

        # Add termination signals for preprocessors
        for _ in range(preprocessing_workers):
            tile_queue.put(None)

        total_tiles = len(valid_tiles)
        start_time = time.time()
        last_log_time = start_time

        # Monitor progress and handle errors
        while len(completed_tiles) + len(failed_tiles) < total_tiles:
            # Check for errors
            while not error_queue.empty():
                try:
                    tile_info, error_msg = error_queue.get_nowait()
                    if tile_info is not None:
                        logger.error(
                            f'Error processing tile {tile_info["tile_number"]}: {error_msg}'
                        )
                        failed_tiles.append(tile_info)
                    else:
                        logger.error(f'System error: {error_msg}')
                except Empty:
                    break

            # Check for completed tiles
            completed_count_before = len(completed_tiles)
            while not completion_queue.empty():
                try:
                    completed_tile_info = completion_queue.get_nowait()
                    completed_tiles.add(completed_tile_info['tile_number'])
                except Empty:
                    break

            # Log progress if new tiles completed or every 30 seconds
            current_time = time.time()
            if (
                len(completed_tiles) > completed_count_before
                or (current_time - last_log_time) >= 30
            ):
                elapsed = current_time - start_time
                if elapsed > 0:
                    rate = len(completed_tiles) / elapsed
                    remaining = (
                        (total_tiles - len(completed_tiles) - len(failed_tiles)) / rate
                        if rate > 0
                        else 0
                    )
                    logger.info(
                        f'Progress: {len(completed_tiles)}/{total_tiles} tiles complete '
                        + f'({(len(completed_tiles) / total_tiles) * 100:.1f}%). '
                        + f'Failed: {len(failed_tiles)}. '
                        + f'Rate: {rate:.2f} tiles/sec. Est. remaining: {remaining / 60:.1f} min.'
                    )
                    last_log_time = current_time

            # Check if GPU worker died unexpectedly
            if not gpu_process.is_alive():
                logger.error('GPU worker process died unexpectedly')
                # Identify already processed tiles
                processed_ids = completed_tiles.union(t['tile_number'] for t in failed_tiles)
                # Add remaining tiles to failed list
                remaining_tiles = [t for t in valid_tiles if t['tile_number'] not in processed_ids]
                failed_tiles.extend(remaining_tiles)
                break

            # Sleep to avoid busy waiting
            time.sleep(1)

        # Send termination signal to GPU worker
        gpu_task_queue.put(None)

        # Send termination signals to writer workers
        for _ in range(writer_workers):
            results_queue.put(None)

        # Set done event
        done_event.set()

        # Wait for workers to finish
        logger.info('Waiting for workers to finish...')
        for p in preprocessors:
            p.join(timeout=10)

        gpu_process.join(timeout=10)

        for w in writers:
            w.join(timeout=10)

        # Final status report
        logger.info(
            f'Processing complete. {len(completed_tiles)} tiles processed successfully, {len(failed_tiles)} tiles failed.'
        )

    except KeyboardInterrupt:
        logger.info('Keyboard interrupt received, stopping workers...')
        done_event.set()
        # Identify already processed tiles
        processed_ids = completed_tiles.union(t['tile_number'] for t in failed_tiles)
        # Add remaining tiles to failed list
        remaining_tiles = [t for t in valid_tiles if t['tile_number'] not in processed_ids]
        failed_tiles.extend(remaining_tiles)

    except Exception as e:
        logger.error(f'Error in GPU processing coordinator: {str(e)}\n{traceback.format_exc()}')
        # Identify already processed tiles
        processed_ids = completed_tiles.union(t['tile_number'] for t in failed_tiles)
        # Add remaining tiles to failed list
        remaining_tiles = [t for t in valid_tiles if t['tile_number'] not in processed_ids]
        failed_tiles.extend(remaining_tiles)

    finally:
        # Clean up resources
        for process_list in [preprocessors, [gpu_process], writers]:
            for p in process_list:
                if p.is_alive():
                    logger.info(f'Terminating process {p.name}')
                    p.terminate()

        # Close queues
        for q in [tile_queue, gpu_task_queue, results_queue, error_queue, completion_queue]:
            q.close()

        logger.info('All workers have been shut down')

    return failed_tiles


def main(config):
    """
    Main function to run the inference pipeline.

    Args:
        config (dict): Configuration dictionary
    """
    logger.info('Starting inference pipeline')
    logger.info(f'Using processing mode: {config["processing_mode"]}')

    # Verify processing mode is valid
    if config['processing_mode'] not in ['cpu', 'gpu']:
        raise ValueError(
            f"Invalid processing mode: {config['processing_mode']}. Must be 'cpu' or 'gpu'"
        )

    # Check GPU availability if GPU mode is selected
    if config['processing_mode'] == 'gpu':
        if not torch.cuda.is_available():
            raise ValueError('GPU processing mode selected but no GPU is available')
        logger.info(f'GPU device: {torch.cuda.get_device_name(0)}')
    else:
        logger.info(f'Using CPU mode with {config["cpu"]["num_workers"]} workers')

    # Find valid tiles
    valid_tiles = find_valid_tiles(config['base_dir'], config['input_tiles'])

    if not valid_tiles:
        logger.error('No valid tiles found. Exiting.')
        return

    logger.info(f'Found {len(valid_tiles)} valid tiles to process')

    # Distribute work according to selected processing mode
    if config['processing_mode'] == 'cpu':
        # Use CPU-only processing (existing implementation)
        failed_tiles = run_cpu_processing(valid_tiles, config)
    else:
        # Use GPU-accelerated processing
        failed_tiles = run_gpu_processing(valid_tiles, config)

    # Report results
    successful_tiles = len(valid_tiles) - len(failed_tiles)
    if len(valid_tiles) > 0:
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
        f.write(f'Processing mode: {config["processing_mode"]}\n')
        f.write(f'Base directory: {config["base_dir"]}\n')
        if config['processing_mode'] == 'cpu':
            f.write(f'Number of workers: {config["cpu"]["num_workers"]}\n')
            f.write('Using shared memory: Yes\n')
        else:
            f.write(f'Preprocessing workers: {config["gpu"]["preprocessing_workers"]}\n')
            f.write(f'Writer workers: {config["gpu"]["writer_workers"]}\n')
        f.write(f'Number of models: {len(config["model_paths"])}\n')
        f.write(f'Total tiles found: {len(valid_tiles)}\n')
        f.write(f'Successfully processed: {successful_tiles}\n')
        f.write(f'Failed: {len(failed_tiles)}\n')
        f.write(f'Success rate: {successful_tiles / len(valid_tiles) * 100:.2f}%\n')

    logger.info(f'Created inference summary report: {summary_file}')


def resume_from_failed(config):
    """
    Resume processing from previously failed tiles using the selected processing mode.

    Args:
        config (dict): Configuration dictionary
    """
    processing_mode = config['processing_mode']
    logger.info(f'Resuming from failed tiles using {processing_mode.upper()} processing mode')

    # Check if failed tiles file exists
    if not os.path.exists('failed_tiles.txt'):
        logger.error('No failed_tiles.txt file found. Cannot resume.')
        return

    # Read failed tile numbers
    with open('failed_tiles.txt') as f:
        failed_tile_numbers = [line.strip() for line in f.readlines()]

    logger.info(f'Found {len(failed_tile_numbers)} failed tiles to retry')

    # Find all valid tiles
    all_valid_tiles = find_valid_tiles(config['base_dir'], config['input_tiles'])

    # Filter to only the failed tiles
    valid_tiles = [tile for tile in all_valid_tiles if tile['tile_number'] in failed_tile_numbers]

    if not valid_tiles:
        logger.error(
            'None of the failed tiles were found in the current base directory. Cannot resume.'
        )
        return

    logger.info(f'Retrying {len(valid_tiles)} failed tiles')

    # Distribute work according to selected processing mode
    start_time = time.time()

    if processing_mode == 'cpu':
        # Use CPU-only processing
        logger.info(f'Using CPU mode with {config["cpu"]["num_workers"]} workers')
        failed_tiles = run_cpu_processing(valid_tiles, config)

    elif processing_mode == 'gpu':
        # Verify GPU is available
        if not torch.cuda.is_available():
            logger.error('GPU processing mode selected but no GPU is available')
            return

        logger.info(f'Using GPU mode with device: {torch.cuda.get_device_name(0)}')
        failed_tiles = run_gpu_processing(valid_tiles, config)

    else:
        logger.error(f'Unknown processing mode: {processing_mode}')
        return

    # Report results
    processing_time = time.time() - start_time
    successful_tiles = len(valid_tiles) - len(failed_tiles)
    success_rate = (successful_tiles / len(valid_tiles)) * 100 if valid_tiles else 0

    logger.info(
        f'Retry completed in {processing_time:.2f} seconds. '
        f'Processed {successful_tiles}/{len(valid_tiles)} previously failed tiles successfully '
        f'({success_rate:.2f}%).'
    )

    if failed_tiles:
        logger.warning(f'{len(failed_tiles)} tiles still failed processing:')
        for tile in failed_tiles[:10]:  # Show only first 10 failed tiles in log
            logger.warning(f'  - {tile["tile_number"]}')

        if len(failed_tiles) > 10:
            logger.warning(f'  ... and {len(failed_tiles) - 10} more')

        # Save remaining failed tiles to a new file
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        retry_file = f'failed_tiles_retry_{timestamp}.txt'
        with open(retry_file, 'w') as f:
            for tile in failed_tiles:
                f.write(f'{tile["tile_number"]}\n')
        logger.info(f'Remaining failed tiles saved to {retry_file}')
    else:
        logger.info('All previously failed tiles processed successfully!')

    # Create a summary report
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'retry_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write('Retry Processing Summary\n')
        f.write('======================\n')
        f.write(f'Date: {pd.Timestamp.now()}\n')
        f.write(f'Processing mode: {processing_mode.upper()}\n')
        f.write(f'Base directory: {config["base_dir"]}\n')
        f.write(f'Number of models: {len(config["model_paths"])}\n')
        f.write(f'Total retry tiles: {len(valid_tiles)}\n')
        f.write(f'Successfully processed: {successful_tiles}\n')
        f.write(f'Failed: {len(failed_tiles)}\n')
        f.write(f'Success rate: {success_rate:.2f}%\n')
        f.write(f'Total processing time: {processing_time:.2f} seconds\n\n')

        # Calculate throughput
        if processing_time > 0 and successful_tiles > 0:
            tiles_per_second = successful_tiles / processing_time
            seconds_per_tile = processing_time / successful_tiles
            f.write(f'Processing rate: {tiles_per_second:.2f} tiles/second\n')
            f.write(f'Average time per tile: {seconds_per_tile:.2f} seconds\n\n')

        f.write('Processing mode details:\n')
        if processing_mode == 'cpu':
            f.write(f'  - Worker processes: {config["cpu"]["num_workers"]}\n')
            f.write(f'  - Inference batch size: {config["cpu"]["inference_batch_size"]}\n')
            f.write(f'  - Process batch size: {config["cpu"]["process_batch_size"]}\n')
        else:
            f.write(f'  - Preprocessing workers: {config["gpu"]["preprocessing_workers"]}\n')
            f.write(f'  - Writer workers: {config["gpu"]["writer_workers"]}\n')
            f.write(f'  - GPU batch size: {config["gpu"]["batch_size"]}\n')
            f.write(f'  - GPU: {torch.cuda.get_device_name(0)}\n')

    logger.info(f'Created retry summary report: {summary_file}')


def run_test(config):
    """
    Run a test on a small set of tiles using the selected processing mode (CPU or GPU).
    Tests the full parallelization pipeline but on a limited dataset.

    Args:
        config (dict): Configuration dictionary
    """
    # Initialize summary variables
    successful_tiles = 0
    success_rate = 0.0
    total_time = 0.0
    failed_tiles = []

    processing_mode = config['processing_mode']
    logger.info(f'Starting TEST MODE inference using {processing_mode.upper()} processing mode')

    # Create test output directory
    test_output_dir = config['test_dir']
    os.makedirs(test_output_dir, exist_ok=True)
    logger.info(f'Created test output directory: {test_output_dir}')

    # Find valid tiles
    all_valid_tiles = find_valid_tiles(config['base_dir'], config['input_tiles'])

    if not all_valid_tiles:
        logger.error('No valid tiles found. Exiting test.')
        return

    # Take a subset of tiles for testing (10-20 tiles is good for testing)
    test_size = config.get('test_size', 10)
    test_tiles = all_valid_tiles[:test_size]
    logger.info(f'Selected {len(test_tiles)} tiles for testing')

    # Create a temporary deepcopy of the config for test modifications
    test_config = copy.deepcopy(config)

    # 1. Create modified config for testing
    # - Redirect outputs to test directory
    # - Limit the number of workers in CPU mode
    # - Adjust queue sizes in GPU mode

    # Make a copy of the original test tiles with modified paths
    redirected_tiles = []
    copy_start = time.time()
    for tile_info in test_tiles:
        original_h5_path = tile_info['h5_path']
        original_parquet_path = tile_info['parquet_path']
        tile_number = tile_info['tile_number']

        # Create test output paths
        test_h5_path = os.path.join(test_output_dir, f'{tile_number}_test.h5')
        test_parquet_path = os.path.join(test_output_dir, f'{tile_number}_test.parquet')

        # First, copy the original files to test directory
        if os.path.exists(original_h5_path):
            shutil.copy2(original_h5_path, test_h5_path)
            logger.info(f'Copied: {original_h5_path} → {test_h5_path}')
        else:
            logger.error(f'Original H5 file not found: {original_h5_path}')
            continue

        if os.path.exists(original_parquet_path):
            shutil.copy2(original_parquet_path, test_parquet_path)
            logger.info(f'Copied: {original_parquet_path} → {test_parquet_path}')
        else:
            logger.error(f'Original parquet file not found: {original_parquet_path}')
            # Clean up H5 since we won't use it
            os.remove(test_h5_path)
            continue

        # Add redirected tile info to our list
        redirected_tiles.append(
            {'tile_number': tile_number, 'h5_path': test_h5_path, 'parquet_path': test_parquet_path}
        )
    logger.info(f'Copied files in {(time.time() - copy_start):.2f} seconds.')

    if not redirected_tiles:
        logger.error('No valid tiles could be prepared for testing. Exiting test.')
        return

    logger.info(f'Prepared {len(redirected_tiles)} tiles for testing with {processing_mode} mode')

    # 2. Run appropriate processing mode on test data
    start_time = time.time()
    failed_tiles = []

    try:
        if processing_mode == 'cpu':
            logger.info(f'Testing CPU processing with {test_config["cpu"]["num_workers"]} workers')

            # Run CPU processing on test tiles
            failed_tiles = run_cpu_processing(redirected_tiles, test_config)

        elif processing_mode == 'gpu':
            # Verify GPU is available for GPU test
            if not torch.cuda.is_available():
                logger.error('GPU processing mode selected but no GPU is available')
                return

            # Adjust settings for GPU test mode
            test_config['gpu']['preprocessing_workers'] = min(
                2, test_config['gpu'].get('preprocessing_workers', 4)
            )
            test_config['gpu']['writer_workers'] = min(
                2, test_config['gpu'].get('writer_workers', 2)
            )
            test_config['gpu']['queue_size'] = min(10, test_config['gpu'].get('queue_size', 50))

            logger.info(
                f'Testing GPU processing with {test_config["gpu"]["preprocessing_workers"]} preprocessors, '
                + f'{test_config["gpu"]["writer_workers"]} writers'
            )

            # Run GPU processing on test tiles
            failed_tiles = run_gpu_processing(redirected_tiles, test_config)
        else:
            logger.error(f'Unknown processing mode for testing: {processing_mode}')
            return

        # Calculate statistics
        successful_tiles = len(redirected_tiles) - len(failed_tiles)
        success_rate = (successful_tiles / len(redirected_tiles)) * 100 if redirected_tiles else 0

        # Log results
        total_time = time.time() - start_time
        logger.info(f'Test completed in {total_time:.2f} seconds')
        logger.info(
            f'Successfully processed {successful_tiles}/{len(redirected_tiles)} test tiles ({success_rate:.2f}%)'
        )

        if failed_tiles:
            logger.warning(f'{len(failed_tiles)} test tiles failed processing:')
            for tile in failed_tiles:
                logger.warning(f'  - {tile["tile_number"]}')

    except Exception as e:
        logger.error(f'Error during {processing_mode} test: {str(e)}\n{traceback.format_exc()}')

    # Create a test summary report
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'test_summary_{timestamp}.txt'
    with open(os.path.join(test_output_dir, summary_file), 'w') as f:
        f.write('Test Inference Summary\n')
        f.write('=====================\n')
        f.write(f'Date: {pd.Timestamp.now()}\n')
        f.write(f'Processing mode: {processing_mode.upper()}\n')
        f.write(f'Base directory: {config["base_dir"]}\n')
        f.write(f'Number of models: {len(config["model_paths"])}\n')
        f.write(f'Total test tiles: {len(redirected_tiles)}\n')
        f.write(f'Successfully processed: {successful_tiles}\n')
        f.write(f'Failed: {len(failed_tiles)}\n')
        f.write(f'Success rate: {success_rate:.2f}%\n')
        f.write(f'Total processing time: {total_time:.2f} seconds\n\n')

        # Calculate throughput
        if total_time > 0 and successful_tiles > 0:
            tiles_per_second = float(successful_tiles) / float(total_time)
            seconds_per_tile = float(total_time) / float(successful_tiles)
            f.write(f'Processing rate: {tiles_per_second:.2f} tiles/second\n')
            f.write(f'Average time per tile: {seconds_per_tile:.2f} seconds\n\n')

        f.write('Processing mode details:\n')
        if processing_mode == 'cpu':
            f.write(f'  - Worker processes: {test_config["cpu"]["num_workers"]}\n')
            f.write(f'  - Inference batch size: {test_config["cpu"]["inference_batch_size"]}\n')
            f.write(f'  - Process batch size: {test_config["cpu"]["process_batch_size"]}\n')
        else:
            f.write(f'  - Preprocessing workers: {test_config["gpu"]["preprocessing_workers"]}\n')
            f.write(f'  - Writer workers: {test_config["gpu"]["writer_workers"]}\n')
            f.write(f'  - GPU batch size: {test_config["gpu"]["batch_size"]}\n')
            f.write(f'  - GPU: {torch.cuda.get_device_name(0)}\n')

    logger.info(f'Created test summary report: {os.path.join(test_output_dir, summary_file)}')
    logger.info(f'Test output files are in: {test_output_dir}')
    return failed_tiles


if __name__ == '__main__':
    # Hardcoded configuration file path
    CONFIG_FILE = 'inference_config.yaml'
    config = None
    logger = logging.getLogger(__name__)
    try:
        # 1. Load Configuration
        config = load_config(CONFIG_FILE)

        # 2. Initialize Logging
        try:
            logger = initialize_logging(config)
            logger.info(f'Loaded configuration from {CONFIG_FILE}')
        except Exception as log_e:
            # Fallback basic logging if full setup fails
            logging.basicConfig(
                level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger()
            logger.error(f'Failed to initialize full logging: {log_e}')
            logger.info(f'Loaded configuration from {CONFIG_FILE} (using basic logging)')

        # 3. Set Multiprocessing Start Method CONDITIONALLY based on Config
        desired_method = None
        mode = config['processing_mode']

        if mode == 'gpu':
            desired_method = 'spawn'  # GPU requires spawn
        elif mode == 'cpu':
            if sys.platform == 'win32':
                desired_method = 'spawn'  # Fork not available on Windows
                logger.warning(
                    "CPU mode selected on Windows. Using 'spawn' start method (fork not available). Shared memory efficiency might be lower."
                )
            else:
                desired_method = 'fork'  # CPU prefers fork for shared memory efficiency
        else:
            # This should ideally be caught by config validation in load_config
            msg = f"Invalid processing_mode '{mode}' found in configuration."
            logger.error(msg)
            raise ValueError(msg)

        if desired_method:
            try:
                current_method = multiprocessing.get_start_method(allow_none=True)
                if current_method is None:
                    multiprocessing.set_start_method(desired_method)
                    logger.info(
                        f"Multiprocessing start method set to '{desired_method}' for '{mode}' mode."
                    )
                elif current_method != desired_method:
                    # Avoid forcing unless absolutely necessary as it might break libraries.
                    logger.warning(
                        f"Multiprocessing start method already set to '{current_method}'. "
                        f"Desired method for '{mode}' mode is '{desired_method}'. Proceeding, but this might cause issues."
                    )
                    # Check if the problematic GPU scenario exists
                    if mode == 'gpu' and current_method != 'spawn':
                        logger.error(
                            "CRITICAL WARNING: GPU mode selected, but start method is not 'spawn'. CUDA operations will likely fail!"
                        )
                    # Check if the potentially problematic CPU scenario exists
                    elif mode == 'cpu' and sys.platform != 'win32' and current_method != 'fork':
                        logger.warning(
                            "WARNING: CPU mode selected, but start method is not 'fork'. Shared memory might not work as expected."
                        )

            except Exception as e:
                logger.error(
                    f"Failed to set multiprocessing start method to '{desired_method}': {e}"
                )
                # If GPU mode needs spawn and failed, it's likely fatal
                if mode == 'gpu' and multiprocessing.get_start_method(allow_none=True) != 'spawn':
                    logger.critical(
                        "FATAL: Cannot proceed with GPU mode without 'spawn' start method."
                    )
                    sys.exit(1)  # Exit if GPU mode cannot be set up correctly

        # 4. Log initial memory (after logging is set up)
        if logger:  # Ensure logger was initialized
            log_memory_usage('Initial Parent Memory')

        # 5. Proceed with main logic (test, resume, or main run)
        if config.get('test_mode', False):
            run_test(config)
        elif config['resume']:
            resume_from_failed(config)
        else:
            main(config)

        if logger:  # Ensure logger was initialized
            log_memory_usage('Final Parent Memory')

    # --- Keep Original Exception Handling ---
    except FileNotFoundError as e:
        err_msg = f'ERROR: Configuration file not found: {CONFIG_FILE}'
        print(err_msg)
        # Use basic logging if logger wasn't set up
        (logger or logging).error(f'Configuration file load failed: {e}')
    except ValueError as e:
        err_msg = f'ERROR: Invalid configuration or value: {e}'
        print(err_msg)
        (logger or logging).error(f'Configuration validation or runtime error: {e}')
    except Exception as e:
        # Catch-all for other unexpected errors
        err_msg = f'An unexpected error occurred: {e}'
        print(err_msg)
        detailed_err_msg = (
            f'An unexpected error occurred in main execution: {e}\n{traceback.format_exc()}'
        )
        (logger or logging).error(detailed_err_msg)
