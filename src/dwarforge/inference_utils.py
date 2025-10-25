import gc
import logging
import os
import shutil
import time
import traceback
from multiprocessing import current_process
from queue import Empty

import h5py
import numpy as np
import pandas as pd
import torch

from dwarforge.make_rbg import preprocess_cutout
from dwarforge.zoobot_utils import ensemble_predict, load_models

logger = logging.getLogger(__name__)


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


def cpu_preprocess_worker(tile_queue, gpu_task_queue, error_queue, done_event, config):
    """
    CPU worker that reads and preprocesses image data from tiles.

    Args:
        tile_queue: Queue containing tile info dictionaries to process
        gpu_task_queue: Queue to place preprocessed data for GPU inference
        error_queue: Queue to report errors
        done_event: Event to signal when preprocessing is done
        config: Configuration dictionary
    """
    process_name = current_process().name
    logger.info(f'Preprocessor {process_name} started')

    process_batch_size = config['gpu']['process_batch_size']

    try:
        while not done_event.is_set():
            try:
                # Get a tile from the queue with timeout
                tile_info = tile_queue.get(timeout=1)

                # Check for termination signal
                if tile_info is None:
                    logger.info(f'Preprocessor {process_name} received termination signal')
                    break

                tile_number = tile_info['tile_number']
                h5_path = tile_info['h5_path']

                logger.info(f'Preprocessor {process_name} processing tile {tile_number}')

                # Read the H5 file
                try:
                    with h5py.File(h5_path, 'r') as h5_file:
                        # Check if images dataset exists
                        if 'images' not in h5_file:
                            error_queue.put((tile_info, f'No images dataset found in {h5_path}'))
                            continue

                        # Read images and object IDs
                        cutouts = h5_file['images'][:]  # type: ignore
                        object_ids = h5_file['unique_id'][:]  # type: ignore

                        if len(cutouts) == 0:  # type: ignore
                            error_queue.put((tile_info, f'No images found in {h5_path}'))
                            continue

                        # Preprocess images
                        preprocessed_images = process_cutouts(
                            cutouts, process_batch_size=process_batch_size
                        )

                        # Create task package for GPU
                        task = {
                            'tile_info': tile_info,
                            'preprocessed_images': preprocessed_images,
                            'object_ids': object_ids,
                        }

                        # Put the preprocessed task in the GPU queue
                        gpu_task_queue.put(task)
                        logger.info(
                            f'Preprocessor {process_name} queued tile {tile_number} for GPU (objects: {len(preprocessed_images)})'
                        )

                except Exception as e:
                    error_msg = f'Error preprocessing tile {tile_number}: {str(e)}\n{traceback.format_exc()}'
                    logger.error(error_msg)
                    error_queue.put((tile_info, error_msg))

            except Empty:
                # Queue is empty but we're not done yet
                time.sleep(0.1)  # Prevent CPU spinning
                continue

    except Exception as e:
        error_msg = (
            f'Fatal error in preprocessor {process_name}: {str(e)}\n{traceback.format_exc()}'
        )
        logger.error(error_msg)
        error_queue.put((None, error_msg))

    finally:
        logger.info(f'Preprocessor {process_name} shutting down')
        # Clean up
        gc.collect()


def gpu_worker(gpu_task_queue, results_queue, error_queue, done_event, config):
    """
    Worker process that runs on GPU to perform model inference.

    Args:
        gpu_task_queue: Queue containing preprocessed image data
        results_queue: Queue to place inference results
        error_queue: Queue to report errors
        done_event: Event to signal when all tasks are processed
        config: Configuration dictionary
    """
    logger.info('GPU worker started')

    # Check for GPU
    if not torch.cuda.is_available():
        error_msg = 'GPU worker started but no CUDA device available'
        logger.error(error_msg)
        error_queue.put((None, error_msg))
        return

    # Set the device
    device = torch.device('cuda')
    logger.info(f'GPU worker using device: {torch.cuda.get_device_name(device.index)}')

    # Load all models once
    try:
        logger.info('GPU worker loading models')
        models = load_models(config['model_paths'])

        if not models:
            error_msg = 'GPU worker failed to load any models'
            logger.error(error_msg)
            error_queue.put((None, error_msg))
            return

        # Move models to GPU and set to eval mode
        for model in models:
            model.to(device)
            model.eval()

        logger.info(f'GPU worker successfully loaded {len(models)} models')

    except Exception as e:
        error_msg = f'Error loading models on GPU: {str(e)}\n{traceback.format_exc()}'
        logger.error(error_msg)
        error_queue.put((None, error_msg))
        return

    # Get batch size from config
    batch_size = config['gpu']['batch_size']

    # Process tiles from queue
    try:
        while not done_event.is_set():
            try:
                # Get a task from the queue with timeout
                task = gpu_task_queue.get(timeout=1)

                # Check for termination signal
                if task is None:
                    logger.info('GPU worker received termination signal')
                    break

                tile_info = task['tile_info']
                tile_number = tile_info['tile_number']

                preprocessed_images = task['preprocessed_images']
                object_ids = task['object_ids']

                logger.info(
                    f'GPU worker processing tile {tile_number} with {len(preprocessed_images)} objects'
                )

                try:
                    # Run ensemble prediction on GPU
                    predictions = ensemble_predict(
                        models, preprocessed_images, batch_size=batch_size, device=device
                    )

                    # Put results in results queue
                    result = {
                        'tile_info': tile_info,
                        'predictions': predictions,
                        'object_ids': object_ids,
                    }

                    results_queue.put(result)
                    logger.info(f'GPU worker completed tile {tile_number}')

                except Exception as e:
                    error_msg = f'Error processing tile {tile_number} on GPU: {str(e)}\n{traceback.format_exc()}'
                    logger.error(error_msg)
                    error_queue.put((tile_info, error_msg))

            except Empty:
                # Queue is empty but we're not done yet
                time.sleep(0.1)  # Prevent CPU spinning
                continue

    except Exception as e:
        error_msg = f'Fatal error in GPU worker: {str(e)}\n{traceback.format_exc()}'
        logger.error(error_msg)
        error_queue.put((None, error_msg))

    finally:
        # Clean up GPU resources
        logger.info('GPU worker shutting down and cleaning up resources')
        for model in models:
            del model

        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()


def cpu_write_worker(results_queue, error_queue, completion_queue, done_event, config):
    """
    CPU worker that writes inference results to H5 and parquet files.

    Args:
        results_queue: Queue containing inference results
        error_queue: Queue to report errors
        completion_queue: Queue to report successful completions
        done_event: Event to signal when all results are processed
        config: Configuration dictionary
    """
    process_name = current_process().name
    logger.info(f'Writer {process_name} started')

    try:
        while not done_event.is_set():
            try:
                # Get a result from the queue with timeout
                result = results_queue.get(timeout=1)

                # Check for termination signal
                if result is None:
                    logger.info(f'Writer {process_name} received termination signal')
                    break

                tile_info = result['tile_info']
                tile_number = tile_info['tile_number']
                h5_path = tile_info['h5_path']
                h5_temp_path = f'{h5_path}.temp'
                parquet_path = tile_info['parquet_path']
                parquet_temp_path = f'{parquet_path}.temp'

                predictions = result['predictions']
                object_ids = result['object_ids']

                logger.info(f'Writer {process_name} saving results for tile {tile_number}')

                try:
                    # === SAFELY UPDATE H5 FILE ===
                    # Copy the original file to temporary path
                    shutil.copy2(h5_path, h5_temp_path)

                    # Update the temporary file
                    with h5py.File(h5_temp_path, 'r+') as h5_temp:
                        # Remove existing predictions if present
                        if 'zoobot_pred_v2' in h5_temp:
                            del h5_temp['zoobot_pred_v2']

                        # Add new predictions
                        h5_temp.create_dataset(
                            'zoobot_pred_v2', data=predictions.astype(np.float32)
                        )
                        h5_temp.flush()

                    # Atomic rename to replace original
                    os.replace(h5_temp_path, h5_path)

                    # === SAFELY UPDATE PARQUET FILE ===
                    # Read original parquet file
                    df = pd.read_parquet(parquet_path)

                    # Create mapping from object ID to prediction
                    prediction_map = dict(zip(object_ids, predictions, strict=False))

                    # Update dataframe with predictions
                    df['zoobot_pred_v2'] = df['unique_id'].map(prediction_map)

                    # Write to temporary file
                    df.to_parquet(parquet_temp_path, index=False)

                    # Atomic rename
                    os.replace(parquet_temp_path, parquet_path)

                    # Signal successful completion
                    completion_queue.put(tile_info)

                    logger.info(
                        f'Writer {process_name} successfully saved results for tile {tile_number}'
                    )

                except Exception as e:
                    error_msg = f'Error writing results for tile {tile_number}: {str(e)}\n{traceback.format_exc()}'
                    logger.error(error_msg)
                    error_queue.put((tile_info, error_msg))

                    # Clean up any temporary files
                    try:
                        if os.path.exists(h5_temp_path):
                            os.remove(h5_temp_path)
                        if os.path.exists(parquet_temp_path):
                            os.remove(parquet_temp_path)
                    except Exception:
                        pass

            except Empty:
                # Queue is empty but we're not done yet
                time.sleep(0.1)
                continue

    except Exception as e:
        error_msg = f'Fatal error in writer {process_name}: {str(e)}\n{traceback.format_exc()}'
        logger.error(error_msg)
        error_queue.put((None, error_msg))

    finally:
        logger.info(f'Writer {process_name} shutting down')
        gc.collect()


def read_tile_data(filepath):
    """
    Read tile numbers from file.

    Args:
        filepath (str): path to file

    Returns:
        numpy.ndarray: array of tile strings
    """
    # Read all lines from the file
    with open(filepath) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Create numpy array
    tile_data = np.array(lines)

    # If array is empty, return empty array
    if tile_data.size == 0:
        return tile_data

    # Return array (will be 0D for single entry, 1D for multiple entries)
    return tile_data
