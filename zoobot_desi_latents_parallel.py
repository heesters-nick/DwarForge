import datetime
import logging
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Literal, Optional, Union

import h5py
import numpy as np
import pandas as pd
import pywt
import torch
from scipy.ndimage import binary_dilation, label
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from zoobot.pytorch.training.representations import ZoobotEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = 1e-8

from logging_setup import setup_logger  # noqa: E402

setup_logger(
    log_dir='./logs',
    name='zoobot_desi_latents_v2',
    logging_level=logging.INFO,
)
logger = logging.getLogger()

NUM_CORES = 6
DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'  # Store as string for workers
DEVICE = torch.device(DEVICE_STR)
EPSILON = 1e-8
MODEL_NAME = 'hf_hub:mwalmsley/zoobot-encoder-convnext_nano'  # Store model name

# --- Global placeholder for worker model ---
worker_model = None


def preprocess_cutout(
    cutout: np.ndarray, mode: str = 'vis', replace_anomaly: bool = True
) -> np.ndarray:
    """Create an RGB image from the cutout data and save or plot it.

    Args:
        cutout (numpy.ndarray): cutout data with shape (channels, height, width)
        mode (str, optional): mode of operation. Defaults to 'training'. Valid options are 'training' or 'vis'. Fills missing channels for visualization.

    Returns:
        numpy.ndarray: preprocessed image cutout

    """
    # map out bands to RGB
    cutout_red = cutout[2]  # i-band
    cutout_green = cutout[1]  # r-band
    cutout_blue = cutout[0]  # g-band

    # adjust zero-point for the g-band
    if np.count_nonzero(cutout_blue) > 0:
        cutout_blue = adjust_flux_with_zp(cutout_blue, 27.0, 30.0)

    if replace_anomaly:
        # replace anomalies
        cutout_red = detect_anomaly(cutout_red)
        cutout_green = detect_anomaly(cutout_green)
        cutout_blue = detect_anomaly(cutout_blue)

    # synthesize missing channel from the existing ones
    # longest valid wavelength is mapped to red, middle to green, shortest to blue
    if mode == 'vis':
        if np.count_nonzero(cutout_red > 1e-10) == 0:
            cutout_red = cutout_green
            cutout_green = (cutout_green + cutout_blue) / 2
        elif np.count_nonzero(cutout_green > 1e-10) == 0:
            cutout_green = (cutout_red + cutout_blue) / 2
        elif np.count_nonzero(cutout_blue > 1e-10) == 0:
            cutout_blue = cutout_red
            cutout_red = (cutout_red + cutout_green) / 2

    # stack the channels in the order red, green, blue
    cutout_prep = np.stack([cutout_red, cutout_green, cutout_blue], axis=-1)

    return cutout_prep


def generate_rgb(
    cutout: np.ndarray,
    scaling_type: Literal['asinh', 'linear'] = 'asinh',
    stretch: float = 125,
    Q: float = 7.0,
    gamma: float = 0.25,
) -> np.ndarray:
    """Create an RGB image from three bands of data preserving relative intensities.

    Processes multi-band astronomical data into a properly scaled RGB image
    suitable for visualization, handling high dynamic range and empty channels.

    Args:
        cutout: 3D array of shape (height, width, 3) with band data
        scaling_type: Type of scaling to apply ("asinh" or "linear")
        stretch: Scaling factor controlling overall brightness
        Q: Softening parameter for asinh scaling (higher = more linear)
        gamma: Gamma correction factor (lower = enhances faint features)

    Returns:
        Normalized RGB image with values in range [0, 1]

    Notes:
        For astronomical data with high dynamic range, "asinh" scaling is
        typically preferred as it preserves both bright and faint details.
    """
    frac = 0.1
    with np.errstate(divide='ignore', invalid='ignore'):
        red = cutout[:, :, 0]
        green = cutout[:, :, 1]
        blue = cutout[:, :, 2]

        # Check for zero channels
        red_is_zero = np.all(red == 0)
        green_is_zero = np.all(green == 0)
        blue_is_zero = np.all(blue == 0)

        # Compute average intensity before scaling choice (avoiding zero channels)
        nonzero_channels = []
        if not red_is_zero:
            nonzero_channels.append(red)
        if not green_is_zero:
            nonzero_channels.append(green)
        if not blue_is_zero:
            nonzero_channels.append(blue)

        if nonzero_channels:
            i_mean = sum(nonzero_channels) / len(nonzero_channels)
        else:
            i_mean = np.zeros_like(red)  # All channels are zero

        if scaling_type == 'asinh':
            # Apply asinh scaling
            if not red_is_zero:
                red = red * np.arcsinh(Q * i_mean / stretch) / (Q * i_mean)
            if not green_is_zero:
                green = green * np.arcsinh(Q * i_mean / stretch) / (Q * i_mean)
            if not blue_is_zero:
                blue = blue * np.arcsinh(Q * i_mean / stretch) / (Q * i_mean)
        elif scaling_type == 'asinh_frac':
            # Apply asinh scaling
            if not red_is_zero:
                red = (
                    red * np.arcsinh(Q * i_mean / stretch) * frac / (np.arcsinh(frac * Q) * i_mean)
                )
            if not green_is_zero:
                green = (
                    green
                    * np.arcsinh(Q * i_mean / stretch)
                    * frac
                    / (np.arcsinh(frac * Q) * i_mean)
                )
            if not blue_is_zero:
                blue = (
                    blue * np.arcsinh(Q * i_mean / stretch) * frac / (np.arcsinh(frac * Q) * i_mean)
                )
        elif scaling_type == 'linear':
            # Apply linear scaling
            if not red_is_zero:
                red = red * stretch
            if not green_is_zero:
                green = green * stretch
            if not blue_is_zero:
                blue = blue * stretch
        else:
            raise ValueError(f'Unknown scaling type: {scaling_type}')

        # Apply gamma correction while preserving sign
        if gamma is not None:
            if not red_is_zero:
                red_mask = abs(red) <= 1e-9  # type: ignore
                red = np.sign(red) * (abs(red) ** gamma)  # type: ignore
                red[red_mask] = 0

            if not green_is_zero:
                green_mask = abs(green) <= 1e-9  # type: ignore
                green = np.sign(green) * (abs(green) ** gamma)  # type: ignore
                green[green_mask] = 0

            if not blue_is_zero:
                blue_mask = abs(blue) <= 1e-9  # type: ignore
                blue = np.sign(blue) * (abs(blue) ** gamma)  # type: ignore
                blue[blue_mask] = 0
        # Stack the channels after scaling and gamma correction
        result = np.stack([red, green, blue], axis=-1).astype(np.float32)

    # back to original axis ordering
    result = np.moveaxis(result, -1, 0)

    return result


def adjust_flux_with_zp(
    flux: np.ndarray, current_zp: Union[float, int], standard_zp: Union[float, int]
) -> np.ndarray:
    """
    Adjust flux values to a standard zero-point.

    Args:
        flux (numpy.ndarray): Flux values to adjust.
        current_zp (float/int): Current zero-point of the flux values.
        standard_zp (float/int): Standard zero-point to adjust to.

    Returns:
        numpy.ndarray: Adjusted flux values.
    """
    adjusted_flux = flux * 10 ** (-0.4 * (current_zp - standard_zp))
    return adjusted_flux


def detect_anomaly(
    image: np.ndarray,
    zero_threshold: float = 0.05,
    min_size: int = 50,
    replace_anomaly: bool = True,
    dilate_mask: bool = True,
    dilation_iters: int = 1,
) -> np.ndarray:
    """
    Detect and replace anomalies in an image using wavelet decomposition.

    This function analyzes an astronomical image to identify anomalous regions
    by performing wavelet decomposition and identifying regions with minimal
    fluctuations below a threshold. It can optionally replace detected anomalous
    pixels with zeros.

    Args:
        image: Input astronomical image to process
        zero_threshold: Fluctuation threshold below which an anomaly is detected
        min_size: Minimum connected pixel count to be considered an anomaly
        replace_anomaly: Whether to set anomalous pixels to zero
        dilate_mask: Whether to expand the detected anomaly mask
        dilation_iters: Number of dilation iterations if dilate_mask is True

    Returns:
        Processed image with anomalies optionally replaced

    Notes:
        This function uses Haar wavelet decomposition to identify regions with
        suspiciously low variation, which often indicate detector artifacts or
        other non-astronomical features in the image.
    """
    # replace nan values with zeros
    image[np.isnan(image)] = 0.0

    # Perform a 2D Discrete Wavelet Transform using Haar wavelets
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs  # Decomposition into approximation and details

    # Create binary masks where wavelet coefficients are below the threshold
    mask_horizontal = np.abs(cH) <= zero_threshold
    mask_vertical = np.abs(cV) <= zero_threshold
    mask_diagonal = np.abs(cD) <= zero_threshold

    masks = [mask_diagonal, mask_horizontal, mask_vertical]

    # Create a global mask to accumulate all anomalies
    global_mask = np.zeros_like(image, dtype=bool)
    # Create masks for each component
    component_masks = np.zeros((3, cA.shape[0], cA.shape[1]), dtype=bool)
    anomalies = np.zeros(3, dtype=bool)
    for i, mask in enumerate(masks):
        # Apply connected-component labeling to find connected regions in the mask
        labeled_array, num_features = label(mask)  # type: ignore

        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

        # Check if any component is larger than the minimum size
        anomaly_detected = np.any(component_sizes[1:] >= min_size)
        anomalies[i] = anomaly_detected

        if not anomaly_detected:
            continue

        # Prepare to accumulate a total mask
        total_feature_mask = np.zeros_like(image, dtype=bool)

        # Loop through all labels to find significant components
        for component_label in range(1, num_features + 1):  # Start from 1 to skip background
            if component_sizes[component_label] >= min_size:
                # Create a binary mask for this component
                component_mask = labeled_array == component_label
                # add component mask to component masks
                component_masks[i] |= component_mask
                # Upscale the mask to match the original image dimensions
                upscaled_mask = np.kron(component_mask, np.ones((2, 2), dtype=bool))
                # Accumulate the upscaled feature mask
                total_feature_mask |= upscaled_mask

        # Accumulate global mask
        global_mask |= total_feature_mask
        # Dilate the masks to catch some odd pixels on the outskirts of the anomaly
        if dilate_mask:
            global_mask = binary_dilation(global_mask, iterations=dilation_iters)
            for j, comp_mask in enumerate(component_masks):
                component_masks[j] = binary_dilation(comp_mask, iterations=dilation_iters)
    # Replace the anomaly with zeros
    if replace_anomaly:
        image[global_mask] = 0.0

    return image


def init_worker(m_name, d_str):
    """Initializer function called once per worker process."""
    global worker_model
    # Using print for initializer phase as logging setup can be complex here
    print(f'[Worker {os.getpid()}] Initializing and loading model {m_name} onto {d_str}...')
    try:
        # Call the existing model loading function
        worker_model = load_zoobot_model(name=m_name, device_str=d_str)
        print(f'[Worker {os.getpid()}] Model loaded successfully in initializer.')
    except Exception as e:
        print(f'[Worker {os.getpid()}] !!! ERROR: Failed to load model in initializer: {e}')
        worker_model = None  # Ensure it's None if loading failed


def load_zoobot_model(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', device_str='cpu'):
    """Loads the Zoobot model onto the specified device."""
    local_device = torch.device(device_str)
    model = ZoobotEncoder.load_from_name(name)
    model.freeze()
    model.eval()
    model = model.to(local_device)
    # Optional: Add logging inside if needed, but basic print in init_worker might suffice
    logger.info(
        f'[Worker {os.getpid()}] Model {name} loaded to {device_str}. Device: {next(model.parameters()).device}'
    )
    return model


def run_inference(model, images_np, batch_size=256, device=None):
    """
    Runs inference on a batch of images using the provided model.

    Args:
        model: The loaded Zoobot model.
        images_np: numpy array of shape (N, 3, H, W), float32.
        batch_size: Inference batch size.
        device: The torch device the model is on.

    Returns:
        latent representations: numpy array of shape (N, representation_dim)
    """
    # 1. Convert to tensor
    images_tensor = torch.tensor(images_np, dtype=torch.float32)

    # 2. Create DataLoader (Consider num_workers=0 for simplicity within a worker process)
    dataset = TensorDataset(images_tensor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=False
    )  # pin_memory=False might be safer in mp

    # 3. Run inference
    latent_representations = []
    model.eval()  # Ensure eval mode

    if device is None:
        device = next(model.parameters()).device  # Infer device from model if not passed

    with torch.no_grad():
        # No tqdm here to avoid interleaved progress bars from workers
        # Progress is tracked by the main process iterating over pool results
        for batch in dataloader:
            inputs = batch[0].to(device)
            latent = model(inputs)
            latent_representations.append(latent.cpu().numpy())

    if not latent_representations:  # Handle case where input images_np was empty
        # Infer representation dimension from the model if possible, otherwise use default
        rep_dim = getattr(
            model, 'representation_dim', 640
        )  # Example: adjust if model has attribute
        return np.empty((0, rep_dim), dtype=np.float32)

    return np.concatenate(latent_representations)


def process_single_tile(args):
    """
    Worker function to process objects within a single tile.
    Assumes model is pre-loaded by initializer into global 'worker_model'.
    """
    # Access the globally loaded model within this worker process
    global worker_model

    # 1. Unpack arguments (model_name and device_str are removed)
    tile, tile_objects_df, base_path, preprocessing_params = args
    tile_formatted = str(tile)
    tile_path = (
        f'{base_path}/{tile_formatted}/gri/{tile_formatted}_matched_cutouts_full_res_final.h5'
    )

    # --- Crucial Check: Ensure model was loaded by initializer ---
    if worker_model is None:
        error_message = (
            f'Model not available in worker {os.getpid()}. Skipping tile {tile_formatted}.'
        )
        # Using print as logger might not be configured per-worker easily
        print(f'ERROR: {error_message}')
        # Define expected metadata keys for empty return
        metadata_keys = ['unique_id', 'ra', 'dec', 'tile', 'z', 'zerr']
        return {
            'latent_vectors': np.empty((0, 640), dtype=np.float32),  # Specify shape and dtype
            'metadata': {
                key: np.array([]) for key in metadata_keys
            },  # Return empty arrays for all keys
            'objects_processed': 0,
            'objects_not_found': len(tile_objects_df),  # Count all as not found if worker is broken
            'error': error_message,
        }
    # --- End Check ---

    # Use the pre-loaded model
    model = worker_model
    try:
        # Infer device from the loaded model itself
        model_device = next(model.parameters()).device
    except Exception:
        # Fallback if getting device fails (shouldn't happen if model loaded OK)
        global DEVICE_STR  # Access the global device string constant if needed
        model_device = torch.device(DEVICE_STR)

    # Initialize results for this tile
    # Define expected metadata keys upfront
    metadata_keys = ['unique_id', 'ra', 'dec', 'tile', 'z', 'zerr']
    tile_latent_vectors_list = []  # Use a list to append results
    tile_metadata_list = {key: [] for key in metadata_keys}  # List for each metadata key

    objects_processed_in_tile = 0
    objects_not_found_in_tile = 0
    error_occurred = False
    error_message = ''

    try:
        # 3. Process the HDF5 file for the tile
        with h5py.File(tile_path, 'r') as src_file:
            # --- Efficient Object Selection & Loading ---
            target_ids = tile_objects_df['unique_id'].values
            num_targets_in_catalog = len(target_ids)

            if 'unique_id' not in src_file:
                objects_not_found_in_tile = num_targets_in_catalog
                raise FileNotFoundError(f"'unique_id' not found in {tile_path}")

            h5_unique_ids = src_file['unique_id'][:]  # type: ignore
            if h5_unique_ids.ndim == 0:  # type: ignore
                h5_unique_ids = np.array([h5_unique_ids])
            if h5_unique_ids.ndim > 1:  # type: ignore
                h5_unique_ids = h5_unique_ids.flatten()  # type: ignore

            h5_id_to_index = {uid: idx for idx, uid in enumerate(h5_unique_ids)}  # type: ignore
            valid_indices: List[int] = []
            catalog_indices_to_keep: List[int] = []

            for cat_idx, target_id in enumerate(target_ids):
                h5_idx = h5_id_to_index.get(target_id)
                if h5_idx is not None:
                    valid_indices.append(h5_idx)
                    catalog_indices_to_keep.append(cat_idx)

            n_valid = len(valid_indices)
            objects_not_found_in_tile = num_targets_in_catalog - n_valid

            if n_valid == 0:
                # No objects to process, return empty (handled outside try/except is cleaner)
                pass  # Let the function finish and return empty results naturally
            else:
                # Continue only if there are valid objects
                valid_catalog_objects = tile_objects_df.iloc[catalog_indices_to_keep]
                valid_indices_arr = np.array(valid_indices)
                sort_order = np.argsort(valid_indices_arr)
                sorted_h5_indices = valid_indices_arr[sort_order]
                valid_catalog_objects_sorted = valid_catalog_objects.iloc[sort_order]

                if 'images' not in src_file:
                    objects_not_found_in_tile += n_valid  # Count these as not found
                    raise FileNotFoundError(f"'images' not found in {tile_path}")

                images = src_file['images'][sorted_h5_indices]  # type: ignore

                # 4. Preprocess images for this tile
                preprocessed_images = np.zeros(
                    (n_valid, 3, images.shape[-2], images.shape[-1]),  # type: ignore
                    dtype=np.float32,
                )
                for i in range(n_valid):
                    cutout = images[i]  # type: ignore
                    # Assuming preprocess_cutout and generate_rgb are defined globally
                    preprocessed = preprocess_cutout(
                        cutout,  # type: ignore
                        mode=preprocessing_params['mode'],
                        replace_anomaly=preprocessing_params['replace_anomaly'],
                    )
                    rgb_image = generate_rgb(
                        preprocessed,
                        scaling_type=preprocessing_params['scaling_type'],
                        stretch=preprocessing_params['stretch'],
                        Q=preprocessing_params['Q'],
                        gamma=preprocessing_params['gamma'],
                    )
                    preprocessed_images[i] = rgb_image

                preprocessed_images_clean = np.nan_to_num(
                    preprocessed_images, nan=0.0, posinf=0.0, neginf=0.0
                )

                # 5. Run inference for this tile's images
                batch_vectors = run_inference(
                    model=model,  # Pass the global model
                    images_np=preprocessed_images_clean,
                    batch_size=256,  # Or make this configurable
                    device=model_device,
                )

                if np.isnan(batch_vectors).any():
                    print(
                        f'WARNING: NaN values detected in latent vectors for tile {tile_formatted}.'
                    )

                # 6. Store results for this tile
                tile_latent_vectors_list.append(batch_vectors)
                tile_metadata_list['unique_id'].append(
                    valid_catalog_objects_sorted['unique_id'].values
                )
                tile_metadata_list['ra'].append(valid_catalog_objects_sorted['ra'].values)
                tile_metadata_list['dec'].append(valid_catalog_objects_sorted['dec'].values)
                tile_metadata_list['tile'].append(np.full(n_valid, tile))
                tile_metadata_list['z'].append(valid_catalog_objects_sorted['z'].values)
                tile_metadata_list['zerr'].append(valid_catalog_objects_sorted['zerr'].values)
                objects_processed_in_tile = n_valid

                # Cleanup memory
                del images, preprocessed_images, preprocessed_images_clean, batch_vectors
                # Optional: Clear CUDA cache if using GPU, though might not be strictly needed per tile now
                # if model_device.type == 'cuda':
                #    torch.cuda.empty_cache()

    except FileNotFoundError as e:
        error_occurred = True
        error_message = f'File/Dataset not found for tile {tile_formatted}: {e}'
        objects_not_found_in_tile = len(tile_objects_df)  # All are considered not found
        objects_processed_in_tile = 0
    except KeyError as e:
        error_occurred = True
        error_message = f'KeyError (missing HDF5 dataset?) for tile {tile_formatted}: {e}'
        objects_not_found_in_tile = len(tile_objects_df)
        objects_processed_in_tile = 0
    except Exception as e:
        error_occurred = True
        error_message = (
            f'Unexpected error processing tile {tile_formatted}: {type(e).__name__} - {e}'
        )
        objects_not_found_in_tile = len(tile_objects_df)
        objects_processed_in_tile = 0
        # Optional: Print traceback for unexpected errors
        # import traceback
        # traceback.print_exc()

    # 7. Prepare results for returning
    final_vectors = (
        np.concatenate(tile_latent_vectors_list)
        if tile_latent_vectors_list
        else np.empty((0, 640), dtype=np.float32)
    )  # Use DTYPE

    final_tile_metadata = {}
    for key, data_list in tile_metadata_list.items():
        if data_list:
            final_tile_metadata[key] = np.concatenate(data_list)
        else:
            # Define empty array types correctly
            if key == 'unique_id':
                dtype = np.int64
            elif key in ['ra', 'dec', 'z', 'zerr']:
                dtype = np.float64
            else:
                dtype = object  # Tile will be object type
            final_tile_metadata[key] = np.array([], dtype=dtype)

    return {
        'latent_vectors': final_vectors,
        'metadata': final_tile_metadata,  # Return the dictionary of concatenated arrays
        'objects_processed': objects_processed_in_tile,
        'objects_not_found': objects_not_found_in_tile,
        'error': error_message if error_occurred else None,
    }


def process_objects_with_model_parallel(
    catalog_path: str,
    output_path: str,
    model_name: str,  # Pass model name string for initializer
    base_path: str = '/projects/unions/ssl/data/raw/tiles/dwarforge',
    preprocessing_params: Optional[Dict[str, Any]] = None,
    num_workers: int = NUM_CORES,  # Use global NUM_CORES
) -> Dict[str, int]:
    """
    Processes objects in parallel using multiple workers with initializer for model loading.
    """
    # 1. Load and prepare the catalog (Main process)
    global logger  # Ensure logger is accessible
    global DEVICE_STR  # Ensure device string is accessible for initargs

    logger.info(f'Starting parallel processing with {num_workers} workers.')
    logger.info(f'Using device for workers: {DEVICE_STR}')
    logger.info('Loading catalog...')
    try:
        catalog = pd.read_csv(catalog_path, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to load catalog '{catalog_path}': {e}")
        raise

    required_cols = ['tile', 'unique_id', 'ra', 'dec', 'z', 'zerr']  # Ensure these match catalog
    missing_cols = [col for col in required_cols if col not in catalog.columns]
    if missing_cols:
        raise ValueError(f'Required columns missing in catalog: {", ".join(missing_cols)}')

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f'Creating output directory: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

    # Set default preprocessing parameters
    if preprocessing_params is None:
        preprocessing_params = {
            'replace_anomaly': True,
            'scaling_type': 'asinh',
            'mode': 'vis',
            'stretch': 125,
            'Q': 7.0,
            'gamma': 0.25,
        }
    logger.info(f'Using preprocessing parameters: {preprocessing_params}')

    # 2. Group objects by tile
    logger.info('Grouping objects by tile...')
    grouped_catalog = catalog.groupby('tile')
    tile_groups = list(grouped_catalog)  # Convert to list for stable order and count
    num_tiles = len(tile_groups)
    total_objects_in_catalog = len(catalog)

    # 3. Prepare arguments for worker processes (MODIFIED)
    # Arguments no longer include model_name or device_str
    worker_args = []
    for tile, tile_objects_df in tile_groups:
        args = (
            tile,
            tile_objects_df.copy(),  # Pass a copy to potentially avoid issues if df is modified? (optional safety)
            base_path,
            preprocessing_params,
        )
        worker_args.append(args)

    # Initialize result accumulators
    all_latent_vectors_list: List[np.ndarray] = []
    # Define expected metadata keys based on process_single_tile return
    metadata_keys = ['unique_id', 'ra', 'dec', 'tile', 'z', 'zerr']
    all_metadata_accum_lists = {key: [] for key in metadata_keys}

    total_objects_processed_accum = 0  # Use a separate accumulator for the counter
    total_objects_not_found_accum = 0  # Use a separate accumulator for the counter

    logger.info(
        f'Processing {num_tiles} unique tiles with {total_objects_in_catalog} total objects...'
    )

    # 4. Create and run the multiprocessing pool (Using Initializer)
    start_method = 'spawn' if DEVICE_STR == 'cuda' else None
    mp_context = mp.get_context(start_method)
    actual_start_method = mp_context.get_start_method()
    logger.info(f'Using multiprocessing context with start method: {actual_start_method}')

    # Make sure num_workers is reasonable
    if num_workers <= 0:
        logger.warning('num_workers <= 0, setting to 1.')
        num_workers = 1

    try:
        with mp_context.Pool(
            processes=num_workers, initializer=init_worker, initargs=(model_name, DEVICE_STR)
        ) as pool:  # Pass initializer and its args
            # Use imap_unordered for potentially better load balancing
            results_iterator = pool.imap_unordered(process_single_tile, worker_args)
            # Set total to the number of objects, unit to 'obj'
            pbar = tqdm(total=total_objects_in_catalog, desc='Processing Objects', unit='obj')
            # Process results as they become available
            for result in results_iterator:
                if result['error']:
                    # Log errors reported by workers
                    logger.error(result['error'])  # Log error message from worker

                objects_in_batch = result['objects_processed']
                not_found_in_batch = result['objects_not_found']

                # Accumulate valid results (check processed count > 0)
                if objects_in_batch > 0:
                    if result['latent_vectors'] is not None and len(result['latent_vectors']) > 0:
                        all_latent_vectors_list.append(result['latent_vectors'])
                        # Accumulate metadata, checking consistency
                        for key in metadata_keys:
                            if key in result['metadata'] and len(result['metadata'][key]) > 0:
                                # Ensure metadata length matches vector length for this batch
                                if len(result['metadata'][key]) == len(result['latent_vectors']):
                                    all_metadata_accum_lists[key].append(result['metadata'][key])
                                else:
                                    logger.warning(
                                        f"!!! Mismatch in length between vectors ({len(result['latent_vectors'])}) and metadata '{key}' ({len(result['metadata'][key])}) for a processed tile. Skipping metadata for this key in this batch."
                                    )
                            # else: logger.warning(f"Metadata key '{key}' missing or empty in result from worker.")
                    else:
                        logger.warning(
                            f'Worker reported {result["objects_processed"]} processed objects but returned empty/None latent vectors. Skipping.'
                        )
                        objects_in_batch = 0

                # Update overall counters
                total_objects_processed_accum += (
                    objects_in_batch  # Add only successfully processed from this batch
                )
                total_objects_not_found_accum += not_found_in_batch

                pbar.update(objects_in_batch + not_found_in_batch)

            pbar.close()

    except Exception as pool_error:
        logger.error(f'An error occurred with the multiprocessing pool: {pool_error}')
        if 'pbar' in locals() and pbar:  # Close pbar if it exists on error
            pbar.close()
        raise  # Re-raise the error for now

    logger.info('Worker pool finished.')

    total_objects_processed = total_objects_processed_accum
    total_objects_not_found = total_objects_not_found_accum

    # 5. Consolidate final results
    logger.info('Concatenating final results...')
    final_latent_vectors = (
        np.concatenate(all_latent_vectors_list)
        if all_latent_vectors_list
        else np.empty((0, 640), dtype=np.float32)
    )

    final_metadata = {}
    for key, data_list in all_metadata_accum_lists.items():
        if data_list:
            final_metadata[key] = np.concatenate(data_list)
        else:
            # Determine appropriate empty dtype based on key
            if key == 'unique_id':
                dtype = np.int64
            elif key in ['ra', 'dec', 'z', 'zerr']:
                dtype = np.float64
            elif key == 'tile':
                dtype = object  # Tiles are likely strings/objects
            else:
                dtype = object  # Default fallback
            final_metadata[key] = np.array([], dtype=dtype)

    # Final consistency check (optional but recommended)
    num_vecs = len(final_latent_vectors)
    # Use a reliable metadata field like unique_id for length check
    num_ids = len(final_metadata.get('unique_id', []))
    if num_vecs != num_ids:
        logger.warning(
            f'!!! FINAL WARNING !!! Mismatch between total latent vectors ({num_vecs}) '
            f'and unique IDs ({num_ids}) after consolidation. Data might be corrupted.'
        )

    # 6. Save consolidated results to HDF5
    actual_objects_saved = len(final_latent_vectors)
    logger.info(f'Saving results for {actual_objects_saved} objects to {output_path}...')
    try:
        with h5py.File(output_path, 'w') as out_file:
            # Save latent vectors first
            out_file.create_dataset('latent_vectors', data=final_latent_vectors)

            # Save metadata
            for key, data in final_metadata.items():
                logger.debug(
                    f"Attempting to save metadata key: '{key}' with dtype: {data.dtype} and shape: {data.shape}"
                )  # Added debug log

                # Ensure data length matches vectors before saving
                if len(data) != actual_objects_saved:
                    logger.error(
                        f"!!! ERROR SAVING: Length mismatch for metadata key '{key}' ({len(data)}) vs vectors ({actual_objects_saved}). Skipping save for this key."
                    )
                    continue  # Skip saving this inconsistent key

                # --- Simplified String/Object Saving ---
                if (
                    data.dtype == object or data.dtype.kind in 'SU'
                ):  # Check for object or string types (S=bytes, U=unicode)
                    logger.debug(f"Saving '{key}' as variable-length string.")
                    try:
                        # Ensure all elements are explicitly converted to Python strings
                        str_data = data.astype(str)
                        # Use the special dtype for variable-length UTF-8 strings
                        dt = h5py.special_dtype(vlen=str)
                        out_file.create_dataset(key, data=str_data, dtype=dt)
                        logger.debug(f"Successfully saved '{key}'.")
                    except Exception as e_vlen:
                        logger.error(
                            f"!!! FAILED to save string/object array '{key}' as vlen strings: {e_vlen}"
                        )
                        # Optionally log more details about the data if it fails
                        logger.error(f"Data sample for failed key '{key}': {data[:5]}")

                # --- Numerical Data Saving ---
                else:
                    logger.debug(f"Saving '{key}' as numerical data.")
                    try:
                        out_file.create_dataset(key, data=data)
                        logger.debug(f"Successfully saved '{key}'.")
                    except Exception as e_num:
                        logger.error(f"!!! FAILED to save numerical array '{key}': {e_num}")

            # Add attributes
            out_file.attrs['processing_date'] = str(
                datetime.datetime.now()
            )  # Ensure datetime is imported
            out_file.attrs['catalog_path'] = catalog_path
            out_file.attrs['image_base_path'] = base_path
            out_file.attrs['model_name'] = model_name
            out_file.attrs['num_workers'] = num_workers
            out_file.attrs['start_method'] = actual_start_method
            out_file.attrs['total_objects_in_catalog'] = total_objects_in_catalog
            # Report counts based on what was actually concatenated and passed length checks
            out_file.attrs['objects_successfully_processed'] = actual_objects_saved
            # Calculate final 'not found' based on catalog size and what was saved
            final_objects_not_found = total_objects_in_catalog - actual_objects_saved
            out_file.attrs['objects_not_found_or_skipped'] = final_objects_not_found

            # Save preprocessing params
            for key, value in preprocessing_params.items():
                try:
                    out_file.attrs[f'preprocessing_{key}'] = value
                except TypeError:
                    logger.warning(
                        f"Could not save preprocessing param '{key}'={value} directly, saving as string."
                    )
                    out_file.attrs[f'preprocessing_{key}'] = str(value)

    except Exception as e:
        logger.error(f"Failed to save output HDF5 file '{output_path}': {e}")
        raise

    # Log final statistics
    logger.info('\n--- Parallel Processing Complete ---')
    logger.info(f'  Total objects in catalog: {total_objects_in_catalog}')
    logger.info(f'  Objects successfully processed & saved: {actual_objects_saved}')
    # Recalculate the final 'not found' based on what was actually saved
    final_objects_not_found = total_objects_in_catalog - actual_objects_saved
    logger.info(f'  Objects not found/skipped/failed: {final_objects_not_found}')
    logger.info(f'  Results saved to: {output_path}')
    logger.info('------------------------------------')

    return {
        'total_objects': total_objects_in_catalog,
        'objects_processed': actual_objects_saved,  # Return count of actually saved objects
        'objects_not_found_or_skipped': final_objects_not_found,
    }


if __name__ == '__main__':
    if DEVICE_STR == 'cuda':
        mp.set_start_method('spawn', force=True)  # Use spawn for CUDA safety
        logger.info("Set multiprocessing start method to 'spawn' for CUDA.")
    elif os.name != 'nt':  # 'fork' is default on Unix-like, 'spawn' on Windows
        try:
            mp.set_start_method('fork', force=True)  # Can try fork on CPU if preferred
            logger.info("Set multiprocessing start method to 'fork'.")
        except RuntimeError:
            logger.info("Could not force 'fork', using default.")

    data_dir = '/arc/projects/unions/ssl/data/raw/tiles/dwarforge'
    table_dir = '/arc/home/heestersnick/dwarforge/tables'
    output_dir = '/arc/home/heestersnick/dwarforge/desi'
    desi_unions_path = os.path.join(table_dir, 'all_desi_unions_matched.csv')

    os.makedirs(output_dir, exist_ok=True)

    # Define preprocessing params
    pp_params = {
        'replace_anomaly': True,
        'scaling_type': 'asinh',
        'mode': 'vis',
        'stretch': 125,
        'Q': 7.0,
        'gamma': 0.25,
    }

    start_time = time.time()
    logger.info(f'Starting parallel processing job with {NUM_CORES} cores...')

    try:
        stats = process_objects_with_model_parallel(
            catalog_path=desi_unions_path,
            output_path=os.path.join(
                output_dir, 'desi_unions_latent_parallel_v2.h5'
            ),  # Changed output filename slightly
            model_name=MODEL_NAME,  # Pass the name, not the loaded model
            base_path=data_dir,
            preprocessing_params=pp_params,
            num_workers=NUM_CORES,  # Explicitly pass number of workers
        )
        logger.info('Parallel processing completed successfully!')

    except Exception:
        logger.exception(
            'An error occurred during the main parallel processing execution.'
        )  # Log traceback
        # stats = None # Or handle partial results if needed

    end_time = time.time()
    logger.info(f'Total execution time: {(end_time - start_time)/60/60:.2f} hours.')
