import datetime
import logging
import os
from typing import Any, Literal

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
DTYPE = torch.float32
EPSILON = 1e-8

from dwarforge.logging_setup import setup_logger  # noqa: E402

setup_logger(
    log_dir='./logs',
    name='zoobot_desi_latents',
    logging_level=logging.INFO,
)
logger = logging.getLogger()


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
    flux: np.ndarray, current_zp: float | int, standard_zp: float | int
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


def load_zoobot_model(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano'):
    model = ZoobotEncoder.load_from_name(name)
    model.freeze()
    model.eval()
    model = model.to(DEVICE)
    logger.info(f'Model is on device: {next(model.parameters()).device}')
    logger.info(f'Model is in eval mode: {all(not p.requires_grad for p in model.parameters())}')
    logger.info(f'Model architecture is: {model.encoder.default_cfg["architecture"]}')
    return model


def run_inference(model, images_np, batch_size=256):
    """
    Args:
        images_np: numpy array of shape (N, 3, H, W)

    Returns:
        latent representations: numpy array of shape (N,640)
    """
    # 1. Verify input normalization
    assert images_np.dtype == np.float32, 'Input must be float32'

    # 2. Convert to tensor
    images_tensor = torch.tensor(images_np, dtype=torch.float32)

    # 3. Create DataLoader
    dataset = TensorDataset(images_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # 4. Run inference
    latent_representations = []
    model.eval()  # Redundant with freeze() but safe

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch[0].to(model.device)
            latent = model(inputs)
            latent_representations.append(latent.cpu().numpy())

    return np.concatenate(latent_representations)


def process_objects_with_model(
    catalog_path: str,
    output_path: str,
    model: Any,  # Ideally replace Any with torch.nn.Module or specific model type
    base_path: str = '/projects/unions/ssl/data/raw/tiles/dwarforge',
    accumulation_size: int = 5000,
    preprocessing_params: dict[str, Any] | None = None,
) -> dict[str, int]:
    """
    Process objects from a catalog with a machine learning model and save results.

    Uses efficient HDF5 slicing for image loading and ensures metadata alignment
    by sorting both images and corresponding metadata based on HDF5 index order.

    Parameters:
    -----------
    catalog_path : str
        Path to the catalog CSV file. Requires columns:
        'tile', 'unique_id', 'ra', 'dec', 'Z', 'LOGM_CIGALE'.
    output_path : str
        Path to the output H5 file where results will be saved.
    model : object
        The machine learning model for latent vector extraction.
    base_path : str, optional
        Base path to the directory containing tile subdirectories.
    accumulation_size : int, optional
        Number of preprocessed images to accumulate before running inference. Default is 5000.
    preprocessing_params : dict, optional
        Parameters for image preprocessing. Defaults will be used if None.

    Returns:
    --------
    dict
        Statistics about the processing (objects processed, not found, etc.)
    """
    # 1. Load and prepare the catalog
    logger.info('Loading catalog...')
    catalog = pd.read_csv(catalog_path, low_memory=False)

    # Check if required columns exist
    required_cols = ['tile', 'unique_id', 'ra', 'dec', 'z', 'zerr']
    for col in required_cols:
        if col not in catalog.columns:
            raise ValueError(f"Required column '{col}' not found in catalog")

    # Ensure unique_id is integer type if needed for matching H5 IDs
    # catalog['unique_id'] = catalog['unique_id'].astype(int) # Uncomment if necessary

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Set default preprocessing parameters if not provided
    if preprocessing_params is None:
        preprocessing_params = {
            'replace_anomaly': True,
            'scaling_type': 'asinh',
            'mode': 'vis',
            'stretch': 125,
            'Q': 7.0,
            'gamma': 0.25,
        }

    # 2. Group objects by tile
    logger.info('Grouping objects by tile...')
    grouped_catalog = catalog.groupby('tile')

    # Initialize data collection
    latent_vectors: list[np.ndarray] = []
    metadata: dict[str, list[np.ndarray]] = {
        'unique_id': [],
        'ra': [],
        'dec': [],
        'tile': [],
        'z': [],
        'zerr': [],
    }  # Lists to hold batches of metadata arrays

    # Counters for statistics
    total_objects = len(catalog)
    objects_processed = 0
    objects_not_found = 0

    # Accumulation buffers
    accumulated_images: list[np.ndarray] = []
    accumulated_metadata: dict[str, list[np.ndarray]] = {
        key: [] for key in metadata.keys()
    }  # Lists to hold batches before inference

    logger.info(
        f'Processing {len(grouped_catalog)} unique tiles with {total_objects} total objects...'
    )

    # 3. Process each tile
    for tile, tile_objects in tqdm(grouped_catalog, desc='Processing tiles'):
        tile_formatted = str(tile)
        # Construct file path (adjust if structure differs)
        tile_path = (
            f'{base_path}/{tile_formatted}/gri/{tile_formatted}_matched_cutouts_full_res_final.h5'
        )

        try:
            with h5py.File(tile_path, 'r') as src_file:
                # --- Efficient Object Selection & Loading ---

                # 1. Get target IDs required for this tile from the catalog
                target_ids = tile_objects['unique_id'].values
                num_targets_in_catalog = len(target_ids)

                # 2. Load unique IDs available in the H5 file
                if 'unique_id' not in src_file:
                    logger.warning(
                        f"Warning: 'unique_id' dataset not found in tile {tile_formatted}. Skipping."
                    )
                    objects_not_found += num_targets_in_catalog
                    continue
                h5_unique_ids = src_file['unique_id'][:]  # type: ignore
                # Ensure h5_unique_ids is 1D array
                if h5_unique_ids.ndim == 0:  # type: ignore
                    h5_unique_ids = np.array([h5_unique_ids])
                if h5_unique_ids.ndim > 1:  # type: ignore
                    h5_unique_ids = h5_unique_ids.flatten()  # Adjust if needed # type: ignore

                # 3. Find matches: Map H5 IDs to their indices
                h5_id_to_index = {uid: idx for idx, uid in enumerate(h5_unique_ids)}  # type: ignore

                # 4. Identify indices in H5 and corresponding catalog rows
                valid_indices: list[int] = []  # Indices within H5 file
                catalog_indices_to_keep: list[int] = []  # Indices within tile_objects

                for cat_idx, target_id in enumerate(target_ids):
                    h5_idx = h5_id_to_index.get(target_id)  # More efficient than 'in' + lookup
                    if h5_idx is not None:
                        valid_indices.append(h5_idx)
                        catalog_indices_to_keep.append(cat_idx)

                n_valid = len(valid_indices)
                num_not_found_in_tile = num_targets_in_catalog - n_valid
                objects_not_found += num_not_found_in_tile

                if n_valid == 0:
                    if num_targets_in_catalog > 0:
                        logger.warning(
                            f'Warning: No matching objects found in file {tile_path} for {num_targets_in_catalog} catalog entries.'
                        )
                    continue  # Skip to the next tile

                # 5. Get initial metadata rows in the original discovery order
                valid_catalog_objects = tile_objects.iloc[catalog_indices_to_keep]

                # --- Sorting for Efficient HDF5 Read & Metadata Alignment ---
                # 6. Get the order needed to sort the H5 indices
                valid_indices_arr = np.array(valid_indices)
                sort_order = np.argsort(valid_indices_arr)

                # 7. Apply sort order to get sorted H5 indices
                sorted_h5_indices = valid_indices_arr[sort_order]

                # 8. Apply the SAME sort order to the metadata DataFrame
                valid_catalog_objects_sorted = valid_catalog_objects.iloc[sort_order]
                # --- End Sorting ---

                # 9. Extract only the required images using sorted H5 indices
                if 'images' not in src_file:
                    logger.warning(
                        f"Warning: 'images' dataset not found in tile {tile_formatted}. Skipping {n_valid} found objects."
                    )
                    objects_not_found += n_valid  # Count these as not found now
                    continue
                images = src_file['images'][sorted_h5_indices]  # type: ignore
                # 'images' now corresponds row-by-row to 'valid_catalog_objects_sorted'

                # 10. Preprocess all extracted images for this tile
                # Assuming H, W are last dims in H5, C=3 is output of generate_rgb
                preprocessed_images = np.zeros(
                    (n_valid, 3, images.shape[-2], images.shape[-1]),  # type: ignore
                    dtype=np.float32,
                )

                for i in range(n_valid):
                    cutout = images[i]  # Already loaded efficiently # type: ignore
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

                # Set problematic values to 0
                preprocessed_images_clean = np.nan_to_num(
                    preprocessed_images, nan=0.0, posinf=0.0, neginf=0.0
                )

                # 11. Add preprocessed images to accumulation buffer
                accumulated_images.append(preprocessed_images_clean)

                # 12. Add SORTED metadata to accumulation buffer
                accumulated_metadata['unique_id'].append(
                    valid_catalog_objects_sorted['unique_id'].to_numpy()
                )
                accumulated_metadata['ra'].append(valid_catalog_objects_sorted['ra'].to_numpy())
                accumulated_metadata['dec'].append(valid_catalog_objects_sorted['dec'].to_numpy())
                accumulated_metadata['tile'].append(
                    np.full(n_valid, tile)
                )  # Use the actual tile variable
                accumulated_metadata['z'].append(valid_catalog_objects_sorted['z'].to_numpy())
                accumulated_metadata['zerr'].append(valid_catalog_objects_sorted['zerr'].to_numpy())

                # 13. Calculate total accumulated images
                total_accumulated = sum(arr.shape[0] for arr in accumulated_images)

                # 14. Run inference when accumulation threshold is met (Original logic)
                # Consider optimizing this part later if it becomes a bottleneck
                while total_accumulated >= accumulation_size:
                    logger.info(f'Running inference on {accumulation_size} accumulated images...')

                    # Concatenate accumulated images (Potential inefficiency here)
                    all_images = np.concatenate(accumulated_images)
                    inference_images = all_images[:accumulation_size]

                    try:
                        batch_vectors = run_inference(
                            model=model, images_np=np.nan_to_num(inference_images, nan=0.0)
                        )
                        if np.isnan(batch_vectors).any():
                            logger.warning(
                                '!!! WARNING: NaN values detected in latent vectors for a batch.'
                            )
                        latent_vectors.append(batch_vectors)  # Add results batch

                        # Process metadata for these images (Potential inefficiency here)
                        remaining_metadata: dict[str, list[np.ndarray]] = {
                            key: [] for key in metadata.keys()
                        }
                        processed_count_in_batch = (
                            0  # Track how much metadata corresponds to inference_images
                        )

                        for key in metadata.keys():
                            # Concatenate all buffered metadata for this key
                            all_metadata_values_list = accumulated_metadata[key]
                            if not all_metadata_values_list:
                                continue  # Skip if empty

                            all_values_key = np.concatenate(all_metadata_values_list)

                            # Store metadata for processed images
                            metadata[key].append(all_values_key[:accumulation_size])
                            processed_count_in_batch = len(
                                all_values_key[:accumulation_size]
                            )  # Update count

                            # Keep remaining metadata if any
                            if len(all_values_key) > accumulation_size:
                                remaining_values = all_values_key[accumulation_size:]
                                remaining_metadata[key] = [remaining_values]  # Store as list
                            else:
                                remaining_metadata[key] = []  # No remainder

                        # Keep remaining images
                        remaining_images = all_images[accumulation_size:]

                        # Update accumulation buffers
                        accumulated_images = [remaining_images] if len(remaining_images) > 0 else []
                        accumulated_metadata = remaining_metadata

                        objects_processed += processed_count_in_batch  # Use actual count

                        # Update total accumulated count *after* buffer update
                        total_accumulated = sum(arr.shape[0] for arr in accumulated_images)

                        # Free GPU memory if available
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(
                            f'Error running inference batch: {type(e).__name__} - {e}. Skipping batch.'
                        )
                        # Decide how to handle buffers - clear them or try to skip bad data?
                        # Simplest: Clear buffers and hope next tile works.
                        accumulated_images = []
                        accumulated_metadata = {key: [] for key in metadata.keys()}
                        total_accumulated = 0
                        break  # Exit the while loop for this tile after error

        except FileNotFoundError:
            num_expected = len(tile_objects)
            logger.error(
                f'Warning: File not found for tile {tile_formatted} at {tile_path}. Skipping {num_expected} objects.'
            )
            objects_not_found += num_expected
        except KeyError as e:
            logger.error(
                f'Error: Dataset missing in H5 file for tile {tile_formatted} (Path: {tile_path}). Missing key: {e}'
            )
            objects_not_found += len(tile_objects)
        except Exception as e:
            logger.error(f'Error processing tile {tile_formatted}: {type(e).__name__} - {e}')
            objects_not_found += len(tile_objects)  # Option: count errors as not found

    # Process any remaining images after the loop (Original logic)
    final_accumulated_count = sum(arr.shape[0] for arr in accumulated_images)
    if final_accumulated_count > 0:
        logger.info(f'Running inference on {final_accumulated_count} remaining images...')
        try:
            all_images = np.concatenate(accumulated_images)
            batch_vectors = run_inference(model=model, images_np=all_images)
            latent_vectors.append(batch_vectors)

            # Process remaining metadata
            for key in metadata.keys():
                if accumulated_metadata[key]:  # Check if list is not empty
                    all_values = np.concatenate(accumulated_metadata[key])
                    metadata[key].append(all_values)
                # Else: no remaining metadata for this key, do nothing

            objects_processed += final_accumulated_count

        except Exception as e:
            logger.error(f'Error processing final batch of images: {type(e).__name__} - {e}')
            # Note: These objects won't be counted in objects_processed if error occurs here

    # Concatenate all collected data (Original logic)
    logger.info('Concatenating final results...')
    all_latent_vectors = (
        np.concatenate(latent_vectors) if latent_vectors else np.empty((0, 640), dtype=np.float32)
    )  # Specify shape if empty

    all_metadata = {}
    for key in metadata:
        if metadata[key]:  # Check list has batches
            all_metadata[key] = np.concatenate(metadata[key])
        else:  # Create empty array of appropriate type if nothing was processed
            if key == 'unique_id':
                dtype = np.int64  # type: ignore[assignment]
            elif key in ['ra', 'dec', 'z', 'zerr']:
                dtype = np.float64  # type: ignore[assignment]
            elif key == 'tile':
                dtype = object  # type: ignore[assignment]  # Or infer from catalog dtype
            else:
                dtype = object  # type: ignore[assignment]
            all_metadata[key] = np.array([], dtype=dtype)

    # Check consistency (optional)
    num_vecs = len(all_latent_vectors)
    num_ids = len(all_metadata.get('unique_id', []))
    if num_vecs != num_ids:
        logger.warning(
            f'!!!WARNING!!! Mismatch between number of latent vectors ({num_vecs}) and unique IDs ({num_ids}). Check accumulation logic.'
        )
        # Fallback: Try to use the minimum length if saving? Or raise error?
        min_len = min(num_vecs, num_ids)
        all_latent_vectors = all_latent_vectors[:min_len]
        for key in all_metadata:
            all_metadata[key] = all_metadata[key][:min_len]
        objects_processed = min_len  # Adjust count

    # Create the output H5 file (Original logic)
    logger.info(f'Saving results for {len(all_latent_vectors)} objects to {output_path}...')
    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset('latent_vectors', data=all_latent_vectors)

        for key, data in all_metadata.items():
            # Handle string/object data saving if needed (e.g., tile)
            if data.dtype == object or data.dtype.kind in 'SU':  # String or object
                data = data.astype(h5py.string_dtype(encoding='utf-8'))
            out_file.create_dataset(key, data=data)

        # Add attributes
        out_file.attrs['processing_date'] = str(datetime.datetime.now())
        out_file.attrs['catalog_path'] = catalog_path
        out_file.attrs['image_base_path'] = base_path
        out_file.attrs['total_objects_in_catalog'] = total_objects
        out_file.attrs['objects_successfully_processed'] = (
            objects_processed  # Use final reliable count
        )
        out_file.attrs['objects_not_found_or_skipped'] = objects_not_found

        for key, value in preprocessing_params.items():
            # Attempt to save param; convert complex types if necessary
            try:
                out_file.attrs[f'preprocessing_{key}'] = value
            except TypeError:
                logger.error(
                    f"Warning: Could not save preprocessing param '{key}'={value} as HDF5 attribute."
                )
                out_file.attrs[f'preprocessing_{key}'] = str(value)

    # Log statistics
    logger.info('\nProcessing complete:')
    logger.info(f'  Total objects in catalog: {total_objects}')
    logger.info(f'  Objects successfully processed (latent vectors generated): {objects_processed}')
    logger.info(f'  Objects not found in H5 files or skipped due to errors: {objects_not_found}')
    # Note: total_objects might not equal processed + not_found if errors occurred during inference/saving

    # Return statistics
    return {
        'total_objects': total_objects,
        'objects_processed': objects_processed,
        'objects_not_found_or_skipped': objects_not_found,
    }


if __name__ == '__main__':
    data_dir = '/arc/projects/unions/ssl/data/raw/tiles/dwarforge'
    table_dir = '/arc/home/heestersnick/dwarforge/tables'
    output_dir = '/arc/home/heestersnick/dwarforge/desi'
    desi_unions_path = os.path.join(table_dir, 'all_desi_unions_matched.csv')

    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = load_zoobot_model()

    # Run the processing
    stats = process_objects_with_model(
        catalog_path=desi_unions_path,
        output_path=os.path.join(output_dir, 'desi_unions_latent.h5'),
        model=model,
        base_path=data_dir,
        accumulation_size=5000,
        preprocessing_params={
            'replace_anomaly': True,
            'scaling_type': 'asinh',
            'mode': 'vis',
            'stretch': 125,
            'Q': 7.0,
            'gamma': 0.25,
        },
    )

    logger.info('Processing completed successfully!')
