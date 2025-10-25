import gc
import logging
import re
import time
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from dwarforge.config import ensure_runtime_dirs, load_settings, settings_to_jsonable
from dwarforge.logging_setup import setup_logger
from dwarforge.make_rbg import preprocess_cutout
from dwarforge.utils import purge_previous_run

logger = logging.getLogger(__name__)

STR_DT = h5py.string_dtype(encoding='utf-8')  # variable-length UTF-8 strings


def transform_list(input_list):
    result = []
    for item in input_list:
        # Extract the numbers from the string using regex
        x, y = re.findall(r'\d+', item)
        # Format each number to a 3-digit format, adding leading zeros if necessary
        formatted_x = x.zfill(3)
        formatted_y = y.zfill(3)
        # Combine them with an underscore
        result.append(f'{formatted_x}_{formatted_y}')
    return result


def combine_h5_files():
    cfg = load_settings('configs/h5_aggregation_config.yaml')
    # remove previous run's database and logs if resume = false
    purge_previous_run(cfg)
    setup_logger(
        log_dir=cfg.paths.log_directory,
        name=cfg.logging.name,
        logging_level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        force=True,
    )
    logger.info('Config and logging initialized.')

    cfg_dict = settings_to_jsonable(cfg)
    # Print settings in human readable format
    cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False)
    logger.info(f'Resolved config (YAML):\n{cfg_yaml}')

    # make sure necessary directories exist
    ensure_runtime_dirs(cfg=cfg)

    label_filters = cfg.h5_aggregation.label_filters
    number_of_tiles = cfg.h5_aggregation.number_of_tiles
    aggregation_dir = cfg.paths.aggregate_directory
    in_file_suffix = cfg.h5_aggregation.in_file_suffix
    out_file_prefix = cfg.h5_aggregation.out_file_prefix
    objects_per_file = cfg.h5_aggregation.objects_per_file
    preprocess = cfg.h5_aggregation.preprocess_cutouts
    prep_mode = cfg.h5_aggregation.preprocessing_mode
    download_dir = cfg.paths.download_directory
    bands_to_combine = cfg.combination.bands_to_combine

    tile_df_path = cfg.h5_aggregation.tile_df_file
    tile_df = pd.read_csv(tile_df_path)

    tiles = np.unique(tile_df['tile'].to_numpy(dtype=str)).tolist()
    # format tile list
    tiles = transform_list(tiles)
    # select input tiles from full tile list
    input_tiles = tiles[:number_of_tiles]

    combined_data = {
        'images': [],
        'ra': [],
        'dec': [],
        'tile': [],
        'known_id': [],
        'label': [],
        'zspec': [],
    }

    file_counter = 1
    object_counter = 0
    total_objects = 0
    total_saved_files = 0

    logger.info(f'{input_tiles}')
    for tile in tqdm(input_tiles, desc='Processing tiles'):
        input_file = download_dir / tile / 'gri' / f'{tile}_{in_file_suffix}.h5'

        if not input_file.exists():
            logger.warning(f'Warning: File not found: {input_file}')
            continue

        try:
            with h5py.File(input_file, 'r') as f:
                labels = np.array(f['label'])
                # Create mask based on label filters
                if label_filters is None:
                    mask = np.ones_like(labels, dtype=bool)
                else:
                    mask = np.zeros_like(labels, dtype=bool)
                    for label_filter in label_filters:
                        if label_filter == 'nan':
                            mask |= np.isnan(labels)
                        else:
                            mask |= labels == label_filter

                if not np.any(mask):
                    continue

                # Get number of matching objects in this tile
                n_objects = np.sum(mask)
                masked_indices = np.where(mask)[0]
                logger.info(f'Found {n_objects} matching objects in tile {tile}')

                # Process objects up to the limit
                remaining_space = objects_per_file - object_counter
                objects_to_take = min(remaining_space, n_objects)

                if objects_to_take > 0:
                    # Take only the number of objects that will fit
                    subset_mask = np.zeros_like(mask, dtype=bool)
                    subset_mask[masked_indices[:objects_to_take]] = True

                    if preprocess:
                        cutouts = np.array(f['images'])[subset_mask]
                        assert cutouts.ndim == 4, (
                            f'Unexpected cutout shape: {cutouts.shape}, expected 4 dimensions'
                        )
                        assert cutouts.shape[1] == 3, (
                            f'Expected 3 bands, got shape: {cutouts.shape}'
                        )
                        cutout_stack = np.zeros_like(cutouts, dtype=np.float32)
                        for i, cutout in enumerate(cutouts):
                            cutout_stack[i] = preprocess_cutout(cutout, mode=prep_mode)
                    else:
                        cutout_stack = np.array(f['images'])[subset_mask]

                    combined_data['images'].extend(cutout_stack)
                    combined_data['ra'].extend(np.array(f['ra'])[subset_mask])
                    combined_data['dec'].extend(np.array(f['dec'])[subset_mask])
                    combined_data['tile'].extend(np.array(f['tile'])[subset_mask])
                    combined_data['known_id'].extend(np.array(f['known_id'])[subset_mask])
                    combined_data['label'].extend(np.array(f['label'])[subset_mask])
                    combined_data['zspec'].extend(np.array(f['zspec'])[subset_mask])

                    object_counter += objects_to_take

                    # If we have exactly objects_per_file objects, save the file
                    if object_counter == objects_per_file:
                        save_combined_file(
                            combined_data,
                            aggregation_dir,
                            out_file_prefix,
                            file_counter,
                            np.array(f['band_names']),
                        )
                        combined_data = {key: [] for key in combined_data}
                        total_objects += object_counter
                        object_counter = 0
                        file_counter += 1
                        total_saved_files += 1
                        gc.collect()

                    # If we have remaining objects in this tile, process them
                    if objects_to_take < n_objects:
                        remaining_mask = np.zeros_like(mask, dtype=bool)
                        remaining_mask[masked_indices[objects_to_take:]] = True

                        if preprocess:
                            cutouts = np.array(f['images'])[remaining_mask]
                            assert cutouts.ndim == 4, (
                                f'Unexpected cutout shape: {cutouts.shape}, expected 4 dimensions'
                            )
                            assert cutouts.shape[1] == 3, (
                                f'Expected 3 bands, got shape: {cutouts.shape}'
                            )
                            cutout_stack = np.zeros_like(cutouts, dtype=np.float32)
                            for i, cutout in enumerate(cutouts):
                                cutout_stack[i] = preprocess_cutout(cutout, mode=prep_mode)
                        else:
                            cutout_stack = np.array(f['images'])[remaining_mask]

                        combined_data['images'].extend(cutout_stack)
                        combined_data['ra'].extend(np.array(f['ra'])[remaining_mask])
                        combined_data['dec'].extend(np.array(f['dec'])[remaining_mask])
                        combined_data['tile'].extend(np.array(f['tile'])[remaining_mask])
                        combined_data['known_id'].extend(np.array(f['known_id'])[remaining_mask])
                        combined_data['label'].extend(np.array(f['label'])[remaining_mask])
                        combined_data['zspec'].extend(np.array(f['zspec'])[remaining_mask])

                        object_counter += n_objects - objects_to_take  # type: ignore

        except Exception as e:
            logger.error(f'Error processing {input_file}: {e}')

    # Save any remaining data
    if object_counter > 0:
        band_names = np.array(bands_to_combine, dtype=STR_DT)
        save_combined_file(
            combined_data, aggregation_dir, out_file_prefix, file_counter, band_names
        )
        total_saved_files += 1
        total_objects += object_counter

    logger.info(f'Saved {total_objects} objects across {total_saved_files} files')


def save_combined_file(combined_data, output_dir, prefix, counter, band_names):
    """Save combined data to a new h5 file."""
    output_file = output_dir / f'{prefix}_{counter}.h5'

    with h5py.File(output_file, 'w', libver='latest') as f:
        f.create_dataset('images', data=np.array(combined_data['images'], dtype=np.float32))
        f.create_dataset('ra', data=np.array(combined_data['ra'], dtype=np.float32))
        f.create_dataset('dec', data=np.array(combined_data['dec'], dtype=np.float32))
        f.create_dataset('tile', data=np.array(combined_data['tile'], dtype=np.int32))
        f.create_dataset('known_id', data=np.array(combined_data['known_id']), dtype=STR_DT)
        f.create_dataset('label', data=np.array(combined_data['label'], dtype=np.float32))
        f.create_dataset('zspec', data=np.array(combined_data['zspec'], dtype=np.float32))
        f.create_dataset('band_names', data=band_names, dtype=STR_DT)


if __name__ == '__main__':
    start = time.time()
    combine_h5_files()
    end = time.time()
    elapsed = end - start
    elapsed_string = str(timedelta(seconds=elapsed))
    hours, minutes, seconds = (
        np.float32(elapsed_string.split(':')[0]),
        np.float32(elapsed_string.split(':')[1]),
        np.float32(elapsed_string.split(':')[2]),
    )
    logger.info(
        f'Done! Execution took {hours} hours, {minutes} minutes, and {seconds:.2f} seconds.'
    )
