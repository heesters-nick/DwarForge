import gc
import logging
import os
import re

import h5py
import numpy as np
import pandas as pd
from dwarforge.logging_setup import setup_logger
from dwarforge.make_rbg import preprocess_cutout
from tqdm import tqdm

setup_logger(
    log_dir='./logs',
    name='combine_h5_files',
    logging_level=logging.INFO,
)
logger = logging.getLogger()


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


def combine_h5_files(
    input_tiles,
    base_dir,
    output_dir,
    objects_per_file=1000,
    label_filters=None,  # None for all, or list like ['nan', 0, 1]
    prefix='combined',
    preprocess=False,
    prep_mode='training',
):
    """
    Combine objects from multiple h5 files into larger files with objects_per_file objects.

    Args:
        input_tiles (list): List of tile number strings in the format '123_456'
        base_dir (str): Base directory containing tile subdirectories
        output_dir (str): Directory to save combined files
        objects_per_file (int): Number of objects per output file
        label_filters (list): List of labels to include, None for all objects
        prefix (str): Prefix for output files
    """
    if label_filters is not None and not isinstance(label_filters, (list, tuple)):
        raise ValueError('label_filters must be None or a list/tuple of values')

    os.makedirs(output_dir, exist_ok=True)

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
        input_file = os.path.join(base_dir, tile, 'gri', f'{tile}_matched_cutouts_full_res.h5')

        if not os.path.exists(input_file):
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
                        assert (
                            cutouts.ndim == 4
                        ), f'Unexpected cutout shape: {cutouts.shape}, expected 4 dimensions'
                        assert (
                            cutouts.shape[1] == 3
                        ), f'Expected 3 bands, got shape: {cutouts.shape}'
                        cutout_stack = np.zeros_like(cutouts, dtype=np.float32)
                        for i, cutout in enumerate(cutouts):
                            cutout_stack[i] = preprocess_cutout(cutout, mode=prep_mode)
                    else:
                        cutout_stack = np.array(f['images'])[subset_mask]

                    combined_data['images'].extend(cutout_stack)
                    combined_data['ra'].extend(np.array(f['ra'])[subset_mask])
                    combined_data['dec'].extend(np.array(f['dec'])[subset_mask])
                    combined_data['tile'].extend([np.array(f['tile'])] * objects_to_take)  # type: ignore
                    combined_data['known_id'].extend(np.array(f['known_id'])[subset_mask])
                    combined_data['label'].extend(np.array(f['label'])[subset_mask])
                    combined_data['zspec'].extend(np.array(f['zspec'])[subset_mask])

                    object_counter += objects_to_take

                    # If we have exactly objects_per_file objects, save the file
                    if object_counter == objects_per_file:
                        save_combined_file(
                            combined_data,
                            output_dir,
                            prefix,
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
                            assert (
                                cutouts.ndim == 4
                            ), f'Unexpected cutout shape: {cutouts.shape}, expected 4 dimensions'
                            assert (
                                cutouts.shape[1] == 3
                            ), f'Expected 3 bands, got shape: {cutouts.shape}'
                            cutout_stack = np.zeros_like(cutouts, dtype=np.float32)
                            for i, cutout in enumerate(cutouts):
                                cutout_stack[i] = preprocess_cutout(cutout)
                        else:
                            cutout_stack = np.array(f['images'])[remaining_mask]

                        combined_data['images'].extend(cutout_stack)
                        combined_data['ra'].extend(np.array(f['ra'])[remaining_mask])
                        combined_data['dec'].extend(np.array(f['dec'])[remaining_mask])
                        combined_data['tile'].extend(
                            [np.array(f['tile'])] * (n_objects - objects_to_take)  # type: ignore
                        )
                        combined_data['known_id'].extend(np.array(f['known_id'])[remaining_mask])
                        combined_data['label'].extend(np.array(f['label'])[remaining_mask])
                        combined_data['zspec'].extend(np.array(f['zspec'])[remaining_mask])

                        object_counter += n_objects - objects_to_take  # type: ignore

        except Exception as e:
            logger.error(f'Error processing {input_file}: {e}')

    # Save any remaining data
    if object_counter > 0:
        try:
            with h5py.File(input_file, 'r') as f:
                band_names = np.array(f['band_names'])
        except Exception as e:
            logger.error(f'Error reading band names from last file: {e}')
            band_names = np.array(['whigs-g', 'cfis_lsb-r', 'ps-i'], dtype='S')

        save_combined_file(combined_data, output_dir, prefix, file_counter, band_names)
        total_saved_files += 1
        total_objects += object_counter

    logger.info(f'Saved {total_objects} objects across {total_saved_files} files')


def save_combined_file(combined_data, output_dir, prefix, counter, band_names):
    """Save combined data to a new h5 file."""
    output_file = os.path.join(output_dir, f'{prefix}_{counter}.h5')

    with h5py.File(output_file, 'w', libver='latest') as f:
        dt = h5py.special_dtype(vlen=str)

        f.create_dataset('images', data=np.array(combined_data['images'], dtype=np.float32))
        f.create_dataset('ra', data=np.array(combined_data['ra'], dtype=np.float32))
        f.create_dataset('dec', data=np.array(combined_data['dec'], dtype=np.float32))
        f.create_dataset('tile', data=np.array(combined_data['tile'], dtype=np.int32))
        f.create_dataset('known_id', data=np.array(combined_data['known_id']), dtype=dt)
        f.create_dataset('label', data=np.array(combined_data['label'], dtype=np.float32))
        f.create_dataset('zspec', data=np.array(combined_data['zspec'], dtype=np.float32))
        f.create_dataset('band_names', data=band_names)


if __name__ == '__main__':
    base_dir = '/arc/projects/unions/ssl/data/raw/tiles/dwarforge'
    project_dir = '/arc/home/heestersnick/dwarforge'
    output_dir = os.path.join('/arc/projects/unions/ssl/data/raw/tiles', 'combined_h5_files')

    label_filter = [0]  # ['nan', 0, 1] or None to include all objects
    file_prefix = 'combined'
    objects_per_file = 5000
    # number of tiles to combine, None -> take all tiles
    number_of_tiles = 500
    # preprocess cutouts, i.e., make rgb?
    preprocess_cutouts = True
    # preprocessing mode, 'vis' or 'training'. vis = missing channel -> average of other two
    preprocessing_mode = 'vis'

    # read input tile df
    tables = os.path.join(project_dir, 'tables')
    os.makedirs(tables, exist_ok=True)
    tile_df_file = 'dwarf_tiles_gri.csv'
    table_path = os.path.join(tables, tile_df_file)
    tile_df = pd.read_csv(table_path)
    # unique tiles
    tiles = np.unique(tile_df['tile'].values)  # type: ignore
    # format tile list
    tiles = transform_list(tiles)
    # select input tiles from full tile list
    input_tiles = tiles[:number_of_tiles]

    logger.info(
        f'Found {len(tiles)} tiles in the input dataframe. Combining data from {number_of_tiles} of them.'
    )

    combine_h5_files(
        input_tiles=input_tiles,
        base_dir=base_dir,
        output_dir=output_dir,
        objects_per_file=objects_per_file,
        label_filters=label_filter,
        prefix=file_prefix,
        preprocess=preprocess_cutouts,
        prep_mode=preprocessing_mode,
    )
