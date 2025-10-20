import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm import tqdm


def match_coordinates(detected_ra, detected_dec, train_ra, train_dec, max_separation=5.0):
    """
    Match coordinates within max_separation arcseconds
    Returns boolean array of matches and indices of matches
    """
    if len(train_ra) == 0:
        return np.zeros(len(detected_ra), dtype=bool), np.array([])

    detected_coords = SkyCoord(ra=detected_ra, dec=detected_dec, unit='deg')
    train_coords = SkyCoord(ra=train_ra, dec=train_dec, unit='deg')

    idx, sep, _ = detected_coords.match_to_catalog_sky(train_coords)
    matches = sep.arcsec <= max_separation

    return matches, idx


def process_single_file(file_path, train_df):
    """
    Read and process a single parquet file, adding training flag
    """
    try:
        df = pd.read_parquet(file_path)

        # Extract tile from file path
        tile = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

        # Initialize training flag column
        df['in_training_data'] = 0

        # Check if this tile exists in training data
        if tile in train_df['tile'].unique():
            # Get training data for this tile
            tile_train_data = train_df[train_df['tile'] == tile]

            # Perform coordinate matching
            coord_matches, match_idx = match_coordinates(
                df['ra'].values,
                df['dec'].values,
                tile_train_data['ra'].values,
                tile_train_data['dec'].values,
            )

            # Check for duplicate matches
            unique_matches, match_counts = np.unique(match_idx[coord_matches], return_counts=True)
            duplicates = unique_matches[match_counts > 1]

            if len(duplicates) > 0:
                print(f'\nTile {tile} has duplicate matches:')
                for dup_idx in duplicates:
                    dup_matches = np.where(match_idx == dup_idx)[0]
                    train_obj = tile_train_data.iloc[dup_idx]
                    print(
                        f'  Training object (ra={train_obj["ra"]:.4f}, dec={train_obj["dec"]:.4f}, id={train_obj["known_id"]})'
                    )
                    print(f'  matches {len(dup_matches)} objects in detection data:')
                    for match_idx in dup_matches:
                        det_obj = df.iloc[match_idx]
                        print(
                            f'    - ra={det_obj["ra"]:.4f}, dec={det_obj["dec"]:.4f}, id={det_obj["ID_known"]}'
                        )

            # Perform ID matching
            id_matches = df['ID_known'].isin(tile_train_data['known_id'])

            # Count matches
            n_coord_matches = np.count_nonzero(coord_matches)
            n_id_matches = np.count_nonzero(id_matches)
            n_training = len(tile_train_data)

            # Compare matches
            if n_coord_matches != n_training or n_id_matches != n_training:
                print(f'\nTile {tile} matching stats:')
                print(f'  - Coordinate matches: {n_coord_matches}/{n_training}')
                print(f'  - ID matches: {n_id_matches}/{n_training}')
                print(f'  - Unique coordinate matches: {len(unique_matches)}/{n_training}')

                # Check overlap between coordinate and ID matches
                both_matches = np.logical_and(coord_matches, id_matches)
                n_both = np.count_nonzero(both_matches)
                print(f'  - Objects matching both criteria: {n_both}')

            # Update training flag (using coordinate matches as primary criterion)
            df.loc[coord_matches, 'in_training_data'] = 1

            # Add a column for ID matches if you want to track both
            df['id_in_training'] = id_matches.astype(int)

        return df

    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return pd.DataFrame()


def gather_data(base_dir, train_df, num_processes=4):
    """
    Gather data from all parquet files and add training object flags
    """
    # Create the pattern to match the parquet files
    pattern = os.path.join(base_dir, '*_*', 'gri', '*_matched_detections.parquet')

    # Get list of all matching files
    parquet_files = glob.glob(pattern)

    if not parquet_files:
        raise ValueError(f'No parquet files found matching pattern: {pattern}')

    print(f'Found {len(parquet_files)} parquet files')

    # Create partial function with train_df
    process_file = partial(process_single_file, train_df=train_df)

    # Use multiprocessing to read and process files in parallel with tqdm
    with Pool(processes=num_processes) as pool:
        dfs = list(
            tqdm(
                pool.imap(process_file, parquet_files),
                total=len(parquet_files),
                desc='Processing files',
                unit='file',
            )
        )

    print('\nCombining dataframes...')
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df
