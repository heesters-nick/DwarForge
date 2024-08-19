import logging
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from utils import tile_str

logger = logging.getLogger()


def match_detections_with_catalogs(tile_nums, params_mto, table_dir, unions_cat=False):
    """
    Match detections to known objects. Return all dwarf candidates.

    Args:
        tile_nums (tuple): tile numbers
        params_mto (dataframe): mto parameter file
        table_dir (str): table directory

    Returns:
        dataframe: all dwarf candidates
    """
    detections = params_mto.loc[params_mto['label'] == 1].reset_index(drop=True)
    c_detections = SkyCoord(detections['ra'], detections['dec'], unit='deg', frame='icrs')
    gen = pd.read_csv(os.path.join(table_dir, 'all_known_dwarfs_processed.csv'), chunksize=500)
    tile_filter = f"tile == '{str(tuple(tile_nums))}'"
    known_dwarfs = pd.concat((x.query(tile_filter) for x in gen), ignore_index=True)
    c_known_dwarfs = SkyCoord(known_dwarfs['ra'], known_dwarfs['dec'], unit='deg', frame='icrs')

    known_matches, known_unmatches, detection_matches = match_cats_old(
        detections, c_detections, known_dwarfs, c_known_dwarfs
    )
    # initialize an empty dataframe for potential detected but not seleced known dwarfs
    non_detection_matches = pd.DataFrame(columns=detections.columns)
    if not known_unmatches.empty:
        c_known_unmatches = SkyCoord(
            known_unmatches['ra'], known_unmatches['dec'], unit='deg', frame='icrs'
        )
        # check if any of the unmatches are in the params_mto where the label is 0
        non_detections = params_mto.loc[params_mto['label'] == 0].reset_index(drop=True)
        c_non_detections = SkyCoord(
            non_detections['ra'], non_detections['dec'], unit='deg', frame='icrs'
        )
        _, known_unmatches, non_detection_matches = match_cats_old(
            non_detections, c_non_detections, known_unmatches, c_known_unmatches
        )

    common_columns = detections.columns.intersection(known_unmatches.columns)
    candidates = pd.concat(
        [detections, non_detection_matches, known_unmatches[common_columns]], ignore_index=True
    )
    candidates['known'] = np.where(
        candidates['ID'].isin(detection_matches['ID'])
        | candidates['ID'].isin(known_matches['ID'])
        | candidates['ID'].isin(known_unmatches['ID']),
        1,
        0,
    )
    candidates['tile'] = f'{tuple(tile_nums)}'

    return candidates


def match_cats_old(det, c_det, known, c_known, max_sep=15.0 * u.arcsec):
    """
    Match detections to known objects, return matches, unmatches

    Args:
        det (dataframe): detections dataframe
        c_det (Skycoord object): Skycoord objects of the detections
        known (dataframe): known object dataframe
        c_known (Skycoord object): Skycoord objects of the known objects
        max_sep (float): maximum separation tollerance

    Returns:
        dataframe: known objects that were detected
        dataframe: known objects that were missed
        dataframe: detections that are known objects
    """
    idx, d2d, _ = c_known.match_to_catalog_3d(c_det)
    sep_constraint = d2d < max_sep
    known_matches = known[sep_constraint].reset_index(drop=True)
    known_unmatches = known[~sep_constraint].reset_index(drop=True)

    det_matches = det.loc[idx[sep_constraint]].reset_index(drop=True)

    return known_matches, known_unmatches, det_matches


def match_cats(df_det, df_label, tile, max_sep=15.0):
    """
    Match detections to known objects, return matches, unmatches

    Args:
        df_det (dataframe): detections dataframe
        df_label (dataframe): dataframe of objects with labels
        tile (tuple): tile numbers
        max_sep (float): maximum separation tollerance

    Returns:
        det_matching_idx (list): indices of detections for which labels are available
        label_matches (dataframe): known objects that were detected
        det_matches (dataframe): detections that are known objects
    """

    c_det = SkyCoord(df_det['ra'], df_det['dec'], unit='deg')
    c_label = SkyCoord(df_label['ra'], df_label['dec'], unit='deg')

    try:
        idx, d2d, _ = c_label.match_to_catalog_3d(c_det)
    except Exception as e:
        logger.error(f'Error while matching catalogs for tile {tile_str(tile)}: {e}')
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    print(f'len idx {len(idx)} for tile {tile_str(tile)}')
    # sep_constraint is a list of True/False
    sep_constraint = d2d < max_sep * u.arcsec
    print(f'len sep contraint: {len(sep_constraint)} for tile: {tile_str(tile)}')
    print(f'non zero in sep contraint {np.count_nonzero(sep_constraint)} for tile {tile_str(tile)}')
    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matching_idx = idx[sep_constraint]  # det_matching_idx is a list of indices
    det_matches = df_det.loc[det_matching_idx].reset_index(drop=True)

    return det_matching_idx, label_matches, label_unmatches, det_matches


def add_labels(tile, band, det_df, det_df_full, dwarfs_df):
    logger.info(f'Adding labels to det params for tile {tile_str(tile)}.')
    det_df_updated = det_df.copy()
    dwarfs_in_tile = dwarfs_df[dwarfs_df['tile'] == tile_str(tile)].reset_index(drop=True)
    logger.info(f'Known dwarfs in tile {tile_str(tile)}: {len(dwarfs_in_tile)}')

    matching_stats = {
        'known_dwarfs_count': len(dwarfs_in_tile),
        'matched_dwarfs_count': 0,
        'unmatched_dwarfs_count': 0,
    }

    if len(dwarfs_in_tile) == 0:
        logger.info(f'No known dwarfs in tile {tile_str(tile)}. Skipping matching process.')
        return det_df_updated, matching_stats

    det_idx_lsb, lsb_matches, lsb_unmatches, _ = match_cats(
        det_df_updated, dwarfs_in_tile, tile, max_sep=10.0
    )
    det_idx_lsb = det_idx_lsb.astype(np.int32)
    # add lsb labels to detections dataframe

    det_df_updated['lsb'] = np.nan
    det_df_updated['ID_known'] = np.nan

    try:
        if len(det_idx_lsb) > 0:
            logger.info(f'Found {len(det_idx_lsb)} lsb detections for tile {tile_str(tile)}.')
            matching_stats['matched_dwarfs_count'] = len(det_idx_lsb)
            det_df_updated.loc[det_idx_lsb, 'lsb'] = 1
            det_df_updated.loc[det_idx_lsb, 'ID_known'] = lsb_matches['ID'].values
            logger.debug(
                f'Added {np.count_nonzero(~np.isnan(det_df_updated["lsb"]))} LSB labels to the detection dataframe for tile {tile} in band {band}.'
            )
    except Exception as e:
        logger.error(f'Error after if len(det_idx_lsb): {e}.')

    try:
        if len(lsb_unmatches) > 0:
            logger.info(f'Found {len(lsb_unmatches)} for tile: {tile_str(tile)}.')
            # Check unfiltered dataframe for known objects that were filtered out
            _, full_lsb_matches, full_lsb_unmatches, full_det_matches = match_cats(
                det_df_full, lsb_unmatches, tile, max_sep=10.0
            )

            if len(full_lsb_matches) > 0:
                matching_stats['matched_dwarfs_count'] += len(full_lsb_matches)
                logger.warning(
                    f'Found {len(full_lsb_matches)} known dwarfs in the unfiltered detections that were filtered out in tile {tile_str(tile)} in band {band}.'
                )

                # Create a new dataframe with the filtered out detections
                new_rows = full_det_matches.copy()
                new_rows['lsb'] = 1
                new_rows['ID_known'] = full_lsb_matches['ID'].values

                # Concatenate the new rows to det_df_updated
                det_df_updated = pd.concat([det_df_updated, new_rows], ignore_index=True)

                logger.info(
                    f'Added back {len(full_lsb_matches)} known dwarfs that were initially filtered out in tile {tile_str(tile)} in band {band}. Their mu values are: {full_lsb_matches["mu"].values}, their r_eff values are {full_lsb_matches["re_arcsec"].values}.'
                )

            matching_stats['unmatched_dwarfs_count'] = len(full_lsb_unmatches)

            if len(full_lsb_unmatches) > 0:
                logger.warning(
                    f'Found {len(full_lsb_unmatches)} undetected but known dwarfs in tile {tile_str(tile)} in band {band}. Their IDs are: {full_lsb_unmatches["ID"].values}.'
                )
    except Exception as e:
        logger.error(f'Error after if len(lsb_unmatches) > 0: {e} for tile {tile_str(tile)}.')

    return det_df_updated, matching_stats
