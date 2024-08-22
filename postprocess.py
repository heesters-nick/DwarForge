import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree

from logging_setup import get_logger
from utils import check_objects_in_neighboring_tiles, tile_str
from warning_manager import set_warnings

logger = get_logger()


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


def match_cats_older(df_det, df_label, tile, max_sep=15.0):
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
        raise
    logger.debug(f'len idx {len(idx)} for tile {tile_str(tile)}')
    # sep_constraint is a list of True/False
    sep_constraint = d2d < max_sep * u.arcsec
    logger.debug(f'len sep contraint: {len(sep_constraint)} for tile: {tile_str(tile)}')
    logger.debug(
        f'non-zero in sep contraint {np.count_nonzero(sep_constraint)} for tile {tile_str(tile)}'
    )
    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matching_idx = idx[sep_constraint]  # det_matching_idx is a list of indices
    det_matches = df_det.loc[det_matching_idx].reset_index(drop=True)

    return det_matching_idx, label_matches, label_unmatches, det_matches


def match_cats(df_det, df_label, tile, header, max_sep=15.0):
    """
    Match detections to known objects preferring larger, lsb objects

    Args:
        df_det (dataframe): detections dataframe
        df_label (dataframe): dataframe of objects with labels
        tile (tuple): tile numbers
        header (header): fits header
        max_sep (float): base maximum separation tolerance in arcseconds

    Returns:
        det_matching_idx (list): indices of detections for which labels are available
        label_matches (dataframe): known objects that were detected
        label_unmatches (dataframe): known objects that were not detected
        det_matches (dataframe): detections that are known objects
        matches (list): list of (known_idx, detection_idx) pairs
    """
    tree = cKDTree(np.column_stack((df_det['ra'], df_det['dec'])))
    matches = []
    potential_matches_df = pd.DataFrame()

    for idx, known in df_label.iterrows():
        known_coords = SkyCoord(known['ra'], known['dec'], unit='deg')

        # Adaptive search radius (keep using re for this, but we'll be more cautious with it later)
        if (
            're' in known
            and known['re'] is not None
            and not np.isnan(known['re'])
            and known['re'] > 0
        ):
            search_radius = max(max_sep, known['re'] * 3) / 3600
        else:
            search_radius = max_sep / 3600

        potential_match_indices = tree.query_ball_point([known['ra'], known['dec']], search_radius)
        potential_matches = df_det.iloc[potential_match_indices]

        print(f'potential matches for {known["ID"]}: {len(potential_matches)}')

        potential_matches_df = pd.concat([potential_matches_df, potential_matches])

        if len(potential_matches) > 0:
            potential_matches_coords = SkyCoord(
                potential_matches['ra'], potential_matches['dec'], unit='deg'
            )
            distances = known_coords.separation(potential_matches_coords).arcsec
            max_n_pix = potential_matches['n_pix'].max()
            max_mu = potential_matches['mu'].max()

            scores = []
            for i, det in potential_matches.iterrows():
                # Size comparison score (now using n_pix)
                size_score = np.log1p(det['n_pix']) / np.log1p(max_n_pix)

                # LSB characteristics score (now incorporating n_pix)
                lsb_score = det['mu'] / max_mu

                # Distance score
                distance_score = 1 / (1 + distances[potential_matches.index.get_loc(i)])

                # Combined score (adjust weights as needed)
                score = lsb_score * 0.2 + size_score * 0.4 + distance_score * 0.4
                logger.debug(f'object: {det["ID"]}; lsb_score: {lsb_score}')
                logger.debug(f'object: {det["ID"]}; size_score: {size_score}')
                logger.debug(f'object: {det["ID"]}; distance_score: {distance_score}')
                logger.debug(f'object: {det["ID"]}; total score: {score}')
                scores.append((i, score))

            best_match = max(scores, key=lambda x: x[1])
            matches.append((idx, best_match[0]))

    if matches:
        label_match_idx, det_match_idx = zip(*matches)
    else:
        label_match_idx, det_match_idx = [], []

    label_matches = df_label.loc[list(label_match_idx)].reset_index(drop=True)
    label_unmatches = df_label.drop(list(label_match_idx)).reset_index(drop=True)
    det_matches = df_det.loc[list(det_match_idx)].reset_index(drop=True)

    return list(det_match_idx), label_matches, label_unmatches, det_matches


def add_labels(tile, band, det_df, det_df_full, dwarfs_df, header):
    logger.debug(f'Adding labels to det params for tile {tile_str(tile)}.')

    warnings = []

    det_df_updated = det_df.copy()
    dwarfs_in_tile = dwarfs_df[dwarfs_df['tile'] == tile_str(tile)].reset_index(drop=True)
    logger.info(f'Known dwarfs in tile {tile_str(tile)}: {len(dwarfs_in_tile)}')
    # Check for objects in neighboring tiles
    additional_dwarfs = check_objects_in_neighboring_tiles(tile_str(tile), dwarfs_df, header)
    if not additional_dwarfs.empty:
        dwarfs_in_tile = pd.concat([dwarfs_in_tile, additional_dwarfs]).reset_index(drop=True)
        logger.info(f'Added {len(additional_dwarfs)} dwarfs from neighboring tiles.')
        logger.info(f'New total in tile {tile_str(tile)}: {len(dwarfs_in_tile)}')

    matching_stats = {
        'known_dwarfs_count': len(dwarfs_in_tile),
        'matched_dwarfs_count': 0,
        'unmatched_dwarfs_count': 0,
    }

    if len(dwarfs_in_tile) == 0:
        logger.debug(f'No known dwarfs in tile {tile_str(tile)}. Skipping matching process.')
        return det_df_updated, matching_stats

    det_idx_lsb, lsb_matches, lsb_unmatches, _ = match_cats(
        det_df_updated, dwarfs_in_tile, tile, header, max_sep=15.0
    )

    # add lsb labels to detections dataframe
    det_df_updated['lsb'] = np.nan
    det_df_updated['ID_known'] = np.nan

    if len(det_idx_lsb) > 0:
        logger.info(f'Found {len(det_idx_lsb)} lsb detections for tile {tile_str(tile)}.')
        matching_stats['matched_dwarfs_count'] = len(det_idx_lsb)
        det_df_updated.loc[det_idx_lsb, 'lsb'] = 1
        # Initialize the column to accept strings
        det_df_updated['ID_known'] = det_df_updated['ID_known'].astype(object)
        det_df_updated.loc[det_idx_lsb, 'ID_known'] = lsb_matches['ID'].values
        logger.debug(
            f'Added {np.count_nonzero(~np.isnan(det_df_updated["lsb"]))} LSB labels to the detection dataframe for tile {tile} in band {band}.'
        )

    if len(lsb_unmatches) > 0:
        warning_msg = f'Found {len(lsb_unmatches)} unmatched dwarf for tile: {tile_str(tile)}.'
        logger.warning(warning_msg)
        warnings.append(warning_msg)
        # Check unfiltered dataframe for known objects that were filtered out
        _, full_lsb_matches, full_lsb_unmatches, full_det_matches = match_cats(
            det_df_full, lsb_unmatches, tile, header, max_sep=15.0
        )

        if len(full_lsb_matches) > 0:
            matching_stats['matched_dwarfs_count'] += len(full_lsb_matches)
            warning_msg = f'Found {len(full_lsb_matches)} known dwarf(s) that was/were filtered out in tile {tile_str(tile)} in band {band}.'
            logger.warning(warning_msg)

            # Create a new dataframe with the filtered out detections
            new_rows = full_det_matches.copy()
            new_rows['lsb'] = 1
            new_rows['ID_known'] = full_lsb_matches['ID'].values

            # Concatenate the new rows to det_df_updated
            det_df_updated = pd.concat([det_df_updated, new_rows], ignore_index=True)

            warning_msg = (
                f"Added back {len(new_rows)} known dwarf(s) that was/were initially filtered out in\n\t"
                f"tile: {tile_str(tile)}\n\t"
                f"band: {band}:\n\t"
                f"ID: {new_rows['ID_known'].values}\n\t"
                f"mu: {[f'{mu:.3f}' for mu in new_rows['mu'].values]}\n\t"
                f"r_eff: {[f'{re:.3f}' for re in new_rows['re_arcsec'].values]}\n\t"
                f"ra: {new_rows['ra'].values}\n\t"
                f"dec: {new_rows['dec'].values}\t"
            )
            logger.warning(warning_msg)
            warnings.append(warning_msg)

        matching_stats['unmatched_dwarfs_count'] = len(full_lsb_unmatches)

        if len(full_lsb_unmatches) > 0:
            warning_msg = (
                f"Found {len(full_lsb_unmatches)} undetected but known dwarfs in\n\t"
                f"tile: {tile_str(tile)}\n\t"
                f"band: {band}\n\t"
                f"ID(s): {full_lsb_unmatches['ID'].values}\n\t"
                f"ra: {full_lsb_unmatches['ra'].values}\n\t"
                f"dec: {full_lsb_unmatches['dec'].values}\n\t"
            )
            logger.warning(warning_msg)
            warnings.append(warning_msg)

    set_warnings(warnings)

    return det_df_updated, matching_stats
