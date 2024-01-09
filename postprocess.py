import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


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

    known_matches, known_unmatches, detection_matches = match_cats(
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
        _, known_unmatches, non_detection_matches = match_cats(
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


def match_cats(det, c_det, known, c_known, max_sep=15.0 * u.arcsec):
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
