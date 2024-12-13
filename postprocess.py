import glob
import os
import time
from itertools import combinations
from typing import Dict, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from logging_setup import get_logger
from utils import (
    check_corrupted_data,
    check_objects_in_neighboring_tiles,
    count_duplicates,
    create_cartesian_kdtree,
    open_fits,
    open_raw_data,
    read_parquet,
    tile_str,
)
from warning_manager import set_warnings

logger = get_logger()


def match_stars(
    df_det, df_label, segmap=None, max_sep=5.0, extended_flag_radius=None, mag_limit=14.0
):
    """
    Match detections to known objects, return matches, unmatches, and extended flag indices
    Args:
        df_det (dataframe): detections dataframe
        df_label (dataframe): dataframe of objects with labels
        max_sep (float): maximum separation tolerance for direct matches
        extended_flag_radius (float): radius to flag detections as potentially associated with stars
    Returns:
        det_matching_idx (list): indices of detections for which labels are available
        label_matches (dataframe): known objects that were detected
        label_unmatches (dataframe): known objects that were not detected
        det_matches (dataframe): detections that are known objects
        extended_flag_idx (list): indices of all detections within extended_flag_radius of known stars
    """
    c_det = SkyCoord(df_det['ra'], df_det['dec'], unit=u.deg)
    c_label = SkyCoord(df_label['ra'], df_label['dec'], unit=u.deg)

    # Find closest matches within max_sep
    idx, d2d, _ = c_label.match_to_catalog_3d(c_det)
    sep_constraint = d2d < max_sep * u.arcsec
    det_matching_idx = idx[sep_constraint]

    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matches = df_det.loc[det_matching_idx].reset_index(drop=True)

    extended_flag_idx, extended_flag_mags = None, None
    if extended_flag_radius is not None:
        bright_stars = df_label[df_label['Gmag'] < mag_limit].reset_index(drop=True)

        # Create a mask for all bright stars at once using cv2.circle
        h, w = segmap.shape
        all_stars_mask = np.zeros((h, w), dtype=np.uint8)
        for x, y in bright_stars[['x', 'y']].values:
            x, y = round(x), round(y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(all_stars_mask, (x, y), round(extended_flag_radius), 1, -1)

        # Find all segment IDs within the masked area
        segment_ids = np.unique(segmap[all_stars_mask > 0])[1:]  # exclude 0 (background)

        extended_flag_idx = (
            segment_ids - 1
        )  # Subtract 1 because SEP IDs start at 1, but DataFrame index starts at 0

        # Find the closest bright star for each flagged segment
        flagged_segments_centers = df_det.loc[extended_flag_idx, ['xpeak', 'ypeak']].values
        bright_stars_coords = bright_stars[['x', 'y']].values
        distances = np.sqrt(
            np.sum((flagged_segments_centers[:, np.newaxis] - bright_stars_coords) ** 2, axis=2)
        )
        closest_star_indices = np.argmin(distances, axis=1)

        extended_flag_mags = bright_stars.loc[closest_star_indices, 'Gmag'].values

    return (
        det_matching_idx,
        label_matches,
        label_unmatches,
        det_matches,
        extended_flag_idx,
        extended_flag_mags,
    )


def match_cats(df_det, df_label, tile, pixel_scale, max_sep=15.0, re_multiplier=3.0):
    tree, _ = create_cartesian_kdtree(df_det['ra'].values, df_det['dec'].values)
    matches = []
    potential_matches_df = pd.DataFrame()
    for idx, known in df_label.iterrows():
        known_coords = SkyCoord(known['ra'], known['dec'], unit='deg')
        known_coords_xyz = known_coords.cartesian.xyz.value

        # Calculate base search radius in degrees
        base_search_radius = max_sep / 3600  # Convert arcseconds to degrees

        # Adaptive search radius (if 're' is available)
        if (
            're' in known
            and known['re'] is not None
            and not np.isnan(known['re'])
            and known['re'] > 0
        ):
            adaptive_radius = known['re'] * re_multiplier / 3600  # Convert to degrees
            search_radius = max(base_search_radius, adaptive_radius)
        else:
            search_radius = base_search_radius

        search_radius_chord = 2 * np.sin(np.deg2rad(search_radius) / 2)

        potential_match_indices = tree.query_ball_point(known_coords_xyz, search_radius_chord)
        potential_matches = df_det.iloc[potential_match_indices]

        logger.debug(f'potential matches for {known["ID"]}: {len(potential_matches)}')

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
                size_score = np.log1p(det['n_pix']) / np.log1p(max_n_pix)
                lsb_score = det['mu'] / max_mu
                distance = distances[potential_matches.index.get_loc(i)]
                distance_score = 1 - (
                    distance / (3600 * search_radius)
                )  # Normalized distance score
                score = lsb_score * 0.2 + size_score * 0.4 + distance_score * 0.4
                logger.debug(
                    f'object: {det["ID"]}; lsb score: {lsb_score:.2f}, size score: {size_score:.2f}, distance score: {distance_score:.2f}'
                )
                logger.debug(
                    f'object: {det["ID"]}; total score: {score:.2f}; distance: {distance:.2f} arcsec'
                )
                scores.append((i, score, distance))
            best_match = max(scores, key=lambda x: x[1])
            matches.append((idx, best_match[0], best_match[2]))

    if matches:
        label_match_idx, det_match_idx, match_distances = zip(*matches)
    else:
        label_match_idx, det_match_idx, match_distances = [], [], []
    label_matches = df_label.loc[list(label_match_idx)].reset_index(drop=True)
    label_unmatches = df_label.drop(list(label_match_idx)).reset_index(drop=True)
    det_matches = df_det.loc[list(det_match_idx)].reset_index(drop=True)
    det_matches['match_distance'] = match_distances
    return list(det_match_idx), label_matches, label_unmatches, det_matches


def add_labels(
    tile, band, det_df, det_df_full, dwarfs_df, fits_path, fits_ext, header, z_class_cat
):
    logger.debug(f'Adding labels to det params for tile {tile_str(tile)}.')

    warnings = []
    corrupted_data_objects = []

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

    # cross match with redshift catalog, add available redshifts
    det_df_updated = add_redshifts(det_df_updated, z_class_cat)

    # add lsb labels to detections dataframe
    det_df_updated['lsb'] = np.nan
    det_df_updated['ID_known'] = np.nan

    if len(dwarfs_in_tile) == 0:
        logger.debug(f'No known dwarfs in tile {tile_str(tile)}. Skipping matching process.')
        return det_df_updated, matching_stats

    det_idx_lsb, lsb_matches, lsb_unmatches, _ = match_cats(
        det_df_updated, dwarfs_in_tile, tile, header, max_sep=15.0
    )

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
        data, header = open_fits(fits_path, fits_ext=0)
        non_corrupted_unmatches = []

        for _, unmatched_obj in lsb_unmatches.iterrows():
            is_corrupted = check_corrupted_data(
                data, header, unmatched_obj['ra'], unmatched_obj['dec']
            )
            if is_corrupted:
                corrupted_data_objects.append(unmatched_obj['ID'])
            else:
                non_corrupted_unmatches.append(unmatched_obj)

        non_corrupted_unmatches = pd.DataFrame(non_corrupted_unmatches)
        matching_stats['corrupted_data_count'] = len(corrupted_data_objects)

        if len(corrupted_data_objects) > 0:
            logger.info(
                f'{len(corrupted_data_objects)}/{len(lsb_unmatches)} unmatched dwarfs are in corrupted data regions.'
            )

        if len(non_corrupted_unmatches) > 0:
            warning_msg = f'Found {len(non_corrupted_unmatches)} unmatched dwarf(s) for tile: {tile_str(tile)} (excluding corrupted data regions).'
            logger.warning(warning_msg)
            warnings.append(warning_msg)
            # Check unfiltered dataframe for known objects that were filtered out
            _, full_lsb_matches, full_lsb_unmatches, full_det_matches = match_cats(
                det_df_full, lsb_unmatches, tile, header, max_sep=15.0
            )

            if len(full_lsb_matches) > 0:
                # matching_stats['matched_dwarfs_count'] += len(full_lsb_matches)
                warning_msg = f'Found {len(full_lsb_matches)} known dwarf(s) that was/were filtered out in tile {tile_str(tile)} in band {band}.'
                logger.warning(warning_msg)

                # Create a new dataframe with the filtered out detections
                new_rows = full_det_matches.copy()
                new_rows['lsb'] = 1
                new_rows['ID_known'] = full_lsb_matches['ID'].values

                # Concatenate the new rows to det_df_updated
                # det_df_updated = pd.concat([det_df_updated, new_rows], ignore_index=True)

                warning_msg = (
                    f"{len(new_rows)} known dwarf(s) that was/were initially filtered out in\n\t"
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

            matching_stats['unmatched_dwarfs_count'] = len(full_lsb_unmatches) + len(
                corrupted_data_objects
            )

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
        else:
            matching_stats['unmatched_dwarfs_count'] = len(corrupted_data_objects)
            logger.info(
                f'All {len(corrupted_data_objects)} unmatched dwarfs are in corrupted data regions.'
            )

    set_warnings(warnings)

    return det_df_updated, matching_stats


def cutout2d_segmented(data, tile_str, segmap, object_id, x, y, size, cutout_in, seg_mode):
    """
    Create 2d cutout from an image, applying a segmentation mask for a specific object.

    Args:
        data (numpy.ndarray): image data
        tile_str (str): tile numbers
        segmap (numpy.ndarray): segmentation map with object IDs
        object_id (int): ID of the object to isolate in the cutout
        x (int): x-coordinate of cutout center
        y (int): y-coordinate of cutout center
        size (int): square cutout size
        cutout_in (numpy.ndarray): empty input cutout
        seg_mode (str): how to treat the segmentation map? multiply / concatenate

    Returns:
        numpy.ndarray: 2d cutout (size x size pixels) with only the specified object
    """
    data_ones = np.ones_like(data, dtype=np.float32)
    cutout_seg = cutout_in.copy()

    y_large, x_large = data.shape
    size_half = size // 2
    y_start = max(0, y - size_half)
    y_end = min(y_large, y + (size - size_half))
    x_start = max(0, x - size_half)
    x_end = min(x_large, x + (size - size_half))

    if y_start >= y_end or x_start >= x_end:
        logger.error(f'Tile: {tile_str}: an object is fully outside of the image.')

    cutout_slice = (
        slice(y_start - y + size_half, y_end - y + size_half),
        slice(x_start - x + size_half, x_end - x + size_half),
    )
    data_slice = slice(y_start, y_end), slice(x_start, x_end)

    # Apply data and segmentation mask
    if seg_mode == 'multiply':
        cutout_in[cutout_slice] = data[data_slice] * (segmap[data_slice] == object_id)
    elif seg_mode == 'concatenate':
        cutout_in[cutout_slice] = data[data_slice]
        cutout_seg[cutout_slice] = data_ones[data_slice] * (segmap[data_slice] == object_id)
    else:
        cutout_in[cutout_slice] = data[data_slice]

    return cutout_in, cutout_seg


def create_segmented_cutouts(data, tile_str, segmap, object_ids, xs, ys, cutout_size, seg_mode):
    """
    Create cutouts for multiple objects efficiently.

    Args:
        data (numpy.ndarray): image data
        tile_str (str): tile numbers
        segmap (numpy.ndarray): segmentation map with object IDs
        object_ids (numpy.ndarray): array of object IDs
        xs (numpy.ndarray): array of x-coordinates
        ys (numpy.ndarray): array of y-coordinates
        cutout_size (int): size of each cutout

    Returns:
        numpy.ndarray: array of cutouts
    """
    cutouts = np.zeros((len(object_ids), cutout_size, cutout_size), dtype=data.dtype)
    cutouts_seg = np.zeros((len(object_ids), cutout_size, cutout_size), dtype=data.dtype)
    cutout_empty = np.zeros((cutout_size, cutout_size), dtype=data.dtype)

    for i, (obj_id, x, y) in enumerate(zip(object_ids, xs, ys)):
        cutouts[i], cutouts_seg[i] = cutout2d_segmented(
            data, tile_str, segmap, obj_id, x, y, cutout_size, cutout_empty.copy(), seg_mode
        )

    return cutouts, cutouts_seg


def make_cutouts(
    data,
    header,
    tile_str,
    df=None,
    ra=None,
    dec=None,
    segmap=None,
    cutout_size=64,
    seg_mode='multiply',
):
    """
    Makes cutouts from the objects passed in the dataframe, data is multiplied
    with the corresponding detection segment.

    Args:
        data (numpy.ndarray): image
        header (header): fits header
        tile_str (str): tile numbers
        df (dataframe, optional): object dataframe. Defaults to None
        ra (numpy.ndarray, optional): right ascention array. Defaults to None
        dec (numpy.ndarray, optional): declination array. Defaults to None
        segmap (numpy.ndarray, optional): segmentation map. Defaults to None
        cutout_size (int, optional): square cutout size. Defaults to 64.
        seg_mode (multiply, optional): segmentation mode. Defaults to 'multiply'

    Returns:
        numpy.ndarray: cutouts of shape (n_cutouts, cutout_size, cutout_size)
    """
    if df is not None:
        xs = np.floor(df.X.values + 0.5).astype(np.int32)
        ys = np.floor(df.Y.values + 0.5).astype(np.int32)
        object_ids = df['ID'].values  # Assuming the index is the object ID
    elif ra is not None and dec is not None:
        wcs = WCS(header)
        xs, ys = wcs.all_world2pix(ra, dec, 0)
        xs, ys = np.floor(xs + 0.5).astype(np.int32), np.floor(ys + 0.5).astype(np.int32)
        object_ids = np.zeros(len(xs))

    cutout_start = time.time()
    cutouts, cutouts_seg = create_segmented_cutouts(
        data, tile_str, segmap, object_ids, xs, ys, cutout_size, seg_mode
    )
    logger.debug(f'{tile_str}: cutouts done in {time.time()-cutout_start:.2f} seconds.')

    return cutouts, cutouts_seg


def load_segmap(file_path):
    path, extension = os.path.splitext(file_path)
    directory = os.path.dirname(file_path)
    try:
        seg_file = [f for f in os.listdir(directory) if f.endswith('_seg.fits')][0]
        seg_path = os.path.join(directory, seg_file)
    except Exception as e:
        print(f'Error finding segmentation map: {e}')
        mto_seg = None
    mto_seg, header_seg = open_fits(seg_path, fits_ext=0)

    return mto_seg


def save_to_h5(
    stacked_cutout,
    stacked_cutout_seg,
    object_df,
    tile_numbers,
    save_path,
    seg_mode,
):
    """
    Save cutout data including metadata to file.

    Args:
        stacked_cutout (numpy.ndarray): stacked numpy array of the image data in different bands
        object_df (dataframe): object dataframe
        tile_numbers (tuple): tile numbers
        save_path (str): path to save the cutout

    Returns:
        None
    """
    logger.info(f'Tile: {tile_numbers} saving cutouts to file: {save_path}')
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(save_path, 'w', libver='latest') as hf:
        hf.create_dataset('images', data=stacked_cutout.astype(np.float32))
        hf.create_dataset('tile', data=np.asarray(tile_numbers), dtype=np.int32)
        hf.create_dataset(
            'known_id', data=np.asarray(object_df['ID_known'].values, dtype='S'), dtype=dt
        )
        hf.create_dataset('mto_id', data=np.asarray(object_df['ID'].values.astype(np.int32)))
        hf.create_dataset('ra', data=object_df['ra'].values.astype(np.float32))
        hf.create_dataset('dec', data=object_df['dec'].values.astype(np.float32))
        hf.create_dataset('label', data=object_df['lsb'].values.astype(np.float32))
        hf.create_dataset('zspec', data=object_df['zspec'].astype(np.float32))
        if seg_mode == 'concatenate':
            hf.create_dataset('segmaps', data=stacked_cutout_seg.astype(np.float32))


def add_redshifts(det_df, z_class_cat):
    margin = 0.0014  # extend the ra and dec ranges by this amount in degrees
    ra_range = (np.min(det_df['ra']) - margin, np.max(det_df['ra']) + margin)
    dec_range = (np.min(det_df['dec'] - margin), np.max(det_df['dec'] + margin))

    # read the redshift/class catalog
    class_z_df = read_parquet(
        z_class_cat,
        ra_range=ra_range,
        dec_range=dec_range,
    )

    det_df['class_label'] = np.nan
    det_df['zspec'] = np.nan

    if len(class_z_df) > 0:
        # match detections to redshift and class catalog
        det_idx_z_class, label_matches_z_class, _, _, _, _ = match_stars(
            det_df, class_z_df, max_sep=5.0
        )
        det_idx_z_class = det_idx_z_class.astype(np.int32)  # make sure index is int

        if len(det_idx_z_class) > 0:
            # add redshift and class labels to detections dataframe
            det_df.loc[det_idx_z_class, 'class_label'] = label_matches_z_class['cspec'].values
            det_df.loc[det_idx_z_class, 'zspec'] = label_matches_z_class['zspec'].values

    return det_df


def match_coordinates(band1, band2, band_data, max_sep=15.0):
    """
    Match reference coordinates to target coordinates.

    Args:
        band1 (str): first band
        band2 (str) (str): second band
        band_data (dict): band data
        max_sep (float): Maximum separation in arcseconds

    Returns:
        matched_ref_indices (array): Indices of matched reference coordinates
        matched_target_indices (array): Indices of matched target coordinates
    """
    reference_coords = SkyCoord(band_data[band1]['ra'], band_data[band1]['dec'], unit=u.deg)
    target_coords = SkyCoord(band_data[band2]['ra'], band_data[band2]['dec'], unit=u.deg)
    idx, d2d, _ = reference_coords.match_to_catalog_3d(target_coords)
    mask = d2d < max_sep * u.arcsec
    return np.where(mask)[0], idx[mask]


def match_coordinates_across_bands_old(
    band_data: Dict[str, Dict], max_sep: float, tile: tuple
) -> Tuple[
    np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    band_names = list(band_data.keys())
    all_matches = {}

    for band1, band2 in combinations(band_names, 2):
        idx1, idx2 = match_coordinates(band1, band2, band_data, max_sep=max_sep)
        all_matches[f'{band1}|{band2}'] = np.column_stack((idx1, idx2))

    num_objects = max(len(band_data[band]['ra']) for band in band_names)

    # Create a unique object ID for each detection in each band
    object_id_map = {band: np.arange(len(band_data[band]['ra'])) for band in band_names}

    # Track unique objects across all bands
    unique_objects = {}
    object_to_unique = {band: {} for band in band_names}

    for match_key, matches in all_matches.items():
        band1, band2 = match_key.split('|')
        for idx1, idx2 in matches:
            obj1 = object_id_map[band1][idx1]
            obj2 = object_id_map[band2][idx2]

            if obj1 in object_to_unique[band1] and obj2 in object_to_unique[band2]:
                # Both objects are already matched, ensure they point to the same unique object
                unique_id1 = object_to_unique[band1][obj1]
                unique_id2 = object_to_unique[band2][obj2]
                if unique_id1 != unique_id2:
                    # Merge the two unique objects
                    unique_objects[unique_id1].update(unique_objects[unique_id2])
                    for b, idx in unique_objects[unique_id2]:
                        object_to_unique[b][object_id_map[b][idx]] = unique_id1
                    del unique_objects[unique_id2]
            elif obj1 in object_to_unique[band1]:
                # Object in band1 is already matched, add band2 object to the same unique object
                unique_id = object_to_unique[band1][obj1]
                unique_objects[unique_id].add((band2, idx2))
                object_to_unique[band2][obj2] = unique_id
            elif obj2 in object_to_unique[band2]:
                # Object in band2 is already matched, add band1 object to the same unique object
                unique_id = object_to_unique[band2][obj2]
                if unique_id not in unique_objects:
                    logger.warning(
                        f'Inconsistency detected tile {tile}: unique_id {unique_id} found in object_to_unique[{band1}] but not in unique_objects.'
                    )
                    logger.warning(f'unique_id: {unique_id}')
                    unique_objects[unique_id] = set()
                unique_objects[unique_id].add((band1, idx1))
                object_to_unique[band1][obj1] = unique_id

            else:
                # New unique object
                unique_id = len(unique_objects)
                unique_objects[unique_id] = {(band1, idx1), (band2, idx2)}
                object_to_unique[band1][obj1] = unique_id
                object_to_unique[band2][obj2] = unique_id

    # Process single-band LSB objects
    single_band_lsbs = []
    for band in band_names:
        df = band_data[band]['df']
        if 'lsb' in df.columns:
            lsb_mask = df['lsb'] == 1  # Assuming 1 indicates LSB objects
            for idx in df.index[lsb_mask]:
                if object_id_map[band][idx] not in object_to_unique[band]:
                    logger.warning(
                        f"tile: {tile}, dwarf {df.loc[idx, 'ID_known']} was only detected in {band}."
                    )
                    single_band_lsbs.append(df.loc[idx, 'ID_known'])
                    unique_id = len(unique_objects)
                    unique_objects[unique_id] = {(band, idx)}
                    object_to_unique[band][object_id_map[band][idx]] = unique_id
    # count duplicates in single_band detections (cross-match failed due to incompatible coordinates)
    duplicates = count_duplicates(single_band_lsbs)
    if duplicates > 0:
        logger.warning(f'Found duplicate single LSB matches in tile {tile}: {single_band_lsbs}')
    # Process unique objects
    multi_band_objects = []
    object_indices = {band: np.full(num_objects, -1, dtype=int) for band in band_names}
    final_ra = []
    final_dec = []
    known_ids = []
    labels = []
    zspecs = []

    for unique_id, detections in unique_objects.items():
        # Include objects detected in at least 2 out of 3 bands and all LSB objects
        # even if detected in only one band
        if len(detections) >= 2 or any(
            band_data[band]['df'].loc[idx, 'lsb'] == 1 for band, idx in detections
        ):
            multi_band_objects.append(unique_id)
            first_band, first_idx = next(iter(detections))

            for band, idx in detections:
                object_indices[band][idx] = unique_id

            final_ra.append(band_data[first_band]['ra'][first_idx])
            final_dec.append(band_data[first_band]['dec'][first_idx])
            known_ids.append(
                band_data[first_band]['df'].loc[first_idx, 'ID_known']
                if 'ID_known' in band_data[first_band]['df'].columns
                else ''
            )
            labels.append(
                band_data[first_band]['df'].loc[first_idx, 'lsb']
                if 'lsb' in band_data[first_band]['df'].columns
                else np.nan
            )
            zspecs.append(
                band_data[first_band]['df'].loc[first_idx, 'zspec']
                if 'zspec' in band_data[first_band]['df'].columns
                else np.nan
            )

    return (
        np.array(multi_band_objects),
        object_indices,
        np.array(final_ra),
        np.array(final_dec),
        np.array(known_ids),
        np.array(labels),
        np.array(zspecs),
    )


def match_coordinates_across_bands(
    band_data: Dict[str, Dict], max_sep: float, tile: tuple
) -> Tuple[
    np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    band_names = list(band_data.keys())
    all_matches = {}

    for band1, band2 in combinations(band_names, 2):
        idx1, idx2 = match_coordinates(band1, band2, band_data, max_sep=max_sep)
        all_matches[f'{band1}|{band2}'] = np.column_stack((idx1, idx2))

    num_objects = max(len(band_data[band]['ra']) for band in band_names)

    # Create a unique object ID for each detection in each band
    object_id_map = {band: np.arange(len(band_data[band]['ra'])) for band in band_names}

    # Track unique objects across all bands
    unique_objects = {}
    object_to_unique = {band: {} for band in band_names}

    # Use a separate counter for unique_id to avoid potential race conditions
    unique_id_counter = 0

    for match_key, matches in all_matches.items():
        band1, band2 = match_key.split('|')
        for idx1, idx2 in matches:
            obj1 = object_id_map[band1][idx1]
            obj2 = object_id_map[band2][idx2]

            if obj1 in object_to_unique[band1] and obj2 in object_to_unique[band2]:
                # Both objects are already matched, ensure they point to the same unique object
                unique_id1 = object_to_unique[band1][obj1]
                unique_id2 = object_to_unique[band2][obj2]
                if unique_id1 != unique_id2:
                    # Merge the two unique objects
                    unique_objects[unique_id1].update(unique_objects[unique_id2])
                    for b, idx in unique_objects[unique_id2]:
                        object_to_unique[b][object_id_map[b][idx]] = unique_id1
                    del unique_objects[unique_id2]
            elif obj1 in object_to_unique[band1]:
                # Object in band1 is already matched, add band2 object to the same unique object
                unique_id = object_to_unique[band1][obj1]
                unique_objects[unique_id].add((band2, idx2))
                object_to_unique[band2][obj2] = unique_id
            elif obj2 in object_to_unique[band2]:
                # Object in band2 is already matched, add band1 object to the same unique object
                unique_id = object_to_unique[band2][obj2]
                unique_objects[unique_id].add((band1, idx1))
                object_to_unique[band1][obj1] = unique_id
            else:
                # New unique object
                unique_id = unique_id_counter
                unique_id_counter += 1
                unique_objects[unique_id] = {(band1, idx1), (band2, idx2)}
                object_to_unique[band1][obj1] = unique_id
                object_to_unique[band2][obj2] = unique_id

    # Process single-band LSB objects
    single_band_lsbs = []
    for band in band_names:
        df = band_data[band]['df']
        if 'lsb' in df.columns:
            lsb_mask = df['lsb'] == 1  # Assuming 1 indicates LSB objects
            for idx in df.index[lsb_mask]:
                if object_id_map[band][idx] not in object_to_unique[band]:
                    logger.warning(
                        f"tile: {tile}, dwarf {df.loc[idx, 'ID_known']} was only detected in {band}."
                    )
                    single_band_lsbs.append(df.loc[idx, 'ID_known'])
                    unique_id = unique_id_counter
                    unique_id_counter += 1
                    unique_objects[unique_id] = {(band, idx)}
                    object_to_unique[band][object_id_map[band][idx]] = unique_id
                    # Ensure the unique_id is valid
                    assert (
                        unique_id in unique_objects
                    ), f'Inconsistency created for unique_id {unique_id}'

    # count duplicates in single_band detections (cross-match failed due to incompatible coordinates)
    duplicates = count_duplicates(single_band_lsbs)
    if duplicates > 0:
        logger.warning(f'Found duplicate single LSB matches in tile {tile}: {single_band_lsbs}')

    # Process unique objects
    multi_band_objects = []
    object_indices = {band: np.full(num_objects, -1, dtype=int) for band in band_names}
    final_ra = []
    final_dec = []
    known_ids = []
    labels = []
    zspecs = []

    for unique_id, detections in unique_objects.items():
        # Include objects detected in at least 2 out of 3 bands and all LSB objects
        # even if detected in only one band
        if len(detections) >= 2 or any(
            band_data[band]['df'].loc[idx, 'lsb'] == 1 for band, idx in detections
        ):
            multi_band_objects.append(unique_id)
            first_band, first_idx = next(iter(detections))

            for band, idx in detections:
                object_indices[band][idx] = unique_id

            final_ra.append(band_data[first_band]['ra'][first_idx])
            final_dec.append(band_data[first_band]['dec'][first_idx])
            known_ids.append(
                band_data[first_band]['df'].loc[first_idx, 'ID_known']
                if 'ID_known' in band_data[first_band]['df'].columns
                else ''
            )
            labels.append(
                band_data[first_band]['df'].loc[first_idx, 'lsb']
                if 'lsb' in band_data[first_band]['df'].columns
                else np.nan
            )
            zspecs.append(
                band_data[first_band]['df'].loc[first_idx, 'zspec']
                if 'zspec' in band_data[first_band]['df'].columns
                else np.nan
            )

    return (
        np.array(multi_band_objects),
        object_indices,
        np.array(final_ra),
        np.array(final_dec),
        np.array(known_ids),
        np.array(labels),
        np.array(zspecs),
    )


def read_band_data(parent_dir, tile_dir, tile, band, in_dict, seg_mode, use_full_res=False):
    # Read the full image, segmentation map, and catalog
    zfill = in_dict[band]['zfill']
    file_prefix = in_dict[band]['name']
    delimiter = in_dict[band]['delimiter']
    suffix = in_dict[band]['suffix']
    fits_ext = in_dict[band]['fits_ext']
    num1, num2 = str(tile[0]).zfill(zfill), str(tile[1]).zfill(zfill)
    filename_prefix = f'{file_prefix}{delimiter}{num1}{delimiter}{num2}{suffix}'
    if use_full_res:
        data_path = os.path.join(parent_dir, tile_dir, band, filename_prefix)
        data, header = open_raw_data(data_path, fits_ext=fits_ext, band=band)
    else:
        data_path = os.path.join(
            parent_dir, tile_dir, band, os.path.splitext(filename_prefix) + '_rebin.fits'
        )
        data, header = open_fits(data_path, fits_ext=0)

    path, extension = os.path.splitext(data_path)

    if seg_mode is not None:
        seg_pattern = f'{path}*_seg.fits'
        seg_path = glob.glob(seg_pattern)
        segmap, _ = open_fits(seg_path, fits_ext=0)
    else:
        segmap = None

    det_pattern = f'{path}*_det_params.parquet'
    det_path = glob.glob(det_pattern)
    det_df = pd.read_parquet(det_path)
    # filter det_df for each band before performing matching
    # exclude likely non-dwarfs beforehand
    if ('lsb' in det_df.columns) and np.count_nonzero(det_df['lsb'] == 1) == 0:
        det_df = filter_candidates(df=det_df, tile=tile, band=band)
    else:
        # only filter non-dwarf dataframe to avoid losing known dwarfs for training
        det_df_other = det_df[det_df['lsb'] != 1].reset_index(drop=True)
        det_df_dwarf = det_df[det_df['lsb'] == 1].reset_index(drop=True)
        det_df = filter_candidates(df=det_df_other, tile=tile, band=band)
        det_df = pd.concat([det_df, det_df_dwarf], ignore_index=True)

    return data, header, segmap, det_df['ra'].values, det_df['dec'].values, det_df


def filter_candidates(df, tile, band):
    df_mod = df.copy()
    df_dwarf = df.loc[df['lsb'] == 1].reset_index(drop=True)

    # Define band-specific conditions
    band_conditions = {
        'cfis_lsb-r': {
            'basic': {
                'mu': (22.0, None),  # (min, max), None means no limit
                're_arcsec': (1.6, 55.0),
                'axis_ratio': (0.17, None),
                'r_10_arcsec': (0.353, 18.2),
                'r_25_arcsec': (0.4, 32.1),
                'r_75_arcsec': (2.16, 102.1),
                'r_90_arcsec': (2.5, 145.1),
                'r_100_arcsec': (2.8, 254.1),
                'r_fwhm_arcsec': (0.4, 13.8),
                'mu_median': (0.34, 28.7),
                'mu_mean': (0.4, 65.1),
                'mu_max': (2.0, 6255.0),
                'total_flux': (55, None),
                'mag': (12.17, 25.7),
            },
            'complex': [
                lambda df_mod: (df_mod['mu'] > (0.6155 * df_mod['mag'] + 11.3832))
                & (df_mod['mu'] > (1.35 * df_mod['mag'] - 6.4)),
                lambda df_mod: (df_mod['mu'] / df_mod['r_75_arcsec'])
                < np.maximum(1.51 * df_mod['mag'] - 24.6, 3.0),
                lambda df_mod: np.log(df_mod['r_90_arcsec'] / df_mod['r_75_arcsec'])
                < (0.295 * np.log(df_mod['r_100_arcsec']) - 0.13),
            ],
        },
        'whigs-g': {
            'basic': {
                'mu': (22.51, None),  # (min, max), None means no limit
                're_arcsec': (1.6, None),
                'axis_ratio': (0.17, None),
                'r_10_arcsec': (0.4, 8.1),
                'r_25_arcsec': (0.72, 15.0),
                'r_75_arcsec': (2.17, 42.0),
                'r_90_arcsec': (2.47, 54.0),
                'r_100_arcsec': (2.8, 79.0),
                'r_fwhm_arcsec': (0.417, 13.5),
                'mu_median': (0.0072, 0.8),
                'mu_mean': (0.017, 2.01),
                'mu_max': (0.066, 226.0),
                'total_flux': (1.77, None),
                'mag': (16.0, 26.5),
            },
            'complex': [
                lambda df_mod: (df_mod['mu'] > (0.7063 * df_mod['mag'] + 9.4420))
                & (df_mod['mu'] > (4.1 * df_mod['mag'] - 78.0)),
                lambda df_mod: np.log(df_mod['r_100_arcsec'] / df_mod['r_90_arcsec'])
                < (-1.2615 * np.log(df_mod['mag']) + 4.3000),
                lambda df_mod: np.log(df_mod['mag'] / df_mod['r_75_arcsec'])
                < np.maximum(6.4500 * np.log(df_mod['mag']) + -17.9000, 0.75),
            ],
        },
        'ps-i': {
            'basic': {
                'mu': (22.0, 29.5),  # (min, max), None means no limit
                're_arcsec': (1.6, 40.0),
                'axis_ratio': (0.17, None),
                'r_10_arcsec': (0.4, 14.95),
                'r_25_arcsec': (0.58, 25.5),
                'r_75_arcsec': (2.1, 58.0),
                'r_90_arcsec': (2.5, 77.0),
                'r_100_arcsec': (2.7, 117.0),
                'r_fwhm_arcsec': (0.4, 20.8),
                'mu_median': (0.445, 40.7),
                'mu_mean': (0.565, 100.8),
                'mu_max': (2.47, 4465.0),
                'total_flux': (66.0, None),
                'mag': (13.1, 26.0),
            },
            'complex': [
                lambda df_mod: (df_mod['r_90_arcsec'] / df_mod['r_75_arcsec'])
                < (0.0100 * df_mod['r_100_arcsec'] + 1.4000),
                lambda df_mod: (df_mod['r_100_arcsec'] / df_mod['r_90_arcsec'])
                < (-0.0850 * df_mod['mag'] + 3.3000),
                lambda df_mod: (df_mod['mag'] / df_mod['r_75_arcsec'])
                < np.maximum(1.4 * df_mod['mag'] - 23, 2.5),
                lambda df_mod: (df_mod['r_90_arcsec'] / df_mod['r_75_arcsec'])
                < (-0.0700 * df_mod['mag'] + 2.97),
                lambda df_mod: (df_mod['r_75_arcsec'] / df_mod['r_25_arcsec'])
                < (0.2 * df_mod['mag'] + 1.0),
                lambda df_mod: (df_mod['r_75_arcsec'] / df_mod['r_25_arcsec'])
                < (-1.1 * df_mod['mag'] + 30.5),
                lambda df_mod: np.log(df_mod['r_100_arcsec'] / df_mod['r_75_arcsec'])
                < (-2.7 * np.log(df_mod['mag']) + 9.05),
                lambda df_mod: np.log(df_mod['r_75_arcsec'] / df_mod['r_25_arcsec'])
                < (0.8 * np.log(df_mod['mag']) - 0.81),
            ],
        },
    }

    if band not in band_conditions:
        logger.error(f'Conditions not implemented for band {band}.')
        return None

    conditions = band_conditions[band]

    # Apply basic conditions
    for column, (min_val, max_val) in conditions['basic'].items():
        if min_val is not None:
            df_mod = df_mod[df_mod[column] > min_val]
        if max_val is not None:
            df_mod = df_mod[df_mod[column] < max_val]

    # Apply complex conditions
    for condition in conditions['complex']:
        df_mod = df_mod[condition]

    # Reset index
    df_mod = df_mod.reset_index(drop=True)

    logger.debug(
        f'{tile}, {band}: filtered out {len(df.loc[df["lsb"]==1])-len(df_mod.loc[df_mod["lsb"]==1])}/{len(df_dwarf)} dwarfs.'
    )
    logger.debug(
        f'{tile}, {band}: filtered out {len(df.loc[df["lsb"].isna()])-len(df_mod.loc[df_mod["lsb"].isna()])}/{len(df.loc[df["lsb"].isna()])} other objects.'
    )

    return df_mod
