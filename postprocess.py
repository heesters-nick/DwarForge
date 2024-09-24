import os
import time

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from logging_setup import get_logger
from utils import (
    check_corrupted_data,
    check_objects_in_neighboring_tiles,
    create_cartesian_kdtree,
    open_fits,
    read_parquet,
    tile_str,
)
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


def match_stars(df_det, df_label, max_sep=2.0):
    """
    Match detections to known objects, return matches, unmatches

    Args:
        df_det (dataframe): detections dataframe
        df_label (dataframe): dataframe of objects with labels
        max_sep (float): maximum separation tollerance

    Returns:
        det_matching_idx (list): indices of detections for which labels are available
        label_matches (dataframe): known objects that were detected
        det_matches (dataframe): detections that are known objects
    """
    c_det = SkyCoord(df_det['ra'], df_det['dec'], unit=u.deg)  # type: ignore
    c_label = SkyCoord(df_label['ra'], df_label['dec'], unit=u.deg)  # type: ignore

    idx, d2d, _ = c_label.match_to_catalog_3d(c_det)
    # sep_constraint is a list of True/False
    sep_constraint = d2d < max_sep * u.arcsec  # type: ignore
    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matching_idx = idx[sep_constraint]  # det_matching_idx is a list of indices
    det_matches = df_det.loc[det_matching_idx].reset_index(drop=True)
    return det_matching_idx, label_matches, label_unmatches, det_matches


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


def cutout2d_segmented(data, tile_str, segmap, object_id, x, y, size, cutout_in):
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

    Returns:
        numpy.ndarray: 2d cutout (size x size pixels) with only the specified object
    """
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
    cutout_in[cutout_slice] = data[data_slice] * (segmap[data_slice] == object_id)

    return cutout_in


def create_segmented_cutouts(data, tile_str, segmap, object_ids, xs, ys, cutout_size):
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
    cutout_empty = np.zeros((cutout_size, cutout_size), dtype=data.dtype)

    for i, (obj_id, x, y) in enumerate(zip(object_ids, xs, ys)):
        cutouts[i] = cutout2d_segmented(
            data, tile_str, segmap, obj_id, x, y, cutout_size, cutout_empty.copy()
        )

    return cutouts


def make_cutouts(data, tile_str, df, segmap, cutout_size=64):
    """
    Makes cutouts from the objects passed in the dataframe, data is multiplied
    with the corresponding detection segment.

    Args:
        data (numpy.ndarray): image
        tile_str (str): tile numbers
        df (dataframe): object dataframe
        segmap (numpy.ndarray): segmentation map
        cutout_size (int, optional): square cutout size. Defaults to 64.

    Returns:
        numpy.ndarray: cutouts of shape (n_cutouts, cutout_size, cutout_size)
    """
    xs = np.floor(df.X.values + 0.5).astype(np.int32)
    ys = np.floor(df.Y.values + 0.5).astype(np.int32)
    object_ids = df['ID'].values  # Assuming the index is the object ID

    cutout_start = time.time()
    cutouts = create_segmented_cutouts(data, tile_str, segmap, object_ids, xs, ys, cutout_size)
    logger.debug(f'{tile_str}: cutouts done in {time.time()-cutout_start:.2f} seconds.')

    return cutouts


def load_segmap(file_path):
    path, extension = os.path.splitext(file_path)
    seg_path = f'{path}_seg{extension}'
    mto_seg, header_seg = open_fits(seg_path, fits_ext=0)

    return mto_seg


def save_to_h5(
    stacked_cutout,
    object_df,
    tile_numbers,
    save_path,
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
        det_idx_z_class, label_matches_z_class, _, _ = match_stars(det_df, class_z_df, max_sep=5.0)
        det_idx_z_class = det_idx_z_class.astype(np.int32)  # make sure index is int

        if len(det_idx_z_class) > 0:
            # add redshift and class labels to detections dataframe
            det_df.loc[det_idx_z_class, 'class_label'] = label_matches_z_class['cspec'].values
            det_df.loc[det_idx_z_class, 'zspec'] = label_matches_z_class['zspec'].values

    return det_df


def match_coordinates(reference_coords, target_coords, max_sep=15.0):
    """
    Match reference coordinates to target coordinates.

    Args:
        reference_coords (SkyCoord): Reference coordinates
        target_coords (SkyCoord): Target coordinates to match against
        max_sep (float): Maximum separation in arcseconds

    Returns:
        matched_ref_indices (array): Indices of matched reference coordinates
        matched_target_indices (array): Indices of matched target coordinates
    """
    idx, d2d, _ = reference_coords.match_to_catalog_3d(target_coords)
    mask = d2d < max_sep * u.arcsec
    return np.where(mask)[0], idx[mask]


# def fuse_cutouts(
#     parent_dir,
#     tiles,
#     in_dict,
#     band_names=['whigs-g', 'cfis_lsb-r', 'ps-i'],
#     parent_dir_destination=None,
#     download_file=False,
# ):
#     logger.info(f'Starting to fuse cutouts for {len(tiles)} tiles in the bands: {band_names}')
#     print(f'First tile: {tiles[0]}')
#     for tile in tqdm(tiles):
#         all_band_data = {}
#         r_band_data = {}

#         tile_dir = f'{tile[0].zfill(3)}_{tile[1].zfill(3)}'
#         # set outpath
#         output_file = f'{tile_dir}_matched_cutouts.h5'
#         out_dir = os.path.join(parent_dir, tile_dir, 'gri')
#         os.makedirs(out_dir, exist_ok=True)
#         output_path = os.path.join(out_dir, output_file)
#         # skip if it already exists
#         if os.path.isfile(output_path):
#             continue

#         try:
#             # Read data for all bands
#             for band in band_names:
#                 zfill = in_dict[band]['zfill']
#                 file_prefix = in_dict[band]['name']
#                 delimiter = in_dict[band]['delimiter']
#                 suffix = in_dict[band]['suffix']
#                 num1, num2 = str(tile[0]).zfill(zfill), str(tile[1]).zfill(zfill)
#                 cutout_file = f'{file_prefix}{delimiter}{num1}{delimiter}{num2}{suffix}_cutouts.h5'
#                 cutout_path = os.path.join(parent_dir, tile_dir, band, cutout_file)
#                 cutout_dict = read_h5(cutout_path)

#                 if band == 'cfis_lsb-r':
#                     r_band_data = cutout_dict.copy()
#                 all_band_data[band] = {
#                     'cutouts': cutout_dict['images'],
#                     'ra': cutout_dict['ra'],
#                     'dec': cutout_dict['dec'],
#                 }

#             # Use r-band as reference
#             r_band = 'cfis_lsb-r'
#             r_band_coords = SkyCoord(
#                 all_band_data[r_band]['ra'], all_band_data[r_band]['dec'], unit=u.deg
#             )

#             # Match cutouts for each non-r band to r-band
#             matched_indices = {r_band: np.arange(len(all_band_data[r_band]['ra']))}
#             for band in band_names:
#                 if band != r_band:
#                     target_coords = SkyCoord(
#                         all_band_data[band]['ra'], all_band_data[band]['dec'], unit=u.deg
#                     )
#                     matched_r_indices, matched_target_indices = match_coordinates(
#                         r_band_coords, target_coords
#                     )
#                     matched_indices[band] = (matched_r_indices, matched_target_indices)

#             # Find common matches across all bands
#             common_r_indices = matched_indices[r_band]
#             for band in band_names:
#                 if band != r_band:
#                     common_r_indices = np.intersect1d(common_r_indices, matched_indices[band][0])

#             # Update matched indices for all bands based on common matches
#             final_indices = {r_band: common_r_indices}
#             for band in band_names:
#                 if band != r_band:
#                     mask = np.isin(matched_indices[band][0], common_r_indices)
#                     final_indices[band] = matched_indices[band][1][mask]

#             num_matched = len(common_r_indices)
#             logger.debug(f'Final number of matched cutouts: {num_matched}')

#             # Create the final array with shape (num_cutouts, num_bands, cutout_size, cutout_size)
#             cutout_size = all_band_data[r_band]['cutouts'].shape[1:]
#             final_cutouts = np.zeros((num_matched, len(band_names), *cutout_size), dtype=np.float32)

#             # Populate the final_cutouts array
#             for i, band in enumerate(band_names):
#                 final_cutouts[:, i] = all_band_data[band]['cutouts'][final_indices[band]]

#             # write fused cutouts to new h5 file
#             dt = h5py.special_dtype(vlen=str)
#             with h5py.File(output_path, 'w', libver='latest') as f:
#                 # Store the matched cutouts
#                 f.create_dataset('images', data=final_cutouts.astype(np.float32))
#                 # Store r-band ra and dec
#                 f.create_dataset('ra', data=all_band_data[r_band]['ra'][final_indices[r_band]])
#                 f.create_dataset('dec', data=all_band_data[r_band]['dec'][final_indices[r_band]])
#                 f.create_dataset('tile', data=r_band_data['tile'], dtype=np.int32)
#                 f.create_dataset(
#                     'known_id', data=r_band_data['known_id'][final_indices[r_band]], dtype=dt
#                 )
#                 f.create_dataset('mto_id', data=r_band_data['mto_id'][final_indices[r_band]])
#                 f.create_dataset('label', data=r_band_data['label'][final_indices[r_band]])
#                 f.create_dataset('zspec', data=r_band_data['zspec'][final_indices[r_band]])
#                 # Store band information
#                 f.create_dataset('band_names', data=np.array(band_names, dtype='S'))

#             logger.debug(f'Created matched cutouts file: {output_path}')

#             if download_file:
#                 name, ext = os.path.splitext(output_file)
#                 dir_destination = os.path.join(parent_dir_destination, tile_dir)
#                 os.makedirs(dir_destination, exist_ok=True)
#                 temp_path = os.path.join(dir_destination, name + '_temp' + ext)
#                 final_path = os.path.join(dir_destination, output_file)

#                 if os.path.exists(final_path):
#                     logger.info(
#                         f'File {output_file} was already downloaded. Skipping to next tile.'
#                     )
#                     continue

#                 logger.info(f'Downloading {output_file}...')

#                 result = subprocess.run(
#                     f'vcp -v {output_path} {temp_path}',
#                     shell=True,
#                     stderr=subprocess.PIPE,
#                     text=True,
#                 )

#                 result.check_returncode()

#                 os.rename(temp_path, final_path)

#                 logger.info(f'Successfully downloaded {output_file} to {dir_destination}.')

#         except Exception as e:
#             logger.error(f'Error processing tile {tile}: {e}')
