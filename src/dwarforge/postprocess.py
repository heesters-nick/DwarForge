import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, concatenate, search_around_sky
from astropy.io.fits import Header
from astropy.units import Unit
from astropy.wcs import WCS
from numpy.typing import NDArray

from dwarforge.utils import (
    check_corrupted_data,
    check_objects_in_neighboring_tiles,
    create_cartesian_kdtree,
    get_coord_median,
    open_fits,
    open_raw_data,
    read_parquet,
    tile_str,
)
from dwarforge.warning_manager import set_warnings

logger = logging.getLogger(__name__)


def match_stars(
    df_det: pd.DataFrame,
    df_label: pd.DataFrame,
    segmap: NDArray | None = None,
    max_sep: float = 5.0,
    extended_flag_radius: float | None = None,
    mag_limit: float = 14.0,
) -> tuple[NDArray, pd.DataFrame, pd.DataFrame, pd.DataFrame, NDArray | None, NDArray | None]:
    """
    Match detections to known objects, return matches, unmatches, and extended flag indices
    Args:
        df_det: detections dataframe
        df_label: dataframe of objects with labels
        segmap: segmentation map
        max_sep: maximum separation tolerance for direct matches
        extended_flag_radius: radius to flag detections as potentially associated with stars
    Returns:
        det_matching_idx: indices of detections for which labels are available
        label_matches: known objects that were detected
        label_unmatches: known objects that were not detected
        det_matches: detections that are known objects
        extended_flag_idx: indices of all detections within extended_flag_radius of known stars
    """
    c_det = SkyCoord(df_det['ra'], df_det['dec'], unit='deg')
    c_label = SkyCoord(df_label['ra'], df_label['dec'], unit='deg')

    # Find closest matches within max_sep
    idx, d2d, _ = c_label.match_to_catalog_3d(c_det)
    sep_constraint = d2d < max_sep * Unit('arcsec')
    det_matching_idx = idx[sep_constraint]

    label_matches = df_label[sep_constraint].reset_index(drop=True)
    label_unmatches = df_label[~sep_constraint].reset_index(drop=True)
    det_matches = df_det.loc[det_matching_idx].reset_index(drop=True)

    extended_flag_idx, extended_flag_mags = None, None
    if extended_flag_radius is not None and segmap is not None:
        bright_stars = df_label[df_label['Gmag'] < mag_limit].reset_index(drop=True)

        # Create a mask for all bright stars at once using cv2.circle
        h, w = segmap.shape
        all_stars_mask = np.zeros((h, w), dtype=np.uint8)
        for x, y in bright_stars[['x', 'y']].values:
            x, y = round(x), round(y)
            extended_flag_radius = round(extended_flag_radius)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(
                    all_stars_mask,
                    center=(x, y),
                    radius=extended_flag_radius,
                    color=255,
                    thickness=-1,
                )

        # Find all segment IDs within the masked area
        segment_ids = np.unique(segmap[all_stars_mask > 0])[1:]  # exclude 0 (background)

        extended_flag_idx = (
            segment_ids - 1
        )  # Subtract 1 because SEP IDs start at 1, but DataFrame index starts at 0

        # Find the closest bright star for each flagged segment
        flagged_segments_centers = df_det.loc[extended_flag_idx, ['xpeak', 'ypeak']].to_numpy()
        bright_stars_coords = bright_stars[['x', 'y']].to_numpy()
        distances = np.sqrt(
            np.sum((flagged_segments_centers[:, np.newaxis] - bright_stars_coords) ** 2, axis=2)
        )
        closest_star_indices = np.asarray(np.argmin(distances, axis=1), dtype=int)

        extended_flag_mags = bright_stars['Gmag'].to_numpy()[closest_star_indices]

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
        known_coords_xyz = known_coords.cartesian.xyz.value  # type: ignore

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

        potential_match_indices = tree.query_ball_point(known_coords_xyz, search_radius_chord)  # type: ignore
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
                distance = distances[potential_matches.index.get_loc(i)]  # type: ignore
                distance_score = 1 - (distance / (3600 * search_radius))  # type: ignore # Normalized distance score
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
                    f'{len(new_rows)} known dwarf(s) that was/were initially filtered out in\n\t'
                    f'tile: {tile_str(tile)}\n\t'
                    f'band: {band}:\n\t'
                    f'ID: {new_rows["ID_known"].values}\n\t'
                    f'mu: {[f"{mu:.3f}" for mu in new_rows["mu"].values]}\n\t'
                    f'r_eff: {[f"{re:.3f}" for re in new_rows["re_arcsec"].values]}\n\t'
                    f'ra: {new_rows["ra"].values}\n\t'
                    f'dec: {new_rows["dec"].values}\t'
                )
                logger.warning(warning_msg)
                warnings.append(warning_msg)

            matching_stats['unmatched_dwarfs_count'] = len(full_lsb_unmatches) + len(
                corrupted_data_objects
            )

            if len(full_lsb_unmatches) > 0:
                warning_msg = (
                    f'Found {len(full_lsb_unmatches)} undetected but known dwarfs in\n\t'
                    f'tile: {tile_str(tile)}\n\t'
                    f'band: {band}\n\t'
                    f'ID(s): {full_lsb_unmatches["ID"].values}\n\t'
                    f'ra: {full_lsb_unmatches["ra"].values}\n\t'
                    f'dec: {full_lsb_unmatches["dec"].values}\n\t'
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


def cutout2d_segmented(
    data: np.ndarray,
    tile_str: str,
    segmap: np.ndarray,
    object_id: int,
    x: int,
    y: int,
    size: int,
    cutout_in: np.ndarray,
    seg_mode: str | None,
):
    """
    Create 2d cutout from an image, applying a segmentation mask for a specific object.

    Args:
        data: image data
        tile_str: tile numbers
        segmap: segmentation map with object IDs
        object_id: ID of the object to isolate in the cutout
        x: x-coordinate of cutout center
        y: y-coordinate of cutout center
        size: square cutout size
        cutout_in: empty input cutout
        seg_mode: how to treat the segmentation map? multiply / concatenate

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
    data: np.ndarray,
    header: Header,
    tile_str: str,
    df: pd.DataFrame | None = None,
    ra: np.ndarray | None = None,
    dec: np.ndarray | None = None,
    segmap: np.ndarray | None = None,
    cutout_size: int = 64,
    seg_mode: str | None = 'multiply',
):
    """
    Makes cutouts from the objects passed in the dataframe, data is multiplied
    with the corresponding detection segment.

    Args:
        data: image
        header: fits header
        tile_str: tile numbers
        df: object dataframe. Defaults to None
        ra: right ascention array. Defaults to None
        dec: declination array. Defaults to None
        segmap: segmentation map. Defaults to None
        cutout_size: square cutout size. Defaults to 64.
        seg_mode: segmentation mode. Defaults to 'multiply'

    Returns:
        numpy.ndarray: cutouts of shape (n_cutouts, cutout_size, cutout_size)
    """
    if df is not None:
        xs = np.floor(df.X.to_numpy() + 0.5).astype(np.int32)
        ys = np.floor(df.Y.to_numpy() + 0.5).astype(np.int32)
        object_ids = df['ID'].to_numpy()  # Assuming the index is the object ID
    elif ra is not None and dec is not None:
        wcs = WCS(header)
        xs, ys = wcs.all_world2pix(ra, dec, 0)
        xs, ys = np.floor(xs + 0.5).astype(np.int32), np.floor(ys + 0.5).astype(np.int32)
        object_ids = np.zeros(len(xs))
    else:
        raise ValueError('Either df or ra and dec must be provided.')

    cutout_start = time.time()
    cutouts, cutouts_seg = create_segmented_cutouts(
        data, tile_str, segmap, object_ids, xs, ys, cutout_size, seg_mode
    )
    logger.debug(f'{tile_str}: cutouts done in {time.time() - cutout_start:.2f} seconds.')

    return cutouts, cutouts_seg


def load_segmap(file_path: Path) -> np.ndarray:
    """Load MTO segmentation map corresponding to the given fits file path."""
    seg_path = next(file_path.parent.glob('*_seg.fits'), None)
    if seg_path is None:
        raise FileNotFoundError(f'No segmentation map found for {file_path.stem}')
    mto_seg, _ = open_fits(seg_path, fits_ext=0)

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
    # Check if either band has no data
    if not band_data[band1]['ra'].size or not band_data[band2]['ra'].size:
        return np.array([], dtype=int), np.array([], dtype=int)

    reference_coords = SkyCoord(band_data[band1]['ra'], band_data[band1]['dec'], unit='deg')
    target_coords = SkyCoord(band_data[band2]['ra'], band_data[band2]['dec'], unit='deg')
    idx, d2d, _ = reference_coords.match_to_catalog_3d(target_coords)
    mask = d2d < max_sep * u.arcsec  # type: ignore
    return np.where(mask)[0], idx[mask]


def match_coordinates_across_bands(
    band_data: dict,
    max_sep: float = 10.0,
    band_priority: list = ['cfis_lsb-r', 'whigs-g', 'ps-i'],
) -> pd.DataFrame:
    """
    Cross-match detections, refining multi-band groups and keeping specific
    single-band detections (lsb=1 or known ID). Final coordinates selected
    based on priority logic applied to the final group members.

    Args:
        band_data: Dictionary containing data for each band.
        max_sep: Maximum separation for initial matching.
        band_priority: List of bands for priority selection.

    Returns:
        DataFrame with refined matches and allowed single-band detections.
    """
    # --- 1. Define Output Structure ---
    band_specific_cols = [
        'ID',
        'X',
        'Y',
        'A',
        'B',
        'theta',
        'total_flux',
        'mu_max',
        'mu_median',
        'mu_mean',
        'R_fwhm',
        'R_e',
        'R10',
        'R25',
        'R75',
        'R90',
        'R100',
        'n_pix',
        'ra',
        'dec',
        're_arcsec',
        'r_fwhm_arcsec',
        'r_10_arcsec',
        'r_25_arcsec',
        'r_75_arcsec',
        'r_90_arcsec',
        'r_100_arcsec',
        'A_arcsec',
        'B_arcsec',
        'axis_ratio',
        'mag',
        'mu',
    ]
    common_cols = ['class_label', 'zspec', 'lsb', 'ID_known']
    columns = ['ra', 'dec'] + common_cols
    for band in band_priority:
        columns.extend(f'{col}_{band}' for col in band_specific_cols)

    # --- 2. Validate and Prepare Input Data ---
    valid_bands = []
    all_coords_list = []
    bands_map = []
    orig_indices_list = []
    dfs = []
    for band in band_priority:
        if band not in band_data or band_data[band] is None:
            continue
        data = band_data[band]
        if not all(k in data for k in ['ra', 'dec', 'df']):
            continue
        if (
            not hasattr(data['ra'], '__len__')
            or not hasattr(data['dec'], '__len__')
            or len(data['ra']) == 0
            or len(data['ra']) != len(data['dec'])
        ):
            continue
        if not isinstance(data['df'], pd.DataFrame) or len(data['df']) != len(data['ra']):
            continue
        if 'ra' not in data['df'].columns or 'dec' not in data['df'].columns:
            continue
        coords = SkyCoord(ra=data['ra'], dec=data['dec'], unit='deg')
        if not np.all(np.isfinite(coords.ra.deg)) or not np.all(np.isfinite(coords.dec.deg)):  # type: ignore
            logger.error(f"  - Skipping band '{band}': Non-finite input coordinates.")
            continue
        logger.debug(f"  - Band '{band}': Found {len(coords)} detections.")
        valid_bands.append(band)
        all_coords_list.append(coords)
        bands_map.extend([band] * len(coords))
        orig_indices_list.append(np.arange(len(coords)))
        dfs.append(data['df'])

    # Need at least one band to potentially keep single detections
    if len(valid_bands) == 0:
        logger.warning('\nNo valid bands found.')
        return pd.DataFrame(columns=['unique_id'] + columns)
    logger.debug(f'\nFound {len(valid_bands)} valid bands: {valid_bands}')

    # --- 3. Initial Matching and Grouping (Union-Find) ---
    initial_groups = defaultdict(set)
    if len(valid_bands) >= 2:
        combined_coords = concatenate(all_coords_list)
        idx1, idx2, _, _ = search_around_sky(
            combined_coords, combined_coords, max_sep * Unit('arcsec')
        )
        mask = (idx1 < idx2) & (
            np.array([bands_map[i] != bands_map[j] for i, j in zip(idx1, idx2)])
        )
        idx1, idx2 = idx1[mask], idx2[mask]
        parent = list(range(len(combined_coords)))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            pu, pv = find(u), find(v)
            parent[pv] = pu if pu != pv else pv

        for i, j in zip(idx1, idx2):
            union(i, j)
        for idx in range(len(combined_coords)):
            initial_groups[find(idx)].add(idx)
    else:  # Only one valid band
        logger.debug('Only one valid band found. Checking for single-band exceptions...')
        # Treat each detection as its own group
        combined_coords = all_coords_list[0]  # Use the single band's coords
        # Need to ensure bands_map is correct for single band case
        bands_map = [valid_bands[0]] * len(combined_coords)
        for idx in range(len(combined_coords)):
            initial_groups[idx].add(idx)  # Each detection is its own group

    # --- 4. Process Initial Groups (Refinement or Single-Band Exception Check) ---
    final_rows = []
    group_counter = 0
    refined_group_count = 0
    single_kept_count = 0

    for group_indices in initial_groups.values():
        group_counter += 1
        if not group_indices:
            continue

        # Determine bands present in this initial group
        # Ensure bands_map has been correctly populated for single-band case too
        group_bands = {bands_map[idx] for idx in group_indices}
        n_bands_in_group = len(group_bands)

        refined_group_indices = set()  # Will store indices kept for the final object
        is_single_band_exception = False

        # Handle Single-Band Groups ---
        if n_bands_in_group == 1:
            the_band = list(group_bands)[0]
            keep_this_single_band_group = False
            single_member_idx_to_keep = -1

            # Check exception criteria for the first member found that qualifies
            for idx in group_indices:
                band_idx_in_valid = valid_bands.index(the_band)
                # Need offset calculation even for single band if matching was done
                current_offset = 0
                if (
                    len(valid_bands) > 1
                ):  # Calculate offset only if multiple bands existed initially
                    current_offset = sum(len(c) for c in all_coords_list[:band_idx_in_valid])

                # Ensure index 'idx' is valid relative to offset
                if idx < current_offset or idx >= current_offset + len(
                    all_coords_list[band_idx_in_valid]
                ):
                    logger.warning(
                        f'Warning: Index {idx} out of bounds for band {the_band} in group {group_counter}. Skipping index.'
                    )
                    continue

                orig_idx = orig_indices_list[band_idx_in_valid][idx - current_offset]

                # Check bounds for iloc
                if orig_idx < 0 or orig_idx >= len(dfs[band_idx_in_valid]):
                    logger.warning(
                        f'Warning: Original index {orig_idx} out of bounds for DataFrame of band {the_band} in group {group_counter}. Skipping index.'
                    )
                    continue

                df_row = dfs[band_idx_in_valid].iloc[orig_idx]

                lsb_check = 'lsb' in df_row and pd.notna(df_row['lsb']) and df_row['lsb'] == 1
                id_known_check = 'ID_known' in df_row and pd.notna(df_row['ID_known'])

                if lsb_check or id_known_check:
                    keep_this_single_band_group = True
                    single_member_idx_to_keep = idx
                    break  # Keep the first qualifying member found

            if keep_this_single_band_group:
                refined_group_indices = {single_member_idx_to_keep}
                single_kept_count += 1
                is_single_band_exception = True  # Mark as exception
                # Skip refinement logic below
            else:
                continue  # Skip this single-band group entirely

        # Handle Multi-Band Groups ---
        elif n_bands_in_group >= 2:
            is_single_band_exception = False  # Mark as multi-band
            group_coords = [combined_coords[idx] for idx in group_indices]
            if not group_coords:
                continue

            band_counts = defaultdict(int)
            members_by_band = defaultdict(list)
            for idx in group_indices:
                band = bands_map[idx]
                band_counts[band] += 1
                members_by_band[band].append(idx)
            has_conflict = any(count > 1 for count in band_counts.values())

            if has_conflict:
                try:  # Select center calculation method
                    group_center = get_coord_median(group_coords)  # type: ignore
                except Exception as e:
                    logger.error(
                        f'  Warning: Could not calculate center for group {group_counter}. Skipping. Error: {e}'
                    )
                    continue
                # Select closest member per band
                current_refined_indices = set()
                for band, member_indices in members_by_band.items():
                    if len(member_indices) == 1:
                        current_refined_indices.add(member_indices[0])
                    else:
                        min_sep = np.inf * u.deg  # type: ignore
                        closest_idx = -1
                        for idx in member_indices:
                            try:
                                coord_obj = combined_coords[idx]
                                if not isinstance(coord_obj, SkyCoord):
                                    continue
                                sep = group_center.separation(coord_obj)
                                if sep < min_sep:
                                    min_sep = sep
                                    closest_idx = idx
                            except Exception as sep_e:
                                logger.error(
                                    f'  Warning: Error calculating separation for idx {idx}. Error: {sep_e}'
                                )
                        if closest_idx != -1:
                            current_refined_indices.add(closest_idx)
                refined_group_indices = current_refined_indices
            else:  # No conflict
                refined_group_indices = group_indices

            # Check if refinement left enough bands
            refined_group_bands = {bands_map[idx] for idx in refined_group_indices}

            if not is_single_band_exception and len(refined_group_bands) < 2:
                continue  # Skip if refinement reduced bands below threshold FOR MULTI-BAND groups

            refined_group_count += 1
        else:  # Should not happen (0 bands)
            continue

        # --- 5. Process Final Group Members (Single or Refined Multi-Band) ---
        if not refined_group_indices:
            continue  # Skip if refinement resulted in empty set

        final_group_bands = {bands_map[idx] for idx in refined_group_indices}

        # Determine Final Coordinates using priority band in the final set of members
        primary_band_final = next((b for b in band_priority if b in final_group_bands), None)
        if primary_band_final is None:
            logger.warning(f'  Warning: No primary band for final group {group_counter}. Skipping.')
            continue
        try:
            primary_idx_final = next(
                idx for idx in refined_group_indices if bands_map[idx] == primary_band_final
            )
        except StopIteration:
            logger.warning(
                f'  Warning: Index for primary band {primary_band_final} not found in final group {group_counter}. Skipping.'
            )
            continue
        final_coord_from_primary = combined_coords[primary_idx_final]

        # Map final indices to original df indices
        band_orig_indices = {}
        for idx in refined_group_indices:
            band = bands_map[idx]
            band_idx_in_valid = valid_bands.index(band)
            # Need offset calculation consistent with how combined_coords was built
            current_offset = 0
            if len(valid_bands) > 1:  # Only relevant if concatenation happened
                current_offset = sum(len(c) for c in all_coords_list[:band_idx_in_valid])

            # Ensure index 'idx' is valid relative to offset before subtraction
            if idx < current_offset or idx >= current_offset + len(
                all_coords_list[band_idx_in_valid]
            ):
                logger.warning(
                    f'Warning: Index {idx} out of bounds for band {band} in band_orig_indices mapping for group {group_counter}. Skipping index.'
                )
                continue

            orig_idx = orig_indices_list[band_idx_in_valid][idx - current_offset]

            # Check bounds before iloc
            if orig_idx < 0 or orig_idx >= len(dfs[band_idx_in_valid]):
                logger.warning(
                    f'Warning: Original index {orig_idx} out of bounds for DataFrame of band {band} in group {group_counter} (band_orig_indices). Skipping index.'
                )
                continue

            band_orig_indices[band] = orig_idx

        # --- 6. Build Output Row ---
        row = {col: np.nan for col in columns}
        row['ra'] = final_coord_from_primary.ra.deg  # type: ignore
        row['dec'] = final_coord_from_primary.dec.deg  # type: ignore

        # Fill common columns
        for col in common_cols:
            for band in band_priority:
                # Use final_group_bands here
                if band in final_group_bands:
                    # Check if band exists in mapping
                    if band not in band_orig_indices:
                        continue
                    orig_idx = band_orig_indices[band]
                    band_df_row = dfs[valid_bands.index(band)].iloc[orig_idx]
                    if col in band_df_row:
                        value = band_df_row[col]
                        is_valid = (value is not None) if (col == 'ID_known') else pd.notna(value)
                        if is_valid:
                            row[col] = value
                            break
        # Fill band-specific columns
        for band in final_group_bands:  # Use final_group_bands here
            # Verify that band exists in mapping
            if band not in band_orig_indices:
                continue
            orig_idx = band_orig_indices[band]
            band_df_row = dfs[valid_bands.index(band)].iloc[orig_idx]
            for col in band_specific_cols:
                col_name = f'{col}_{band}'
                if col in band_df_row:
                    row[col_name] = band_df_row[col]
        final_rows.append(row)

    if not final_rows:
        logger.warning('Warning: No matching objects found.')
        result_df = pd.DataFrame(columns=['unique_id'] + columns)
        # Ensure unique_id exists even if empty
        if 'unique_id' in result_df.columns:
            result_df = result_df.drop(columns=['unique_id'])
        result_df.insert(0, 'unique_id', [])
        return result_df  # Return empty DataFrame with correct columns

    # Create the DataFrame *before* adding unique_id and resolving conflicts
    result_df = pd.DataFrame(final_rows, columns=columns)

    # --- 8. Resolve ID_known Conflicts using Surface Brightness ('mu') ---
    indices_to_drop = set()  # Use a set to automatically handle duplicates

    # Check if ID_known column exists and has data
    if 'ID_known' in result_df.columns and result_df['ID_known'].notna().any():
        # Find IDs that appear more than once (duplicates)
        id_counts = result_df.dropna(subset=['ID_known'])['ID_known'].value_counts()
        duplicated_ids = id_counts[id_counts > 1].index.tolist()

        if duplicated_ids:
            logger.debug(f'  - Found {len(duplicated_ids)} ID_known values with conflicts.')
            # Define the mu column names to check based on the bands used
            mu_cols_to_check = [
                f'mu_{band}' for band in valid_bands if f'mu_{band}' in result_df.columns
            ]

            for conflict_id in duplicated_ids:
                # Get rows with the current conflicting ID
                conflict_rows = result_df[result_df['ID_known'] == conflict_id]

                # Find the best (minimum) mu value for each conflicting row
                # Handle rows where all relevant mu values might be NaN
                best_mus = conflict_rows[mu_cols_to_check].min(
                    axis=1, skipna=True
                )  # Get min mu per row

                # If all mu values for all conflicting rows are NaN, keep the first one arbitrarily
                if best_mus.isna().all():
                    idx_to_keep = conflict_rows.index[0]
                    logger.debug(
                        f'    - ID {conflict_id}: All mu values NaN. Keeping first occurrence (index {idx_to_keep}).'
                    )
                else:
                    # Find the index corresponding to the overall minimum mu value (highest surf bright)
                    # idxmin will skip NaNs by default
                    idx_to_keep = best_mus.idxmax()
                    logger.debug(
                        f'    - ID {conflict_id}: Keeping row index {idx_to_keep} (Best mu: {best_mus[idx_to_keep]:.4f}).'
                    )

                # Add indices of all *other* rows with this ID to the drop set
                indices_to_drop.update(conflict_rows.index.difference([idx_to_keep]))

            # Drop the conflicting rows
            if indices_to_drop:
                logger.debug(f'  - Dropping {len(indices_to_drop)} rows due to ID_known conflicts.')
                result_df = result_df.drop(index=list(indices_to_drop))
        else:
            logger.debug('  - No duplicate ID_known values found.')
    else:
        logger.debug(
            "  - No 'ID_known' column or no valid IDs found, skipping conflict resolution."
        )

    # --- 9. Add Final Unique IDs ---
    # Ensure unique_id column doesn't exist before inserting
    if 'unique_id' in result_df.columns:
        result_df = result_df.drop(columns=['unique_id'])
    # Insert unique IDs based on the final filtered DataFrame
    result_df.insert(0, 'unique_id', np.arange(len(result_df)))
    logger.debug(f'Total objects in final DataFrame after conflict resolution: {len(result_df)}')

    return result_df


def read_band_data(
    parent_dir: Path,
    tile_dir: Path,
    tile: tuple[int, int],
    band: str,
    in_dict: dict,
    seg_mode: str | None,
    use_full_res: bool = False,
) -> tuple[
    np.ndarray,
    Header,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    # Read the full image, segmentation map, and catalog
    zfill = in_dict[band]['zfill']
    file_prefix = in_dict[band]['name']
    delimiter = in_dict[band]['delimiter']
    suffix = in_dict[band]['suffix']
    fits_ext = in_dict[band]['fits_ext']

    base_dir = parent_dir / tile_dir / band

    num1, num2 = str(tile[0]).zfill(zfill), str(tile[1]).zfill(zfill)
    filename = f'{file_prefix}{delimiter}{num1}{delimiter}{num2}{suffix}'
    data_path = base_dir / filename

    if use_full_res:
        data, header = open_raw_data(file_path=data_path, fits_ext=fits_ext, band=band)
    else:
        data_path = data_path.with_stem(f'{data_path.stem}_rebin').with_suffix('.fits')
        data, header = open_fits(file_path=data_path, fits_ext=0)

    if seg_mode is not None:
        seg_matches = list(data_path.parent.glob(f'{data_path.stem}*_seg.fits'))
        if seg_matches:
            segmap, _ = open_fits(seg_matches[0], fits_ext=0)
        else:
            segmap = None
    else:
        segmap = None

    det_matches = list(data_path.parent.glob(f'{data_path.stem}*_det_params.parquet'))
    if not det_matches:
        logger.info(f'Tile {tile}: detection catalog not found for band {band}.')
        return data, header, segmap, np.zeros(1), np.zeros(1), pd.DataFrame()

    det_df = pd.read_parquet(det_matches[0])
    # filter det_df for each band before performing matching
    # exclude likely non-dwarfs beforehand
    if 'lsb' not in det_df.columns:
        det_df = filter_candidates(df=det_df, tile=tile, band=band)
    elif ('lsb' in det_df.columns) and np.count_nonzero(det_df['lsb'] == 1) == 0:
        det_df = filter_candidates(df=det_df, tile=tile, band=band)
    else:
        # only filter non-dwarf dataframe to avoid losing known dwarfs for training
        det_df_other = det_df[det_df['lsb'] != 1].reset_index(drop=True)
        det_df_dwarf = det_df[det_df['lsb'] == 1].reset_index(drop=True)
        det_df = filter_candidates(df=det_df_other, tile=tile, band=band)
        det_df = pd.concat([det_df, det_df_dwarf], ignore_index=True)
        det_df = det_df.sort_values('ID')  # keep original order

    if det_df is None:
        det_df = pd.DataFrame(columns=['ra', 'dec'])

    return data, header, segmap, det_df['ra'].to_numpy(), det_df['dec'].to_numpy(), det_df


def filter_candidates(df: pd.DataFrame, tile: tuple[int, int], band: str) -> pd.DataFrame | None:
    df_mod = df.copy()
    if 'lsb' in df.columns:
        df_dwarf = df.loc[df['lsb'] == 1].reset_index(drop=True)
    else:
        df_dwarf = pd.DataFrame(columns=df.columns)

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

    if 'lsb' in df.columns:
        logger.debug(
            f'{tile}, {band}: filtered out {len(df.loc[df["lsb"] == 1]) - len(df_mod.loc[df_mod["lsb"] == 1])}/{len(df_dwarf)} dwarfs.'
        )
        logger.debug(
            f'{tile}, {band}: filtered out {len(df.loc[df["lsb"].isna()]) - len(df_mod.loc[df_mod["lsb"].isna()])}/{len(df.loc[df["lsb"].isna()])} other objects.'
        )

    return df_mod
