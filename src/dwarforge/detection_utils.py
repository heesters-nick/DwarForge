import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
import sep
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from dwarforge.postprocess import match_stars
from dwarforge.utils import open_fits
from scipy.ndimage import binary_dilation, label

logger = logging.getLogger(__name__)


def run_mto(
    file_path: Path,
    band: str,
    mto_path: Path,
    with_segmap: bool = False,
    move_factor: float = 0.5,
    min_distance: float = 0.1,
):
    # Validate inputs early
    if not Path(file_path).is_file():
        raise FileNotFoundError(f'MTO input not found: {file_path}')
    if not Path(mto_path).is_file():
        raise FileNotFoundError(f'MTO script not found: {mto_path}')

    file_path = Path(file_path)
    mto_path = Path(mto_path)

    fits_name = file_path.name
    tile_name = file_path.stem
    file_dir = file_path.parent

    seg_path = file_dir / f'{tile_name}_seg.fits'
    param_path = file_dir / f'{tile_name}_det_params.csv'

    # Build argv list (no shell) and always use the *current* Python
    argv = [sys.executable, str(mto_path), str(file_path), '-par_out', str(param_path)]
    if with_segmap:
        argv = [
            sys.executable,
            str(mto_path),
            str(file_path),
            '-out',
            str(seg_path),
            '-par_out',
            str(param_path),
            '-move_factor',
            str(move_factor),
            '-min_distance',
            str(min_distance),
        ]

    mto_start = time.time()
    logger.info(f'Running MTO on file: {fits_name} (band={band})')
    try:
        res = subprocess.run(argv, check=True, capture_output=True, text=True)
        # Optional: log stdout at debug so you have it when needed
        if res.stdout:
            # rewrite with f strings
            logger.debug(f'MTO stdout for {fits_name}:\n{res.stdout}')
        logger.info(
            f'Successfully ran MTO on tile {tile_name} for band {band} in {np.round((time.time() - mto_start), 2)} s.'
        )
    except subprocess.CalledProcessError as e:
        # Show EVERYTHING so the cause is clear
        logger.error(
            f'Tile {tile_name}: MTO failed in {band} (exit={e.returncode})\nCMD: {" ".join(argv)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}'
        )
        # Re-raise if you want the pipeline to stop; or return paths with a failure signal
        raise

    return param_path, seg_path


def param_phot(param_path, header, zp=30.0, mu_lim=22.0, re_lim=1.6, band='cfis_lsb-r'):
    params_field = pd.read_csv(param_path)
    params_field['ra'], params_field['dec'] = WCS(header).all_pix2world(
        params_field['X'], params_field['Y'], 0
    )
    pixel_scale = abs(header['CD1_1'] * 3600)
    params_field['re_arcsec'] = params_field.R_e * pixel_scale
    params_field['r_fwhm_arcsec'] = params_field.R_fwhm * pixel_scale
    params_field['r_10_arcsec'] = params_field.R10 * pixel_scale
    params_field['r_25_arcsec'] = params_field.R25 * pixel_scale
    params_field['r_75_arcsec'] = params_field.R75 * pixel_scale
    params_field['r_90_arcsec'] = params_field.R90 * pixel_scale
    params_field['r_100_arcsec'] = params_field.R100 * pixel_scale
    params_field['A_arcsec'] = params_field.A * pixel_scale
    params_field['B_arcsec'] = params_field.B * pixel_scale
    params_field['axis_ratio'] = np.minimum(
        params_field['A_arcsec'], params_field['B_arcsec']
    ) / np.maximum(params_field['A_arcsec'], params_field['B_arcsec'])
    params_field['mag'] = -2.5 * np.log10(params_field.total_flux) + zp
    params_field['mu'] = np.where(
        (params_field['A_arcsec'] > 0) & (params_field['B_arcsec'] > 0),
        params_field.mag
        + 0.752
        + 2.5
        * np.log10(
            np.pi * params_field.re_arcsec**2 * params_field.B_arcsec / params_field.A_arcsec
        ),
        params_field.mag + 0.752 + 2.5 * np.log10(np.pi * params_field.re_arcsec**2),
    )
    params_field = params_field[~params_field.isin([np.inf, -np.inf]).any(axis=1)].reset_index(
        drop=True
    )
    mto_all = params_field.copy()

    # Define band-specific conditions
    band_conditions = {
        'cfis_lsb-r': {
            'basic': {
                'mu': (mu_lim, None),
                're_arcsec': (re_lim, 55.0),
                # 'axis_ratio': (0.17, None),
                # 'r_10_arcsec': (0.39, 19.0),
                # 'r_90_arcsec': (2.3, 150.0),
                # 'r_fwhm_arcsec': (0.4, 10.6),
                # 'mu_median': (0.3, 29.0),
                # 'mu_mean': (0.4, 70.0),
                # 'mu_max': (1.1, 5700.0),
                # 'total_flux': (55, None),
                # 'mag': (13.8, 25.7),
            },
            'complex': [
                lambda df: df['mu'] > (0.6060 * df['mag'] + 11.6293),
                lambda df: df['mu_max'] < (1.2e10 * np.exp(0.84 * df['mag'])),
                lambda df: df['r_90_arcsec'] < (12.0000 * df['r_10_arcsec'] + 20.0000),
                lambda df: df['r_90_arcsec'] < (3.0 * df['re_arcsec'] + 9.0000),
                lambda df: df['mu_median'] < (4.0 * df['re_arcsec'] + 4.0000),
                lambda df: df['mu_median'] < (4.0 * df['r_10_arcsec'] + 12.0000),
                lambda df: df['mu_median'] < (0.3 * df['r_90_arcsec'] + 15.0000),
                lambda df: df['mu_max'] < (120.0 * df['r_90_arcsec'] + 650.0000),
            ],
        },
        'whigs-g': {
            'basic': {
                'mu': (mu_lim, None),
                're_arcsec': (re_lim, 55.0),
                # 'axis_ratio': (0.17, None),
                # 'r_10_arcsec': (0.39, 8.0),
                # 'r_90_arcsec': (2.5, 40.0),
                # 'r_fwhm_arcsec': (0.4, 7.5),
                # 'mu_median': (0.1, 12.0),
                # 'mu_mean': (0.25, 17.3),
                # 'mu_max': (1.7, 2600.0),
                # 'total_flux': (39, None),
                # 'mag': (15.3, 23.0),
            },
            'complex': [
                lambda df: df['mu'] > (0.6981 * df['mag'] + 8.7072),
                lambda df: df['mu_max'] < (-20.0 * df['mag'] + 500.0),
                lambda df: df['r_90_arcsec'] > (4.3025 * df['r_10_arcsec'] - 1.5000),
                lambda df: df['r_90_arcsec'] < (1.9000 * df['re_arcsec'] + 3.5000),
                lambda df: df['mu_median'] < (0.1500 * df['re_arcsec'] + 3.3),
                lambda df: df['mu_median'] < (3.0 * df['r_10_arcsec'] + 1.0000),
                lambda df: df['mu_median'] < (0.0800 * df['r_90_arcsec'] + 3.5),
                lambda df: df['mu_max'] < (14.0 * df['r_90_arcsec'] + 20),
            ],
        },
        'ps-i': {
            'basic': {
                'mu': (mu_lim, None),  # (min, max), None means no limit
                're_arcsec': (re_lim, 46.0),
                # 'axis_ratio': (0.17, None),
                # 'r_10_arcsec': (0.39, 16.2),
                # 'r_90_arcsec': (2.51, 85.0),
                # 'r_fwhm_arcsec': (0.4, 14.2),
                # 'mu_median': (0.35, 34.0),
                # 'mu_mean': (0.4, 103.0),
                # 'mu_max': (2, 8300.0),
                # 'total_flux': (68, None),
                # 'mag': (15.8, 25.3),
            },
            'complex': [
                lambda df: df['mu'] > (0.5864 * df['mag'] + 11.9705),
                lambda df: df['mu_max']
                < (
                    5593.3712
                    / (1 + np.exp(0.9174 * (df['mag'] - 16.1930)))
                    * np.exp(-0.0000 * df['mag'])
                    + 200.0000
                ),
                lambda df: df['r_90_arcsec'] < (12.0000 * df['r_10_arcsec'] + 3.0),
                lambda df: df['r_90_arcsec'] < (3.0 * df['re_arcsec'] + 3.8),
                lambda df: df['mu_median'] < (2.0 * df['re_arcsec'] + 8.0000),
                lambda df: df['mu_median'] < (4.0 * df['r_10_arcsec'] + 12.0000),
                lambda df: df['mu_median'] < (4 * df['r_90_arcsec'] + 9.5),
                lambda df: df['mu_max'] < (35 * df['r_90_arcsec'] + 100.0000),
                lambda df: df['r_25_arcsec'] < (2.5 * df['r_10_arcsec'] + 1.0000),
            ],
        },
    }

    if band not in band_conditions:
        logger.error(f'Conditions not implemented for band {band}.')
        return None, None

    conditions = band_conditions[band]

    # Apply basic conditions
    for column, (min_val, max_val) in conditions['basic'].items():
        if min_val is not None:
            params_field = params_field[params_field[column] > min_val]
        if max_val is not None:
            params_field = params_field[params_field[column] < max_val]

    # Apply complex conditions
    # for condition in conditions['complex']:
    #     params_field = params_field[condition]

    # Remove streaks
    params_field = params_field[
        (params_field['axis_ratio'] >= 0.17) | (params_field['n_pix'] <= 1000)
    ]

    # Reset index
    params_field = params_field.reset_index(drop=True)

    return params_field, mto_all


def source_detection(
    image: np.ndarray,
    header: Header,
    file_path: Path,
    star_df: pd.DataFrame,
    thresh: float = 3.0,
    minarea: int = 5,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.005,
    bkg_dim: int = 50,
    save_segmap: bool = False,
    extended_flag_radius: float = 9.0,
    mag_limit: float = 14.0,
) -> tuple[pd.DataFrame, np.ndarray, Any, np.ndarray]:
    """
    Source detection using SEP (Source Extractor in Python).

    Args:
        image: image data
        header: FITS header of the image
        file_path: Path to the image file, used for saving outputs
        star_df: DataFrame containing star catalog for matching
        thresh: detection threshold. Defaults to 3.
        minarea: minimum number of pixels for a valid detection. Defaults to 5.
        deblend_nthresh: number of thresholds for deblending. Defaults to 32.
        deblend_cont: minimum contrast ratio for deblending. Defaults to 0.005.
        bkg_dim: size of the background mesh. Defaults to 50.
        save_segmap: whether to save the segmentation map. Defaults to False.
        extended_flag_radius: radius in arcseconds to flag extended sources near stars. Defaults to 9.0.
        mag_limit: magnitude limit for star catalog matching. Defaults to 14.0.

    Returns:
        objects: A record array containing the properties of the detected objects.
        data_sub: The image with the background subtracted.
        bkg: The background estimation model.
        segmap: The segmentation map.
    """
    logger.debug('starting sep')
    # avoid changing the original image
    image_copy = image.copy()
    # set zeros to nan so sep can ignore them
    image_copy[image_copy == 0] = np.nan
    sep_start = time.time()
    bkg = sep.Background(image_copy, maskthresh=5, bw=bkg_dim)
    # non-zero mask
    mask = image != 0
    # only subtract background where the image is non-zero
    data_sub = np.where(mask, image_copy - bkg.back(), image_copy)

    objects, segmap = sep.extract(  # noqa: F821
        data_sub,
        thresh=thresh,
        err=bkg.globalrms,
        minarea=minarea,
        segmentation_map=True,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
    )
    logger.debug(f'finished sep in {time.time()-sep_start:.2f} seconds.')

    if save_segmap:
        out_path_seg = file_path.with_stem(f'{file_path.stem}_sep_seg')
        seg_hdu = fits.PrimaryHDU(data=segmap.astype(np.float32), header=header)
        seg_hdu.writeto(out_path_seg, overwrite=True)

    # Convert the structured numpy array to a DataFrame and ensure it's a copy
    objects_df = pd.DataFrame(objects).copy()
    # add world coordinates to df
    objects_df['ra'], objects_df['dec'] = WCS(header).all_pix2world(
        objects_df['xpeak'], objects_df['ypeak'], 0
    )

    det_matching_idx, gaia_matches, _, _, extended_flag_idx, extended_flag_mags = match_stars(
        objects_df,
        star_df,
        segmap=segmap,
        max_sep=5,
        extended_flag_radius=extended_flag_radius,
        mag_limit=mag_limit,
    )
    objects_df['star'] = 0
    objects_df.loc[det_matching_idx, 'star'] = 1  # type: ignore
    objects_df.loc[det_matching_idx, 'Gmag'] = gaia_matches['Gmag'].values  # type: ignore
    objects_df['star_cand'] = 0
    objects_df.loc[extended_flag_idx, 'star_cand'] = 1  # type: ignore
    objects_df['Gmag_closest'] = np.nan
    objects_df.loc[extended_flag_idx, 'Gmag_closest'] = extended_flag_mags  # type: ignore
    objects_df.insert(0, 'ID', objects_df.index + 1)

    data_sub[np.isnan(data_sub)] = 0.0

    return objects_df, data_sub, bkg, segmap


def source_detection_with_dynamic_limit(
    data_ano_mask: np.ndarray,
    header_binned: Header,
    file_path: Path,
    sorted_star_df: pd.DataFrame,
    thresh: float = 1.0,
    minarea: int = 4,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.0005,
    bkg_dim: int = 50,
    save_segmap: bool = False,
    extended_flag_radius: float = 9.0,
    mag_limit: float = 14.0,
    initial_limit: int = 500000,
    max_limit: int = 10000000,
    increment_factor: int = 2,
) -> tuple[pd.DataFrame, np.ndarray, Any, np.ndarray] | tuple[None, None, None, None]:
    """
    Wrapper for source_detection that dynamically adjusts SEP's internal pixel stack limit.
    Args:
        data_ano_mask: Image data with anomalies masked.
        header_binned: FITS header of the binned image.
        file_path: Path to the image file, used for saving outputs.
        sorted_star_df: DataFrame containing star catalog for matching.
        thresh: detection threshold. Defaults to 1.0.
        minarea: minimum number of pixels for a valid detection. Defaults to 4.
        deblend_nthresh: number of thresholds for deblending. Defaults to 32.
        deblend_cont: minimum contrast ratio for deblending. Defaults to 0.0005.
        bkg_dim: size of the background mesh. Defaults to 50.
        save_segmap: whether to save the segmentation map. Defaults to False.
        extended_flag_radius: radius in arcseconds to flag extended sources near stars. Defaults to 9.0.
        mag_limit: magnitude limit for star catalog matching. Defaults to 14.0.
        initial_limit: initial pixel stack limit for SEP. Defaults to 500000.
        max_limit: maximum pixel stack limit for SEP. Defaults to 10000000.
        increment_factor: factor by which to increase the limit on failure. Defaults to 2.
    Returns:
        tuple: A tuple containing:
            - objects: A DataFrame containing the properties of the detected objects.
            - data_sub: The image with the background subtracted.
            - bkg: The background estimation model.
            - segmap: The segmentation map.
        If detection fails even at max_limit, returns (None, None, None, None).
    """
    current_limit = initial_limit
    sep.set_extract_pixstack(current_limit)
    sep.set_sub_object_limit(1024)

    while current_limit <= max_limit:
        try:
            objects_sep, data_sub_sep, bkg_sep, segmap_sep = source_detection(
                data_ano_mask,
                header_binned,
                file_path,
                sorted_star_df,
                thresh=thresh,
                minarea=minarea,
                deblend_nthresh=deblend_nthresh,
                deblend_cont=deblend_cont,
                bkg_dim=bkg_dim,
                save_segmap=save_segmap,
                extended_flag_radius=extended_flag_radius,
                mag_limit=mag_limit,
            )
            return objects_sep, data_sub_sep, bkg_sep, segmap_sep

        except Exception as e:
            if 'internal pixel buffer full' in str(e):
                logger.warning(
                    f'SEP pixel stack limit of {current_limit} exceeded. Increasing limit.'
                )
                current_limit *= increment_factor
                sep.set_extract_pixstack(current_limit)
            else:
                logger.error(f'Error in source_detection: {e}')
                raise

    logger.error(f'source_detection failed even with maximum pixel stack limit of {max_limit}')
    return None, None, None, None


def detect_anomaly(
    image: np.ndarray | Path,
    header: Header,
    file_path: Path,
    zero_threshold: float = 0.0025,
    min_size: int = 50,
    replace_anomaly: bool = False,
    dilate_mask: bool = False,
    dilation_iters: int = 1,
    save_to_file: bool = False,
    band: str = 'cfis_lsb-r',
) -> tuple[np.ndarray, Path]:
    """
    Detect anomalies in an image using wavelet transforms and connected-component labeling.
    Args:
        image: 2D array of image data or path to the FITS file.
        header: FITS header of the image.
        file_path: Path to the image file, used for saving outputs.
        zero_threshold: Threshold below which wavelet coefficients are considered zero. Defaults to 0.0025.
        min_size: Minimum size of connected components to be considered anomalies. Defaults to 50.
        replace_anomaly: Whether to replace detected anomalies with zeros. Defaults to False.
        dilate_mask: Whether to dilate the anomaly mask. Defaults to False.
        dilation_iters: Number of iterations for mask dilation. Defaults to 1.
        save_to_file: Whether to save the processed image to a new FITS file. Defaults to False.
        band: Band of the image, e.g. 'cfis_lsb-r', 'whigs-g', 'ps-i'. Affects thresholding. Defaults to 'cfis_lsb-r'.
    Returns:
        tuple: A tuple containing:
            - The processed image with anomalies handled as specified.
            - Path to the saved FITS file if save_to_file is True, otherwise None.
    """
    # Takes both data or path to file
    if isinstance(image, Path):
        image, header = open_fits(image, fits_ext=0)

    # replace nan values with zeros
    image[np.isnan(image)] = 0.0
    # replace saturated pixels with zeros
    image[image >= 65536] = 0.0
    # replace highly negative values with zeros
    if band == 'ps-i':
        image[image < -2.0] = 0.0
    else:
        image[image < -5.0] = 0.0

    # Perform a 2D Discrete Wavelet Transform using Haar wavelets
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs  # Decomposition into approximation and details

    # Create binary masks where wavelet coefficients are below the threshold
    mask_horizontal = np.abs(cH) <= zero_threshold
    mask_vertical = np.abs(cV) <= zero_threshold
    mask_diagonal = np.abs(cD) <= zero_threshold

    masks = [mask_diagonal, mask_horizontal, mask_vertical]

    global_mask = np.zeros_like(image, dtype=bool)
    component_masks = np.zeros((3, cA.shape[0], cA.shape[1]), dtype=bool)
    anomalies = np.zeros(3, dtype=bool)
    for i, mask in enumerate(masks):
        # Apply connected-component labeling to find connected regions in the mask
        labeled_array, num_features = label(mask)  # type: ignore

        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

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
            for i, comp_mask in enumerate(component_masks):
                component_masks[i] = binary_dilation(comp_mask, iterations=dilation_iters)
    # Replace the anomaly with gaussian sky noise
    if replace_anomaly:
        image[global_mask] = 0.0

    out_path = file_path.with_stem(f'{file_path.stem}_ano_mask')
    if save_to_file:
        new_hdu = fits.PrimaryHDU(data=image.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return image, out_path
