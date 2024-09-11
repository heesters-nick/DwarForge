import os
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import pywt

# import sep
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, label

from logging_setup import get_logger
from preprocess import open_fits

logger = get_logger()


def run_mto_old(
    tile_nums,
    file_path,
    model_dir,
    mto_path,
    band,
    mu_lim,
    reff_lim,
    bkg,
    zp,
    header,
    with_segmap=False,
):
    fits_name = os.path.basename(file_path)
    tile_name = os.path.splitext(fits_name)[0]
    file_dir = os.path.dirname(file_path)
    out_path = os.path.join(file_dir, tile_name + '_seg.fits')
    mto_start = time.time()
    logger.info(f'Running MTO on file: {fits_name}')
    param_path = os.path.join(file_dir, tile_name + '_det_params.csv')
    exec_str = f'python {mto_path} {file_path} -par_out {param_path}'
    exec_str_seg = f'python {mto_path} {file_path} -out {out_path} -par_out {param_path}'
    if with_segmap:
        exec_str = exec_str_seg
    try:
        result_mto = subprocess.run(exec_str, shell=True, stderr=subprocess.PIPE, text=True)
        result_mto.check_returncode()
        logger.info(
            f'Successfully ran MTO on tile {tuple(tile_nums)} for band {band}. \n Finished in  {np.round((time.time() - mto_start)/60, 3)} minutes.'
        )

    except subprocess.CalledProcessError as e:
        logger.error(f'Tile {tuple(tile_nums)} failed to download in {band}.')
        logger.exception(f'Subprocess error details: {e}')

        logger.info('Trying to run MTO with Source Extractor background estimation..')
        try:
            mto_start_se = time.time()
            exec_str_se = f'python {mto_path} {file_path} -out {out_path} -par_out {param_path} -bg_variance {bkg.background_rms_median**2} -bg_mean {np.mean(bkg.background())}'
            result_mto = subprocess.run(exec_str_se, shell=True, stderr=subprocess.PIPE, text=True)
            result_mto.check_returncode()
            logger.info(
                f'Successfully ran MTO using Source Extractor background estimation on tile {tuple(tile_nums)} for band {band}. \n Finished in  {np.round((time.time() - mto_start_se)/60, 3)} minutes.'
            )

        except subprocess.CalledProcessError as e:
            logger.error(
                f'Failed running MTO on Tile {tuple(tile_nums)} in {band} with Source Extractor background estimation.'
            )
            logger.exception(f'Subprocess error details: {e}')
            return None, None

    except Exception as e:
        logger.error(f'Tile {tuple(tile_nums)} in {band}: an unexpected error occurred: {e}')
        return None, None

    logger.info('Loading detections.., selecting dwarfs..')
    sel_start = time.time()
    if os.path.exists(param_path):
        pixel_scale = abs(header['CD1_1'] * 3600)
        params_field = pd.read_csv(param_path)
        params_field['ra'], params_field['dec'] = WCS(header).all_pix2world(
            params_field['x'], params_field['y'], 0
        )
        params_field['re_arcsec'] = params_field.R_e * pixel_scale
        params_field['r_fwhm_arcsec'] = params_field.R_fwhm * pixel_scale
        params_field['r_10_arcsec'] = params_field.R10 * pixel_scale
        params_field['r_90_arcsec'] = params_field.R90 * pixel_scale
        params_field['A_arcsec'] = params_field.A * pixel_scale
        params_field['B_arcsec'] = params_field.B * pixel_scale
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
        # drop all sources that are not dwarfs
        params_field = params_field.loc[
            (params_field['mu'] < mu_lim) & (params_field['re_arcsec'] < reff_lim)
        ].reset_index(drop=True)

        df_for_pred = params_field[
            [
                'total_flux',
                'mu_max',
                'mu_median',
                'mu_mean',
                're_arcsec',
                'r_fwhm_arcsec',
                'r_10_arcsec',
                'r_90_arcsec',
                'A_arcsec',
                'B_arcsec',
                'mag',
                'mu',
            ]
        ]
        # Load the saved model from the file
        loaded_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))

        # Use the loaded model for predictions or further analysis
        predictions = loaded_model.predict(df_for_pred)
        params_field['label'] = predictions
        logger.info(
            f'Dwarfs selected: {np.count_nonzero(params_field["label"] == 1)}. Took {np.round(time.time() - sel_start, 3)} seconds.'
        )
        params_field.to_csv(param_path, index=False)

    else:
        params_field = None
        logger.error('No parameter file found. Check what went wrong.')

    return params_field


def run_mto(
    file_path,
    band,
    mto_path='./mto.py',
    with_segmap=False,
    move_factor=0.5,
    min_distance=0.1,
):
    # paths
    fits_name = os.path.basename(file_path)
    tile_name = os.path.splitext(fits_name)[0]
    file_dir = os.path.dirname(file_path)
    out_path = os.path.join(file_dir, tile_name + '_seg.fits')

    mto_start = time.time()
    logger.info(f'Running MTO on file: {fits_name}')
    param_path = os.path.join(file_dir, tile_name + '_det_params.csv')
    exec_str = f'python {mto_path} {file_path} -par_out {param_path}'
    exec_str_seg = f'python {mto_path} {file_path} -out {out_path} -par_out {param_path} -move_factor {move_factor} -min_distance {min_distance}'
    if with_segmap:
        exec_str = exec_str_seg
    try:
        result_mto = subprocess.run(exec_str, shell=True, stderr=subprocess.PIPE, text=True)
        result_mto.check_returncode()
        logger.info(
            f'Successfully ran MTO on tile {tile_name} for band {band} in {np.round((time.time() - mto_start), 2)} seconds.'
        )

    except subprocess.CalledProcessError as e:
        logger.error(f'Tile {tile_name}: MTO failed in {band}. Subprocess error details: {e}')

    return param_path


def param_phot(param_path, header, zp=30.0, mu_min=19.0, reff_min=1.4):
    params_field = pd.read_csv(param_path)
    params_field['ra'], params_field['dec'] = WCS(header).all_pix2world(
        params_field['X'], params_field['Y'], 0
    )
    pixel_scale = abs(header['CD1_1'] * 3600)
    params_field['re_arcsec'] = params_field.R_e * pixel_scale
    params_field['r_fwhm_arcsec'] = params_field.R_fwhm * pixel_scale
    params_field['r_10_arcsec'] = params_field.R10 * pixel_scale
    params_field['r_90_arcsec'] = params_field.R90 * pixel_scale
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
    params_field = params_field.loc[
        (params_field['mu'] > mu_min)
        & (params_field['re_arcsec'] > reff_min)
        & (params_field['axis_ratio'] > 0.1)
    ].reset_index(drop=True)
    # Remove streaks
    params_field = params_field[
        (params_field['axis_ratio'] >= 0.17) | (params_field['n_pix'] <= 1000)
    ]

    return params_field, mto_all


def source_detection(image, thresh=3.0, minarea=5):
    """
    Source detection using SEP (Source Extractor in Python).

    Args:
        image (numpy.ndarray): image data
        thresh (float, optional): detection threshold. Defaults to 3.
        minarea (int, optional): minimum number of pixels for a valid detection. Defaults to 5.

    Returns:
        objects (numpy.ndarray): A record array containing the properties of the detected objects.
        data_sub (numpy.ndarray): The image with the background subtracted.
        bkg (sep.Background): The background estimation model.
        segmap (numpy.ndarray): The segmentation map.
    """
    logger.info('starting sep')
    image_c = image.byteswap().newbyteorder()
    bkg = sep.Background(image_c, maskthresh=thresh, bw=128)  # noqa: F821
    data_sub = image_c - bkg
    objects, segmap = sep.extract(  # noqa: F821
        data_sub,
        thresh=thresh,
        err=bkg.globalrms,
        minarea=minarea,
        segmentation_map=True,
        deblend_nthresh=32,
        deblend_cont=0.005,
    )
    logger.info('finished sep')
    return objects, data_sub, bkg, segmap


def detect_anomaly(
    image,
    zero_threshold=0.0025,
    min_size=50,
    replace_anomaly=False,
    dilate_mask=False,
    dilation_iters=1,
):
    # Takes both data or path to file
    if isinstance(image, str):
        image, header = open_fits(image, fits_ext=0)

    # replace nan values with zeros
    image[np.isnan(image)] = 0.0
    # replace highly negative values with zeros
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

    return image
