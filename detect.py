import os
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import pywt
import sep
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, label

from logging_setup import get_logger
from postprocess import match_stars
from utils import open_fits

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


def param_phot_old(param_path, header, zp=30.0, mu_lim=22.0, re_lim=1.6, band='cfis_lsb-r'):
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

    if band == 'cfis_lsb-r':
        conditions = {
            'mu': (mu_lim, None),  # (min, max), None means no limit
            're_arcsec': (re_lim, None),
            'axis_ratio': (0.17, None),
            'r_10_arcsec': (0.39, 19.0),
            'r_25_arcsec': (0.7, None),
            'r_75_arcsec': (1.98, None),
            'r_90_arcsec': (2.3, 150.0),
            'r_100_arcsec': (2.0, None),
            'r_fwhm_arcsec': (0.4, 10.6),
            'mu_median': (0.3, 29.0),
            'mu_mean': (0.4, 70.0),
            'mu_max': (1.7, 5700.0),
            'total_flux': (55, None),
            'mag': (13.8, 25.7),
        }
    elif band == 'whigs-g':
        conditions = {
            'mu': (mu_lim, None),  # (min, max), None means no limit
            're_arcsec': (re_lim, None),
            # 'axis_ratio': (0.17, None),
            # 'r_10_arcsec': (0.39, 19.0),
            # 'r_25_arcsec': (0.7, None),
            # 'r_75_arcsec': (1.98, None),
            # 'r_90_arcsec': (2.3, 150.0),
            # 'r_100_arcsec': (2.0, None),
            # 'r_fwhm_arcsec': (0.4, 10.6),
            # 'mu_median': (0.3, 29.0),
            # 'mu_mean': (0.4, 70.0),
            # 'mu_max': (1.7, 5700.0),
            # 'total_flux': (55, None),
            # 'mag': (13.8, 25.7),
        }
    elif band == 'ps-i':
        conditions = {
            'mu': (mu_lim, None),  # (min, max), None means no limit
            're_arcsec': (re_lim, None),
            'axis_ratio': (0.17, None),
            'r_10_arcsec': (0.39, 19.0),
            'r_25_arcsec': (0.7, None),
            'r_75_arcsec': (1.98, None),
            'r_90_arcsec': (2.3, 150.0),
            'r_100_arcsec': (2.0, None),
            'r_fwhm_arcsec': (0.4, 10.6),
            'mu_median': (0.3, 29.0),
            'mu_mean': (0.4, 70.0),
            'mu_max': (1.7, 5700.0),
            'total_flux': (55, None),
            'mag': (13.8, 25.7),
        }
    else:
        logger.warning(f'No custom cuts implemented yet for band {band}. Using r-band cuts.')
        conditions = {
            'mu': (mu_lim, None),  # (min, max), None means no limit
            're_arcsec': (re_lim, None),
            'axis_ratio': (0.17, None),
            'r_10_arcsec': (0.39, 19.0),
            'r_25_arcsec': (0.7, None),
            'r_75_arcsec': (1.98, None),
            'r_90_arcsec': (2.3, 150.0),
            'r_100_arcsec': (2.0, None),
            'r_fwhm_arcsec': (0.4, 10.6),
            'mu_median': (0.3, 29.0),
            'mu_mean': (0.4, 70.0),
            'mu_max': (1.7, 5700.0),
            'total_flux': (55, None),
            'mag': (13.8, 25.7),
        }

    for column, (min_val, max_val) in conditions.items():
        if min_val is not None:
            params_field = params_field[params_field[column] > min_val]
        if max_val is not None:
            params_field = params_field[params_field[column] < max_val]

    # Remove streaks
    params_field = params_field[
        (params_field['axis_ratio'] >= 0.17) | (params_field['n_pix'] <= 1000)
    ]
    # # mag vs mu filter
    # params_field = params_field[params_field['mu'] > (0.6060 * params_field['mag'] + 11.6293)]
    # # mag vs mu_max filter
    # params_field = params_field[
    #     params_field['mu_max'] < (1.2e10 * np.exp(0.84 * params_field['mag']))
    # ]
    # # r_90 vs r_10
    # params_field = params_field[
    #     params_field['r_90_arcsec'] < (12.0000 * params_field['r_10_arcsec'] + 20.0000)
    # ]
    # # r_90 vs re
    # params_field = params_field[
    #     params_field['r_90_arcsec'] < (3.0 * params_field['re_arcsec'] + 9.0000)
    # ]
    # # mu_median vs re
    # params_field = params_field[
    #     params_field['mu_median'] < (4.0 * params_field['re_arcsec'] + 4.0000)
    # ]
    # # mu_median vs r_10
    # params_field = params_field[
    #     params_field['mu_median'] < (4.0 * params_field['r_10_arcsec'] + 12.0000)
    # ]
    # # mu_median vs r_90
    # params_field = params_field[
    #     params_field['mu_median'] < (0.3 * params_field['r_90_arcsec'] + 15.0000)
    # ]
    # # mu_max vs r_90
    # params_field = params_field[
    #     params_field['mu_max'] < (120.0 * params_field['r_90_arcsec'] + 650.0000)
    # ]

    # params_field = params_field.loc[
    #     (params_field['mu'] > mu_min)
    #     & (params_field['re_arcsec'] > reff_min)
    #     & (params_field['axis_ratio'] > 0.1)
    # ].reset_index(drop=True)
    # # Remove streaks
    # params_field = params_field[
    #     (params_field['axis_ratio'] >= 0.17) | (params_field['n_pix'] <= 1000)
    # ]

    # Remove previous index and reset
    params_field = params_field.reset_index(drop=True)

    return params_field, mto_all


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
                'axis_ratio': (0.17, None),
                'r_10_arcsec': (0.39, 19.0),
                'r_90_arcsec': (2.3, 150.0),
                'r_fwhm_arcsec': (0.4, 10.6),
                'mu_median': (0.3, 29.0),
                'mu_mean': (0.4, 70.0),
                'mu_max': (1.1, 5700.0),
                'total_flux': (55, None),
                'mag': (13.8, 25.7),
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
                'axis_ratio': (0.17, None),
                'r_10_arcsec': (0.39, 8.0),
                'r_90_arcsec': (2.5, 40.0),
                'r_fwhm_arcsec': (0.4, 7.5),
                'mu_median': (0.1, 12.0),
                'mu_mean': (0.25, 17.3),
                'mu_max': (1.7, 2600.0),
                'total_flux': (39, None),
                'mag': (15.3, 23.0),
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
                'axis_ratio': (0.17, None),
                'r_10_arcsec': (0.39, 16.2),
                'r_90_arcsec': (2.51, 85.0),
                'r_fwhm_arcsec': (0.4, 14.2),
                'mu_median': (0.35, 34.0),
                'mu_mean': (0.4, 103.0),
                'mu_max': (2, 8300.0),
                'total_flux': (68, None),
                'mag': (15.8, 25.3),
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
        print(f'Conditions not implemented for band {band}.')
        return None, None

    conditions = band_conditions[band]

    # Apply basic conditions
    for column, (min_val, max_val) in conditions['basic'].items():
        if min_val is not None:
            params_field = params_field[params_field[column] > min_val]
        if max_val is not None:
            params_field = params_field[params_field[column] < max_val]

    # Apply complex conditions
    for condition in conditions['complex']:
        params_field = params_field[condition]

    # Remove streaks
    params_field = params_field[
        (params_field['axis_ratio'] >= 0.17) | (params_field['n_pix'] <= 1000)
    ]

    # Reset index
    params_field = params_field.reset_index(drop=True)

    return params_field, mto_all


def source_detection(
    image,
    header,
    file_path,
    star_df,
    thresh=3.0,
    minarea=5,
    deblend_nthresh=32,
    deblend_cont=0.005,
    save_segmap=False,
):
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
    logger.debug('starting sep')
    # avoid changing the original image
    image_copy = image.copy()
    # set zeros to nan so sep can ignore them
    image_copy[image_copy == 0] = np.nan

    sep_start = time.time()
    bkg = sep.Background(image_copy, maskthresh=thresh, bw=100)
    # non-zero mask
    mask = image != 0
    # only subtract background where the image is non-zero
    data_sub = np.where(mask, image_copy - bkg.back(), image_copy)
    # detect objects
    objects, segmap = sep.extract(
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
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)

        filename_seg = f'{name}_sep_seg{extension}'
        out_path_seg = os.path.join(directory, filename_seg)
        seg_hdu = fits.PrimaryHDU(data=segmap.astype(np.float32), header=header)
        seg_hdu.writeto(out_path_seg, overwrite=True)

    # Convert the structured numpy array to a DataFrame and ensure it's a copy
    objects_df = pd.DataFrame(objects).copy()
    # add world coordinates to df
    objects_df['ra'], objects_df['dec'] = WCS(header).all_pix2world(
        objects_df['x'], objects_df['y'], 0
    )
    # match detections to gaia stars
    det_matching_idx, _, _, _ = match_stars(objects_df, star_df, max_sep=1.5)
    objects_df['star'] = 0
    objects_df.loc[det_matching_idx, 'star'] = 1
    objects_df.insert(0, 'ID', objects_df.index + 1)

    # set nan values back to 0
    data_sub[np.isnan(data_sub)] = 0.0

    return objects_df, data_sub, bkg, segmap


def detect_anomaly(
    image,
    header,
    file_path,
    zero_threshold=0.0025,
    min_size=50,
    replace_anomaly=False,
    dilate_mask=False,
    dilation_iters=1,
    save_to_file=False,
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

    if save_to_file:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_ano_mask{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=image.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return image, out_path
