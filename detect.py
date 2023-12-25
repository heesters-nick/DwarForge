import logging
import os
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import sep
from astropy.wcs import WCS


def run_mto(
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
    logging.info(f'Running MTO on file: {fits_name}')
    param_path = os.path.join(file_dir, tile_name + '_det_params.csv')
    exec_str = f'python {mto_path} {file_path} -par_out {param_path}'
    exec_str_seg = f'python {mto_path} {file_path} -out {out_path} -par_out {param_path}'
    if with_segmap:
        exec_str = exec_str_seg
    try:
        result_mto = subprocess.run(exec_str, shell=True, stderr=subprocess.PIPE, text=True)
        result_mto.check_returncode()
        logging.info(
            f'Successfully ran MTO on tile {tuple(tile_nums)} for band {band}. \n Finished in  {np.round((time.time() - mto_start)/60, 3)} minutes.'
        )

    except subprocess.CalledProcessError as e:
        logging.error(f'Tile {tuple(tile_nums)} failed to download in {band}.')
        logging.exception(f'Subprocess error details: {e}')

        logging.info('Trying to run MTO with Source Extractor background estimation..')
        try:
            mto_start_se = time.time()
            exec_str_se = f'python {mto_path} {file_path} -out {out_path} -par_out {param_path} -bg_variance {bkg.background_rms_median**2} -bg_mean {np.mean(bkg.background())}'
            result_mto = subprocess.run(exec_str_se, shell=True, stderr=subprocess.PIPE, text=True)
            result_mto.check_returncode()
            logging.info(
                f'Successfully ran MTO using Source Extractor background estimation on tile {tuple(tile_nums)} for band {band}. \n Finished in  {np.round((time.time() - mto_start_se)/60, 3)} minutes.'
            )

        except subprocess.CalledProcessError as e:
            logging.error(
                f'Failed running MTO on Tile {tuple(tile_nums)} in {band} with Source Extractor background estimation.'
            )
            logging.exception(f'Subprocess error details: {e}')
            return None, None

    except Exception as e:
        logging.error(f'Tile {tuple(tile_nums)} in {band}: an unexpected error occurred: {e}')
        return None, None

    logging.info('Loading detections.., selecting dwarfs..')
    sel_start = time.time()
    if os.path.exists(param_path):
        pixel_scale = abs(header['CD1_1'] * 3600)
        params_field = pd.read_csv(param_path)
        params_field['ra'], params_field['dec'] = WCS(header).all_pix2world(
            params_field['X'], params_field['Y'], 0
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
        logging.info(
            f'Dwarfs selected: {np.count_nonzero(params_field["label"] == 1)}. Took {np.round(time.time() - sel_start, 3)} seconds.'
        )

    else:
        params_field = None
        logging.error('No parameter file found. Check what went wrong.')

    return params_field


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
    logging.info('starting sep')
    image_c = image.byteswap().newbyteorder()
    bkg = sep.Background(image_c, maskthresh=thresh, bw=128)
    data_sub = image_c - bkg
    objects, segmap = sep.extract(
        data_sub,
        thresh=thresh,
        err=bkg.globalrms,
        minarea=minarea,
        segmentation_map=True,
        deblend_nthresh=32,
        deblend_cont=0.005,
    )
    logging.info('finished sep')
    return objects, data_sub, bkg, segmap
