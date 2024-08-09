import logging
import os
import time

import cv2
import numpy as np
import pandas as pd
from astride import Streak
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from utils import (
    func_PCA,
    get_background,
    piecewise_function_with_break_global,
    piecewise_linear,
    power_law,
    query_gaia_stars,
)

logger = logging.getLogger()


def open_fits(file_path, fits_ext):
    """
    Open fits file and return data and header.

    Args:
        file_path (str): name of the fits file
        fits_ext (int): extension of the fits file

    Returns:
        data (numpy.ndarray): image data
        header (fits header): header of the fits file
    """
    logger.debug(f'Opening fits file {os.path.basename(file_path)}..')
    with fits.open(file_path, memmap=True) as hdul:
        data = hdul[fits_ext].data  # type: ignore
        header = hdul[fits_ext].header  # type: ignore
    logger.debug(f'Fits file {os.path.basename(file_path)} opened.')
    return data, header


def find_stars(file_path, tile_dim, coord, search_radius, wcs):
    """
    Find stars within a given search radius around a specific coordinate.

    Args:
        file_path (str): Path to the fits file.
        tile_dim (tuple): Dimensions of the tile.
        coord (SkyCoord): Coordinate around which to search for stars.
        search_radius (float): Radius within which to search for stars.
        wcs (WCS): World Coordinate System object.

    Returns:
    - star_df (DataFrame): DataFrame containing information about the stars found in and around the field.
    """
    logger.info(f'Querying Gaia for tile {os.path.basename(file_path)}..')
    star_df = query_gaia_stars(coord, search_radius)
    if star_df.empty:
        logger.info(f'No stars found in tile {os.path.basename(file_path)}.')
        return star_df
    c_star = SkyCoord(ra=star_df.ra, dec=star_df.dec, unit='deg', frame='icrs')
    x_star, y_star = wcs.world_to_pixel(c_star)

    mask = (
        (-700 < x_star)
        & (x_star < (tile_dim[1] + 700))
        & (-700 < y_star)
        & (y_star < (tile_dim[0] + 700))
    )

    star_df = star_df[mask].reset_index(drop=True)
    star_df['x'], star_df['y'] = x_star[mask], y_star[mask]
    logger.info(f'Stars found in tile {os.path.basename(file_path)}.')

    return star_df


def star_fit(df, survey=None, with_plot=False, folder=None):
    """
    Fits star size, diffraction spike length, and thickness to its Gaia G-band magnitude.

    Args:
        df (DataFrame): Input DataFrame for stars in and around the field.
        survey (str, optional): Survey name ('UNIONS', 'DECALS', or 'LBT').
        with_plot (bool, optional): Whether to generate and save plots. Defaults to False.
        folder (str, optional): Path to the folder where the output plots will be saved.

    Returns:
        df (DataFrame): Updated DataFrame for stars in and around the field.
    """
    # fit star size
    # use UNIONS parameters as default
    break_point_s = 9.49
    param_s = np.array([7.21366775e05, 6.13780414e-01, 1.26614049e02, 8.26137977e05])
    df['A'] = piecewise_function_with_break_global(np.array(df.Gmag), break_point_s, *param_s)
    df['R_ISO'] = np.sqrt(df.A / np.pi)

    if survey == 'UNIONS':
        param_s = np.array([7.21366775e05, 6.13780414e-01, 1.26614049e02, 8.26137977e05])
        break_point_s = 9.49
        df['A'] = piecewise_function_with_break_global(np.array(df.Gmag), break_point_s, *param_s)
        df['R_ISO'] = np.sqrt(df.A / np.pi)
    if survey == 'DECALS':
        param_s = np.array([1.40400983e-02, 4.94599527e00, -2.24130151e02])
        df['A'] = power_law(np.array(df.Gmag), *param_s)
        df['R_ISO'] = np.sqrt(df.A / np.pi)
    if survey == 'LBT':
        param_s = np.array([7.21366775e05, 6.13780414e-01, 1.26614049e02, 8.26137977e05])
        break_point_s = 9.49
        df['A'] = piecewise_function_with_break_global(np.array(df.Gmag), break_point_s, *param_s)
        df['R_ISO'] = np.sqrt(df.A / np.pi)

    # fit length of diffraction spikes
    param_l = np.array([1.36306644e03, 3.45428885e-01, 3.30158864e00, 1.09029943e03])
    break_point_l = 15.29
    df['diffs_len_fit'] = piecewise_function_with_break_global(df['Gmag'], break_point_l, *param_l)
    df['diffs_len_fit'] = piecewise_function_with_break_global(df['Gmag'], break_point_l, *param_l)

    # fit thickness of diffraction spikes
    param_t = np.array([15.6, -0.5, -0.05, 1.5])
    df['diffs_thick_fit'] = piecewise_linear(df['Gmag'], *param_t)
    df['diffs_thick_fit'] = piecewise_linear(df['Gmag'], *param_t)

    if with_plot and folder is not None:
        # star size
        plt.figure(figsize=(10, 10))
        x = np.arange(6, 21.4, 0.1)
        plt.scatter(
            np.array(df.Gmag),
            np.array(df.npix),
            marker='o',
            s=10,
            edgecolor='blue',
            facecolor='None',
            label='Stars in the field.',
        )
        plt.ylim([0, 50000])
        plt.plot(
            x,
            piecewise_function_with_break_global(x, break_point_s, *param_s),
            label=f'Curve fit: piecewise function with a break at {break_point_s}',
        )
        plt.xlabel('Gmag (Gaia)', fontsize=15)
        plt.ylabel('npix', fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig(folder + 'star_size_global_fit.png', dpi=300)

        # length of diffraction spikes
        plt.figure(figsize=(10, 10))
        x = np.arange(6, 21.4, 0.1)
        plt.scatter(
            np.array(df.Gmag),
            np.array(df.diffs_len),
            marker='o',
            s=10,
            edgecolor='blue',
            facecolor='None',
            label='Stars in the field.',
        )
        plt.ylim([0, 2000])
        plt.plot(
            x,
            piecewise_function_with_break_global(x, break_point_l, *param_l),
            label='Curve fit: piecewise function with a break at {}.'.format(break_point_l),
        )
        plt.xlabel('Gmag (Gaia)', fontsize=15)
        plt.ylabel('diffs_len', fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig(folder + 'star_len_global_fit.png', dpi=300)

        # thickness of diffraction spikes
        plt.figure(figsize=(10, 10))
        x = np.arange(6, 21.4, 0.1)
        plt.scatter(
            np.array(df.Gmag),
            np.array(df.avg_thickness),
            marker='o',
            s=10,
            edgecolor='blue',
            facecolor='None',
            label='Stars in the field.',
        )
        plt.ylim([0, 8])
        plt.plot(
            x,
            piecewise_linear(x, *param_t),
            label='Curve fit: piecewise linear function.',
        )
        plt.xlabel('Gmag (Gaia)', fontsize=15)
        plt.ylabel('diffs_thick', fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig(folder + 'star_thick_global_fit.png', dpi=300)

    return df


def star_mask(file_path, data, header, folder=None, survey=None):
    """
    Mask stars in the field.

    Args:
        file_path (str): path to the fits file
        data (numpy.ndarray): image data
        header (fits header): header of the fits file
        survey (str, optional): survey name, defaults to UNIONS
        folder (str, optional): folder where potential plots will be saved

    Returns:
        masked image data (array)
    """
    logger.info(f'Removing background for tile {os.path.basename(file_path)}..')
    bkg_start = time.time()
    data[data == 0] = np.nan
    # data_sub, bkg_orig = remove_background(data)
    data_sub, bkg_rms = get_background(data)
    logger.info(
        f'Background for tile {os.path.basename(file_path)} removed! Took {np.round(time.time() - bkg_start)} seconds.'
    )

    wcs = WCS(header)
    tile_width, tile_height = header['NAXIS1'], header['NAXIS2']
    pix_scale = abs(header['CD1_1'] * 3600)
    dim_x_arcsec = tile_width * pix_scale
    dim_y_arcsec = tile_height * pix_scale
    tile_coord = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit='deg', frame='icrs')
    # extend search radius to include stars that are just outside the field
    search_radius = np.sqrt(dim_x_arcsec**2 + dim_y_arcsec**2) / 2 + 200

    # query gaia stars in the field
    star_df = find_stars(file_path, [tile_height, tile_width], tile_coord, search_radius, wcs)
    star_df = star_fit(star_df)

    logger.info(f'Masking stars for tile {os.path.basename(file_path)}..')
    x_arr, y_arr, r_arr, len_arr, thick_arr = (
        star_df.x.values,
        star_df.y.values,
        star_df.R_ISO.values,
        star_df.diffs_len_fit.values,
        star_df.diffs_thick_fit.values,
    )

    # create an empty image the size of the field
    stellar_mask = np.zeros_like(data)
    x_arr_int = [int(x) for x in x_arr]
    y_arr_int = [int(x) for x in y_arr]
    centers = np.column_stack((x_arr_int, y_arr_int))

    # use UNIONS parameters as default
    r_arr_int = [int(x * 1.5) for x in r_arr]
    len_arr_int = [int(x * 3.5) for x in len_arr]
    thick_arr_int = [int(x * 5) for x in thick_arr]

    if survey == 'DECALS':
        r_arr_int = [int(x * 1.7) for x in r_arr]
        len_arr_int = [int(x * 1.2) for x in len_arr]
        thick_arr_int = [int(x * 2) for x in thick_arr]
    if survey == 'UNIONS':
        r_arr_int = [int(x * 1.5) for x in r_arr]
        len_arr_int = [int(x * 3.5) for x in len_arr]
        thick_arr_int = [int(x * 5) for x in thick_arr]
    if survey == 'LBT':
        r_arr_int = [int(x * 1.5) for x in r_arr]
        len_arr_int = [int(x * 3.5) for x in len_arr]
        thick_arr_int = [int(x * 5) for x in thick_arr]

    for center, r, len, thick in zip(centers, r_arr_int, len_arr_int, thick_arr_int):
        if survey == 'DECALS':
            cv2.drawMarker(
                stellar_mask,
                center,
                color=(255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=len,
                thickness=thick,
            )
        else:
            cv2.drawMarker(
                stellar_mask,
                center,
                color=(255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=len,
                thickness=thick,
            )

        cv2.circle(stellar_mask, center, r, (255, 255, 255), -1)
    stellar_mask[stellar_mask > 0] = 1

    # save mask to file
    # hdu_mask = fits.PrimaryHDU(star_mask, wcs.to_header())
    # hdu_mask.writeto(folder + fits_name + '_star_mask_cv2.fits', overwrite=True)

    # add high negative values to mask and replace with gaussian noise
    neg_mask = data < -9.0
    nan_mask = np.isnan(data)
    star_neg_mask = (stellar_mask == 1) | (neg_mask == 1) | (nan_mask == 1)
    if survey == 'DECALS':
        saturated_mask = data >= 9.9
        star_neg_mask = (
            (stellar_mask == 1) | (neg_mask == 1) | (nan_mask == 1) | (saturated_mask == 1)
        )
    if survey == 'LBT':
        saturated_mask = data >= 12
        star_neg_mask = (
            (stellar_mask == 1) | (neg_mask == 1) | (nan_mask == 1) | (saturated_mask == 1)
        )
    gaussian_noise = np.random.normal(0, bkg_rms, np.count_nonzero(star_neg_mask))
    data_masked = np.copy(data_sub)
    data_masked[star_neg_mask != 0] = gaussian_noise
    logger.info(f'Stars masked for tile {os.path.basename(file_path)}')

    with open('bkg_rms_median.txt', 'w') as f:
        f.write(str(bkg_rms))

    return data_masked, bkg_rms


def streak_mask(data_masked, file_path, bkg_rms, table_dir, header, psf_multiplier=2.0):
    """
    Masks streaks in the image.

    Args:
            data_masked (numpy.ndarray): star masked image data
            fits_name (str): fits file name
            bkg_rms (float): background global rms
            data_dir (str): data directory
            table_dir (str): table directory
            header (fits header): fits header

    Returns:
            numpy.ndarray: streak masked image data
    """
    logger.info(f'Masking streaks in tile {os.path.basename(file_path)}.')
    seeing_arcsec = header['IQFINAL']
    pix_scale = abs(header['CD1_1'] * 3600)
    seeing_pix = seeing_arcsec / pix_scale
    kernel = Gaussian2DKernel(
        x_stddev=psf_multiplier * seeing_pix * gaussian_fwhm_to_sigma, mode='oversample', factor=1
    )
    data_conv_streak = convolve(data_masked, kernel)

    # zero padding on the image, otherwise streaks starting at the border of an image will be missed
    data_conv_padded = np.pad(data_conv_streak, pad_width=1)
    # save as a new fits file with old header
    # save as float32 to avoid overflow

    hdu_new = fits.PrimaryHDU(data_conv_padded, WCS(header).to_header())
    out_file = os.path.join(os.path.splitext(file_path)[0] + '_star_masked_smooth_padded.fits')
    hdu_new.writeto(out_file, overwrite=True)

    streak = Streak(
        out_file,
        contour_threshold=2,
        area_cut=1000,
    )
    streak.detect()
    # delete the padded file
    os.remove(out_file)
    if streak.streaks is None:
        logger.info(f'No streaks found in tile {os.path.basename(file_path)}.')
        return data_masked
    streaks = pd.DataFrame.from_dict(streak.streaks)  # type: ignore
    # streak.write_outputs()
    # streak.plot_figures()
    # run PCA on the points tracing the streak outlines
    for i in range(len(streaks)):
        ar, thick, pca = func_PCA((streaks.loc[i].x - 1), (streaks.loc[i].y - 1))
        slope = pca.components_[0, 1] / pca.components_[0, 0]
        intercept = pca.mean_[1] - slope * pca.mean_[0]
        streaks.at[i, 'axis_ratio'] = ar
        streaks.at[i, 'thickness'] = thick
        streaks.at[i, 'slope_pca'] = slope
        streaks.at[i, 'intercept_pca'] = intercept

    # loop over detections and check if the center coordinates of the detections are in the beam of at least one
    # other detection. If they are, they are likely to belong to the same streak
    for i in range(len(streaks)):
        x_center, y_center = streaks.x_center[i] - 1, streaks.y_center[i] - 1
        in_beam = 0
        for j in streaks.index[streaks.axis_ratio < 0.15]:
            if i == j:
                continue
            slope, inter, thick = (
                streaks.slope_pca[j],
                streaks.intercept_pca[j],
                streaks.thickness[j],
            )
            distance = abs(slope * (x_center - 1) - (y_center - 1) + inter) / np.sqrt(slope**2 + 1)
            if distance < thick:
                in_beam += 1
        if in_beam > 0:
            streaks.at[i, 'real_streak'] = 1
        elif streaks.axis_ratio[i] < 0.15:
            streaks.at[i, 'real_streak'] = 1
        else:
            streaks.at[i, 'real_streak'] = 0

    if len(streaks.index[streaks['real_streak'] == 1] > 0):
        # mask streaks
        masks = []
        for i in streaks.index[streaks['real_streak'] == 1]:
            binary_image = np.zeros(data_masked.shape, dtype=np.uint8)
            polygon = [(x, y) for x, y in zip((streaks.loc[i].x - 1), (streaks.loc[i].y - 1))]
            mask_image = Image.fromarray(binary_image)
            draw = ImageDraw.Draw(mask_image)
            draw.polygon(polygon, fill=1)
            mask = np.array(mask_image, dtype=np.uint8)
            masks.append(mask)
        # remove x and y from streaks dataframe
        streaks = streaks.drop(['x', 'y'], axis=1)
        streaks.to_csv(table_dir + 'detected_streaks.csv')

        # combine individual masks into a single one
        combined_mask = np.logical_or.reduce(masks).astype(np.uint8)
        # define amount by which to dilate the final mask
        border_size = 12
        # dilate the final mask
        expanded_mask = binary_dilation(combined_mask, iterations=border_size).astype(np.uint8)

        # replace masked region with gaussian noise
        gaussian_noise = np.random.normal(0, bkg_rms, np.count_nonzero(expanded_mask))
        data_masked[expanded_mask != 0] = gaussian_noise
        logger.info(f'Streaks masked in tile {os.path.basename(file_path)}.')
    else:
        logger.info(f'No streaks found in tile {os.path.basename(file_path)}.')

    return data_masked


def smooth_image(data_masked, file_path, header, psf_multiplier=2.0):
    """
    Smooth the image using a gaussian kernel with a sigma corresponding to a fraction of the seeing FWHM.

    Args:
        data_masked (numpy.ndarray): streak masked image data
        file_path (str): path to the fits file
        header (fits header): fits header
        psf_multiplier (float, optional): Multiplier for the PSF. Defaults to 2.

    Returns:
        numpy.ndarray: smoothed image data
    """
    logger.info(f'Smoothing tile {os.path.basename(file_path)}..')
    smoothing_start = time.time()
    wcs = WCS(header)
    seeing_arcsec = header['IQFINAL']
    pix_scale = abs(header['CD1_1'] * 3600)
    seeing_pix = seeing_arcsec / pix_scale
    kernel_start = time.time()
    kernel = Gaussian2DKernel(
        x_stddev=psf_multiplier * seeing_pix * gaussian_fwhm_to_sigma, mode='oversample', factor=1
    )
    logger.info(
        f'Kernel for tile {os.path.basename(file_path)} created. Took {np.round(time.time() - kernel_start)} seconds.'
    )
    data_conv = convolve(data_masked, kernel)
    hdu_new = fits.PrimaryHDU(data_conv, wcs.to_header())
    out_file = os.path.splitext(file_path)[0] + '_star_masked_smooth.fits'
    hdu_new.writeto(out_file, overwrite=True)
    logger.info(
        f'Tile {os.path.basename(file_path)} smoothed. Took {np.round(time.time() - smoothing_start)} seconds.'
    )
    pass


def save_processed(data_binned, header, file_path, bin_size, preprocess_type, update_header=True):
    # out path
    directory, filename = os.path.split(file_path)
    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)
    # Create the new filename
    new_filename = f'{name}_{preprocess_type}{extension}'
    # Combine the directory and new filename
    out_path = os.path.join(directory, new_filename)
    # wcs
    wcs = WCS(header)

    if update_header:
        wcs.wcs.cd *= bin_size
        wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / bin_size + 0.5

        # Update the image dimensions in the header
        new_header = header.copy()
        new_header['NAXIS1'] = data_binned.shape[1]
        new_header['NAXIS2'] = data_binned.shape[0]

        # Update the CD matrix in the header
        new_header['CD1_1'] = wcs.wcs.cd[0, 0]
        new_header['CD1_2'] = wcs.wcs.cd[0, 1]
        new_header['CD2_1'] = wcs.wcs.cd[1, 0]
        new_header['CD2_2'] = wcs.wcs.cd[1, 1]

        # Update CRPIX values
        new_header['CRPIX1'] = wcs.wcs.crpix[0]
        new_header['CRPIX2'] = wcs.wcs.crpix[1]

        # Update the WCS information in the header
        new_header.update(wcs.to_header())
    else:
        new_header = header.copy()

    # Create a new HDU with the rebinned data and updated header
    new_hdu = fits.PrimaryHDU(data=data_binned, header=new_header)

    # save new fits file
    new_hdu.writeto(out_path, overwrite=True)

    return out_path, new_header


def adjust_flux_with_zp(flux, current_zp, standard_zp):
    adjusted_flux = flux * 10 ** (-0.4 * (current_zp - standard_zp))
    return adjusted_flux


def bin_image_cv2(image, bin_size):
    return cv2.resize(
        image,
        (image.shape[1] // bin_size, image.shape[0] // bin_size),
        interpolation=cv2.INTER_AREA,
    )


def prep_tile(file_path, fits_ext, zp, bin_size=4):
    """
    Preprocess a tile for detection with MTO. This includes masking stars and streaks, and smoothing the image.

    Args:
        file_path (str): path to the fits file
        fits_ext (int): fits extension, 0 or 1
        zp (float): photometric zeropoint for this tile
        bin_size (int, optional): bin factor, defaults to 4
    Returns:
        str, str: path to preprocessed file, header of preprocessed file
    """
    # read in data and header
    data, header = open_fits(file_path, fits_ext)
    # adjust zeropoint
    if zp != 30.0:
        data = adjust_flux_with_zp(data, current_zp=zp, standard_zp=30.0)
    # bin image to increase signal-to-noise + aid LSB detection
    binned_image = bin_image_cv2(data, bin_size)
    # save preprocessed image to fits
    file_path_binned, header_binned = save_processed(
        binned_image, header, file_path, bin_size, preprocess_type='rebin'
    )
    return file_path_binned, header_binned
