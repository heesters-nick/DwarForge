import os
import time

import cv2
import numpy as np
import pandas as pd
from astride import Streak
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, label

from detect import detect_anomaly
from logging_setup import get_logger
from utils import (
    delete_file,
    func_PCA,
    generate_positive_trunc_normal,
    open_fits,
    piecewise_function_with_break_global,
    piecewise_linear,
    power_law,
    query_gaia_stars,
    tile_str,
)

logger = get_logger()


def remove_background(
    image, header, file_path, bw=200, bh=200, estimator=MedianBackground(), save_file=False
):
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = estimator
    bkg = Background2D(
        image, (bw, bh), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator
    )
    data_sub = image - bkg.background

    if save_file:
        directory, filename = os.path.split(file_path)
        # Split the filename into name and extension
        name, extension = os.path.splitext(filename)
        # Create the new filename
        new_filename = f'{name}_bkg_sub{extension}'
        # Combine the directory and new filename
        out_path = os.path.join(directory, new_filename)
        # Create a new HDU with the rebinned data and updated header
        new_hdu = fits.PrimaryHDU(data=data_sub.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)
    else:
        out_path = None

    return data_sub, bkg, out_path


def replace_with_local_background(
    data, star_df, bright_star_mask, r_scale, l_scale, bkg, bkg_factor, embedded_threshold=0.9
):
    h, w = data.shape
    result = data.copy()
    total_stars = len(star_df)
    skipped_stars = 0

    bright_star_mask = bright_star_mask.astype(np.uint8)
    replaced_stars_mask = np.zeros_like(bright_star_mask, dtype=np.uint8)

    visualization_mask = np.zeros((h, w, 3), dtype=np.uint8)

    def create_annular_segments(
        shape, center, inner_r, outer_r, spike_len, spike_thick, n_segments=4
    ):
        y, x = center
        mask = np.zeros(shape, dtype=np.uint8)

        yy, xx = np.ogrid[: shape[0], : shape[1]]
        distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

        # Use arctan2 for full 360-degree angle range
        angles = np.arctan2(yy - y, xx - x)
        # Shift angles to be in [0, 2π) range
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        # set annulus extend
        annulus_extend = max(int(np.round(outer_r / 2)) + 2, spike_len + 1)
        # define annulus mask
        annulus = (distances >= inner_r + 1) & (distances <= annulus_extend)

        segment_size = 2 * np.pi / n_segments
        for i in range(n_segments):
            start_angle = i * segment_size
            end_angle = (i + 1) * segment_size

            # Handle the case where the segment crosses the 0/2π boundary
            if start_angle < end_angle:
                segment = (angles >= start_angle) & (angles < end_angle)
            else:
                segment = (angles >= start_angle) | (angles < end_angle)

            mask[(annulus & segment)] = 255

        # Remove diffraction spike areas
        spike_mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawMarker(
            spike_mask,
            (x, y),
            color=255,
            markerType=cv2.MARKER_CROSS,
            markerSize=spike_len,
            thickness=spike_thick,
        )
        mask[spike_mask > 0] = 0

        return mask

    for _, star in star_df.iterrows():
        y, x = np.round(star.y).astype(np.int32), np.round(star.x).astype(np.int32)
        r = max(np.round(star.R_ISO * r_scale).astype(np.int32), 1)
        spike_len = int(np.round(star.diffs_len_fit * l_scale))
        spike_thick = max(1, int(np.round(star.diffs_thick_fit)))

        outer_r = max(r + 7, spike_len + 2)  # Ensure we capture enough background

        y_min, y_max = max(0, y - outer_r), min(h, y + outer_r + 1)
        x_min, x_max = max(0, x - outer_r), min(w, x + outer_r + 1)
        local_region = result[y_min:y_max, x_min:x_max]
        local_y, local_x = y - y_min, x - x_min

        # Create star region mask including diffraction spikes
        star_region = np.zeros_like(local_region, dtype=np.uint8)
        cv2.circle(star_region, (local_x, local_y), r, 255, -1, lineType=cv2.LINE_AA)
        if spike_len > 0:
            cv2.drawMarker(
                star_region,
                (local_x, local_y),
                color=255,
                markerType=cv2.MARKER_CROSS,
                markerSize=spike_len,
                thickness=spike_thick,
            )

        # Check if star is embedded
        check_annulus = cv2.circle(
            np.zeros_like(local_region, dtype=np.uint8), (local_x, local_y), r + 2, 255, 1
        )
        check_pixels = local_region[check_annulus > 0]
        median_background = bkg.background_median
        rms_background = bkg.background_rms_median
        is_embedded = (
            np.sum(check_pixels > max(1, bkg_factor * rms_background + median_background))
            / len(check_pixels)
            >= embedded_threshold
        )

        if is_embedded:
            r_reduced = max(int(np.round(r * 0.5)), 1)
            l_reduced = spike_len
            star_region_reduced = np.zeros_like(local_region, dtype=np.uint8)
            cv2.circle(
                star_region_reduced, (local_x, local_y), r_reduced, 255, -1, lineType=cv2.LINE_AA
            )
            if spike_len > 0:
                l_reduced = max(int(np.round(spike_len * 0.4)), 1)
                cv2.drawMarker(
                    star_region_reduced,
                    (local_x, local_y),
                    color=255,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=l_reduced,
                    thickness=spike_thick,
                )

            # Create a narrow annulus around the reduced star region for background estimation
            inner_r_bg = r_reduced
            outer_r_bg = l_reduced  # Up to reduced length of diffraction spikes
            bg_segments = create_annular_segments(
                local_region.shape,
                (local_y, local_x),
                inner_r_bg,
                outer_r_bg,
                l_reduced,
                spike_thick,
            )
        else:
            star_region_reduced = star_region
            # Use the original background estimation for non-embedded stars
            bg_segments = create_annular_segments(
                local_region.shape, (local_y, local_x), r, outer_r, spike_len, spike_thick
            )

        # Exclude all stars (bright and dim) from the background segments
        local_bright_star_mask = bright_star_mask[y_min:y_max, x_min:x_max]
        bg_segments[local_bright_star_mask > 0] = 0

        visualization_mask[y_min:y_max, x_min:x_max, 0] |= (
            star_region_reduced  # Red for star region
        )
        visualization_mask[y_min:y_max, x_min:x_max, 1] |= (
            bg_segments  # Green for background segments
        )

        if np.any((star_region_reduced > 0) & (local_bright_star_mask > 0)):
            skipped_stars += 1
            continue

        bg_data = local_region[bg_segments > 0]

        if len(bg_data) == 0 or np.count_nonzero(np.isnan(bg_data)) / len(bg_data) > 0.9:
            skipped_stars += 1
            continue

        mean, median, std = sigma_clipped_stats(bg_data, sigma=3.0)

        num_pixels = np.sum(star_region_reduced > 0)
        if is_embedded:
            # For embedded stars, use a more conservative approach
            random_values = generate_positive_trunc_normal(
                bg_data, median, min(std, 3.0), num_pixels
            )
        else:
            random_values = np.random.normal(median, min(std, 3.0), num_pixels)

        local_result = local_region.copy()
        local_result[star_region_reduced > 0] = random_values

        result[y_min:y_max, x_min:x_max] = local_result

        replaced_stars_mask[y_min:y_max, x_min:x_max] |= star_region_reduced

    logger.debug(f'Skipped {skipped_stars}/{total_stars} stars.')
    return result, replaced_stars_mask.astype(bool), visualization_mask


# def replace_with_local_background_old(
#     data, star_df, bright_star_mask, r_scale, l_scale, bkg, bkg_factor, embedded_threshold=0.9
# ):
#     h, w = data.shape
#     result = data.copy()
#     total_stars = len(star_df)
#     skipped_stars = 0

#     bright_star_mask = bright_star_mask.astype(np.uint8)
#     replaced_stars_mask = np.zeros_like(bright_star_mask, dtype=np.uint8)

#     visualization_mask = np.zeros((h, w, 3), dtype=np.uint8)

#     def create_annular_segments(shape, center, inner_r, outer_r, l, t, n_segments=4):
#         y, x = center
#         mask = np.zeros(shape, dtype=np.uint8)

#         yy, xx = np.ogrid[: shape[0], : shape[1]]
#         distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

#         # Use arctan2 for full 360-degree angle range
#         angles = np.arctan2(yy - y, xx - x)
#         # Shift angles to be in [0, 2π) range
#         angles = (angles + 2 * np.pi) % (2 * np.pi)
#         # set annulus extend
#         annulus_extend = max(int(np.round(outer_r / 2)) + 2, l + 1)
#         # define annulus mask
#         annulus = (distances >= inner_r + 1) & (distances <= int(np.round(outer_r / 2)) + 2)

#         segment_size = 2 * np.pi / n_segments
#         for i in range(n_segments):
#             start_angle = i * segment_size
#             end_angle = (i + 1) * segment_size

#             # Handle the case where the segment crosses the 0/2π boundary
#             if start_angle < end_angle:
#                 segment = (angles >= start_angle) & (angles < end_angle)
#             else:
#                 segment = (angles >= start_angle) | (angles < end_angle)

#             mask[(annulus & segment)] = 255

#         # Remove diffraction spike areas
#         spike_mask = np.zeros(shape, dtype=np.uint8)
#         cv2.drawMarker(
#             spike_mask, (x, y), color=255, markerType=cv2.MARKER_CROSS, markerSize=l, thickness=t
#         )
#         mask[spike_mask > 0] = 0

#         return mask

#     for _, star in star_df.iterrows():
#         y, x = np.round(star.y).astype(np.int32), np.round(star.x).astype(np.int32)
#         r = max(np.round(star.R_ISO * r_scale).astype(np.int32), 1)
#         l = int(np.round(star.diffs_len_fit * l_scale))
#         t = max(1, int(np.round(star.diffs_thick_fit)))

#         outer_r = max(r + 7, l + 2)  # Ensure we capture enough background

#         y_min, y_max = max(0, y - outer_r), min(h, y + outer_r + 1)
#         x_min, x_max = max(0, x - outer_r), min(w, x + outer_r + 1)
#         local_region = result[y_min:y_max, x_min:x_max]
#         local_y, local_x = y - y_min, x - x_min

#         # Create star region mask including diffraction spikes
#         star_region = np.zeros_like(local_region, dtype=np.uint8)
#         cv2.circle(star_region, (local_x, local_y), r, 255, -1, lineType=cv2.LINE_AA)
#         if l > 0:
#             cv2.drawMarker(
#                 star_region,
#                 (local_x, local_y),
#                 color=255,
#                 markerType=cv2.MARKER_CROSS,
#                 markerSize=l,
#                 thickness=t,
#             )

#         # Check if star is embedded
#         check_annulus = cv2.circle(
#             np.zeros_like(local_region, dtype=np.uint8), (local_x, local_y), r + 2, 255, 1
#         )
#         check_pixels = local_region[check_annulus > 0]
#         median_background = bkg.background_median
#         rms_background = bkg.background_rms_median
#         is_embedded = (
#             np.sum(check_pixels > max(1, bkg_factor * rms_background + median_background))
#             / len(check_pixels)
#             >= embedded_threshold
#         )
#         #         is_embedded = np.sum(check_pixels > max(1, bkg_factor*median_background)) / len(check_pixels) >= embedded_threshold

#         if is_embedded:
#             r_reduced = max(int(np.round(r * 0.5)), 1)
#             l_reduced = l
#             star_region_reduced = np.zeros_like(local_region, dtype=np.uint8)
#             cv2.circle(
#                 star_region_reduced, (local_x, local_y), r_reduced, 255, -1, lineType=cv2.LINE_AA
#             )
#             if l > 0:
#                 l_reduced = max(int(np.round(l * 0.4)), 1)
#                 cv2.drawMarker(
#                     star_region_reduced,
#                     (local_x, local_y),
#                     color=255,
#                     markerType=cv2.MARKER_CROSS,
#                     markerSize=l_reduced,
#                     thickness=t,
#                 )

#             # Create a narrow annulus around the reduced star region for background estimation
#             inner_r_bg = r_reduced
#             outer_r_bg = l_reduced  # Up to reduced length of diffraction spikes
#             bg_segments = create_annular_segments(
#                 local_region.shape, (local_y, local_x), inner_r_bg, outer_r_bg, l_reduced, t
#             )
#         else:
#             star_region_reduced = star_region
#             # Use the original background estimation for non-embedded stars
#             bg_segments = create_annular_segments(
#                 local_region.shape, (local_y, local_x), r, outer_r, l, t
#             )

#         # Exclude all stars (bright and dim) from the background segments
#         local_bright_star_mask = bright_star_mask[y_min:y_max, x_min:x_max]
#         bg_segments[local_bright_star_mask > 0] = 0

#         visualization_mask[y_min:y_max, x_min:x_max, 0] |= (
#             star_region_reduced  # Red for star region
#         )
#         visualization_mask[y_min:y_max, x_min:x_max, 1] |= (
#             bg_segments  # Green for background segments
#         )

#         if np.any((star_region_reduced > 0) & (local_bright_star_mask > 0)):
#             skipped_stars += 1
#             continue

#         bg_data = local_region[bg_segments > 0]

#         if len(bg_data) == 0 or np.count_nonzero(np.isnan(bg_data)) / len(bg_data) > 0.9:
#             skipped_stars += 1
#             continue

#         mean, median, std = sigma_clipped_stats(bg_data, sigma=3.0)

#         num_pixels = np.sum(star_region_reduced > 0)
#         if is_embedded:
#             # For embedded stars, use a more conservative approach
#             random_values = generate_positive_trunc_normal(
#                 bg_data, median, min(std, 3.0), num_pixels
#             )
#         else:
#             random_values = np.random.normal(median, min(std, 3.0), num_pixels)

#         local_result = local_region.copy()
#         local_result[star_region_reduced > 0] = random_values

#         result[y_min:y_max, x_min:x_max] = local_result

#         replaced_stars_mask[y_min:y_max, x_min:x_max] |= star_region_reduced

#     print(f'Skipped {skipped_stars}/{total_stars} stars.')
#     return result, replaced_stars_mask.astype(bool), visualization_mask


def find_stars(tile, header):
    """
    Find stars within a given search radius around a specific coordinate.

    Args:
        tile (str): tile numbers
        header (header): fits header

    Returns:
    - star_df (DataFrame): DataFrame containing information about the stars found in and around the field.
    """
    wcs = WCS(header)
    pixel_scale = abs(header['CD1_1'] * 3600)
    tile_width, tile_height = header['NAXIS1'], header['NAXIS2']
    dim_x_arcsec = tile_width * pixel_scale
    dim_y_arcsec = tile_height * pixel_scale
    tile_coord = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit='deg', frame='icrs')
    # extend search radius to include stars that are just outside the field
    search_radius = np.sqrt(dim_x_arcsec**2 + dim_y_arcsec**2) / 2 + 200
    #     print(f'search_radius: {search_radius} arcsec')

    logger.debug(f'Querying Gaia for tile {tile}..')
    star_df = query_gaia_stars(tile_coord, search_radius)
    star_df.dropna(inplace=True)
    if star_df.empty:
        logger.info(f'No stars found in tile {tile}.')
        return star_df
    c_star = SkyCoord(ra=star_df.ra, dec=star_df.dec, unit='deg', frame='icrs')
    x_star, y_star = wcs.world_to_pixel(c_star)

    mask = (0 < x_star) & (x_star < tile_width) & (0 < y_star) & (y_star < tile_height)

    star_df = star_df[mask].reset_index(drop=True)
    star_df['x'], star_df['y'] = x_star[mask], y_star[mask]
    logger.info(f'{len(star_df)} stars found in tile {tile}.')

    return star_df


def star_fit(df_if, survey='UNIONS'):
    df = df_if.copy()
    # fit star size
    if survey == 'UNIONS':
        param_s = np.array([7.21366775e05, 6.13780414e-01, 1.26614049e02, 8.26137977e05])
        break_point_s = 9.49
        df['A'] = piecewise_function_with_break_global(np.array(df.Gmag), *param_s, break_point_s)
        df['R_ISO'] = np.sqrt(df.A / np.pi)
    if survey == 'DECALS':
        param_s = np.array([1.40400983e-02, 4.94599527e00, -2.24130151e02])
        df['A'] = power_law(np.array(df_if.Gmag), *param_s)
        df['R_ISO'] = np.sqrt(df_if.A / np.pi)

    # fit length of diffraction spikes
    param_l = np.array([1.36306644e03, 3.45428885e-01, 3.30158864e00, 1.09029943e03])
    break_point_l = 15.29
    df['diffs_len_fit'] = piecewise_function_with_break_global(df['Gmag'], *param_l, break_point_l)

    # fit thickness of diffraction spikes
    param_t = np.array([15.6, -0.5, -0.05, 1.5])
    df['diffs_thick_fit'] = piecewise_linear(df['Gmag'], *param_t)

    return df


def mask_stars(
    tile_str,
    data,
    header,
    file_path,
    bkg,
    r_scale=0.6,
    l_scale=0.8,
    gmag_lim=13.2,
    save_to_file=False,
    survey='UNIONS',
):
    star_df = find_stars(tile_str, header)
    full_star_df = star_fit(star_df)

    # check if there is a bright star in the field
    gmag_brightest = np.min(star_df['Gmag'].values)

    bkg_factor = 5.0
    if gmag_brightest < 9.0:
        bright_star_flag = True
        if gmag_brightest < 7.0:
            bkg_factor = 30.0
        else:
            bkg_factor = 10.0
    else:
        bright_star_flag = False

    dim_star_df = full_star_df.loc[star_df['Gmag'] > gmag_lim].reset_index(drop=True)
    bright_star_df = full_star_df.loc[star_df['Gmag'] <= gmag_lim].reset_index(drop=True)

    h, w = data.shape
    bright_star_mask = np.zeros((h, w), dtype=np.uint8)

    # Create mask for bright stars (not to be replaced)
    for _, star in bright_star_df.iterrows():
        x, y = round(star.x), round(star.y)
        r = max(round(star.R_ISO * r_scale), 1)
        cv2.circle(bright_star_mask, (x, y), r, 255, -1, lineType=cv2.LINE_AA)
        # Add diffraction spikes for bright stars
        spike_len = round(star.diffs_len_fit * l_scale)
        spike_thick = round(star.diffs_thick_fit * 5.0)

        cv2.drawMarker(
            bright_star_mask,
            (x, y),
            color=255,
            markerType=cv2.MARKER_CROSS,
            markerSize=spike_len,
            thickness=spike_thick,
        )

    data, replaced_star_mask, visualization_mask = replace_with_local_background(
        data, dim_star_df, bright_star_mask, r_scale, l_scale, bkg, bkg_factor
    )

    if save_to_file:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_star_mask{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return data, replaced_star_mask, out_path, bright_star_flag, gmag_brightest, visualization_mask


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
    new_hdu = fits.PrimaryHDU(data=data_binned.astype(np.float32), header=new_header)
    # save new fits file
    new_hdu.writeto(out_path, overwrite=True)

    return data_binned, out_path, new_header


def adjust_flux_with_zp(flux, current_zp, standard_zp):
    adjusted_flux = flux * 10 ** (-0.4 * (current_zp - standard_zp))
    return adjusted_flux


def bin_image_cv2(image, bin_size):
    return cv2.resize(
        image,
        (image.shape[1] // bin_size, image.shape[0] // bin_size),
        interpolation=cv2.INTER_AREA,
    )


def mask_hot_pixels(image, threshold, bkg, max_size=3, sigma=5):
    # Create a mask of pixels above the threshold
    hot_pixels = image > threshold
    neighbor_threshold = bkg.background_median + sigma * bkg.background_rms_median

    # Label connected regions
    labeled, num_features = label(hot_pixels)

    # Get the size of each labeled region
    sizes = np.bincount(labeled.ravel())

    # Create a mask of regions that are small enough
    mask_size = sizes <= max_size
    mask_size[0] = 0  # Remove background

    # Apply the size mask to the labeled image
    potential_hot_pixels = mask_size[labeled]

    # Create a structure for dilation (3x3 square)
    structure = np.ones((3, 3), dtype=bool)

    # Dilate the potential hot pixels
    dilated = binary_dilation(potential_hot_pixels, structure=structure)

    # The outline is the difference between the dilated image and the original
    outline = dilated & ~potential_hot_pixels

    not_hot = np.zeros_like(image, dtype=bool)
    not_hot[outline] = image[outline] <= neighbor_threshold

    # Identify hot pixels
    hot_pixel_mask = np.zeros_like(image, dtype=bool)

    # Dilate the not_hot mask to touch the potential hot pixels
    touching_not_hot = binary_dilation(not_hot, structure=structure)

    # Hot pixels are those that are potential hot pixels and touch a not_hot pixel
    hot_pixel_mask = potential_hot_pixels & touching_not_hot

    # Create a copy of the image and mask the hot pixels
    masked_image = image.copy()
    masked_image[hot_pixel_mask] = np.median(image)  # Replace with median value

    return masked_image, hot_pixel_mask


def prep_tile(tile, file_path, fits_ext, zp, bin_size=4):
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
    prep_start = time.time()
    # read in data and header
    data, header = open_fits(file_path, fits_ext)
    # adjust zeropoint
    if zp != 30.0:
        data = adjust_flux_with_zp(data, current_zp=zp, standard_zp=30.0)
    # bin image to increase signal-to-noise + aid LSB detection
    binned_image = bin_image_cv2(data, bin_size)
    # save binned image to fits
    data_binned, file_path_binned, header_binned = save_processed(
        binned_image, header, file_path, bin_size, preprocess_type='rebin'
    )
    start = time.time()
    # detect data anomalies and set them to zero
    data_ano_mask, file_path_ano_mask = detect_anomaly(
        data_binned, header_binned, file_path_binned, replace_anomaly=True, save_to_file=True
    )
    logger.info(f'{tile_str(tile)}: detected anomalies in {time.time()-start:.2f} seconds.')
    # estimate the background
    start = time.time()
    _, bkg, _ = remove_background(
        data_ano_mask,
        header_binned,
        file_path_ano_mask,
        bw=100,
        bh=100,
        estimator=MedianBackground(),
    )
    logger.info(f'{tile_str(tile)}: estimated background in {time.time()-start:.2f} seconds.')
    # mask hot pixels
    start = time.time()
    data_ano_mask, hot_mask = mask_hot_pixels(
        data_ano_mask, threshold=70, bkg=bkg, max_size=3, sigma=5
    )
    logger.info(f'{tile_str(tile)}: masked hot pixels in {time.time()-start:.2f} seconds.')
    start = time.time()
    # find and mask dim stars up to a magnitude of gmag_lim
    (
        data_star_mask,
        star_mask,
        file_path_star_mask,
        bright_star_flag,
        gmag_brightest,
        visualization_mask,
    ) = mask_stars(
        tile_str(tile),
        data_ano_mask,
        header_binned,
        file_path_binned,
        bkg,
        r_scale=0.55,
        l_scale=0.8,
        gmag_lim=11.0,
        save_to_file=True,
    )
    logger.info(f'{tile_str(tile)}: masked stars in {time.time()-start:.2f} seconds.')

    # If there is a bright star in the image MTO struggles to accurately estimate the background
    # so we remove it in these cases
    start = time.time()
    if bright_star_flag:
        # narrower background window for fields with brighter stars
        if gmag_brightest <= 8:
            bkg_dim = 50
        elif gmag_brightest <= 7:
            bkg_dim = 25
        else:
            bkg_dim = 100
        data_sub, _, file_path_sub = remove_background(
            data_star_mask,
            header_binned,
            file_path_star_mask,
            bw=bkg_dim,
            bh=bkg_dim,
            estimator=MedianBackground(),
            save_file=True,
        )
        logger.info(f'{tile_str(tile)}: subtracted background in {time.time()-start:.2f} seconds.')
        data_prepped, file_path_prepped = data_sub, file_path_sub
        # delete data products from intermediate steps
        delete_file(file_path_ano_mask)
        delete_file(file_path_star_mask)

    else:
        data_prepped, file_path_prepped = data_star_mask, file_path_star_mask
        # delete data products from intermediate steps
        delete_file(file_path_ano_mask)

    logger.info(f'{tile_str(tile)}: tile prepped in {time.time()-prep_start:.2f} seconds.')

    return data_prepped, file_path_prepped, header_binned
