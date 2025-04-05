import os
import time

import cv2
import numpy as np
import pandas as pd
import sep
from astride import Streak
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import binary_dilation, label

from detect import detect_anomaly, source_detection_with_dynamic_limit
from logging_setup import get_logger
from utils import (
    delete_file,
    dist_peak_center,
    estimate_axis_ratio,
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
        image,
        (bw, bh),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,  # type: ignore
    )
    # non-zero mask
    mask = image > 0
    data_sub = np.where(mask, image - bkg.background, image)

    if save_file and file_path is not None:
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
    data,
    star_df,
    bright_star_mask,
    r_scale,
    l_scale,
    bkg,
    bkg_factor,
    segmap,
    bkg_mean,
    bkg_median,
    bkg_std,
    object_df,
    grid_size,
    band,
):
    h, w = data.shape
    result = data.copy()
    total_stars = len(star_df)
    skipped_stars = 0
    undetected_stars = 0

    bright_star_mask = bright_star_mask.astype(np.uint8)
    replaced_stars_mask = np.zeros_like(bright_star_mask, dtype=np.uint8)
    visualization_mask = np.zeros((h, w, 3), dtype=np.uint8)
    border_visualization = np.zeros((h, w), dtype=np.uint8)

    star_case_rows = []

    grid_y, grid_x = np.indices((h, w))
    grid_i, grid_j = grid_y // grid_size, grid_x // grid_size

    def create_annular_segments(
        shape, center, inner_r, outer_r, spike_len, spike_thick, n_segments=4, star_case='isolated'
    ):
        y, x = center
        mask = np.zeros(shape, dtype=np.uint8)

        yy, xx = np.ogrid[: shape[0], : shape[1]]
        distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

        # Use arctan2 for full 360-degree angle range
        angles = np.arctan2(yy - y, xx - x)
        # Shift angles to be in [0, 2π) range
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        if star_case == 'not_deblended':
            annulus_extend = max(round(inner_r * 1.5), round(inner_r + 2))
        else:
            # set annulus extend
            annulus_extend = max(
                round(outer_r / 2) + 2,
                round(inner_r * 1.5),
                round(spike_len / 2) + round(0.3 * spike_len / 2),
            )
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
            color=255,  # type: ignore
            markerType=cv2.MARKER_CROSS,
            markerSize=spike_len,
            thickness=spike_thick,
        )  # type: ignore
        mask[spike_mask > 0] = 0

        return mask

    def process_bright_star(
        local_region,
        local_y,
        local_x,
        y,
        x,
        r,
        spike_len,
        spike_thick,
        outer_r,
        local_segmap,
        star_segment,
        local_object_df,
    ):
        star_region = np.zeros_like(local_region, dtype=np.uint8)
        cv2.circle(star_region, (local_x, local_y), r, 255, -1, lineType=cv2.LINE_AA)  # type: ignore

        # Create a mask of non-star objects
        non_star_mask = np.zeros_like(local_segmap, dtype=bool)
        for segment_id in np.unique(local_segmap):
            if segment_id != 0 and segment_id != star_segment:
                obj = local_object_df[local_object_df['ID'] == segment_id]
                if (
                    not obj.empty and obj['star'].iloc[0] == 0 and obj['star_cand'].iloc[0] == 0
                ):  # Non-star object
                    non_star_mask |= local_segmap == segment_id

        # Remove non-star objects from the star region
        star_region[non_star_mask] = 0

        bg_segments = create_annular_segments(
            local_region.shape,
            (local_y, local_x),
            r,
            outer_r,
            spike_len,
            spike_thick,
        )

        # Remove non-star objects and other star objects from background segments
        bg_segments[local_segmap != 0] = 0

        # Diffraction spikes are masked in any case if present
        if spike_len > 0:
            cv2.drawMarker(
                star_region,
                (local_x, local_y),
                color=255,  # type: ignore
                markerType=cv2.MARKER_CROSS,
                markerSize=spike_len,
                thickness=spike_thick,
            )  # type: ignore

        return star_region, bg_segments

    def process_embedded(
        local_region,
        local_y,
        local_x,
        y,
        x,
        r,
        spike_len,
        spike_thick,
        outer_r,
        local_segmap,
        star_segment,
        r_sep,
    ):
        star_region = local_segmap == star_segment

        bg_segments = create_annular_segments(
            local_region.shape,
            (local_y, local_x),
            r_sep,
            max(round(r_sep * 1.5), r_sep + 3),
            0,
            spike_thick,
        )

        # Remove the star region from bg_segments
        bg_segments[star_region] = 0

        return star_region, bg_segments

    def process_not_deblended(
        local_region,
        local_y,
        local_x,
        y,
        x,
        r,
        spike_len,
        spike_thick,
        outer_r,
        local_segmap,
        star_segment,
    ):
        r_reduced = max(round(r * 0.3), 1)
        l_reduced = max(round(spike_len * 0.4), 1)

        #         print(f'x: {x}, y: {y}, r_reduced: {r_reduced}, l_reduced: {l_reduced}')

        star_region_reduced = np.zeros_like(local_region, dtype=np.uint8)
        cv2.circle(
            star_region_reduced,
            (local_x, local_y),
            r_reduced,
            255,  # type: ignore
            -1,
            lineType=cv2.LINE_AA,  # type: ignore
        )  # type: ignore
        if spike_len > 0:
            cv2.drawMarker(
                star_region_reduced,
                (local_x, local_y),
                color=255,  # type: ignore
                markerType=cv2.MARKER_CROSS,
                markerSize=l_reduced,
                thickness=spike_thick,
            )  # type: ignore

        inner_r_bg = r_reduced
        outer_r_bg = r_reduced + 3
        bg_segments = create_annular_segments(
            local_region.shape,
            (local_y, local_x),
            inner_r_bg,
            outer_r_bg,
            l_reduced,
            spike_thick,
            star_case='not_deblended',
        )

        return star_region_reduced, bg_segments

    def process_other_bands(
        local_region,
        local_y,
        local_x,
        y,
        x,
        r,
        r_sep,
        outer_r,
        local_segmap,
        star_segment,
        local_object_df,
        is_bright_star,
        star_case,
    ):
        if is_bright_star:
            star_region = np.zeros_like(local_region, dtype=np.uint8)
            cv2.circle(
                star_region,
                (local_x, local_y),
                round(r_sep * 2.5),
                255,  # type: ignore
                -1,
                lineType=cv2.LINE_AA,
            )  # type: ignore
            star_region = binary_dilation(star_region, iterations=1).astype(np.uint8)

            # Remove non-star objects from the star region
            non_star_mask = np.zeros_like(local_segmap, dtype=bool)
            for segment_id in np.unique(local_segmap):
                if segment_id != 0 and segment_id != star_segment:
                    obj = local_object_df[local_object_df['ID'] == segment_id]
                    if (
                        not obj.empty and obj['star'].iloc[0] == 0 and obj['star_cand'].iloc[0] == 0
                    ):  # Non-star object
                        non_star_mask |= local_segmap == segment_id

            star_region[non_star_mask] = 0

        else:
            star_region = (local_segmap == star_segment).astype(np.uint8)

        if star_case in ['embedded']:
            bg_segments = create_annular_segments(
                local_region.shape,
                (local_y, local_x),
                round(r_sep),
                max(round(r_sep * 1.5), r_sep + 3),
                0,  # No spike_len for other bands
                1,  # Minimal spike_thick
            )
            bg_segments[star_region] = 0
        elif is_bright_star and star_case in ['not_deblended']:
            r_reduced = max(round(r * 0.3), 1)
            star_region = np.zeros_like(local_region, dtype=np.uint8)
            cv2.circle(star_region, (local_x, local_y), r_reduced, 255, -1, lineType=cv2.LINE_AA)  # type: ignore

            inner_r_bg = r_reduced
            outer_r_bg = r_reduced + 3
            bg_segments = create_annular_segments(
                local_region.shape,
                (local_y, local_x),
                inner_r_bg,
                outer_r_bg,
                0,
                spike_thick,
                star_case='not_deblended',
            )
            bg_segments[star_region > 0] = 0
        elif is_bright_star and (star_case not in ['embedded', 'not_deblended']):
            bg_segments = create_annular_segments(
                local_region.shape,
                (local_y, local_x),
                round(2.5 * r_sep),
                round(4 * r_sep),
                0,  # No spike_len for other bands
                1,  # Minimal spike_thick
            )
            bg_segments[star_region] = 0
        elif not is_bright_star and star_case in ['not_deblended']:
            star_region = np.zeros_like(local_region, dtype=np.uint8)
            cv2.circle(star_region, (local_x, local_y), r, 255, -1, lineType=cv2.LINE_AA)  # type: ignore

            inner_r_bg = r
            outer_r_bg = r + 3
            bg_segments = create_annular_segments(
                local_region.shape,
                (local_y, local_x),
                inner_r_bg,
                outer_r_bg,
                0,
                1,
                star_case='not_deblended',
            )
            bg_segments[star_region > 0] = 0
        else:
            bg_segments = None

        return star_region, bg_segments

    def check_star_overlap(star_region, local_bright_star_mask, overlap_threshold=0.8):
        # Calculate the total area of the star
        star_area = np.sum(star_region > 0)

        # Calculate the overlapping area
        overlap_area = np.sum((star_region > 0) & (local_bright_star_mask > 0))

        # Calculate the overlap fraction
        overlap_fraction = overlap_area / star_area if star_area > 0 else 0

        # Check if the overlap fraction exceeds the threshold
        if overlap_fraction >= overlap_threshold:
            return True
        else:
            return False

    for _, star in star_df.iterrows():
        y, x = min(round(star.y), h - 1), min(round(star.x), w - 1)
        star_segment = segmap[y, x]
        r = max(round(star.R_ISO * r_scale), 1)
        spike_len = round(star.diffs_len_fit * l_scale) if band == 'cfis_lsb-r' else 0
        spike_thick = max(1, round(star.diffs_thick_fit)) if band == 'cfis_lsb-r' else 1

        star_case, star_border_mask, r_sep = determine_star_case(segmap, x, y, r, object_df)
        star_case_rows.append({'x': x, 'y': y, 'star_case': star_case})
        # border_visualization |= star_border_mask

        if star_case == 'undetected':
            undetected_stars += 1
            continue

        outer_r = max(round(r * 1.5), r + 7, round(spike_len / 2) + 2)
        y_min, y_max = max(0, y - outer_r), min(h, y + outer_r + 1)
        x_min, x_max = max(0, x - outer_r), min(w, x + outer_r + 1)
        local_y, local_x = y - y_min, x - x_min
        local_region = result[y_min:y_max, x_min:x_max]
        local_segmap = segmap[y_min:y_max, x_min:x_max]
        local_bright_star_mask = bright_star_mask[y_min:y_max, x_min:x_max]
        local_object_df = object_df[object_df['ID'].isin(np.unique(local_segmap))]

        is_bright_star = star.Gmag < 15.0

        if band == 'cfis_lsb-r':
            if is_bright_star and star_case not in ['embedded', 'not_deblended']:
                star_region, bg_segments = process_bright_star(
                    local_region,
                    local_y,
                    local_x,
                    y,
                    x,
                    r,
                    spike_len,
                    spike_thick,
                    outer_r,
                    local_segmap,
                    star_segment,
                    local_object_df,
                )
            elif star_case in ['adjacent', 'isolated']:
                star_region = (local_segmap == star_segment).astype(np.uint8)
                star_region = binary_dilation(star_region, iterations=1).astype(np.uint8)
                if spike_len > 0:
                    cv2.drawMarker(
                        star_region,
                        (local_x, local_y),
                        color=255,  # type: ignore
                        markerType=cv2.MARKER_CROSS,
                        markerSize=spike_len,
                        thickness=spike_thick,
                    )  # type: ignore
                bg_segments = np.zeros_like(star_region)
            elif star_case == 'embedded':
                star_region, bg_segments = process_embedded(
                    local_region,
                    local_y,
                    local_x,
                    y,
                    x,
                    r,
                    spike_len,
                    spike_thick,
                    outer_r,
                    local_segmap,
                    star_segment,
                    r_sep,
                )
            elif star_case == 'not_deblended':
                star_region, bg_segments = process_not_deblended(
                    local_region,
                    local_y,
                    local_x,
                    y,
                    x,
                    r,
                    spike_len,
                    spike_thick,
                    outer_r,
                    local_segmap,
                    star_segment,
                )
        else:
            star_region, bg_segments = process_other_bands(
                local_region,
                local_y,
                local_x,
                y,
                x,
                r,
                r_sep,
                outer_r,
                local_segmap,
                star_segment,
                local_object_df,
                is_bright_star,
                star_case,
            )

        if check_star_overlap(star_region, local_bright_star_mask, overlap_threshold=0.99):
            skipped_stars += 1
            continue

        # Exclude bright stars and other segments from background estimation
        if bg_segments is not None and star_case not in ['embedded', 'not_deblended']:
            bg_segments[local_bright_star_mask > 0] = 0
            bg_segments[(local_segmap > 0) & (local_segmap != star_segment)] = 0

        if bg_segments is not None and (
            is_bright_star or star_case in ['embedded', 'not_deblended']
        ):
            bg_data = local_region[bg_segments > 0]

            if len(bg_data) == 0 or np.count_nonzero(np.isnan(bg_data)) / len(bg_data) > 0.9:
                skipped_stars += 1
                continue

            mean, median, std = sigma_clipped_stats(bg_data, sigma=3.0)
            num_pixels = np.sum(star_region > 0)
            if star_case in ['embedded', 'not_deblended']:
                random_values = generate_positive_trunc_normal(bg_data, median, std, num_pixels)
            else:
                random_values = np.random.normal(median, min(std, 3), num_pixels)  # type: ignore

            local_region[star_region > 0] = random_values
        else:  # Use background grid for other cases in non-cfis_lsb-r bands
            local_grid_i = grid_i[y_min:y_max, x_min:x_max]
            local_grid_j = grid_j[y_min:y_max, x_min:x_max]

            medians = bkg_median[local_grid_i[star_region > 0], local_grid_j[star_region > 0]]
            stds = np.minimum(
                bkg_std[local_grid_i[star_region > 0], local_grid_j[star_region > 0]], 3.0
            )

            replacement_values = np.random.normal(medians, stds)
            local_region[star_region > 0] = replacement_values

        # visualization_mask[y_min:y_max, x_min:x_max, 0] |= star_region.astype(
        #     np.uint8
        # )  # Red for star region
        # if bg_segments is not None:
        #     visualization_mask[y_min:y_max, x_min:x_max, 1] |= bg_segments.astype(
        #         np.uint8
        #     )  # Green for background segments

        # replaced_stars_mask[y_min:y_max, x_min:x_max] |= star_region.astype(np.uint8)
        result[y_min:y_max, x_min:x_max] = local_region

    logger.debug(f'Skipped {skipped_stars}/{total_stars} stars.')
    logger.debug(f'Undetected {undetected_stars}/{total_stars} stars.')
    star_case_df = pd.DataFrame(star_case_rows)
    return (
        result,
        replaced_stars_mask.astype(bool),
        visualization_mask,
        star_case_df,
        border_visualization,
    )


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
    star_df.dropna(inplace=True)  # type: ignore
    if star_df.empty:  # type: ignore
        logger.info(f'No stars found in tile {tile}.')
        return star_df
    c_star = SkyCoord(ra=star_df.ra, dec=star_df.dec, unit='deg', frame='icrs')  # type: ignore
    x_star, y_star = wcs.world_to_pixel(c_star)

    mask = (0 < x_star) & (x_star < tile_width) & (0 < y_star) & (y_star < tile_height)

    star_df = star_df[mask].reset_index(drop=True)  # type: ignore
    star_df['x'], star_df['y'] = x_star[mask], y_star[mask]
    logger.debug(f'{len(star_df)} stars found in tile {tile}.')

    return star_df


def star_fit(df_if, survey='UNIONS'):
    df = df_if.copy()
    # fit star size
    if survey == 'UNIONS':
        param_s = np.array([7.21366775e05, 6.13780414e-01, 1.26614049e02, 8.26137977e05])
        break_point_s = 9.49
        df['A'] = piecewise_function_with_break_global(np.array(df.Gmag), *param_s, break_point_s)  # type: ignore
        df['R_ISO'] = np.sqrt(df.A / np.pi)
    if survey == 'DECALS':
        param_s = np.array([1.40400983e-02, 4.94599527e00, -2.24130151e02])
        df['A'] = power_law(np.array(df_if.Gmag), *param_s)
        df['R_ISO'] = np.sqrt(df_if.A / np.pi)

    # fit length of diffraction spikes
    param_l = np.array([1.36306644e03, 3.45428885e-01, 3.30158864e00, 1.09029943e03])
    break_point_l = 15.29
    df['diffs_len_fit'] = piecewise_function_with_break_global(df['Gmag'], *param_l, break_point_l)  # type: ignore

    # fit thickness of diffraction spikes
    param_t = np.array([15.6, -0.5, -0.05, 1.5])
    df['diffs_thick_fit'] = piecewise_linear(df['Gmag'], *param_t)

    return df


def mask_stars(
    tile_str,
    data,
    header,
    file_path,
    star_df,
    bkg,
    bkg_mean,
    bkg_median,
    bkg_std,
    segmap,
    object_df,
    r_scale=0.6,
    l_scale=0.8,
    gmag_lim=13.2,
    save_to_file=False,
    survey='UNIONS',
    grid_size=25,
    band='cfis_lsb-r',
):
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
    visualization_mask = np.zeros((h, w), dtype=np.uint8)
    replaced_star_mask = np.zeros((h, w), dtype=np.uint8)

    # Create mask for bright stars (not to be replaced)
    for _, star in bright_star_df.iterrows():
        x, y = round(star.x), round(star.y)
        r = max(round(star.R_ISO * r_scale), 1)
        cv2.circle(bright_star_mask, (x, y), round(0.6 * r), 255, -1, lineType=cv2.LINE_AA)  # type: ignore
        # Add diffraction spikes for bright stars
        spike_len = round(star.diffs_len_fit * l_scale)
        spike_thick = round(star.diffs_thick_fit * 5.0)

        cv2.drawMarker(
            bright_star_mask,
            (x, y),
            color=255,  # type: ignore
            markerType=cv2.MARKER_CROSS,
            markerSize=round(0.8 * spike_len),
            thickness=round(0.9 * spike_thick),
        )  # type: ignore

    data, replaced_star_mask, visualization_mask, star_case_df, vis_mask_border = (
        replace_with_local_background(
            data,
            dim_star_df,
            bright_star_mask,
            r_scale,
            l_scale,
            bkg,
            bkg_factor,
            segmap,
            bkg_mean,
            bkg_median,
            bkg_std,
            object_df,
            grid_size,
            band,
        )
    )

    if save_to_file:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_star_mask{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return (
        data,
        replaced_star_mask,
        out_path,
        bright_star_flag,
        gmag_brightest,
        visualization_mask,
        star_case_df,
        vis_mask_border,
    )


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


def mask_hot_pixels(
    image,
    header,
    file_path,
    threshold,
    bkg,
    max_size=3,
    sigma=5,
    neighbor_ratio=0.2,
    save_file=False,
):
    # Create a mask of pixels above the threshold
    hot_pixels = image > threshold
    neighbor_threshold = bkg.background_median + sigma * bkg.background_rms_median

    # Label connected regions
    labeled, num_features = ndimage.label(hot_pixels)  # type: ignore

    # Get the size of each labeled region
    sizes = np.bincount(labeled.ravel())

    # Create a mask of regions that are small enough
    mask_size = sizes <= max_size
    mask_size[0] = 0  # Remove background

    # Apply the size mask to the labeled image
    potential_hot_pixels = mask_size[labeled]  # type: ignore

    # Create a structure for dilation (3x3 square)
    structure = np.ones((3, 3), dtype=bool)

    # Dilate the potential hot pixels
    dilated = ndimage.binary_dilation(potential_hot_pixels, structure=structure)

    # The outline is the difference between the dilated image and the original
    outline = dilated & ~potential_hot_pixels

    # Count the number of surrounding pixels below the threshold
    below_threshold = np.zeros_like(image, dtype=int)
    below_threshold[outline] = (image[outline] <= neighbor_threshold).astype(int)

    # Sum the counts for each potential hot pixel
    below_threshold_sum = ndimage.sum(
        below_threshold, labels=labeled, index=np.arange(1, num_features + 1)
    )

    # Calculate the total number of surrounding pixels for each region
    surrounding_pixels_count = ndimage.sum(
        outline, labels=labeled, index=np.arange(1, num_features + 1)
    )

    # Calculate the ratio of surrounding pixels below threshold
    below_threshold_ratio = np.divide(
        below_threshold_sum,
        surrounding_pixels_count,
        where=surrounding_pixels_count != 0,
        out=np.zeros_like(below_threshold_sum),
    )

    # Create a mask for hot pixels based on the ratio
    hot_pixel_mask = np.zeros_like(image, dtype=bool)
    hot_pixel_mask[potential_hot_pixels] = (
        below_threshold_ratio[labeled[potential_hot_pixels] - 1] >= neighbor_ratio  # type: ignore
    )

    # Create a copy of the image and mask the hot pixels
    masked_image = image.copy()
    masked_image[hot_pixel_mask] = np.median(image)  # Replace with median value

    if save_file and file_path is not None:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_hot_mask{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=masked_image.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return masked_image, hot_pixel_mask


def prep_tile(tile, file_path, fits_ext, zp, band, bin_size=4):
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
    # read in data and header, fits_ext is always 0 after decompressing and
    # saving data to the 0th extension in the g and z band
    data, header = open_fits(file_path, fits_ext=0)
    # bin image to increase signal-to-noise + aid LSB detection
    binned_image = bin_image_cv2(data, bin_size)
    # save binned image to fits
    data_binned, file_path_binned, header_binned = save_processed(
        binned_image, header, file_path, bin_size, preprocess_type='rebin'
    )
    start = time.time()
    # detect data anomalies and set them to zero
    data_ano_mask, file_path_ano_mask = detect_anomaly(
        data_binned,
        header_binned,
        file_path_binned,
        zero_threshold=0.005,
        replace_anomaly=True,
        dilate_mask=True,
        save_to_file=False,
        band=band,
    )
    logger.debug(f'{tile_str(tile)}: detected anomalies in {time.time()-start:.2f} seconds.')
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
    logger.debug(f'{tile_str(tile)}: estimated background in {time.time()-start:.2f} seconds.')
    # mask hot pixels
    start = time.time()
    data_ano_mask, _ = mask_hot_pixels(
        data_ano_mask,
        header_binned,
        file_path_binned,
        threshold=70,
        bkg=bkg,
        max_size=3,
        sigma=3,
        neighbor_ratio=0.2,
        save_file=False,
    )
    logger.debug(f'{tile_str(tile)}: masked hot pixels in {time.time()-start:.2f} seconds.')
    # find stars in the image from gaia
    star_df = find_stars(tile_str(tile), header_binned)
    # sort df to mask dim stars first, overlapping bright stars will mask over them, leaving a smoother image
    sorted_star_df = star_df.sort_values(by='Gmag', ascending=False)  # type: ignore
    # detect objects using SEP
    try:
        objects_sep, data_sub_sep, bkg_sep, segmap_sep = source_detection_with_dynamic_limit(
            data_ano_mask,
            header_binned,
            file_path_binned,
            sorted_star_df,
            thresh=1.0,
            minarea=4,
            deblend_nthresh=32,
            deblend_cont=0.0005,
            bkg_dim=50,
            save_segmap=False,
            extended_flag_radius=9.0,
            mag_limit=14.0,
            initial_limit=500000,
            max_limit=10000000,
            increment_factor=2,
        )

        if objects_sep is None:
            raise Exception('source_detection failed after multiple attempts')

        # get photometry of detected objects
        pixel_scale = abs(header_binned['CD1_1'] * 3600)
        objects_sep = aperture_photometry_mag_auto(
            data_sub_sep, objects_sep, zp=zp, pixel_scale=pixel_scale
        )

        # correct oversubtracted background for g-band
        bkg_mean, bkg_median, bkg_std = None, None, None
        if band == 'whigs-g':
            start_dilate = time.time()
            (
                data_ano_mask,
                bkg_mean,
                bkg_median,
                bkg_std,
                ring_mask,
                dilated_mask,
                large_objects_mask,
            ) = correct_oversubtracted_background(
                data_ano_mask,
                file_path_binned,
                header_binned,
                segmap_sep,
                grid_size=25,
                dilation_size=65,
                min_pixels=600,
                min_axis_ratio=0.2,
                save_file=False,
            )
            logger.debug(f'Dilation took: {time.time()-start_dilate:.2f} seconds.')

        start = time.time()

        # mask tiny/small objects
        data_ano_mask, segmap_updated, objects_updated, bkg_mean, bkg_median, bkg_std = (
            mask_small_objects(
                objects_sep,
                segmap_sep,
                data_ano_mask,
                header_binned,
                file_path,
                bkg_mean,
                bkg_median,
                bkg_std,
                max_npix=39,
                max2_npix=100,
                re_max=1.5,
                grid_size=25,
                save_file=False,
            )
        )
        logger.debug(f'Small stuff masked in {time.time()-start:.2f} seconds.')
    except Exception as e:
        logger.error(f'Error using SEP: {e}')
        # Create background statistics map
        bkg_mean, bkg_median, bkg_std = (
            np.zeros((100, 100), dtype=np.float32),
            np.zeros((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )
        segmap_updated = np.zeros_like(data_ano_mask, dtype=np.float32)
        objects_updated = pd.DataFrame()

    start = time.time()
    # find and mask dim stars up to a magnitude of gmag_lim
    (
        data_star_mask,
        star_mask,
        file_path_star_mask,
        bright_star_flag,
        gmag_brightest,
        visualization_mask,
        star_case_df,
        visualization_mask_border,
    ) = mask_stars(
        tile_str(tile),
        data_ano_mask,
        header_binned,
        file_path_binned,
        sorted_star_df,
        bkg,
        bkg_mean,
        bkg_median,
        bkg_std,
        segmap_updated,
        objects_updated,
        r_scale=0.9,
        l_scale=0.8,
        gmag_lim=10.5,
        save_to_file=True,
        grid_size=25,
        band=band,
    )
    logger.debug(f'{tile_str(tile)}: masked stars in {time.time()-start:.2f} seconds.')

    # If there is a bright star in the image MTO struggles to accurately estimate the background
    # so we remove it in these cases
    start = time.time()
    if bright_star_flag:
        # narrower background window for fields with brighter stars
        if gmag_brightest <= 8:
            bkg_dim = 50
        elif gmag_brightest <= 7:
            bkg_dim = 25
        elif gmag_brightest <= 6:
            bkg_dim = 10
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
        logger.debug(f'{tile_str(tile)}: subtracted background in {time.time()-start:.2f} seconds.')
        data_prepped, file_path_prepped = data_sub, file_path_sub
        # delete data products from intermediate steps
        delete_file(file_path_ano_mask)
        delete_file(file_path_star_mask)

    else:
        data_prepped, file_path_prepped = data_star_mask, file_path_star_mask
        # delete data products from intermediate steps
        delete_file(file_path_ano_mask)

    logger.info(f'{tile_str(tile)}: tile prepped in {time.time()-prep_start:.2f} seconds.')

    return data_binned, data_prepped, file_path_prepped, header_binned


def aperture_photometry_mag_auto(
    data_sub, objects, zp=30, pixel_scale=0.7431
):  # https://sep.readthedocs.io/en/v1.0.x/apertures.html?highlight=aperture
    kronrad, krflag = sep.kron_radius(
        data_sub,
        objects['x'].values,
        objects['y'].values,
        objects['a'].values,
        objects['b'].values,
        objects['theta'].values,
        6.0,
    )
    # fix theta according to following bug: https://github.com/kbarbary/sep/issues/110
    objects = objects.copy()  # Create a copy to avoid SettingWithCopyWarning
    mask = objects['theta'] > np.pi / 2
    objects.loc[mask, 'theta'] -= np.pi

    flux, fluxerr, flag = sep.sum_ellipse(
        data_sub,
        objects['x'].values,
        objects['y'].values,
        objects['a'].values,
        objects['b'].values,
        objects['theta'].values,
        2.5 * kronrad,
        subpix=1,
    )
    flag |= krflag  # combine flags into 'flag'
    r_min = 1.75  # minimum diameter = 3.5
    cflux, cfluxerr, cflag = sep.sum_circle(
        data_sub, objects['x'].values, objects['y'].values, r_min, subpix=1
    )

    # get effective radius
    r_eff = sep.flux_radius(
        data_sub,
        objects['x'].values,
        objects['y'].values,
        6.0 * objects['a'].values,
        0.5,
        normflux=flux,
        subpix=5,
    )
    r_eff = np.array(r_eff) * pixel_scale

    flux = cflux
    fluxerr = cfluxerr  # noqa: F841
    mag = -2.5 * np.log10(flux) + zp
    sb_mean = mag + 2.5 * np.log10(np.pi * r_eff[0] ** 2) + 0.7526

    objects.loc[:, 'mag'] = mag
    objects.loc[:, 'sb_mean'] = sb_mean
    objects.loc[:, 're'] = r_eff[0]

    return objects


def replace_object_pixels(local_region, obj_region, bg_data):
    mean, median, std = sigma_clipped_stats(bg_data, sigma=3.0)
    num_pixels = np.sum(obj_region > 0)
    random_values = np.random.normal(median, min(std, 3.0), num_pixels)  # type: ignore

    result = local_region.copy()
    result[obj_region > 0] = random_values
    return result


def create_annular_segments(shape, center, inner_r, outer_r, n_segments=4):
    y, x = center
    mask = np.zeros(shape, dtype=np.uint8)

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    angles = np.arctan2(yy - y, xx - x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    annulus = (inner_r + 1 <= distances) & (distances <= round(outer_r / 2) + 2)

    segment_size = 2 * np.pi / n_segments
    for i in range(n_segments):
        start_angle = i * segment_size
        end_angle = (i + 1) * segment_size

        segment = (start_angle <= angles) & (angles < end_angle)
        mask[annulus & segment] = 255

    return mask


def create_background_map(data, segmap, grid_size=25, min_fraction=0.5):
    # Create a copy of the data with objects masked
    bg_only = np.where(segmap == 0, data, np.nan)

    h, w = data.shape
    grid_h, grid_w = h // grid_size, w // grid_size

    # Initialize arrays to store background statistics
    bg_mean = np.full((grid_h, grid_w), np.nan)
    bg_median = np.full((grid_h, grid_w), np.nan)
    bg_std = np.full((grid_h, grid_w), np.nan)

    for i in range(grid_h):
        for j in range(grid_w):
            y_start, y_end = i * grid_size, min((i + 1) * grid_size, h)
            x_start, x_end = j * grid_size, min((j + 1) * grid_size, w)

            cell_data = bg_only[y_start:y_end, x_start:x_end]

            # Remove NaN values
            cell_data_clean = cell_data[~np.isnan(cell_data)]

            # Check if we have enough background pixels
            if len(cell_data_clean) / cell_data.size >= min_fraction:
                # Use sigma_clipped_stats on non-NaN data
                mean, median, std = sigma_clipped_stats(cell_data_clean, sigma=3.0)
                bg_mean[i, j] = mean
                bg_median[i, j] = median
                bg_std[i, j] = std

    # Fill invalid cells with nearest valid cell values
    bg_mean = fill_invalid_cells_nearest(bg_mean)
    bg_median = fill_invalid_cells_nearest(bg_median)
    bg_std = fill_invalid_cells_nearest(bg_std)

    return bg_mean, bg_median, bg_std


def fill_invalid_cells_nearest(array):
    mask = np.isnan(array)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return array[tuple(idx)]  # type: ignore


def mask_small_objects(
    objects_df,
    segmap,
    image,
    header,
    file_path,
    bkg_mean,
    bkg_median,
    bkg_std,
    max_npix,
    max2_npix,
    re_max,
    grid_size=25,
    save_file=False,
):
    if bkg_mean is None:
        # Create background statistics map
        bkg_mean, bkg_median, bkg_std = create_background_map(image, segmap, grid_size)

    # Create boolean masks for different object categories
    very_small_objects_mask = (objects_df['npix'] < max_npix) & (objects_df['star'] != 1)
    medium_objects_mask = (
        (objects_df['npix'] >= max_npix)
        & (objects_df['npix'] < max2_npix)
        & (objects_df['re'] < re_max)
        & (objects_df['star'] != 1)
    )
    large_objects_mask = ~(very_small_objects_mask | medium_objects_mask)

    # Get the IDs of objects in each category
    very_small_object_ids = objects_df.loc[very_small_objects_mask, 'ID'].values
    medium_object_ids = objects_df.loc[medium_objects_mask, 'ID'].values
    large_object_ids = objects_df.loc[large_objects_mask, 'ID'].values  # noqa: F841

    # Create boolean masks for the segmentation map
    very_small_objects_segmap_mask = np.isin(segmap, very_small_object_ids)
    medium_objects_segmap_mask = np.isin(segmap, medium_object_ids)

    # Label connected components of very small objects
    labeled_very_small_objects, _ = label(very_small_objects_segmap_mask)  # type: ignore

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_very_small_objects.ravel())[1:]

    # Create a mask for very small object groups that are below the threshold
    very_small_groups_mask = np.isin(
        labeled_very_small_objects, np.where(component_sizes < max_npix)[0] + 1
    )

    # Combine masks for all objects to be masked
    objects_to_mask = very_small_groups_mask | medium_objects_segmap_mask

    # Create grid index arrays
    h, w = image.shape
    grid_y, grid_x = np.indices((h, w))
    grid_i, grid_j = grid_y // grid_size, grid_x // grid_size

    # Get medians and stds for the pixels to be masked
    medians = bkg_median[grid_i[objects_to_mask], grid_j[objects_to_mask]]
    stds = np.minimum(bkg_std[grid_i[objects_to_mask], grid_j[objects_to_mask]], 3.0)

    # Generate replacement values
    replacement_values = np.random.normal(medians, stds)

    # Create a copy of the image only if necessary
    masked_image = image if not save_file else image.copy()

    # Apply the mask and replace values
    masked_image[objects_to_mask] = replacement_values

    # Identify which objects were actually masked
    masked_very_small_ids = very_small_object_ids[
        np.isin(very_small_object_ids, segmap[very_small_groups_mask])
    ]
    masked_medium_ids = medium_object_ids
    all_masked_ids = np.concatenate([masked_very_small_ids, masked_medium_ids])

    # Create a mask for all objects to be removed from the segmentation map
    objects_to_mask_segmap = np.isin(segmap, all_masked_ids)

    # Update the segmentation map
    masked_segmap = segmap.copy()
    masked_segmap[objects_to_mask_segmap] = 0

    # Remove masked objects from objects_df
    updated_objects_df = objects_df[~objects_df['ID'].isin(all_masked_ids)].reset_index(drop=True)

    if save_file:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        filename_tiny = f'{name}_tiny{extension}'
        out_path_tiny = os.path.join(directory, filename_tiny)
        tiny_hdu = fits.PrimaryHDU(data=masked_image.astype(np.float32), header=header)
        tiny_hdu.writeto(out_path_tiny, overwrite=True)

    return masked_image, masked_segmap, updated_objects_df, bkg_mean, bkg_median, bkg_std


def determine_star_case(
    segmap, x, y, r, object_df, expansion=5, min_d_peak_center=15, max_mag=13.5
):
    h, w = segmap.shape

    # Get the segment ID at the star's location
    star_segment = segmap[y, x]

    # Check if the star is in the background (segment ID is 0)
    if star_segment == 0:
        return 'undetected', np.zeros((h, w), dtype=np.uint8), None

    # Try to get the bounding box information for this star from object_df
    star_obj = object_df[object_df['ID'] == star_segment]
    if star_obj.empty:
        # If there's no entry in object_df, treat it as undetected
        return 'undetected', np.zeros((h, w), dtype=np.uint8), 0

    # Get bounding box information
    xmin, xmax = round(star_obj['xmin'].iloc[0]), round(star_obj['xmax'].iloc[0])
    ymin, ymax = round(star_obj['ymin'].iloc[0]), round(star_obj['ymax'].iloc[0])
    # number of pixels assigned to this object
    npix = star_obj['npix'].iloc[0]
    # gmag if object is a star
    gmag = star_obj['Gmag'].iloc[0]
    # Get euclidean distance between the photometric center and the center of the segment
    d_peak_center, d_norm = dist_peak_center(star_obj)
    # Get object radius
    r_sep = np.sqrt(npix / np.pi)

    # If the radius of the matched object is significantly larger than the expected star radius from
    # the magnitude fit, then the star is embedded in a bright object and was not deblended.
    potentially_not_deblended = (r_sep > 1.0 * r) or (
        d_peak_center >= min_d_peak_center and gmag > max_mag
    )

    # check if the expencted segment of the star to be classified is adjacent to or overlapping with
    # other bright stars. If so, it should not be classified as not_deblended
    border_segments_check = np.array([])
    if potentially_not_deblended:
        star_mask_check = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(star_mask_check, (x, y), r, 1, -1, lineType=cv2.LINE_AA)  # type: ignore

        # Dilate the star mask to get its outer edge
        kernel_check = np.ones((3, 3), np.uint8)
        dilated_mask_check = cv2.dilate(star_mask_check, kernel_check, iterations=1)

        # The border is where the dilated mask is 1 and the original mask is 0
        border_mask_check = cv2.subtract(dilated_mask_check, star_mask_check)

        # Check the segment IDs in the border region
        border_segments_check = np.unique(segmap[border_mask_check == 1])

        # Check if any surrounding segment is a star
        adjacent_to_star = False
        for segment_id in border_segments_check:
            if segment_id != 0 and segment_id != star_segment:
                adj_obj = object_df[object_df['ID'] == segment_id]
                if not adj_obj.empty and (
                    (adj_obj['star'].iloc[0] == 1 and adj_obj['Gmag'].iloc[0] < max_mag)
                    or (
                        adj_obj['star_cand'].iloc[0] == 1
                        and adj_obj['Gmag_closest'].iloc[0] < max_mag
                    )
                ):
                    adjacent_to_star = True
                    break

    # Expand the bounding box
    xmin_expanded = max(0, xmin - expansion)
    xmax_expanded = min(w, xmax + expansion)
    ymin_expanded = max(0, ymin - expansion)
    ymax_expanded = min(h, ymax + expansion)

    local_segmap = segmap[ymin_expanded:ymax_expanded, xmin_expanded:xmax_expanded]
    star_mask = local_segmap == star_segment

    # Dilate the star mask to get its outer edge
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(star_mask.astype(np.uint8), kernel, iterations=1)

    # The border is where the dilated mask is 1 and the original mask is 0
    border_mask = (dilated_mask == 1) & (star_mask == 0)

    # Check the values surrounding the star segment
    surrounding_values = local_segmap[border_mask]

    # Create a visualization mask
    vis_mask = np.zeros((h, w), dtype=np.uint8)
    vis_mask[ymin_expanded:ymax_expanded, xmin_expanded:xmax_expanded][border_mask] = 255

    # Determine the case
    if np.all(surrounding_values == 0):
        case = 'isolated'
    elif potentially_not_deblended and not adjacent_to_star:  # and 0 not in border_segments_check:
        case = 'not_deblended'
        vis_mask = np.zeros((h, w), dtype=np.uint8)
    elif potentially_not_deblended and adjacent_to_star:
        case = 'adjacent'
    elif np.any(surrounding_values == 0):
        case = 'adjacent'
    else:
        case = 'embedded'

    return case, vis_mask, round(r_sep)


def create_ring_mask(
    file_path, header, segmap, min_pixels=600, min_axis_ratio=0.2, dilation_size=5, save_file=False
):
    # Convert segmap to binary
    binary_mask = (segmap > 0).astype(int)

    # Label connected components
    labeled_objects, num_features = ndimage.label(binary_mask)  # type: ignore

    # Count pixels in each labeled region
    object_sizes = np.bincount(labeled_objects.ravel())[1:]

    # Identify large objects
    large_object_candidates = np.where(object_sizes >= min_pixels)[0] + 1

    # Calculate axis ratios only for large object candidates
    axis_ratios = np.array(
        [estimate_axis_ratio(labeled_objects, label) for label in large_object_candidates]
    )

    # Filter large objects based on axis ratio
    large_object_labels = large_object_candidates[axis_ratios >= min_axis_ratio]

    # Create mask of large objects
    large_objects_mask = np.isin(labeled_objects, large_object_labels)

    # Dilate large objects
    dilated_large_objects = ndimage.binary_dilation(large_objects_mask, iterations=dilation_size)

    # Combine dilated large objects with original binary mask
    final_mask = np.logical_or(dilated_large_objects, binary_mask).astype(int)

    # Create ring mask
    ring_mask = np.logical_and(final_mask, np.logical_not(binary_mask))

    if save_file:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_seg_dilated{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=final_mask.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

        new_filename = f'{name}_ring_mask{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=ring_mask.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

        new_filename = f'{name}_large_mask{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=large_objects_mask.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return final_mask, ring_mask, large_objects_mask


def correct_oversubtracted_background(
    image,
    file_path,
    header,
    segmap,
    grid_size=25,
    dilation_size=5,
    min_pixels=600,
    min_axis_ratio=0.2,
    save_file=False,
):
    # Create dilated mask, ring mask, and large objects mask
    dilated_mask, ring_mask, large_objects_mask = create_ring_mask(
        file_path, header, segmap, min_pixels, min_axis_ratio, dilation_size, save_file=save_file
    )

    # Create background map using the dilated mask
    bg_mean, bg_median, bg_std = create_background_map(image, dilated_mask, grid_size)

    # Create grid index arrays
    h, w = image.shape
    grid_y, grid_x = np.indices((h, w))
    grid_i, grid_j = grid_y // grid_size, grid_x // grid_size

    # Get medians and stds for the pixels in the ring
    ring_medians = bg_median[grid_i[ring_mask], grid_j[ring_mask]]
    ring_stds = bg_std[grid_i[ring_mask], grid_j[ring_mask]]

    # Generate replacement values
    replacement_values = np.random.normal(ring_medians, ring_stds)

    # Create a copy of the image and replace values
    corrected_image = image.copy()
    corrected_image[ring_mask] = replacement_values

    if save_file:
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f'{name}_oversub_corr{extension}'
        out_path = os.path.join(directory, new_filename)
        new_hdu = fits.PrimaryHDU(data=corrected_image.astype(np.float32), header=header)
        # save new fits file
        new_hdu.writeto(out_path, overwrite=True)

    return corrected_image, bg_mean, bg_median, bg_std, ring_mask, dilated_mask, large_objects_mask
