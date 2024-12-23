import logging
import warnings

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, binary_fill_holes, label


def process_channels(
    img,
    scaling_type='linear',
    stretch=0.01,
    Q=5.0,
    gamma=0.25,
):
    """
    Create an RGB image from three bands of data preserving relative channel intensities.
    Handles channels that are all zeros.
    """
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Check for zero channels
    red_is_zero = np.all(red == 0)
    green_is_zero = np.all(green == 0)
    blue_is_zero = np.all(blue == 0)

    # Compute average intensity before scaling choice (avoiding zero channels)
    nonzero_channels = []
    if not red_is_zero:
        nonzero_channels.append(red)
    if not green_is_zero:
        nonzero_channels.append(green)
    if not blue_is_zero:
        nonzero_channels.append(blue)

    if nonzero_channels:
        i_mean = sum(nonzero_channels) / len(nonzero_channels)
    else:
        i_mean = np.zeros_like(red)  # All channels are zero

    if scaling_type == 'asinh':
        # Apply asinh scaling only to non-zero channels
        if not red_is_zero:
            red = red * np.arcsinh(stretch * Q * (i_mean)) / (Q * i_mean)
        if not green_is_zero:
            green = green * np.arcsinh(stretch * Q * (i_mean)) / (Q * i_mean)
        if not blue_is_zero:
            blue = blue * np.arcsinh(stretch * Q * (i_mean)) / (Q * i_mean)
    elif scaling_type == 'linear':
        # Apply linear scaling without normalization
        if not red_is_zero:
            red = red * stretch
        if not green_is_zero:
            green = green * stretch
        if not blue_is_zero:
            blue = blue * stretch
    else:
        raise ValueError(f'Unknown scaling type: {scaling_type}')

    # Apply gamma correction only to non-zero channels
    if gamma is not None:
        if not red_is_zero:
            red_mask = abs(red) <= 1e-9
            red = np.sign(red) * (abs(red) ** gamma)  # Preserve sign
            red[red_mask] = 0

        if not green_is_zero:
            green_mask = abs(green) <= 1e-9
            green = np.sign(green) * (abs(green) ** gamma)
            green[green_mask] = 0

        if not blue_is_zero:
            blue_mask = abs(blue) <= 1e-9
            blue = np.sign(blue) * (abs(blue) ** gamma)
            blue[blue_mask] = 0

    result = np.stack([red, green, blue], axis=-1).astype(np.float32)
    return result


def adjust_flux_with_zp(flux, current_zp, standard_zp):
    adjusted_flux = flux * 10 ** (-0.4 * (current_zp - standard_zp))
    return adjusted_flux


def find_percentile_from_target(cutouts, target_value):
    """
    Determines the first percentile from 100 to 0 where the value is less than or equal to the target value

    Args:
        cutouts (list): list of numpy.ndarrays for each band in the order [i, r, g]
        target_value (float): target value to compare against

    Returns:
        dict: dictionary containing the first percentiles where values are <= target_value for each band
    """
    results = {}
    bands = ['R', 'G', 'B']  # Define band names according to the order of input arrays
    percentiles = np.arange(100, 0, -0.01)  # Creating percentiles from 100 to 0 with 0.01 steps

    for band, cutout in zip(bands, cutouts):
        # We calculate values at each percentile
        values_at_percentiles = np.nanpercentile(cutout, percentiles)

        # Check for the first value that is <= target value
        idx = np.where(values_at_percentiles <= target_value)[0]
        if idx.size > 0:
            results[band] = percentiles[idx[0]]
        else:
            results[band] = 100.0

    return results


def desaturate(image, saturation_percentile, interpolate_neg=False, min_size=10, fill_holes=True):
    """
    Desaturate saturated pixels in an image using interpolation.

    Args:
        image (numpy.ndarray): single band image data
        saturation_percentile (float): percentile to use as saturation threshold
        interpolate_neg (bool, optional): interpolate patches of negative values. Defaults to False.
        min_size (int, optional): number of pixels in a patch to perform interpolation of neg values. Defaults to 10.
        fill_holes (bool, optional): fill holes in generated saturation mask. Defaults to True.

    Returns:
        numpy.ndarray: desaturated image, mask of saturated pixels
    """
    # Check if image is all zeros
    if np.all(image == 0):
        return image, np.zeros_like(image, dtype=bool)
    # Assuming image is a 2D numpy array for one color band
    # Identify saturated pixels
    mask = image >= np.nanpercentile(image, saturation_percentile)
    mask = binary_dilation(mask, iterations=2)

    if interpolate_neg:
        neg_mask = image <= 0.9

        labeled_array, num_features = label(neg_mask)  # type: ignore
        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

        # Prepare to accumulate a total mask
        total_feature_mask = np.zeros_like(image, dtype=np.float64)

        # Loop through all labels to find significant components
        for component_label in range(1, num_features + 1):  # Start from 1 to skip background
            if component_sizes[component_label] >= min_size:
                # Create a binary mask for this component
                component_mask = labeled_array == component_label
                # add component mask to component masks
                # Accumulate the upscaled feature mask
                total_feature_mask |= component_mask

        total_feature_mask = binary_dilation(total_feature_mask, iterations=1)
        mask = np.logical_or(mask, total_feature_mask)

    if fill_holes:
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
        filled_padded_mask = binary_fill_holes(padded_mask)
        if filled_padded_mask is not None:
            mask = filled_padded_mask[1:-1, 1:-1]

    y, x = np.indices(image.shape)

    # Coordinates of non-saturated pixels
    x_nonsat = x[np.logical_not(mask)]
    y_nonsat = y[np.logical_not(mask)]
    values_nonsat = image[np.logical_not(mask)]

    # Coordinates of saturated pixels
    x_sat = x[mask]
    y_sat = y[mask]

    # Interpolate to find values at the positions of the saturated pixels
    interpolated_values = griddata(
        (y_nonsat.flatten(), x_nonsat.flatten()),  # points
        values_nonsat.flatten(),  # values
        (y_sat.flatten(), x_sat.flatten()),  # points to interpolate
        method='linear',  # 'linear', 'nearest' or 'cubic'
    )
    # If any of the interpolated values are NaN, use nearest interpolation
    if np.any(np.isnan(interpolated_values)):
        interpolated_values = griddata(
            (y_nonsat.flatten(), x_nonsat.flatten()),  # points
            values_nonsat.flatten(),  # values
            (y_sat.flatten(), x_sat.flatten()),  # points to interpolate
            method='nearest',  # 'linear', 'nearest' or 'cubic'
        )

    # Replace saturated pixels in the image
    new_image = image.copy()
    new_image[y_sat, x_sat] = interpolated_values

    return new_image, mask


def preprocess_cutout(
    cutout,
    scaling='asinh',
    Q=7,
    stretch=0.008,
    gamma=0.25,
    mode='training',
    with_desaturation=False,
):
    """
    Create an RGB image from the cutout data and save or plot it.

    Args:
        cutout (numpy.ndarray): cutout data
        scaling (str, optional): scaling type. Defaults to 'linear'. Valid options are 'linear' or 'asinh'.
        Q (float, optional): softening parameter for asinh scaling. Defaults to 5.
        stretch (float, optional): scaling factor. Defaults to 0.01.
        gamma (float, optional): gamma correction factor. Defaults to 0.25.
        mode (str, optional): mode of operation. Defaults to 'training'. Valid options are 'training' or 'vis'. Fills missing channels for visualization.
        with_desaturation (bool, optional): whether to apply desaturation. Defaults to False.

    Returns:
        numpy.ndarray: preprocessed image cutout
    """

    # Define warning filter for specific warnings
    warnings.filterwarnings(
        'ignore', category=RuntimeWarning, message='invalid value encountered in log'
    )
    warnings.filterwarnings(
        'ignore', category=RuntimeWarning, message='invalid value encountered in power'
    )
    warnings.filterwarnings(
        'ignore', category=RuntimeWarning, message='invalid value encountered in cast'
    )
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
    warnings.filterwarnings(
        'ignore',
        category=RuntimeWarning,
        message='RuntimeWarning: invalid value encountered in divide',
    )

    def local_warn_handler(message, category, filename, lineno, file=None, line=None):
        if category in [RuntimeWarning, UserWarning]:  # Filter specific warning types
            # Custom message with context about the image
            log = f'Warning: {filename}:{lineno}: {category.__name__}: {message}'
            logging.warning(log)  # Log the warning with contextual info
        else:
            # Let other warnings through normally
            warnings.showwarning_default(message, category, filename, lineno, file, line)

    # Store the default warning handler
    warnings.showwarning_default = warnings.showwarning
    # Set our custom handler
    warnings.showwarning = local_warn_handler

    # Use context manager for temporary warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

    cutout_red = cutout[2]  # i-band
    cutout_green = cutout[1]  # r-band
    cutout_blue = cutout[0]  # g-band

    if np.count_nonzero(cutout_blue) > 0:
        cutout_blue = adjust_flux_with_zp(cutout_blue, 27.0, 30.0)  # g-band

    if mode == 'vis':
        if np.count_nonzero(cutout_red > 1e-10) == 0:
            cutout_red = cutout_green
            cutout_green = (cutout_green + cutout_blue) / 2
        elif np.count_nonzero(cutout_green > 1e-10) == 0:
            cutout_green = (cutout_red + cutout_blue) / 2
        elif np.count_nonzero(cutout_blue > 1e-10) == 0:
            cutout_blue = cutout_red
            cutout_red = (cutout_red + cutout_green) / 2

    if with_desaturation:
        percentile = 99.9
        saturation_percentile_threshold = 1000.0
        high_saturation_threshold = 20000.0
        interpolate_neg = False
        min_size = 1000
        percentile_red = np.nanpercentile(cutout_red, percentile)
        percentile_green = np.nanpercentile(cutout_green, percentile)
        percentile_blue = np.nanpercentile(cutout_blue, percentile)

        # Check if there are saturated pixels in the image
        if np.any(
            np.array([percentile_red, percentile_green, percentile_blue])
            > saturation_percentile_threshold
        ):
            # If any band is highly saturated choose a lower percentile target to bring out more lsb features
            if np.any(
                np.array([percentile_red, percentile_green, percentile_blue])
                > high_saturation_threshold
            ):
                percentile_target = 200.0
            else:
                percentile_target = 1000.0

            # Find individual saturation percentiles for each band
            percentiles = find_percentile_from_target(
                [cutout_red, cutout_green, cutout_blue], percentile_target
            )
            cutout_red_desat, _ = desaturate(
                cutout_red,
                saturation_percentile=percentiles['R'],  # type: ignore
                interpolate_neg=interpolate_neg,
                min_size=min_size,
            )
            cutout_green_desat, _ = desaturate(
                cutout_green,
                saturation_percentile=percentiles['G'],  # type: ignore
                interpolate_neg=interpolate_neg,
                min_size=min_size,
            )
            cutout_blue_desat, _ = desaturate(
                cutout_blue,
                saturation_percentile=percentiles['B'],  # type: ignore
                interpolate_neg=interpolate_neg,
                min_size=min_size,
            )

            rgb = np.stack(
                [cutout_red_desat, cutout_green_desat, cutout_blue_desat], axis=-1
            )  # Stack data in [R, G, B] order
        else:
            rgb = np.stack([cutout_red, cutout_green, cutout_blue], axis=-1)
    else:
        rgb = np.stack([cutout_red, cutout_green, cutout_blue], axis=-1)

    # Create RGB image
    img_linear = process_channels(
        rgb,
        scaling_type=scaling,
        stretch=stretch,
        Q=Q,
        gamma=gamma,
    )

    # restore original cutout shape (channel, cutout_size, cutout_size)
    img_linear = np.moveaxis(img_linear, -1, 0)

    # Restore default warning behavior after function completes
    warnings.showwarning = warnings.showwarning_default

    return img_linear
