import ast
import logging
import os
import re
import shutil
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sep

# import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from numpy import linalg as LA
from photutils.background import Background2D
from scipy.spatial import cKDTree
from scipy.stats import truncnorm
from sklearn.decomposition import PCA
from vos import Client

logger = logging.getLogger(__name__)


def func_PCA(x, y):
    """
    Perform principal component analysis on a set of x and y coordinates.

    Args:
        x (numpy.ndarray): x coordinates
        y (numpy.ndarray): y coordinates

    Returns:
        axis ratio (float): axis ratio of the ellipse
        minor axis (float): minor axis of the ellipse
        pca (sklearn.decomposition.PCA): PCA object
    """
    # Create a 2D array of x and y coordinates
    xy = np.column_stack((x, y))

    # Perform principal component analysis
    pca = PCA()
    pca.fit(xy)

    # Get the eigenvalues and eigenvectors
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_  # noqa: F841

    # Get the axis lengths and axis ratio
    major_axis = 2 * np.sqrt(eigenvalues[0])
    minor_axis = 2 * np.sqrt(eigenvalues[1])
    axis_ratio = minor_axis / major_axis
    return axis_ratio, minor_axis, pca


def query_gaia_stars(target_coord, r_arcsec, max_retries=3, retry_delay=5):
    """
    Query Gaia DR3 for non-galaxy sources within the defined search radius.

    Args:
        target_coord (numpy.2darray): coordinates of the tile center.
        r_arcsec (float): search radius in arcseconds.
        max_retries (int): maximum number of retires after a failed attempt.
        retry_delay (int): seconds of delay between retries.

    Returns:
        table (dataframe): non-galaxy Gaia sources within the search radius.
    """
    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'  # type: ignore # Select Data Release 3
    Gaia.ROW_LIMIT = -1  # unlimited rows  # type: ignore
    columns = ['source_id', 'ra', 'dec', 'phot_g_mean_mag', 'in_galaxy_candidates']

    for attempt in range(max_retries):
        try:
            # Query Gaia DR3 for sources within the defined search radius
            job = Gaia.cone_search_async(
                target_coord, radius=u.Quantity(r_arcsec, u.arcsec), columns=columns
            )
            table = job.get_results()

            # Check if the results come back empty, if so return an empty dataframe
            if table is None:
                return pd.DataFrame()
            else:
                table = table.to_pandas()
            # Remove sources that are galaxy candidates
            table = table.loc[~table['in_galaxy_candidates']].reset_index(drop=True)
            table.rename(columns={'phot_g_mean_mag': 'Gmag'}, inplace=True)
            # Remove sources with Gmag >= 20.6
            table = table.loc[table['Gmag'] < 20.6].reset_index(drop=True)
            return table
        except Exception as e:
            if attempt < max_retries - 1:
                logger.error(f'Error occurred: {str(e)}. Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)
            else:
                logger.error(f'Max retries reached. Last error: {str(e)}')
                return pd.DataFrame()


def remove_background(image):
    """
    Remove the background from an image using background estimation and subtraction.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        data_sub (numpy.ndarray): The image with the background subtracted.
        orig_back (astropy.modeling.models.MedianBackground): The original background estimation model.
        bkg_sub (astropy.modeling.models.MedianBackground): The background estimation model after subtracting the background.
    """
    sigma_clip = SigmaClip(sigma=3.0)
    # bkg_estim = MedianBackground()
    orig_bkg = Background2D(
        image,
        (200, 200),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        # bkg_estimator=bkg_estim,  # type: ignore
    )
    data_sub = image - orig_bkg.background
    # bkg_sub = Background2D(
    #     data_sub,
    #     (200, 200),
    #     filter_size=(3, 3),
    #     sigma_clip=sigma_clip,
    #     bkg_estimator=bkg_estim,  # type: ignore
    # )
    orig_back = orig_bkg
    return data_sub, orig_back


def get_background(data, thresh=1, bw=200, bh=200, mask=None):
    image_c = data.byteswap().newbyteorder()
    bkg = sep.Background(image_c, maskthresh=thresh, bw=bw, bh=bh, mask=mask)  # noqa: F821
    bkg.subfrom(image_c)
    return image_c, bkg.globalrms


def piecewise_function_with_break_global(x, a, b, c, d, break_point):
    """
    Compute the piecewise function with a break point.

    Args:
        x (float or array-like): The input values.
        a (float): Coefficient for the exponential term before the break point.
        b (float): Coefficient for the exponential term.
        c (float): Constant term before the break point.
        d (float): Coefficient for the exponential term after the break point.
        break_point (float): The break point value.

    Returns:
        float or array-like: The computed values of the piecewise function.

    """
    result = np.where(
        x < break_point,
        a * np.exp(-b * (x - 6.1368)) + c,
        d * np.exp(-b * (x - 6.1368)) + c,
    )
    return np.where(result < 0, 0, result)


def power_law(x, a, b, c):
    """
    Calculate the power law function.

    Args:
        x (float or array-like): The input value(s).
        a (float): The coefficient of the power law.
        b (float): The exponent of the power law.
        c (float): The constant offset.

    Returns:
        float or array-like: The result of the power law function.
    """
    result = (a * x) ** (-b) + c
    return np.where(result < 0, 0, result)


def piecewise_linear(x, x0, a1, a2, b):
    """
    Computes the piecewise linear function.

    Args:
        x (float or array-like): Input array or scalar.
        x0 (float): Threshold value for the piecewise function.
        a1 (float): Slope of the function for x <= x0.
        a2 (float): Slope of the function for x > x0.
        b (float): Intercept of the function.

    Returns:
        float or array-like: Output array or scalar with the same shape as x.
    """
    return np.where(x <= x0, a1 * (x - x0) + b, a2 * (x - x0) + b)


def relate_coord_tile(coords=None, nums=None):
    """
    Conversion between tile numbers and coordinates.

    Args:
        right ascention, declination (tuple): ra and dec coordinates
        nums (tuple): first and second tile numbers

    Returns:
        tuple: depending on the input, return the tile numbers or the ra and dec coordinates
    """
    if coords:
        ra, dec = coords
        xxx = ra * 2 * np.cos(np.radians(dec))
        yyy = (dec + 90) * 2
        return int(xxx), int(yyy)
    else:
        xxx, yyy = nums  # type: ignore
        dec = yyy / 2 - 90
        ra = xxx / 2 / np.cos(np.radians(dec))
        return np.round(ra, 12), np.round(dec, 12)


def tile_coordinates(name):
    """
    Extract RA and Dec from tile name

    Args:
        name (str): .fits file name of a given tile

    Returns:
        ra, dec (tuple): RA and Dec of the tile center
    """
    parts = name.split('.')
    if name.startswith('calexp'):
        parts = parts[0].split('_')
    xxx, yyy = map(int, parts[1:3])
    ra = np.round(xxx / 2 / np.cos(np.radians((yyy / 2) - 90)), 6)
    dec = np.round((yyy / 2) - 90, 6)
    return ra, dec


def update_available_tiles(path, in_dict, save=True):
    """
    Update available tile lists from the VOSpace. Takes a few mins to run.

    Args:
        path (str): path to save tile lists.
        in_dict (dict): band dictionary
        save (bool): save new lists to disk, default is True.

    Returns:
        None
    """

    for band in np.array(list(in_dict.keys())):
        vos_dir = in_dict[band]['vos']
        band_filter = in_dict[band]['band']
        suffix = in_dict[band]['suffix']

        start_fetch = time.time()
        try:
            logger.info(f'Retrieving {band_filter}-band tiles...')
            # avoid adding other recently added files that are in different format
            if band == 'whigs-g':
                band_tiles = Client().glob1(vos_dir, f'calexp*{suffix}')
            else:
                band_tiles = Client().glob1(vos_dir, f'*{suffix}')
            end_fetch = time.time()
            logger.info(
                f'Retrieving {band_filter}-band tiles completed. Took {np.round((end_fetch - start_fetch) / 60, 3)} minutes.'
            )
            if save:
                np.savetxt(os.path.join(path, f'{band}_tiles.txt'), band_tiles, fmt='%s')
        except Exception as e:
            logger.error(f'Error fetching {band_filter}-band tiles: {e}')


def load_available_tiles(path: Path, in_dict: dict) -> dict:
    """
    Load tile lists from disk.
    Args:
        path: path to files
        in_dict: band dictionary

    Returns:
        dictionary of available tiles for the selected bands
    """

    band_tiles = {}
    for band in np.array(list(in_dict.keys())):
        tiles = np.loadtxt(os.path.join(path, f'{band}_tiles.txt'), dtype=str)
        band_tiles[band] = tiles

    return band_tiles


def get_tile_numbers(name):
    """
    Extract tile numbers from tile name
    :param name: .fits file name of a given tile
    :return two three digit tile numbers
    """

    if name.startswith('calexp'):
        pattern = re.compile(r'(?<=[_-])(\d+)(?=[_.])')
    else:
        pattern = re.compile(r'(?<=\.)(\d+)(?=\.)')

    matches = pattern.findall(name)

    return tuple(map(int, matches))


def extract_tile_numbers(tile_dict: dict, in_dict: dict) -> list[list[tuple[int, int]]]:
    """
    Extract tile numbers from .fits file names.

    Args:
        tile_dict: lists of file names from the different bands
        in_dict: band dictionary

    Returns:
        num_lists: list of lists containing available tile numbers in the different bands
    """

    num_lists = []
    for band in list(in_dict.keys()):
        tiles_for_band = [get_tile_numbers(name) for name in tile_dict[band]]
        num_lists.append(tiles_for_band)
    return num_lists


class TileAvailability:
    def __init__(self, tile_nums, in_dict, at_least=False, band=None):
        self.all_tiles = tile_nums
        self.tile_num_sets = [set(map(tuple, tile_array)) for tile_array in self.all_tiles]
        self.unique_tiles = sorted(set.union(*self.tile_num_sets))
        self.availability_matrix = self._create_availability_matrix()
        self.counts = self._calculate_counts(at_least)
        self.band_dict = in_dict

    def _create_availability_matrix(self):
        array_shape = (len(self.unique_tiles), len(self.all_tiles))
        availability_matrix = np.zeros(array_shape, dtype=int)

        for i, tile in enumerate(self.unique_tiles):
            for j, tile_num_set in enumerate(self.tile_num_sets):
                availability_matrix[i, j] = int(tile in tile_num_set)

        return availability_matrix

    def _calculate_counts(self, at_least):
        counts = np.sum(self.availability_matrix, axis=1)
        bands_available, tile_counts = np.unique(counts, return_counts=True)

        counts_dict = dict(zip(bands_available, tile_counts))

        if at_least:
            at_least_counts = np.zeros_like(bands_available)
            for i, count in enumerate(bands_available):
                at_least_counts[i] = np.sum(tile_counts[i:])
            counts_dict = dict(zip(bands_available, at_least_counts))

        return counts_dict

    def get_availability(self, tile_nums):
        try:
            index = self.unique_tiles.index(tuple(tile_nums))
        except ValueError:
            logger.warning(f'Tile number {tile_nums} not available in any band.')
            return [], []
        except TypeError:
            return [], []
        bands_available = np.where(self.availability_matrix[index] == 1)[0]
        return [list(self.band_dict.keys())[i] for i in bands_available], bands_available

    def band_tiles(self, band=None):
        tile_array = np.array(self.unique_tiles)[
            self.availability_matrix[:, list(self.band_dict.keys()).index(band)] == 1
        ]
        return [tuple(tile) for tile in tile_array]

    def get_tiles_for_bands(self, bands=None) -> list[tuple[int, int]]:
        """
        Get all tiles that are available in specified bands.
        If no bands are specified, return all unique tiles.

        Args:
            bands (str or list): Band name(s) to check for availability.
                                 Can be a single band name or a list of band names.

        Returns:
            list: List of tuples representing the tiles available in all specified bands.
        """
        if bands is None:
            return self.unique_tiles

        if isinstance(bands, str):
            bands = [bands]

        try:
            band_indices = [list(self.band_dict.keys()).index(band) for band in bands]
        except ValueError as e:
            logger.error(f'Invalid band name: {e}')
            return []

        # Get tiles available in all specified bands
        available_tiles = np.where(self.availability_matrix[:, band_indices].all(axis=1))[0]

        return [self.unique_tiles[i] for i in available_tiles]

    def stats(self, band=None):
        logger.info('Number of currently available tiles per band:')
        max_band_name_length = max(map(len, self.band_dict.keys()))  # for output format
        for band_name, count in zip(
            self.band_dict.keys(), np.sum(self.availability_matrix, axis=0)
        ):
            logger.info(f'{band_name.ljust(max_band_name_length)}: {count}')

        logger.info('Number of tiles available in different bands:')
        for bands_available, count in sorted(self.counts.items(), reverse=True):
            logger.info(f'In {bands_available} bands: {count}')

        logger.info(f'Number of unique tiles available: {len(self.unique_tiles)}')

        if band:
            logger.info(f'Number of tiles available in combinations containing the {band}-band:\n')

            all_bands = list(self.band_dict.keys())
            all_combinations = []
            for r in range(1, len(all_bands) + 1):
                all_combinations.extend(combinations(all_bands, r))
            combinations_w_r = [x for x in all_combinations if band in x]

            for band_combination in combinations_w_r:
                band_combination_str = ''.join([str(x).split('-')[-1] for x in band_combination])
                band_indices = [
                    list(self.band_dict.keys()).index(band_c) for band_c in band_combination
                ]
                common_tiles = np.sum(self.availability_matrix[:, band_indices].all(axis=1))
                logger.info(f'{band_combination_str}: {common_tiles}')


def delete_file(file_path: Path):
    if file_path is None:
        return
    try:
        file_path.unlink(missing_ok=True)
        logger.debug(f'File {file_path.name} has been deleted successfully')
    except FileNotFoundError:
        logger.exception(f'File {file_path.name} does not exist')
    except PermissionError:
        logger.exception(f'Permission denied: unable to delete {file_path.name}')
    except Exception as e:
        logger.exception(f'An error occurred while deleting the file {file_path.name}: {e}')


def tile_str(tile):
    return f'({tile[0]}, {tile[1]})'


def extract_tile_numbers_from_job(s):
    # Remove the outer parentheses and split by comma
    parts = s.strip('()').split(',')
    # Extract the numbers from each part
    # numbers = [int(part.split('(')[1].split(')')[0]) for part in parts]
    numbers = [int(part) for part in parts]
    return tuple(numbers)


def get_neighboring_tile_numbers(tile):
    tile = ast.literal_eval(tile)
    x, y = map(int, tile)
    neighbors = [
        (x - 1, y - 1),
        (x - 1, y),
        (x - 1, y + 1),
        (x, y - 1),
        (x, y + 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
    ]
    return [f'({nx:03d}, {ny:03d})' for nx, ny in neighbors if 0 <= nx < 1000 and 0 <= ny < 1000]


def check_objects_in_neighboring_tiles(tile, dwarfs_df, header):
    wcs = WCS(header)
    # Get neighboring tile numbers
    neighboring_tiles = get_neighboring_tile_numbers(tile)

    # Filter dwarfs in neighboring tiles
    neighboring_dwarfs = dwarfs_df[dwarfs_df['tile'].isin(neighboring_tiles)]

    # Check which of these dwarfs are actually within the current tile's boundaries
    dwarfs_in_current_tile = neighboring_dwarfs[
        neighboring_dwarfs.apply(
            lambda row: wcs.footprint_contains(
                SkyCoord(row['ra'], row['dec'], unit='deg', frame='icrs')
            ),
            axis=1,
        )
    ]

    return dwarfs_in_current_tile


def get_dwarf_tile_list(dwarf_cat: Path, in_dict: dict, bands: list[str]) -> list[tuple[int, int]]:
    try:
        bands = [in_dict[band]['band'] for band in bands]
        dwarf_cat_filtered = get_df_for_bands(dwarf_cat, bands)
        dwarf_tiles_for_bands = dwarf_cat_filtered['tile'].values
    except Exception as e:
        print(f'Error getting known dwarf tiles in r: {e}')
    try:
        str_to_tuple = [ast.literal_eval(item) for item in dwarf_tiles_for_bands]
    except Exception as e:
        print(f'Error in str_to_tuple: {e}')
    unique_tiles = list(set(str_to_tuple))
    return unique_tiles


def check_bands(bands_str, to_check):
    if isinstance(bands_str, str):
        if bands_str.startswith('['):
            # Handle string representation of a list
            try:
                bands_list = ast.literal_eval(bands_str)
                return all(band in bands_list for band in to_check)
            except Exception:
                return False
        else:
            # Handle simple string format
            return all(band in bands_str for band in to_check)
    return False  # Return False for NaN values


def get_df_for_bands(dwarf_cat: Path, check_for_bands: list[str]) -> pd.DataFrame:
    dwarf_cat_df = pd.read_csv(dwarf_cat)
    df_select = dwarf_cat_df.loc[
        (~dwarf_cat_df['tile'].isna())
        & (dwarf_cat_df['bands'].apply(lambda x: check_bands(x, check_for_bands)))
    ].reset_index(drop=True)
    return df_select


def check_corrupted_data(data, header, ra, dec, radius_arcsec=15.0):
    """
    Check if the data around a specific coordinate is corrupted.

    This function examines a square region around the given coordinates in a FITS image
    and determines if the data is likely corrupted. First anomalies are detected and replaced
    with zeros then the fraction of zero-value pixels determines the corruption flag.

    Args:
        data (numpy.ndarray): fits image data
        header (header): fits header
        ra (float): Right Ascension of the object in degrees.
        dec (float): Declination of the object in degrees.
        radius_arcsec (float, optional): Radius around the object to check, in arcseconds. Defaults to 15.0.

    Returns:
        bool: True if the data is likely corrupted (mostly zeros), False otherwise.

    Raises:
        ValueError: If the WCS information cannot be extracted from the FITS header.
    """

    wcs = WCS(header)

    if wcs is None:
        raise ValueError('Unable to extract WCS information from the FITS header.')

    # Convert sky coordinates to pixel coordinates
    x, y = wcs.all_world2pix(ra, dec, 0)
    x, y = int(np.round(x)), int(np.round(y))

    # Calculate pixel scale and radius in pixels
    if 'CDELT1' in header:
        pixel_scale = abs(header['CDELT1']) * 3600  # arcsec/pixel
    elif 'CD1_1' in header:
        pixel_scale = abs(header['CD1_1']) * 3600  # arcsec/pixel
    else:
        raise ValueError('Unable to determine pixel scale from FITS header.')

    radius_pixels = round(radius_arcsec / pixel_scale)

    # Extract the region around the coordinate
    y_min = max(0, y - radius_pixels)
    y_max = min(data.shape[0], y + radius_pixels + 1)
    x_min = max(0, x - radius_pixels)
    x_max = min(data.shape[1], x + radius_pixels + 1)
    region = data[y_min:y_max, x_min:x_max]

    # Check if more than 90% of the pixels are zero
    zero_fraction = np.sum(region == 0) / region.size
    return zero_fraction > 0.6


def generate_positive_trunc_normal(annulus_data, mean, std, size):
    #     print(f'mean: {mean}, std: {std}')

    if len(annulus_data) != 0:
        lower, upper = np.percentile(annulus_data, 60), 10**5
    #         print(f'percentile: {np.percentile(annulus_data, 60)}')
    #         print(f'frac > 0: {np.sum(annulus_data > lower)/len(annulus_data)}')
    else:
        lower, upper = 0, 10**5

    scale = max(std, 0.1)
    loc = mean

    try:
        # Calculate the standard truncated normal distribution's limits
        a, b = (lower - loc) / scale, (upper - loc) / scale
    except Exception as e:
        print(f'Error: {e}')

    # Generate values
    positive_values = truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)
    return positive_values


def open_fits(file_path: Path, fits_ext: int) -> tuple[np.ndarray, Header]:
    """
    Open fits file and return data and header.

    Args:
        file_path: name of the fits file
        fits_ext: extension of the fits file

    Returns:
        data (numpy.ndarray): image data
        header (fits header): header of the fits file
    """
    logger.debug(f'Opening fits file {os.path.basename(file_path)}..')
    start_opening = time.time()
    with fits.open(file_path, memmap=True) as hdul:  # type: ignore
        data = hdul[fits_ext].data.astype(np.float32)  # type: ignore
        header = hdul[fits_ext].header  # type: ignore
    logger.debug(
        f'Fits file {os.path.basename(file_path)} opened in {time.time() - start_opening:.2f} seconds.'
    )
    return data, header


def create_cartesian_kdtree(ra, dec):
    """
    Create a KD-Tree using Cartesian coordinates converted from RA and Dec.

    :param ra: Right Ascension in degrees
    :param dec: Declination in degrees
    :return: cKDTree object and the corresponding SkyCoord object
    """
    coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    xyz = coords.cartesian.xyz.value.T  # type: ignore
    tree = cKDTree(xyz)  # type: ignore
    return tree, coords


def delete_folder_contents(folder_path):
    logger.info(f'Deleting contents of folder: {folder_path}')
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def read_parquet(parquet_path, ra_range, dec_range, columns=None):
    """
    Read parquet file and return a pandas dataframe.

    Args:
        parquet_path (str): path to parquet file
        ra_range (tuple): range of right ascension to select
        dec_range (tuple): range of declination to select
        columns (list): columns to select

    Returns:
        df (dataframe): pandas dataframe containing the selected data
    """
    logger.debug('Reading redshift catalog.')
    filter_coords = [
        ('ra', '>=', ra_range[0]),
        ('ra', '<=', ra_range[1]),
        ('dec', '>=', dec_range[0]),
        ('dec', '<=', dec_range[1]),
    ]
    df = pq.read_table(parquet_path, memory_map=True, filters=filter_coords).to_pandas()
    if columns:
        df = df[columns]
    logger.debug(f'Read {len(df)} objects from catalog {os.path.basename(parquet_path)}.')
    return df


def is_mostly_zeros(file_path, file_path_binned, fits_ext, band):
    if file_path_binned.is_file():
        return False
    # HSC data is compressed, decompress for faster read speed later
    if band in ['whigs-g', 'wishes-z']:
        data, _ = decompress_fits(file_path)
    elif band in ['ps-i']:
        data, _ = adjust_psdr3_header(file_path)
    else:
        data, _ = open_fits(file_path, fits_ext)

    frac_zeros = np.count_nonzero(data == 0.0) / len(data.flatten())
    if frac_zeros > 1.0:
        return True
    else:
        return False


def transform_path(path):
    # Remove the leading slash if it exists
    if path.startswith('/'):
        path = path[1:]

    # Replace the first slash with a colon
    parts = path.split('/', 1)
    if len(parts) > 1:
        return f'/{parts[0]}:{parts[1]}'
    else:
        return path  # Return the original path if there's no slash to replace


def dist_peak_center(obj):
    x, y = round(obj['xpeak'].iloc[0]), round(obj['ypeak'].iloc[0])
    xmin, xmax = round(obj['xmin'].iloc[0]), round(obj['xmax'].iloc[0])
    ymin, ymax = round(obj['ymin'].iloc[0]), round(obj['ymax'].iloc[0])
    x_center, y_center = np.mean([xmin, xmax]), np.mean([ymin, ymax])
    npix = obj['npix'].iloc[0]
    # Get euclidean distance between the photometric center and the center of the segment
    d_peak_center = LA.norm(np.array([x_center, y_center]) - np.array([x, y]))
    # normalized by area
    d_norm = d_peak_center / npix

    return d_peak_center, d_norm


def estimate_axis_ratio(labeled_array, label):
    y, x = np.nonzero(labeled_array == label)
    if len(y) < 2:  # Handle single-pixel objects
        return 1.0
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    y -= y.mean()
    x -= x.mean()
    inertia = np.array([[np.sum(y**2), np.sum(x * y)], [np.sum(x * y), np.sum(x**2)]])
    eigenvalues = np.linalg.eigvals(inertia)
    minor, major = np.sqrt(np.abs(eigenvalues))
    return minor / major if major != 0 else 1.0


def ensure_list(variable):
    if isinstance(variable, str):
        return [variable]
    elif isinstance(variable, list):
        return variable
    else:
        raise TypeError('Input should be either a string or a list.')


def decompress_fits(file_path, fits_ext=1):
    """
    Decompress fits file by reading and saving again.

    Args:
        file_path (str): path to file
        fits_ext (int, optional): data extension. Defaults to 1.

    Returns:
        data (numpy.ndarray): data
        header (header): fits header
    """
    data, header = open_fits(file_path, fits_ext=fits_ext)
    new_hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    # save new fits file
    new_hdu.writeto(file_path, overwrite=True)

    return data, header


def count_duplicates(lst):
    return sum(1 for count in Counter(lst).values() if count > 1)


def adjust_psdr3_header(file_path):
    data, header = open_fits(file_path, fits_ext=0)

    new_header = header.copy()
    # Update CRPIX values
    new_header['CRPIX1'] += 0.5 * 0.25 / 0.186  # type: ignore
    new_header['CRPIX2'] += 0.5 * 0.25 / 0.186  # type: ignore

    # Create a new HDU with the rebinned data and updated header
    new_hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=new_header)

    # save new fits file
    new_hdu.writeto(file_path, overwrite=True)

    return data, new_header


def open_raw_data(file_path: Path, fits_ext: int, band: str) -> tuple[np.ndarray, Header]:
    # HSC data is compressed, decompress for faster read speed later
    if band in ['whigs-g', 'wishes-z']:
        data, header = decompress_fits(file_path)
    elif band in ['ps-i']:
        data, header = adjust_psdr3_header(file_path)
    else:
        data, header = open_fits(file_path, fits_ext)

    return data, header


def get_coord_median(coords: list[SkyCoord]) -> SkyCoord:
    """
    Calculates the coordinate-wise median (median RA, median Dec).
    Handles RA wrap-around. Robust against non-finite inputs checked beforehand.
    """
    if not coords:
        raise ValueError('Input list cannot be empty.')
    try:
        common_frame = coords[0].frame
        coords_in_frame = [c.transform_to(common_frame) for c in coords]

        # Extract ra and dec values in degrees
        ras = np.array([c.ra.deg for c in coords_in_frame])  # type: ignore
        decs = np.array([c.dec.deg for c in coords_in_frame])  # type: ignore

        # Median Dec (simple)
        median_dec = np.median(decs)

        # Median RA (handle wrap-around)
        # Check if the RA range suggests wrap-around
        ras_sorted = np.sort(ras)
        ra_range = ras_sorted[-1] - ras_sorted[0]
        if ra_range > 180.0:
            # Shift RAs where RA < 180 by +360 degrees
            ras_shifted = np.where(ras < 180.0, ras + 360.0, ras)
            median_ra_shifted = np.median(ras_shifted)
            # Ensure the result is back in the 0-360 range
            median_ra = median_ra_shifted % 360.0
        else:
            # No wrap-around suspected, use simple median
            median_ra = np.median(ras)

        # Create SkyCoord from median ra/dec
        median_skycoord = SkyCoord(ra=median_ra, dec=median_dec, unit='deg', frame=common_frame)
        return median_skycoord
    except Exception as e:
        # Add more context to the error
        logger.error(f'Error in get_coord_median for input coords (first few shown): {coords[:3]}')
        raise ValueError(f'Could not calculate coordinate median: {e}')


def purge_previous_run(cfg) -> None:
    """If resume == False, delete existing log files and the progress database."""
    if cfg.monitoring.progress.resume:
        return

    # Logs: remove current + rotated files for this logger name
    log_dir: Path = cfg.paths.log_directory
    stem = cfg.logging.name
    for p in log_dir.glob(f'{stem}.log*'):
        p.unlink(missing_ok=True)

    # Progress DB: remove the SQLite file
    db: Path = cfg.paths.progress_db_path
    db.unlink(missing_ok=True)
