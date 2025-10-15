import logging
import os
import posixpath
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs.utils import skycoord_to_pixel
from dwarforge.kd_tree import TileWCS, query_tree
from dwarforge.utils import TileAvailability, relate_coord_tile, tile_str

logger = logging.getLogger(__name__)


def tile_finder(
    avail: TileAvailability,
    catalog: pd.DataFrame | None,
    coord_c: SkyCoord | None,
    tile_info_dir: Path,
    band_constr: int = 5,
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None, pd.DataFrame | None]:
    """
    Finds tiles a list of objects are in.

    Args:
        avail: object to retrieve available tiles
        catalog: object catalog
        coord_c: astropy SkyCoord object of the coordinates
        tile_info_dir: tile information directory
        band_constr: minimum number of bands that should be available. Defaults to 5.

    Returns:
        unique_tiles: unique tiles the objects are in
        tiles_x_bands: unique tiles with at least band_constr bands available
        catalog: updated catalog with tile information
    """
    if catalog is None or coord_c is None:
        return None, None, None
    available_tiles = avail.unique_tiles
    tiles_matching_catalog = np.empty(len(catalog), dtype=object)
    pix_coords = np.empty((len(catalog), 2), dtype=np.float64)
    bands = np.empty(len(catalog), dtype=object)
    n_bands = np.empty(len(catalog), dtype=np.int32)
    for i, obj_coord in enumerate(coord_c):
        tile_numbers, _ = query_tree(
            available_tiles,
            np.array([obj_coord.ra.deg, obj_coord.dec.deg]),  # type: ignore
            tile_info_dir,
        )
        tiles_matching_catalog[i] = tile_numbers
        # check how many bands are available for this tile
        bands_tile, band_idx_tile = avail.get_availability(tile_numbers)
        bands[i], n_bands[i] = bands_tile, len(band_idx_tile)
        if len(bands_tile) == 0:
            bands[i] = np.nan
            pix_coords[i] = np.nan, np.nan
            continue
        wcs = TileWCS()
        wcs.set_coords(relate_coord_tile(nums=tile_numbers))
        pix_coord = skycoord_to_pixel(obj_coord, wcs.wcs_tile, origin=1)
        pix_coords[i] = pix_coord

    # add tile numbers and pixel coordinates to catalog
    catalog['tile'] = tiles_matching_catalog
    catalog['x'] = pix_coords[:, 0]
    catalog['y'] = pix_coords[:, 1]
    catalog['bands'] = bands
    catalog['n_bands'] = n_bands
    unique_tiles = list(set(tiles_matching_catalog.tolist()))
    tiles_x_bands = [
        tile for tile in unique_tiles if len(avail.get_availability(tile)[1]) >= band_constr
    ]

    return unique_tiles, tiles_x_bands, catalog


def tile_band_specs(tile: tuple, in_dict: dict, band: str, download_dir: Path) -> dict:
    """
    Get the necessary information for downloading a tile in a specific band.

    Args:
        tile: tile numbers
        in_dict: band dictionary containing the necessary info on the file properties
        band: band name
        download_dir: download directory

    Returns:
        tuple: tile_fitsfilename, file_path after download complete, rebin_path after tile rebin, temp_path while download ongoing, vos_path (path to file on server), fits extension of the data, zero point
    """
    vos_dir = in_dict[band]['vos']
    prefix = in_dict[band]['name']
    suffix = in_dict[band]['suffix']
    delimiter = in_dict[band]['delimiter']
    zfill = in_dict[band]['zfill']
    fits_ext = in_dict[band]['fits_ext']
    zp = in_dict[band]['zp']

    tile_dir = Path(download_dir) / f'{tile[0]:0>3}_{tile[1]:0>3}'
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_band_dir = tile_dir / band
    tile_band_dir.mkdir(parents=True, exist_ok=True)

    tile_fitsfilename = (
        f'{prefix}{delimiter}{tile[0]:0>{zfill}}{delimiter}{tile[1]:0>{zfill}}{suffix}'
    )
    final_path = tile_band_dir / tile_fitsfilename
    temp_path = final_path.with_name(final_path.stem + '_temp.fits')
    final_path_binned = final_path.with_name(final_path.stem + '_rebin.fits')
    vos_path = posixpath.join(vos_dir, tile_fitsfilename)

    return {
        'fitsfilename': tile_fitsfilename,
        'final_path': final_path,
        'final_path_binned': final_path_binned,
        'temp_path': temp_path,
        'vos_path': vos_path,
        'fits_ext': fits_ext,
        'zp': zp,
        'tile_dir': tile_dir,
    }


def download_tile_one_band(
    tile_numbers: tuple,
    tile_fitsname: str,
    final_path: Path,
    final_path_binned: Path,
    temp_path: Path,
    vos_path: str,
    band: str,
    full_res: bool = False,
) -> bool:
    """
    Download a tile in a specific band.

    Args:
        tile_numbers: tile numbers
        tile_fitsname: tile fits filename
        final_path: path to file after download complete
        final_path_binned: path to file after rebinning
        temp_path: path to file while download ongoing
        vos_path: path to file on server
        band: band name
        full_res: download full resolution even if rebinned version exists. Defaults to False.

    Returns:
        success/failure
    """
    if final_path.is_file():
        logger.info(f'File {tile_fitsname} was already downloaded for band {band}.')
        return True
    if final_path_binned.is_file() and not full_res:
        logger.info(f'File {tile_fitsname} already exists in rebinned form for band {band}.')
        return True

    try:
        logger.info(f'Downloading {tile_fitsname} for band {band}...')
        start_time = time.time()
        result = subprocess.run(
            ['vcp', '-v', vos_path, str(temp_path)],
            text=True,
            capture_output=True,
            check=False,
        )

        # Surface output in logs to help diagnose remote errors
        if result.stdout:
            logger.debug('vcp stdout:\n%s', result.stdout)
        if result.stderr:
            logger.debug('vcp stderr:\n%s', result.stderr)

        result.check_returncode()
        # change to path mode
        temp_path.rename(final_path)
        logger.info(
            f'Successfully downloaded tile {tile_str(tile_numbers)} for band {band} in {np.round(time.time()-start_time, 1)} seconds.'
        )
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f'Failed downloading tile {tile_str(tile_numbers)} for band {band}.')
        logger.error(f'Subprocess error details: {e}')
        return False

    except FileNotFoundError:
        logger.error(f'Failed downloading tile {tile_str(tile_numbers)} for band {band}.')
        logger.exception(f'Tile {tile_str(tile_numbers)} not available in {band}.')
        return False

    except Exception as e:
        logger.error(f'Tile {tile_str(tile_numbers)} in {band}: an unexpected error occurred: {e}')
        return False


def download_tile_all_bands(avail, tile_numbers, in_dict, download_dir):
    """
    Download a tile for the available bands.

    Args:
        avail (object): object to retrieve available tiles
        tile_numbers (tuple): tile numbers
        in_dict (dict): band dictionary containing the necessary info on the file properties
        download_dir (str): download directory

    Returns:
        bool: success/failure
    """
    tile_dir = download_dir + f'{str(tile_numbers[0]).zfill(3)}_{str(tile_numbers[1]).zfill(3)}'
    avail_idx = avail.get_availability(tile_numbers)[1]
    for band in np.array(list(in_dict.keys()))[avail_idx]:
        vos_dir = in_dict[band]['vos']
        prefix = in_dict[band]['name']
        suffix = in_dict[band]['suffix']
        delimiter = in_dict[band]['delimiter']
        zfill = in_dict[band]['zfill']

        os.makedirs(tile_dir, exist_ok=True)
        tile_fitsfilename = f'{prefix}{delimiter}{str(tile_numbers[0]).zfill(zfill)}{delimiter}{str(tile_numbers[1]).zfill(zfill)}{suffix}'
        # use a temporary name while the file is downloading
        temp_name = '.'.join(tile_fitsfilename.split('.')[:-1]) + '_temp.fits'
        temp_path = os.path.join(tile_dir, temp_name)
        final_path = os.path.join(tile_dir, tile_fitsfilename)
        vos_path = os.path.join(vos_dir, tile_fitsfilename)
        # Check if the directory exists, and create it if not
        if os.path.exists(os.path.join(tile_dir, tile_fitsfilename)):
            logger.info(f'File {tile_fitsfilename} was already downloaded.')
        else:
            logger.info(f'Downloading {tile_fitsfilename}..')
            try:
                result_download = subprocess.run(
                    f'vcp -v {vos_path} {temp_path}',
                    shell=True,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                result_download.check_returncode()
                os.rename(temp_path, final_path)
                logger.info(f'Tile {tuple(tile_numbers)} downloaded successfully in {band}.')

            except subprocess.CalledProcessError as e:
                logger.error(f'Tile {tuple(tile_numbers)} failed to download in {band}.')
                logger.exception(f'Subprocess error details: {e}')
                return False

            except FileNotFoundError:
                logger.error(f'Failed downloading tile {tuple(tile_numbers)} for band {band}.')
                logger.exception(f'Tile {tuple(tile_numbers)} not available in {band}.')
                return False

            except Exception as e:
                logger.exception(
                    f'Tile {tuple(tile_numbers)} in {band}: an unexpected error occurred: {e}'
                )
                return False

    return True


def make_cutout(data, x, y, size):
    """
    Creates an image cutout centered on the object.

    Args:
        data (numpy.ndarray): image data
        x (float): x coordinate of the cutout center
        y (float): y coordinate of the cutout center
        size (int): cutout size in pixels

    Returns:
        img_cutout (numpy.ndarray): cutout
    """
    img_cutout = Cutout2D(data, (x, y), size, mode='partial', fill_value=0).data

    if (
        np.count_nonzero(np.isnan(img_cutout)) >= 0.05 * size**2
        or np.count_nonzero(img_cutout) == 0
    ):
        return np.zeros((size, size))  # Don't use this cutout

    img_cutout[np.isnan(img_cutout)] = 0

    return img_cutout


def make_cutouts_all_bands(avail, tile, obj_in_tile, download_dir, in_dict, size):
    """
    Loops over all five bands for a given tile, creates cutouts of the targets and adds them to the band dictionary.

    Args:
        avail (object): object to retrieve available tiles
        tile (tuple): tile numbers
        obj_in_tile (dataframe): dataframe containing the known objects in this tile
        download_dir (str): directory storing the tiles
        in_dict (dict): band dictionary
        size (int): square cutout size in pixels
    Returns:
        cutout (numpy.ndarray): stacked cutout data
    """
    avail_idx = avail.get_availability(tile)[1]
    cutout = np.zeros((len(obj_in_tile), len(in_dict), size, size))
    for j, band in enumerate(np.array(list(in_dict.keys()))[avail_idx]):
        prefix = in_dict[band]['name']
        suffix = in_dict[band]['suffix']
        delimiter = in_dict[band]['delimiter']
        fits_ext = in_dict[band]['fits_ext']
        zfill = in_dict[band]['zfill']
        tile_dir = download_dir + f'{str(tile[0]).zfill(zfill)}_{str(tile[1]).zfill(zfill)}'
        tile_fitsfilename = f'{prefix}{delimiter}{str(tile[0]).zfill(zfill)}{delimiter}{str(tile[1]).zfill(zfill)}{suffix}'
        with fits.open(os.path.join(tile_dir, tile_fitsfilename), memmap=True) as hdul:  # type: ignore
            data = hdul[fits_ext].data  # type: ignore
        for i, (x, y) in enumerate(zip(obj_in_tile.x.values, obj_in_tile.y.values)):
            cutout[i, j] = make_cutout(data, x, y, size)
    return cutout


def save_to_h5(stacked_cutout, tile_numbers, ids, ras, decs, size, avail_bands, cutout_dir):
    """
    Save cutout data including metadata to file.

    Args:
        stacked_cutout (numpy.ndarray): stacked numpy array of the image data in different bands
        tile_numbers (tuple): tile numbers
        ids (list): object IDs
        ras (numpy.ndarray): right ascension coordinate array
        decs (numpy.ndarray): declination coordinate array
        size (int): cutout size in pixels
        avail_bands (list): available bands
        cutout_dir (str): directory to save the cutouts

    Returns:
        None
    """
    save_path = os.path.join(
        cutout_dir,
        f'{str(tile_numbers[0]).zfill(3)}_{str(tile_numbers[1]).zfill(3)}_{size}x{size}_{avail_bands}.h5',
    )
    logger.info(f'Saving file: {save_path}')
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(save_path, 'w', libver='latest') as hf:
        hf.create_dataset('images', data=stacked_cutout.astype(np.float32))
        hf.create_dataset('tile', data=np.asarray(tile_numbers), dtype=np.int32)
        hf.create_dataset('cfis_id', data=np.asarray(ids, dtype='S'), dtype=dt)
        hf.create_dataset('ra', data=ras.astype(np.float32))
        hf.create_dataset('dec', data=decs.astype(np.float32))
    pass


def read_h5(cutout_dir):
    """
    Reads cutout data from HDF5 file
    :param cutout_dir: cutout directory
    :return: cutout data
    """
    with h5py.File(cutout_dir, 'r') as f:
        # Create empty dictionaries to store data for each group
        cutout_data = {}

        # Loop through datasets
        for dataset_name in f:
            data = np.array(f[dataset_name])
            cutout_data[dataset_name] = data
    return cutout_data
