import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from dwarforge.config import Inputs
from dwarforge.kd_tree import build_tree
from dwarforge.tile_cutter import tile_finder
from dwarforge.utils import (
    TileAvailability,
    extract_tile_numbers,
    load_available_tiles,
    update_available_tiles,
)

logger = logging.getLogger(__name__)


def query_availability(
    update: bool, in_dict: dict, show_stats: bool, build_kdtree: bool, tile_info_dir: Path
) -> tuple[TileAvailability, list[list[tuple[int, int]]]]:
    """
    Gather information on the currently available tiles.

    Args:
        update: update the available tiles
        in_dict: band dictionary
        show_stats: show stats on the currently available tiles
        build_kdtree: build a kd tree from the currently available tiles
        tile_info_dir: path to save the tile information

    Returns:
        A pair with the availability object and the band-by-tile listings.
    """
    # update information on the currently available tiles
    if update:
        update_available_tiles(tile_info_dir, in_dict)
    # extract the tile numbers from the available tiles
    all_bands = extract_tile_numbers(load_available_tiles(tile_info_dir, in_dict), in_dict)
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict)
    # build the kd tree
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats()
    return availability, all_bands


def import_coordinates(
    coordinates: list[tuple[float, float]],
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame, SkyCoord]:
    """
    Process coordinates provided in the config file.

    Args:
        coordinates: ra, dec coordinates
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Raises:
        ValueError: error if the number of coordinates is not even

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    catalog = pd.DataFrame(coordinates, columns=[ra_key, dec_key], dtype=np.float32)
    # assign IDs to the coordinates
    catalog[id_key] = pd.RangeIndex(start=1, stop=len(catalog) + 1, step=1)
    logger.info('Coordinates received from config: %s', coordinates)
    coord_c = SkyCoord(
        ra=catalog[ra_key].to_numpy(), dec=catalog[dec_key].to_numpy(), unit='deg', frame='icrs'
    )
    return catalog, coord_c


def import_dataframe(
    dataframe_path: Path,
    ra_key: str,
    dec_key: str,
    id_key: str,
) -> tuple[pd.DataFrame | None, SkyCoord | None]:
    """
    Process a DataFrame provided in the config file.

    Args:
        dataframe_path: path to the DataFrame
        ra_key: right ascention key
        dec_key: declination key
        id_key: ID key

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    logging.info('Dataframe read from config file.')
    catalog = pd.read_csv(dataframe_path)

    if (
        ra_key not in catalog.columns
        or dec_key not in catalog.columns
        or id_key not in catalog.columns
    ):
        logging.error(
            'One or more keys not found in the DataFrame. Please provide the correct keys '
            'for right ascention, declination and object ID \n'
            'if they are not equal to the default keys: ra, dec, ID.'
        )
        return None, None

    coord_c = SkyCoord(
        catalog[ra_key].to_numpy(), catalog[dec_key].to_numpy(), unit='deg', frame='icrs'
    )

    return catalog, coord_c


def import_tiles(
    tiles: list[tuple[int, int]], availability: TileAvailability, band_constr: int
) -> list[tuple[int, int]]:
    """
    Process tiles provided in the config file.

    Args:
        tiles: tile numbers
        availability: instance of the TileAvailability class
        band_constr: minimum number of bands that should be available

    Raises:
        ValueError: provide two three digit numbers for each tile

    Returns:
        list: list of tiles that are available in r and at least two other bands
    """
    logging.info(f'Tiles read from config file: {tiles}')

    return [tile for tile in tiles if len(availability.get_availability(tile)[1]) >= band_constr]


def input_to_tile_list(
    availability: TileAvailability,
    band_constr: int,
    inputs: Inputs,
    tile_info_dir: Path,
    ra_key_default: str = 'ra',
    dec_key_default: str = 'dec',
    id_key_default: str = 'ID',
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None, pd.DataFrame | None]:
    """
    Process the input to get a list of tiles that are available in r and at least two other bands.

    Args:
        availability: instance of the TileAvailability class
        band_constr: minimum number of bands that should be available
        inputs: input dictionary with coordinates, a dataframe, or tiles
        tile_info_dir: path to tile information.
        ra_key_default: default right ascention key. Defaults to 'ra'.
        dec_key_default: default declination key. Defaults to 'dec'.
        id_key_default: default ID key. Defaults to 'ID'.

    Returns:
        list: list of tiles that are available in r and at least two other bands
        catalog (dataframe): updated catalog with tile information
    """
    source = inputs.source
    if source == 'coordinates':
        catalog, coord_c = import_coordinates(
            inputs.coordinates, ra_key_default, dec_key_default, id_key_default
        )
    elif source == 'dataframe':
        catalog, coord_c = import_dataframe(
            inputs.dataframe.path,
            inputs.dataframe.columns.ra,
            inputs.dataframe.columns.dec,
            inputs.dataframe.columns.id,
        )
    elif source == 'tiles':
        return None, import_tiles(inputs.tiles, availability, band_constr), None
    else:
        logging.info(
            'No coordinates, DataFrame or tiles provided. Processing all available tiles..'
        )
        return None, None, None

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        availability, catalog, coord_c, tile_info_dir, band_constr
    )

    return unique_tiles, tiles_x_bands, catalog
