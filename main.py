import argparse
import logging
import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from vos import Client

from detect import run_mto
from kd_tree import build_tree
from postprocess import match_detections_with_catalogs
from preprocess import prep_tile
from tile_cutter import (
    download_tile_all_bands,
    download_tile_one_band,
    make_cutouts_all_bands,
    save_to_h5,
    tile_finder,
)
from utils import (
    TileAvailability,
    extract_tile_numbers,
    load_available_tiles,
    update_available_tiles,
)

client = Client()

# To work with the client you need to get CANFAR X509 certificates
# Run these lines on the command line:
# cadc-get-cert -u yourusername
# cp ${HOME}/.ssl/cadcproxy.pem .

band_dict = {
    'cfis-u': {
        'name': 'CFIS',
        'band': 'u',
        'vos': 'vos:cfis/tiles_DR5/',
        'suffix': '.u.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
    'whigs-g': {
        'name': 'calexp-CFIS',
        'band': 'g',
        'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme/',
        'suffix': '.fits',
        'delimiter': '_',
        'fits_ext': 1,
        'zfill': 0,
    },
    'cfis_lsb-r': {
        'name': 'CFIS_LSB',
        'band': 'r',
        'vos': 'vos:cfis/tiles_LSB_DR5/',
        'suffix': '.r.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
    'ps-i': {
        'name': 'PS-DR3',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR3/tiles/',
        'suffix': '.i.fits',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
    },
    'wishes-z': {
        'name': 'WISHES',
        'band': 'z',
        'vos': 'vos:cfis/wishes_1/coadd/',
        'suffix': '.z.fits',
        'delimiter': '.',
        'fits_ext': 1,
        'zfill': 0,
    },
}

# retrieve from the VOSpace and update the currently available tiles; takes some time to run
update_tiles = False
# build kd tree with updated tiles otherwise use the already saved tree
if update_tiles:
    build_new_kdtree = True
else:
    build_new_kdtree = False
# return the number of available tiles that are available in at least 5, 4, 3, 2, 1 bands
at_least = False
# show stats on currently available tiles, remember to update
show_tile_statistics = False
# define the minimum number of bands that should be available for a tile
band_constraint = 3
# print per tile availability
print_per_tile_availability = False
# download the tiles
download_tiles = False
# Plot cutouts from one of the tiles after execution
with_plot = False
# Plot a random cutout from one of the tiles after execution else plot all cutouts
plot_random_cutout = False
# Show plot
show_plot = False
# Save plot
save_plot = False

# paths
# define the root directory
parent_directory = '/home/nick/astro/DwarForge/'
table_directory = os.path.join(parent_directory, 'tables/')
os.makedirs(table_directory, exist_ok=True)
# define catalog file
catalog_file = 'all_known_dwarfs.csv'
catalog_script = pd.read_csv(os.path.join(table_directory, catalog_file))
# define the keys for ra, dec, and id in the catalog
ra_key_script, dec_key_script, id_key_script = 'ra', 'dec', 'ID'
# define where the information about the currently available tiles should be saved
tile_info_directory = os.path.join(parent_directory, 'tile_info/')
os.makedirs(tile_info_directory, exist_ok=True)
# define where the tiles should be saved
download_directory = os.path.join(parent_directory, 'data/')
os.makedirs(download_directory, exist_ok=True)
# define where models should be saved
model_directory = os.path.join(parent_directory, 'models/')
os.makedirs(model_directory, exist_ok=True)
# define where the logs should be saved
log_dir = os.path.join(parent_directory, 'logs/')
os.makedirs(log_dir, exist_ok=True)
# define where the cutouts should be saved
cutout_directory = os.path.join(parent_directory, 'h5_files/')
os.makedirs(cutout_directory, exist_ok=True)
# define where figures should be saved
figure_directory = os.path.join(parent_directory, 'figures/')
os.makedirs(figure_directory, exist_ok=True)

# tile parameters
cutout_size = 200
h5_filename = 'cutout_stacks_ugriz_lsb_200x200'
num_workers = 9  # specifiy the number of parallel workers following machine capabilities


def query_availability(update, in_dict, at_least_key, show_stats, build_kdtree, tile_info_dir):
    """
    Gather information on the currently available tiles.

    Args:
        update (bool): update the available tiles
        in_dict (dict): band dictionary
        at_least_key (bool): print the number of tiles in at least (not exactly) 5, 4, ... bands
        show_stats (bool): show stats on the currently available tiles
        build_kdtree (bool): build a kd tree from the currently available tiles
        tile_info_dir (str): path to save the tile information

    Returns:
        TileAvailability: availability of the tiles
    """
    # update information on the currently available tiles
    if update:
        update_available_tiles(tile_info_dir)
    # extract the tile numbers from the available tiles
    u, g, lsb_r, i, z = extract_tile_numbers(load_available_tiles(tile_info_dir))
    all_bands = [u, g, lsb_r, i, z]
    # create the tile availability object
    availability = TileAvailability(all_bands, in_dict, at_least_key)
    # build the kd tree
    if build_kdtree:
        build_tree(availability.unique_tiles, tile_info_dir)
    # show stats on the currently available tiles
    if show_stats:
        availability.stats()
    return availability


def import_coordinates(coordinates, ra_key_default, dec_key_default, id_key_default):
    """
    Process coordinates provided from the command line.

    Args:
        coordinates (nested list): ra, dec coordinates
        ra_key_default (str): default right ascention key
        dec_key_default (str): default declination key
        id_key_default (str): default ID key

    Raises:
        ValueError: error if the number of coordinates is not even

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    coordinates = coordinates[0]
    if (len(coordinates) == 0) or len(coordinates) % 2 != 0:
        raise ValueError('Provide even number of coordinates.')

    ras, decs, ids = (
        coordinates[::2],
        coordinates[1::2],
        list(np.arange(1, len(coordinates) // 2 + 1)),
    )
    ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
    df_coordinates = pd.DataFrame({id_key: ids, ra_key: ras, dec_key: decs})

    formatted_coordinates = ' '.join([f'({ra}, {dec})' for ra, dec in zip(ras, decs)])
    logging.info(f'Coordinates received from the command line: {formatted_coordinates}')
    catalog = df_coordinates
    coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')
    return catalog, coord_c


def import_dataframe(
    dataframe_path, ra_key, dec_key, id_key, ra_key_default, dec_key_default, id_key_default
):
    """
    Process a DataFrame provided from the command line.

    Args:
        dataframe_path (str): path to the DataFrame
        ra_key (str): right ascention key
        dec_key (str): declination key
        id_key (str): ID key
        ra_key_default (str): default right ascention key
        dec_key_default (str): default declination key
        id_key_default (str): default ID key

    Returns:
        tuple: dataframe, SkyCoord object of the coordinates
    """
    logging.info('Dataframe received from command line.')
    catalog = pd.read_csv(dataframe_path)

    if ra_key is None or dec_key is None or id_key is None:
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default

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

    coord_c = SkyCoord(catalog[ra_key].values, catalog[dec_key].values, unit='deg', frame='icrs')

    return catalog, coord_c


def import_tiles(tiles, availability, band_constr):
    """
    Process tiles provided from the command line.

    Args:
        tiles (nested list): tile numbers
        availability (TileAvailability): instance of the TileAvailability class
        band_constr (int): minimum number of bands that should be available

    Raises:
        ValueError: provide two three digit numbers for each tile

    Returns:
        list: list of tiles that are available in r and at least two other bands
    """
    tiles = tiles[0]
    if (len(tiles) == 0) or len(tiles) % 2 != 0:
        raise ValueError('Provide two three digit numbers for each tile.')

    tile_list = [tuple(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    logging.info(f'Tiles received from command line: {tiles}')

    return [
        tile
        for tile in tile_list
        if 'r' in availability.get_availability(tile)[0]
        and len(availability.get_availability(tile)[1]) >= band_constr
    ]


def complete_tile_processing_workflow(
    avail,
    in_dict,
    tile_nums,
    download_dir,
    cutout_dir,
    table_dir,
    model_dir,
    mto_path,
    band,
    with_download,
    with_detection,
    with_cat_match,
    with_cutouts,
    size,
    zp,
    mu_lim,
    reff_lim,
    id_key,
    ra_key,
    dec_key,
    in_cat=None,
    with_streak_mask=False,
):
    vos_dir = in_dict[band]['vos']
    prefix = in_dict[band]['name']
    suffix = in_dict[band]['suffix']
    delimiter = in_dict[band]['delimiter']
    fits_ext = in_dict[band]['fits_ext']
    zfill = in_dict[band]['zfill']
    tile_dir = download_dir + f'{str(tile_nums[0]).zfill(3)}_{str(tile_nums[1]).zfill(3)}'
    os.makedirs(tile_dir, exist_ok=True)
    tile_fitsfilename = f'{prefix}{delimiter}{str(tile_nums[0]).zfill(zfill)}{delimiter}{str(tile_nums[1]).zfill(zfill)}{suffix}'
    temp_name = '.'.join(tile_fitsfilename.split('.')[:-1]) + '_temp.fits'
    temp_path = os.path.join(tile_dir, temp_name)
    final_path = os.path.join(tile_dir, tile_fitsfilename)
    vos_path = os.path.join(vos_dir, tile_fitsfilename)
    obj_to_cut = None
    avail_bands = ''.join(avail.get_availability(tile_nums)[0])

    if with_download:
        download_tile_one_band(tile_nums, tile_fitsfilename, final_path, temp_path, vos_path, band)
    else:
        logging.info('Skipping download.')

    if with_detection:
        bkg, header = prep_tile(final_path, fits_ext, table_dir, with_streak_mask)
        params_mto = run_mto(
            tile_nums, final_path, model_dir, mto_path, band, mu_lim, reff_lim, bkg, zp, header
        )

        if with_cat_match:
            obj_to_cut = match_detections_with_catalogs(tile_nums, params_mto, table_dir)

        else:
            logging.info('Skipping catalog matching.')
            obj_to_cut = params_mto.loc[params_mto['label'] == 1].reset_index(drop=True)

    if with_cutouts:
        if in_cat is not None:
            obj_to_cut = in_cat
        if obj_to_cut is not None:
            if download_tile_all_bands(avail, tile_nums, in_dict, download_dir):
                logging.info('Successfully downloaded the remaining bands.')
                cutout = make_cutouts_all_bands(
                    avail, tile_nums, obj_to_cut, download_dir, in_dict, size
                )
                save_to_h5(
                    cutout,
                    tile_nums,
                    obj_to_cut[id_key].values,
                    obj_to_cut[ra_key].values,
                    obj_to_cut[dec_key].values,
                    size,
                    avail_bands,
                    cutout_dir,
                )
        else:
            logging.info('Skipping download and cutout, no objects to cut out.')
    else:
        logging.info('No tiles were harmed while running this script.')


def input_to_tile_list(
    availability,
    band_constr,
    coordinates=None,
    dataframe_path=None,
    tiles=None,
    ra_key=None,
    dec_key=None,
    id_key=None,
    tile_info_dir=None,
    ra_key_default='ra',
    dec_key_default='dec',
    id_key_default='ID',
):
    """
    Process the input to get a list of tiles that are available in r and at least two other bands.

    Args:
        availability (TileAvailability): instance of the TileAvailability class
        band_constr (int): minimum number of bands that should be available
        coordinates (nested list, optional): coordinates from the command line. Defaults to None.
        dataframe_path (str, optional): path to dataframe. Defaults to None.
        tiles (nested list, optional): tiles from the command line. Defaults to None.
        ra_key (str, optional): right ascention key. Defaults to None.
        dec_key (str_, optional): declination key. Defaults to None.
        id_key (str, optional): ID key. Defaults to None.
        tile_info_dir (str, optional): path to save the tile information. Defaults to None.
        ra_key_default (str, optional): default right ascention key. Defaults to 'ra'.
        dec_key_default (str, optional): default declination key. Defaults to 'dec'.
        id_key_default (str, optional): default ID key. Defaults to 'ID'.

    Returns:
        list: list of tiles that are available in r and at least two other bands
        catalog (dataframe): updated catalog with tile information
    """

    if coordinates is not None:
        catalog, coord_c = import_coordinates(
            coordinates, ra_key_default, dec_key_default, id_key_default
        )
    elif dataframe_path is not None:
        catalog, coord_c = import_dataframe(
            dataframe_path, ra_key, dec_key, id_key, ra_key_default, dec_key_default, id_key_default
        )
    elif tiles is not None:
        return import_tiles(tiles, availability, band_constr), None
    else:
        logging.info('No coordinates or DataFrame provided. Processing all available LSB-r tiles..')
        ra_key, dec_key, id_key = ra_key_default, dec_key_default, id_key_default
        return [
            tile
            for tile in availability.unique_tiles
            if 'r' in availability.get_availability(tile)[0]
            and len(availability.get_availability(tile)[1]) >= band_constr
        ], None

    unique_tiles, tiles_x_bands, catalog = tile_finder(
        availability, catalog, coord_c, tile_info_dir, band_constr
    )

    return [
        tile
        for tile in unique_tiles
        if 'r' in availability.get_availability(tile)[0]
        and len(availability.get_availability(tile)[1]) >= band_constr
    ], catalog


def main(
    update,
    band_dict,
    at_least,
    show_tile_stats,
    build_kdtree,
    tile_info_dir,
    coordinates,
    dataframe_path,
    tiles,
    ra_key,
    ra_key_default,
    dec_key,
    dec_key_default,
    id_key,
    id_key_default,
    in_dict,
    band_constr,
    download_dir,
):
    # query availability of the tiles
    availability = query_availability(
        update, band_dict, at_least, show_tile_stats, build_kdtree, tile_info_dir
    )
    # get the list of tiles for which r and at least two more bands are available
    tiles_r_plus, input_catalog = input_to_tile_list(
        coordinates,
        dataframe_path,
        tiles,
        ra_key,
        dec_key,
        id_key,
        availability,
        band_constr,
        tile_info_dir,
        ra_key_default,
        dec_key_default,
        id_key_default,
    )


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coordinates',
        nargs='+',
        type=float,
        action='append',
        metavar=('ra', 'dec'),
        help='list of pairs of coordinates to make cutouts from',
    )
    parser.add_argument('--dataframe', type=str, help='path to a CSV file containing the DataFrame')
    parser.add_argument('--ra_key', type=str, help='right ascension key in the DataFrame')
    parser.add_argument('--dec_key', type=str, help='declination key in the DataFrame')
    parser.add_argument('--id_key', type=str, help='id key in the DataFrame')
    parser.add_argument(
        '--tiles',
        type=int,
        nargs='+',
        action='append',
        metavar=('tile'),
        help='list of tiles to make cutouts from',
    )
    args = parser.parse_args()

    # define the arguments for the main function
    # arg_dict_main = {
    # 	'cat_default': catalog_script,
    # 	'ra_key_default': ra_key_script,
    # 	'dec_key_default': dec_key_script,
    # 	'id_key_default': id_key_script,
    # 	'tile_info_dir': tile_info_directory,
    # 	'in_dict': band_dict,
    # 	'at_least_key': at_least,
    # 	'band_constr': band_constraint,
    # 	'download_dir': download_directory,
    # 	'cutout_dir': cutout_directory,
    # 	'figure_dir': figure_directory,
    # 	'table_dir': table_directory,
    # 	'size': cutout_size,
    # 	'h5_name': h5_filename,
    # 	'workers': num_workers,
    # 	'update': update_tiles,
    # 	'show_stats': show_tile_statistics,
    # 	'dl_tiles': download_tiles,
    # 	'build_kdtree': build_new_kdtree,
    # 	'coordinates': args.coordinates,
    # 	'dataframe_path': args.dataframe,
    # 	'tiles': args.tiles,
    # 	'ra_key': args.ra_key,
    # 	'dec_key': args.dec_key,
    # 	'id_key': args.id_key,
    # 	'show_plt': show_plot,
    # 	'save_plt': save_plot,
    # }

    # start = time.time()
    # main(**arg_dict_main)
    # end = time.time()
    # elapsed = end - start
    # elapsed_string = str(timedelta(seconds=elapsed))
    # hours, minutes, seconds = (
    # 	elapsed_string.split(':')[0],
    # 	elapsed_string.split(':')[1],
    # 	elapsed_string.split(':')[2],
    # )
    # print(f'Done! Execution took {hours} hours, {minutes} minutes, and {seconds} seconds.')
