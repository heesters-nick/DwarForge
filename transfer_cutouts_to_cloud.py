import os
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
from tqdm.asyncio import tqdm  # Import tqdm from tqdm.asyncio
from vos import Client
from google.cloud import storage
from google.api_core import retry
import argparse

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/nick/astro/DwarForge/tables/service_key_google_cloud.json'

band_dict = {
    'cfis-u': {
        'name': 'CFIS',
        'band': 'u',
        'vos': 'vos:cfis/tiles_DR5/',
        'suffix': '.u',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'whigs-g': {
        'name': 'calexp-CFIS',
        'band': 'g',
        'vos': 'vos:cfis/whigs/stack_images_CFIS_scheme/',
        'suffix': '',
        'delimiter': '_',
        'fits_ext': 1,
        'zfill': 0,
        'zp': 27.0,
    },
    'cfis_lsb-r': {
        'name': 'CFIS_LSB',
        'band': 'r',
        'vos': 'vos:cfis/tiles_LSB_DR5/',
        'suffix': '.r',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'ps-i': {
        'name': 'PSS.DR4',
        'band': 'i',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
        'suffix': '.i',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
    'wishes-z': {
        'name': 'WISHES',
        'band': 'z',
        'vos': 'vos:cfis/wishes_1/coadd/',
        'suffix': '.z',
        'delimiter': '.',
        'fits_ext': 1,
        'zfill': 0,
        'zp': 27.0,
    },
    'ps-z': {
        'name': 'PSS.DR4',
        'band': 'ps-z',
        'vos': 'vos:cfis/panstarrs/DR4/resamp/',
        'suffix': '.z',
        'delimiter': '.',
        'fits_ext': 0,
        'zfill': 3,
        'zp': 30.0,
    },
}

@retry.Retry()
def upload_to_gcs(local_file_path, bucket_name, gcs_file_path):
    """Upload a file to Google Cloud Storage with retry."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    blob.upload_from_filename(local_file_path)

async def download_file(session, url, local_path):
    async with session.get(url) as response:
        if response.status == 200:
            with open(local_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
    return False

async def process_file(client, filepath, bucket_name, band):
    filename = os.path.basename(filepath)
    tile_numbers = filename.split('.')[1:3]
    tile_id = f'{tile_numbers[0].zfill(3)}_{tile_numbers[1].zfill(3)}'
    
    gcs_file_path = f"{tile_id}/{band}/{filename}"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    
    if await blob.exists():
        return 'exists'
    
    temp_local_path = f'/tmp/{filename}'
    
    try:
        async with aiohttp.ClientSession() as session:
            success = await download_file(session, filepath, temp_local_path)
        
        if success:
            await asyncio.to_thread(upload_to_gcs, temp_local_path, bucket_name, gcs_file_path)
            return 'success'
        else:
            return 'error'
    except Exception:
        return 'error'
    finally:
        if os.path.exists(temp_local_path):
            os.remove(temp_local_path)

async def gather_cutout_data(parent_dir_source, bucket_name, in_dict, band='cfis_lsb-r', max_workers=20, num_tiles=None):
    print(f"Starting gather_cutout_data with parameters:")
    print(f"parent_dir_source: {parent_dir_source}")
    print(f"bucket_name: {bucket_name}")
    print(f"band: {band}")
    print(f"max_workers: {max_workers}")
    print(f"num_tiles: {num_tiles}")

    client = Client()
    
    print("Listing directories...")
    tile_dirs = client.listdir(parent_dir_source)
    print(f"Found {len(tile_dirs)} directories")
    if num_tiles is not None:
        tile_dirs = tile_dirs[:num_tiles]
        print(f"Limited to {num_tiles} directories")
    
    zfill = in_dict[band]['zfill']
    file_prefix = in_dict[band]['name']
    delimiter = in_dict[band]['delimiter']
    suffix = in_dict[band]['suffix']
    
    download_and_upload_tasks = []
    
    async for tile_dir in tqdm(tile_dirs, desc="Gathering data"):
        tile_nums = tile_dir.split('_')
        num1, num2 = tile_nums[0].zfill(zfill), tile_nums[1].zfill(zfill)
        
        tile_dir_band = os.path.join(parent_dir_source, tile_dir, band)
        if not client.isdir(tile_dir_band):
            continue
        
        expected_filename = f'{file_prefix}{delimiter}{num1}{delimiter}{num2}{suffix}_cutouts.h5'
        filepath = os.path.join(tile_dir_band, expected_filename) 
        
        if not client.isfile(filepath):
            continue
        
        download_and_upload_tasks.append((client, filepath, bucket_name, band))
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    async for result in tqdm(
        asyncio.as_completed([process_file(*task) for task in download_and_upload_tasks]),
        total=len(download_and_upload_tasks),
        desc="Processing files"
    ):
        if result == 'success':
            success_count += 1
        elif result == 'error':
            error_count += 1
        else:  # 'exists'
            skipped_count += 1
    
    print(f"Processing complete. Successes: {success_count}, Errors: {error_count}, Skipped (already exists): {skipped_count}")

async def main():
    parser = argparse.ArgumentParser(description='Upload data to Google Cloud Storage')
    parser.add_argument('--source_dir', type=str, default='arc:/projects/unions/ssl/data/raw/tiles/dwarforge', help='Source directory in VOS')
    parser.add_argument('--bucket_name', type=str, default='unions_cutouts', help='GCS bucket name')
    parser.add_argument('--band', type=str, default='cfis_lsb-r', help='Band to process')
    parser.add_argument('--max_workers', type=int, default=20, help='Maximum number of worker processes')
    parser.add_argument('--num_tiles', type=int, default=None, help='Number of tiles to process (None for all)')
    
    args = parser.parse_args()

    print("Starting main function with arguments:")
    print(args)

    await gather_cutout_data(args.source_dir, bucket_name=args.bucket_name, in_dict=band_dict, 
                             band=args.band, max_workers=args.max_workers, num_tiles=args.num_tiles)

if __name__ == "__main__":
    asyncio.run(main())
