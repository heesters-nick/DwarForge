import logging
import os
import pickle
import time

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm

from logging_setup import setup_logger

setup_logger(
    log_dir='./logs',
    name='file_transfer',
    logging_level=logging.INFO,
)
logger = logging.getLogger()


def upload_to_drive(file_path, folder_id):
    """Upload a single file to a specific Google Drive folder"""
    logger.info(f'Starting upload of {file_path} to folder {folder_id}')

    # Load credentials
    with open('auth/token.pickle', 'rb') as token:
        creds = pickle.load(token)

    # Build the Drive API service
    service = build('drive', 'v3', credentials=creds)

    # Prepare the file metadata
    file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id]}

    # Prepare the media upload with progress tracking
    media = MediaFileUpload(
        file_path,
        resumable=True,
        chunksize=1024 * 1024,  # 1MB chunks
    )

    # Create the file
    logger.info('Starting file upload...')
    request = service.files().create(body=file_metadata, media_body=media, fields='id')

    # Upload with progress tracking
    response = None
    file_size = os.path.getsize(file_path)
    with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
        while response is None:
            status, response = request.next_chunk()
            if status:
                pbar.update(status.resumable_progress - pbar.n)

    logger.info(f"Upload complete! File ID: {response.get('id')}")
    return response.get('id')


# Example usage
if __name__ == '__main__':
    # Replace these with your actual file path and folder ID
    file_path = 'cutouts/lsb_gri.h5'
    folder_id = '1ahQ7_zqLlwVluaw1o_mWyM2A8JV4KaJA'  # find this in the url of the drive folder
    try:
        transfer_start = time.time()
        file_id = upload_to_drive(file_path, folder_id)
        logger.info(
            f'Success! You can view your file at: https://drive.google.com/file/d/{file_id}/view'
        )
        logger.info(f'The transfer took {(time.time()-transfer_start)/60.:.2f} minutes.')
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
