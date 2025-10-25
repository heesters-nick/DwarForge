import hashlib
import json
import logging
import os
import time

# Disable the file_cache warning
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm

from dwarforge.config import load_settings
from dwarforge.logging_setup import setup_logger

warnings.filterwarnings('ignore', message='file_cache is only supported with oauth2client<4.0.0')

cfg = load_settings('configs/default.yaml')
setup_logger(
    log_dir=str(cfg.paths.log_directory),
    name=cfg.logging.name,
    logging_level=getattr(logging, cfg.logging.level.upper()),
    force=True,
)
logger = logging.getLogger(__name__)


class DriveUploader:
    def __init__(self, token_path='auth/token.json', max_retries=3, checkpoint_dir='./checkpoints'):
        self.token_path = Path(token_path)
        self.max_retries = max_retries
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.file_cache_path = self.checkpoint_dir / 'file_cache.json'
        self.file_cache = self._load_file_cache()

    def _load_file_cache(self):
        """Load the cache of previously uploaded files"""
        if self.file_cache_path.exists():
            with open(self.file_cache_path) as f:
                return json.load(f)
        return {}

    def _save_file_cache(self):
        """Save the cache of uploaded files"""
        with open(self.file_cache_path, 'w') as f:
            json.dump(self.file_cache, f, indent=2)

    def _get_fresh_service(self):
        """Create a fresh service instance"""
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        return build('drive', 'v3', credentials=creds, cache_discovery=False)

    def _get_file_info(self, file_path):
        """Get file information including hash and modification time"""
        stat = os.stat(file_path)
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)

        return {'md5_hash': hash_md5.hexdigest(), 'size': stat.st_size, 'mtime': stat.st_mtime}

    def _get_drive_files_batch(self, folder_id):
        """Get all active (non-trashed) files in the folder in a single API call"""
        try:
            service = self._get_fresh_service()
            files_dict = {}
            page_token = None

            while True:
                # Explicitly check for non-trashed files in the specific folder
                query = f"'{folder_id}' in parents and trashed=false"
                response = (
                    service.files()
                    .list(
                        q=query,
                        spaces='drive',
                        fields='nextPageToken, files(id, name, md5Checksum)',
                        pageToken=page_token,
                    )
                    .execute()
                )

                for file in response.get('files', []):
                    files_dict[file['name']] = {
                        'id': file['id'],
                        'md5Checksum': file.get('md5Checksum'),
                    }

                page_token = response.get('nextPageToken')
                if not page_token:
                    break

            return files_dict
        except Exception as e:
            logger.warning(f'Error getting file list from Drive: {str(e)}')
            return {}

    def _upload_or_update_file(self, file_path, folder_id, existing_files=None, retry_count=0):
        """Upload a new file or update existing file if changed"""

        file_info = self._get_file_info(file_path)
        filename = os.path.basename(file_path)

        try:
            # Check if file exists in Drive and is not trashed
            existing_file = existing_files.get(filename) if existing_files is not None else None

            # Create fresh service instance for upload
            service = self._get_fresh_service()

            media = MediaFileUpload(file_path, resumable=True, chunksize=1024 * 1024)

            if existing_file:
                # If file exists, verify its hash
                drive_md5 = existing_file.get('md5Checksum')
                if drive_md5 and drive_md5 == file_info['md5_hash']:
                    logger.info(f'File {filename} already exists and is identical, skipping.')
                    return {
                        'file_path': file_path,
                        'file_id': existing_file['id'],
                        'status': 'skipped',
                    }
                else:
                    # Update existing file
                    request = service.files().update(
                        fileId=existing_file['id'], media_body=media, fields='id'
                    )
                    operation = 'Updating'
            else:
                # Upload new file
                file_metadata = {'name': filename, 'parents': [folder_id]}
                request = service.files().create(body=file_metadata, media_body=media, fields='id')
                operation = 'Uploading'

            response = None
            with tqdm(
                total=file_info['size'],
                unit='B',
                unit_scale=True,
                desc=f'{operation} {filename}',
                leave=True,
            ) as pbar:
                while response is None:
                    try:
                        status, response = request.next_chunk()
                        if status:
                            pbar.update(status.resumable_progress - pbar.n)
                    except HttpError as e:
                        if retry_count < self.max_retries:
                            logger.warning(
                                f'Retry {retry_count + 1} for {filename} due to: {str(e)}'
                            )
                            time.sleep(2**retry_count)
                            return self._upload_or_update_file(
                                file_path, folder_id, existing_files, retry_count + 1
                            )
                        else:
                            raise e

            # Update cache with new file info
            self.file_cache[file_path] = {
                'file_id': response.get('id'),
                'md5_hash': file_info['md5_hash'],
                'size': file_info['size'],
                'mtime': file_info['mtime'],
                'last_upload': datetime.now().isoformat(),
            }
            self._save_file_cache()

            return {
                'file_path': file_path,
                'file_id': response.get('id'),
                'status': 'updated' if existing_file else 'uploaded',
            }

        except Exception as e:
            logger.error(f'Error processing {filename}: {str(e)}')
            return {'file_path': file_path, 'error': str(e), 'status': 'failed'}

    def batch_upload(self, source_path, folder_id, max_workers=10, file_pattern='*'):
        """Upload or update multiple files"""
        source_path = Path(source_path)
        if not source_path.exists():
            raise ValueError(f'Source path {source_path} does not exist')

        files_to_process = list(source_path.glob(file_pattern))
        if not files_to_process:
            logger.info('No matching files found to process.')
            return []

        # Get existing files in one batch
        logger.info('Checking for existing files in Drive...')
        existing_files = self._get_drive_files_batch(folder_id)

        total_size = sum(os.path.getsize(f) for f in files_to_process)
        logger.info(
            f'Found {len(files_to_process)} files to process (Total: {total_size / 1e9:.2f} GB)'
        )

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._upload_or_update_file, str(file_path), folder_id, existing_files
                ): file_path
                for file_path in files_to_process
            }

            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

                if result['status'] in ['uploaded', 'updated']:
                    logger.info(
                        f'{result["status"].capitalize()}: {result["file_path"]} -> '
                        f'https://drive.google.com/file/d/{result["file_id"]}/view'
                    )

        # Summary
        uploaded = sum(1 for r in results if r['status'] == 'uploaded')
        updated = sum(1 for r in results if r['status'] == 'updated')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'failed')

        logger.info('\nOperation Summary:')
        logger.info(f'Total time: {(time.time() - start_time) / 60:.2f} minutes')
        logger.info(f'New uploads: {uploaded}')
        logger.info(f'Updated files: {updated}')
        logger.info(f'Skipped (unchanged): {skipped}')
        logger.info(f'Failed: {failed}')

        if failed > 0:
            logger.info('\nFailed files:')
            for result in results:
                if result['status'] == 'failed':
                    logger.info(f'{result["file_path"]}: {result["error"]}')

        return results


if __name__ == '__main__':
    folder_id = '1Zy-KmrUP25HkBujNJCy_6QZR3EdJjrng'  # Set the folder ID of the destination folder on Google Drive
    base_dir = '/arc/projects/unions/ssl/data/raw/tiles'
    project_dir = '/arc/home/heestersnick/dwarforge'
    source_directory = os.path.join(project_dir, 'desi')

    try:
        uploader = DriveUploader(
            token_path='auth/token.json',  # Use token.json for authentication
            checkpoint_dir='./upload_checkpoints',
        )
        results = uploader.batch_upload(
            source_path=source_directory,
            folder_id=folder_id,
            max_workers=5,
            file_pattern='*prep_v1.h5',  # Set suffix of files that should be uploaded
        )
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
