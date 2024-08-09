import logging
import sqlite3

logger = logging.getLogger()


def init_db():
    logger.info('Initializing progress database')
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS processed_tiles
                 (tile_id TEXT, band TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (tile_id, band))""")
    conn.commit()
    conn.close()


def get_unprocessed_jobs(tile_availability, process_band=None, process_all_bands=False):
    logger.info('Checking for unprocessed jobs')
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()

    unprocessed = []
    if process_band:
        # Get tiles available in the specified band
        tiles_to_process = tile_availability.band_tiles(process_band)
    else:
        tiles_to_process = tile_availability.unique_tiles

    for tile in tiles_to_process:
        available_bands, _ = tile_availability.get_availability(tile)
        for band in available_bands:
            if process_band and not process_all_bands and band != process_band:
                continue
            c.execute(
                'SELECT 1 FROM processed_tiles WHERE tile_id = ? AND band = ?', (str(tile), band)
            )
            if c.fetchone() is None:
                unprocessed.append((tile, band))

    conn.close()
    return unprocessed


def update_progress(tile, band):
    logger.info(f'Updating progress for tile {tile} in band {band}')
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute(
        """
        INSERT OR REPLACE INTO processed_tiles (tile_id, band, timestamp)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """,
        (str(tile), band),
    )
    conn.commit()
    conn.close()
