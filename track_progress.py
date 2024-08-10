import datetime
import logging
import sqlite3

logger = logging.getLogger()


def init_db():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS processed_tiles
                 (tile_id TEXT, 
                  band TEXT, 
                  start_time DATETIME,
                  end_time DATETIME,
                  status TEXT,
                  error_message TEXT,
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


def update_progress(tile, band, status, db_lock, error_message=None):
    with db_lock:
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        now = datetime.now().isoformat()  # type: ignore
        if status == 'started':
            c.execute(
                """
                INSERT OR REPLACE INTO processed_tiles (tile_id, band, start_time, status)
                VALUES (?, ?, ?, ?)
            """,
                (str(tile), band, now, status),
            )
        elif status in ['completed', 'failed']:
            c.execute(
                """
                UPDATE processed_tiles 
                SET end_time = ?, status = ?, error_message = ?
                WHERE tile_id = ? AND band = ?
            """,
                (now, status, error_message, str(tile), band),
            )
        conn.commit()
        conn.close()


def get_progress_summary():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status = 'started' THEN 1 ELSE 0 END) as in_progress
        FROM processed_tiles
    """)
    result = c.fetchone()
    conn.close()
    return result
