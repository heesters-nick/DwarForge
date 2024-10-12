import sqlite3
import time
from functools import wraps
from queue import Queue
from threading import Lock

from logging_setup import get_logger

logger = get_logger()


class SQLiteConnectionPool:
    def __init__(self, database, max_connections=5):
        self.database = database
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = Lock()

    def get_connection(self):
        if self.connections.qsize() < self.max_connections and self.connections.empty():
            return sqlite3.connect(self.database)
        return self.connections.get()

    def return_connection(self, connection):
        if self.connections.qsize() < self.max_connections:
            self.connections.put(connection)
        else:
            connection.close()

    def execute(self, query, params=None):
        with self.lock:
            connection = self.get_connection()
            try:
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                result = cursor.fetchall()
                connection.commit()
                return result
            finally:
                self.return_connection(connection)

    def close(self):
        while not self.connections.empty():
            connection = self.connections.get()
            connection.close()
        # In case there are any connections currently in use
        with self.lock:
            for _ in range(self.max_connections):
                connection = self.get_connection()
                connection.close()


def retry_on_db_locked(max_attempts=10, initial_wait=0.1, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            wait_time = initial_wait
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if 'database is locked' in str(e):
                        attempt += 1
                        logger.warning(
                            f'Database locked, attempt {attempt} of {max_attempts}. Retrying in {wait_time:.2f} seconds.'
                        )
                        if attempt == max_attempts:
                            logger.error(f'Max attempts reached. Final error: {str(e)}')
                            raise
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        raise
            return func(*args, **kwargs)  # Final attempt

        return wrapper

    return decorator


def init_connection_pool(database, max_connections):
    return SQLiteConnectionPool(database, max_connections)
