"""Database connection helpers."""
from contextlib import contextmanager
import psycopg2
import config


@contextmanager
def get_conn():
    """Yield a psycopg2 connection, guaranteed to close.

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            print(cur.fetchone())
    """
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    try:
        yield conn
    finally:
        conn.close()
