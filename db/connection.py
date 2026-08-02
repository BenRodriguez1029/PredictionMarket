"""Database connection helpers."""
from contextlib import contextmanager
import psycopg2
import config
from sqlalchemy import create_engine

# SQLAlchemy engine
_engine = None

def get_engine():
    """Return a shared SQLAlchemy engine (created lazily on first use)."""
    global _engine
    if _engine is None:
        url = (
            f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASSWORD}"
            f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        )
        _engine = create_engine(url)
    return _engine

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
