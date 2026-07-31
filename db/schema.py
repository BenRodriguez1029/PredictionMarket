from db.connection import get_conn

"""

This module defines the database schema and a one-shot setup function.
All statements use IF NOT EXISTS, so setup is idempotent and safe to re-run.

"""


# Schema

def create_tables(conn) -> None:
    """
    Create all tables and indexes on the given connection.

    Does not commit — the caller owns the transaction boundary.

    Parameters
    ----------
    conn : psycopg2 connection
        An open connection. The caller is responsible for commit/close.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id              SERIAL PRIMARY KEY,
                source          TEXT        NOT NULL DEFAULT 'polymarket',
                event_id        TEXT,
                market_id       TEXT        NOT NULL UNIQUE,
                title           TEXT,
                description     TEXT,
                category        TEXT,
                event_time      TIMESTAMPTZ,
                active          BOOLEAN,
                closed          BOOLEAN,   
                closed_time       TIMESTAMPTZ, 
                resolved          BOOLEAN,
                resolved_outcome  INT,
                liquidity       NUMERIC(18,6),
                volume          NUMERIC(18,6),
                volume_24h      NUMERIC(18,6),
                best_bid        NUMERIC(10,6),
                best_ask        NUMERIC(10,6),
                mid_price       NUMERIC(10,6),
                spread          NUMERIC(10,6),
                first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                id              SERIAL PRIMARY KEY,
                market_id       TEXT        NOT NULL
                                    REFERENCES markets(market_id) ON DELETE CASCADE,
                outcome_index   INT         NOT NULL,
                name            TEXT        NOT NULL,
                price           NUMERIC(10,6),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (market_id, outcome_index)
            );

            CREATE TABLE IF NOT EXISTS market_snapshots (
                id              SERIAL PRIMARY KEY,
                market_id       TEXT        NOT NULL,
                best_bid        NUMERIC(10,6),
                best_ask        NUMERIC(10,6),
                mid_price       NUMERIC(10,6),
                spread          NUMERIC(10,6),
                volume_24h      NUMERIC(18,6),
                liquidity       NUMERIC(18,6),
                active          BOOLEAN,
                snapshot_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_markets_market_id
                ON markets(market_id);
            CREATE INDEX IF NOT EXISTS idx_markets_event_id
                ON markets(event_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_market_id
                ON outcomes(market_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_market_id
                ON market_snapshots(market_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_time
                ON market_snapshots(snapshot_at DESC);
        """)


# Setup

def setup_db() -> None:
    """
    Create all tables, then commit. Safe to call repeatedly.
    
    """
    with get_conn() as conn:
        create_tables(conn)
        conn.commit()
    print("[db] tables ready")


if __name__ == "__main__":
    setup_db()
    print("Setup complete")