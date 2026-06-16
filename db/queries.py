import psycopg2
import psycopg2.extras
import pandas as pd
from db.connection import get_conn

"""

This module holds all database read/write functions.
Each function takes an open connection as its first argument and does NOT commit —
the caller (pipeline.py) owns the transaction so a failure rolls the whole batch back.

"""


# Writes

def upsert_markets(conn, markets: list[dict]) -> None:
    """
    Insert one row per market; update all mutable fields if it already exists.

    Deduplicates within the incoming batch first (last record per market_id wins),
    since ON CONFLICT cannot handle the same key appearing twice in one statement.

    Parameters
    ----------
    conn : psycopg2 connection
        Open connection; not committed here.
    markets : list[dict]
        Normalized market dicts from the ingest pipeline.
    """
    if not markets:
        return

    # Collapse duplicate market_ids within the batch
    seen = {}
    for m in markets:
        seen[m["market_id"]] = m
    markets = list(seen.values())

    rows = [
        (
            m.get("source"),
            m.get("event_id"),
            m["market_id"],
            m.get("title"),
            m.get("description"),
            m.get("event_time"),
            m.get("active"),
            m.get("closed"),
            m.get("liquidity"),
            m.get("volume"),
            m.get("volume_24h"),
            m.get("best_bid"),
            m.get("best_ask"),
            m.get("mid_price"),
            m.get("spread"),
            m.get("timestamp"),    # written into updated_at
        )
        for m in markets
    ]

    sql = """
        INSERT INTO markets
            (source, event_id, market_id, title, description, event_time,
             active, closed, liquidity, volume, volume_24h,
             best_bid, best_ask, mid_price, spread, updated_at)
        VALUES %s
        ON CONFLICT (market_id) DO UPDATE SET
            source      = EXCLUDED.source,
            event_id    = EXCLUDED.event_id,
            title       = EXCLUDED.title,
            description = EXCLUDED.description,
            event_time  = EXCLUDED.event_time,
            active      = EXCLUDED.active,
            closed      = EXCLUDED.closed,
            liquidity   = EXCLUDED.liquidity,
            volume      = EXCLUDED.volume,
            volume_24h  = EXCLUDED.volume_24h,
            best_bid    = EXCLUDED.best_bid,
            best_ask    = EXCLUDED.best_ask,
            mid_price   = EXCLUDED.mid_price,
            spread      = EXCLUDED.spread,
            updated_at  = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows)


def upsert_outcomes(conn, markets: list[dict]) -> None:
    """
    Flatten each market's nested outcomes into rows and upsert them.

    Deduplicates within the batch by (market_id, outcome_index); on conflict
    the name and price are refreshed.

    Parameters
    ----------
    conn : psycopg2 connection
        Open connection; not committed here.
    markets : list[dict]
        Normalized market dicts, each with an 'outcomes' list.
    """
    # Build rows keyed by (market_id, index) to drop in-batch duplicates
    rows_dict = {}
    for m in markets:
        for idx, outcome in enumerate(m.get("outcomes") or []):
            key = (m["market_id"], idx)
            rows_dict[key] = (
                m["market_id"],
                idx,
                outcome.get("name"),
                outcome.get("price"),
                m.get("timestamp"),
            )

    rows = list(rows_dict.values())
    if not rows:
        return

    sql = """
        INSERT INTO outcomes (market_id, outcome_index, name, price, updated_at)
        VALUES %s
        ON CONFLICT (market_id, outcome_index) DO UPDATE SET
            name       = EXCLUDED.name,
            price      = EXCLUDED.price,
            updated_at = EXCLUDED.updated_at
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows)


def insert_snapshots(conn, markets: list[dict]) -> None:
    """
    Append one snapshot row per market. Never updated — this is the price history.

    Parameters
    ----------
    conn : psycopg2 connection
        Open connection; not committed here.
    markets : list[dict]
        Normalized market dicts for this pull.
    """
    if not markets:
        return

    rows = [
        (
            m["market_id"],
            m.get("best_bid"),
            m.get("best_ask"),
            m.get("mid_price"),
            m.get("spread"),
            m.get("volume_24h"),
            m.get("liquidity"),
            m.get("active"),
            m.get("timestamp"),
        )
        for m in markets
    ]

    sql = """
        INSERT INTO market_snapshots
            (market_id, best_bid, best_ask, mid_price, spread,
             volume_24h, liquidity, active, snapshot_at)
        VALUES %s
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows)



def get_snapshot_history(market_ids: list[str]) -> pd.DataFrame:
    """
    Full snapshot history for a given list of market_ids, ordered by time.

    One row per snapshot, not just the latest — used for time-series plots.
    """
    sql = """
        SELECT market_id, best_bid, best_ask, mid_price, spread,
               volume_24h, liquidity, snapshot_at
        FROM market_snapshots
        WHERE market_id = ANY(%s)
        ORDER BY market_id, snapshot_at
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=(market_ids,))
    return df