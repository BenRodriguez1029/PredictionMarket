import psycopg2
import psycopg2.extras
import pandas as pd
from db.connection import get_engine, get_conn

"""

This module holds all database read/write functions.
Each function takes an open connection as its first argument and does NOT commit —
the caller (pipeline.py) owns the transaction so a failure rolls the whole batch back.

"""

# ----Writes----

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
            m.get("category"),
            m.get("event_time"),
            m.get("active"),
            m.get("closed"),
            m.get("closed_time"),
            m.get("resolved"),           
            m.get("resolved_outcome"),  
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
            (source, event_id, market_id, title, description, category, event_time,
             active, closed, closed_time, resolved, resolved_outcome,
             liquidity, volume, volume_24h,
             best_bid, best_ask, mid_price, spread, updated_at)
        VALUES %s
        ON CONFLICT (market_id) DO UPDATE SET
            source      = EXCLUDED.source,
            event_id    = EXCLUDED.event_id,
            title       = EXCLUDED.title,
            description = EXCLUDED.description,
            category    = COALESCE(markets.category, EXCLUDED.category),
            event_time  = EXCLUDED.event_time,
            active      = EXCLUDED.active,
            closed      = EXCLUDED.closed,
            closed_time      = EXCLUDED.closed_time,
            resolved         = EXCLUDED.resolved,
            resolved_outcome = EXCLUDED.resolved_outcome,
            liquidity        = EXCLUDED.liquidity,
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


def update_resolutions(conn, updates: list[tuple]) -> None:
    """
    Mark markets resolved. updates is a list of
    (resolved, outcome, closed_time, market_id) tuples.
    """
    if not updates:
        return
    sql = """
        UPDATE markets
        SET resolved = %s, resolved_outcome = %s, closed_time = %s, updated_at = NOW()
        WHERE market_id = %s
    """
    with conn.cursor() as cur:
        cur.executemany(sql, updates)


# ----Reads----

# General

_MARKET_COLS_ALL = [
    "market_id", "event_id", "title", "description", "category",
    "event_time", "closed_time", "active", "closed", "resolved",
    "resolved_outcome", "liquidity", "volume", "volume_24h",
    "best_bid", "best_ask", "mid_price", "spread",
    "first_seen_at", "updated_at",
]
 
_SNAPSHOT_COLS_ALL = [
    "market_id", "snapshot_at", "best_bid", "best_ask", "mid_price",
    "spread", "volume_24h", "liquidity", "active",
]
 
_MARKET_COLS_DEFAULT = [
    "market_id", "event_id", "title", "description", "category",
    "event_time", "closed_time", "active", "closed", "resolved",
    "resolved_outcome", "liquidity", "volume", "volume_24h",
    "best_bid", "best_ask", "mid_price", "spread",
]

_SNAPSHOT_COLS_DEFAULT = _SNAPSHOT_COLS_ALL


def _resolve_columns(columns, allowed: list[str], default: list[str],
                     prefix: str = "") -> str:
    """
    Validate a requested column list against an allowlist and return a
    comma-separated SQL string. `columns` may be a list, a comma-separated
    string, or None (uses default).
    """
    if columns is None:
        cols = default
    elif isinstance(columns, str):
        cols = [c.strip() for c in columns.split(",") if c.strip()]
    else:
        cols = list(columns)
 
    bad = [c for c in cols if c not in allowed]
    if bad:
        raise ValueError(f"Unknown/disallowed columns: {bad}. Allowed: {allowed}")
 
    return ", ".join(f"{prefix}{c}" for c in cols)


#Markets - by explicit ID list

def get_markets_by_ids(market_ids: list[str], columns=None) -> pd.DataFrame:
    """
    Fetch market metadata rows for an explicit list of market_ids.
 
    Parameters
    ----------
    market_ids : list[str]
        Exact markets to fetch. Empty list returns an empty DataFrame.
    columns : list[str] or str, optional
        Which columns to return. Default: the standard market column set.
    """
    if not market_ids:
        return pd.DataFrame()
    col_sql = _resolve_columns(columns, _MARKET_COLS_ALL, _MARKET_COLS_DEFAULT)
    sql = f"""
        SELECT {col_sql}
        FROM markets
        WHERE market_id = ANY(%s)
    """
    
    return pd.read_sql(sql, get_engine(), params=(list(market_ids),))
 
 
# Markets — by filter
 
def get_markets_by_filter(
    resolved: bool = None,
    category=None,
    closed: bool = None,
    active: bool = None,
    has_outcome: bool = None,
    columns=None,
) -> pd.DataFrame:
    """
    Fetch market metadata rows matching the given criteria. All filters are
    optional; None means "don't filter on this". Filters combine with AND.
 
    Parameters
    ----------
    resolved : bool, optional
        Restrict to resolved (True) or not-yet-resolved (False) markets.
    category : str or list[str], optional
        Restrict to one category or any of several.
    closed : bool, optional
        Restrict by the closed flag.
    active : bool, optional
        Restrict by the active flag.
    has_outcome : bool, optional
        If True, require resolved_outcome IS NOT NULL; if False, IS NULL.
    columns : list[str] or str, optional
        Which columns to return. Default: the standard market column set.
    """
    col_sql = _resolve_columns(columns, _MARKET_COLS_ALL, _MARKET_COLS_DEFAULT)
 
    conditions = []
    params = []
 
    if resolved is not None:
        conditions.append("resolved = %s")
        params.append(resolved)
    if category is not None:
        cats = [category] if isinstance(category, str) else list(category)
        conditions.append("category = ANY(%s)")
        params.append(cats)
    if closed is not None:
        conditions.append("closed = %s")
        params.append(closed)
    if active is not None:
        conditions.append("active = %s")
        params.append(active)
    if has_outcome is not None:
        conditions.append(
            "resolved_outcome IS NOT NULL" if has_outcome
            else "resolved_outcome IS NULL"
        )
 
    where = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"""
        SELECT {col_sql}
        FROM markets
        WHERE {where}
    """
    return pd.read_sql(sql, get_engine(), params=params)
 
 
# Snapshots — by explicit ID list
 
def get_snapshots_by_ids(
    market_ids: list[str],
    start=None,
    end=None,
    usable_only: bool = True,
    columns=None,
) -> pd.DataFrame:
    """
    Fetch snapshot history for an explicit list of market_ids, ordered by
    (market_id, snapshot_at) so it's ready to group and analyze.
 
    Parameters
    ----------
    market_ids : list[str]
        Exact markets to fetch snapshots for.
    start, end : datetime, optional
        Time window on snapshot_at.
    usable_only : bool
        If True, only rows with a non-null mid_price.
    columns : list[str] or str, optional
        Which columns to return. Default: the standard snapshot column set.
    """
    if not market_ids:
        return pd.DataFrame()
 
    col_sql = _resolve_columns(columns, _SNAPSHOT_COLS_ALL,
                               _SNAPSHOT_COLS_DEFAULT, prefix="s.")
 
    conditions = ["s.market_id = ANY(%s)"]
    params = [list(market_ids)]
 
    if usable_only:
        conditions.append("s.mid_price IS NOT NULL")
    if start is not None:
        conditions.append("s.snapshot_at >= %s")
        params.append(start)
    if end is not None:
        conditions.append("s.snapshot_at <= %s")
        params.append(end)
 
    where = " AND ".join(conditions)
    sql = f"""
        SELECT {col_sql}
        FROM market_snapshots s
        WHERE {where}
        ORDER BY s.market_id, s.snapshot_at
    """
    return pd.read_sql(sql, get_engine(), params=params)
 
 
# Snapshots — by filter
 
def get_snapshots_by_filter(
    resolved: bool = None,
    category=None,
    has_outcome: bool = None,
    min_snaps: int = None,
    min_range: float = None,
    start=None,
    end=None,
    usable_only: bool = True,
    columns=None,
) -> pd.DataFrame:
    """
    Fetch snapshots for markets matching the given criteria — filtering markets
    and returning their snapshot data in a SINGLE query (no id round-trip).
 
    Market-level filters (resolved, category, has_outcome, min_snaps, min_range)
    decide WHICH markets; snapshot-level filters (start, end, usable_only) decide
    WHICH rows. Result is ordered by (market_id, snapshot_at).
 
    Parameters
    ----------
    resolved : bool, optional
        Restrict to resolved / unresolved markets.
    category : str or list[str], optional
        Restrict to one or several categories.
    has_outcome : bool, optional
        Require resolved_outcome present (True) or absent (False).
    min_snaps : int, optional
        Market must have at least this many usable snapshots.
    min_range : float, optional
        Market's mid_price must have spanned at least this range.
    start, end : datetime, optional
        Time window on snapshot_at.
    usable_only : bool
        If True, only rows with a non-null mid_price.
    columns : list[str] or str, optional
        Which snapshot columns to return. Default: the standard snapshot set.
 
    Notes
    -----
    Parameter order matters: values are appended in the exact order their
    placeholders appear in the assembled SQL. Do not reorder the blocks below
    without re-checking the params list stays aligned.
    """
    col_sql = _resolve_columns(columns, _SNAPSHOT_COLS_ALL,
                               _SNAPSHOT_COLS_DEFAULT, prefix="s.")
 
    conditions = []
    params = []
 
    # snapshot-level filters (outer query)
    if usable_only:
        conditions.append("s.mid_price IS NOT NULL")
    if start is not None:
        conditions.append("s.snapshot_at >= %s")
        params.append(start)
    if end is not None:
        conditions.append("s.snapshot_at <= %s")
        params.append(end)
 
    # market-level attribute filters (via join)
    if resolved is not None:
        conditions.append("m.resolved = %s")
        params.append(resolved)
    if category is not None:
        cats = [category] if isinstance(category, str) else list(category)
        conditions.append("m.category = ANY(%s)")
        params.append(cats)
    if has_outcome is not None:
        conditions.append(
            "m.resolved_outcome IS NOT NULL" if has_outcome
            else "m.resolved_outcome IS NULL"
        )
 
    # market-level aggregate filters (via subquery)
    if min_snaps is not None or min_range is not None:
        having = []
        if min_snaps is not None:
            having.append("COUNT(*) >= %s")
        if min_range is not None:
            having.append("MAX(mid_price) - MIN(mid_price) >= %s")
        conditions.append(f"""
            s.market_id IN (
                SELECT market_id
                FROM market_snapshots
                WHERE mid_price IS NOT NULL
                GROUP BY market_id
                HAVING {" AND ".join(having)}
            )
        """)
        if min_snaps is not None:
            params.append(min_snaps)
        if min_range is not None:
            params.append(min_range)
 
    where = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"""
        SELECT {col_sql}
        FROM market_snapshots s
        JOIN markets m ON m.market_id = s.market_id
        WHERE {where}
        ORDER BY s.market_id, s.snapshot_at
    """
    return pd.read_sql(sql, get_engine(), params=params)


# Specialized

def get_snapshot_history(market_ids: list[str]) -> pd.DataFrame:
    """
    Full snapshot history for a given list of market_ids, ordered by time.

    One row per snapshot, not just the latest — used for time-series plots.
    """
    sql = """
        SELECT market_id, best_bid, best_ask, mid_price, spread,
               volume_24h, liquidity, snapshot_at, active
        FROM market_snapshots
        WHERE market_id = ANY(%s)
        ORDER BY market_id, snapshot_at
    """

    df = pd.read_sql(sql, get_engine(), params=(market_ids,))
    return df


def get_resolved_markets(conn) -> list[dict]: # Have newer function for this, but keep until we rework old analysis code
    """
    Fetch cleanly-resolved markets with outcome and resolution anchor time.

    resolution_time = closed_time when available, else event_time (fallback).
    closed_time is the actual settlement anchor; event_time is the scheduled
    event conclusion, which can be later than actual resolution.
    """
    sql = """
        SELECT market_id, resolved_outcome,
               COALESCE(closed_time, event_time) AS resolution_time,
               closed_time, event_time, category
        FROM markets
        WHERE resolved = true
          AND resolved_outcome IS NOT NULL
          AND COALESCE(closed_time, event_time) IS NOT NULL
        ORDER BY resolution_time DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    

def get_resolved_market_ids(conn) -> set:
    """
    Set of market_ids already marked resolved — used to skip them in live ingest
    so we stop accumulating post-resolution junk snapshots.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT market_id FROM markets WHERE resolved = true")
        return {row[0] for row in cur.fetchall()}
    