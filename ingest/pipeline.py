from db.connection import get_conn
from db.schema import create_tables
from db.queries import upsert_markets, upsert_outcomes, insert_snapshots, update_resolutions
from ingest.polymarket import get_polymarket_data, normalize_polymarket_data, fetch_market_by_id, _determine_resolution, _parse_iso
import config 

"""

This module orchestrates ingestion. Two entry points:
    - ingest(markets):    write already-normalized markets to the DB
    - run_ingest(limit):  fetch + normalize + write, the full end-to-end pull

"""


# Database write

def ingest(markets: list[dict]) -> None:
    """
    Write normalized markets to all three tables in a single transaction.

    Upserts markets and outcomes, appends snapshots, then commits once.
    Sharing one connection means a failure anywhere rolls the whole batch back,
    so the DB never ends up with markets but no matching outcomes.

    Parameters
    ----------
    markets : list[dict]
        Normalized market dicts from normalize_polymarket_data().
    """
    if not markets:
        print("[db] nothing to ingest")
        return
    
    # erase duplicates
    seen = {}
    for m in markets:
        seen[m["market_id"]] = m
    markets = list(seen.values())


    with get_conn() as conn:
        upsert_markets(conn, markets)
        upsert_outcomes(conn, markets)
        insert_snapshots(conn, markets)
        conn.commit()

    print(f"[db] ingested {len(markets)} markets")


# Full pipeline

def run_ingest(limit: int = 1000) -> None:
    """
    Run the full pipeline: ensure tables exist, fetch, normalize, write.

    This is the single entry point for a scheduled pull.

    Parameters
    ----------
    limit : int
        Maximum number of events to fetch from the Gamma API.
    """
    # Make sure the schema exists before the first write
    with get_conn() as conn:
        create_tables(conn)
        conn.commit()

    df = get_polymarket_data(limit=limit)
    if df.empty:
        print("[ingest] no data returned from API")
        return

    markets = normalize_polymarket_data(df)
    ingest(markets)
    return len(markets)


def check_resolutions(batch_limit: int = 500) -> None:
    """
    For tracked markets not yet resolved, re-fetch their state and update
    resolved/resolved_outcome if they've now cleanly resolved.

    Pre-filters by event_time before hitting the API:
      - skips markets too recent to have settled (within the settle buffer)
      - skips markets so old that non-resolution implies junk (> max age)
    """
    sql_select = """
        SELECT market_id FROM markets
        WHERE (resolved IS NULL OR resolved = false)
          AND event_time IS NOT NULL
          AND event_time <= NOW() - (INTERVAL '1 hour' * %s)
          AND event_time >= NOW() - (INTERVAL '1 day'  * %s)
        ORDER BY event_time DESC
        LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_select, (
            config.RESOLUTION_SETTLE_BUFFER_HOURS,
            config.RESOLUTION_MAX_AGE_DAYS,
            batch_limit,
        ))
        market_ids = [row[0] for row in cur.fetchall()]

    if not market_ids:
        print("[resolution] nothing to check")
        return

    newly_resolved = []
    for i, market_id in enumerate(market_ids, 1):
        raw = fetch_market_by_id(market_id)
        if raw is None:
            continue
        resolved, outcome = _determine_resolution(raw)
        if resolved:
            closed_time = _parse_iso(raw.get("closedTime"))   # added
            newly_resolved.append((resolved, outcome, closed_time, market_id))
        if i % 100 == 0:
            print(f"[resolution] {i}/{len(market_ids)} checked, {len(newly_resolved)} resolved so far")

    with get_conn() as conn:
        update_resolutions(conn, newly_resolved)
        conn.commit()

    print(f"[resolution] checked {len(market_ids)}, marked {len(newly_resolved)} resolved")

    return market_ids, newly_resolved


if __name__ == "__main__":
    run_ingest(limit=1000)
