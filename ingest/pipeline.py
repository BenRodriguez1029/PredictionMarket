from db.connection import get_conn
from db.schema import create_tables
from db.queries import get_resolved_market_ids, upsert_markets, upsert_outcomes, insert_snapshots, update_resolutions
from ingest.polymarket import _extract_category, get_polymarket_data, normalize_polymarket_data, fetch_market_by_id, _determine_resolution, _parse_iso
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


        resolved_ids = get_resolved_market_ids(conn)
        fresh = [m for m in markets if m["market_id"] not in resolved_ids]
        skipped = len(markets) - len(fresh)
        insert_snapshots(conn, fresh)

        conn.commit()

    print(f"[db] ingested {len(markets)} markets ({skipped} snapshot-skipped as resolved)")


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




import requests
from db.connection import get_conn
import config
 
 
# ---- category extraction ---------------------------------------------------
 
def _extract_category(event: dict) -> str | None:
    """
    Pick one broad category from an event's tags.
 
    Returns the first label found in config.CATEGORY_PRIORITY; else
    config.CATEGORY_FALLBACK if the event has tags but none match; else
    None if the event has no tags at all.
    """
    tags = event.get("tags") or []
    labels = {t.get("label") for t in tags if t.get("label")}
    if not labels:
        return None
    for category in config.CATEGORY_PRIORITY:
        if category in labels:
            return category
    return config.CATEGORY_FALLBACK
 
 
# ---- event fetching --------------------------------------------------------
 
def _fetch_events_page(offset: int, limit: int, closed: bool, archived: bool) -> list[dict]:
    """Fetch one page of events. Returns [] on error or empty."""
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "limit": limit,
        "offset": offset,
        "closed": str(closed).lower(),
        "archived": str(archived).lower(),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[category] events fetch failed at offset {offset}: {e}")
        return []
 
 
# ---- main backfill ---------------------------------------------------------
 
def backfill_categories(max_events_per_bucket: int = 50000, page_size: int = 100) -> None:
    """
    Backfill markets.category from event tags. Safe to re-run (only touches
    markets where category IS NULL).
    """
    # 1. Which markets still need a category? Store as a set of strings.
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT market_id FROM markets WHERE category IS NULL")
        needed = {str(row[0]) for row in cur.fetchall()}
 
    if not needed:
        print("[category] nothing to backfill — all markets categorized")
        return
    print(f"[category] {len(needed)} markets need a category")
 
    # 2. Scan events across all three visibility buckets, collecting matches.
    updates: dict[str, str] = {}   # market_id -> category
    buckets = [
        ("open",            dict(closed=False, archived=False)),
        ("closed",          dict(closed=True,  archived=False)),
        ("closed+archived", dict(closed=True,  archived=True)),
    ]
 
    for label, flags in buckets:
        print(f"[category] scanning {label} events...")
        offset = 0
        while offset < max_events_per_bucket:
            events = _fetch_events_page(offset, page_size, **flags)
            if not events:
                break
 
            for ev in events:
                category = _extract_category(ev)
                if not category:
                    continue
                for m in (ev.get("markets") or []):
                    mid = str(m.get("id"))
                    if mid in needed and mid not in updates:
                        updates[mid] = category
 
            offset += page_size
            if offset % 1000 == 0:
                print(f"[category]   {label}: scanned {offset} events, "
                      f"{len(updates)} matched so far")
 
    if not updates:
        print("[category] no matching categories found — nothing written")
        return
 
    print(f"[category] matched {len(updates)} markets; writing...")
 
    # 3. Write, counting rows ACTUALLY changed (not attempted).
    sql = "UPDATE markets SET category = %s, updated_at = NOW() WHERE market_id = %s"
    changed = 0
    with get_conn() as conn, conn.cursor() as cur:
        for mid, cat in updates.items():
            cur.execute(sql, (cat, mid))
            changed += cur.rowcount
        conn.commit()
 
    print(f"[category] committed — {changed} rows actually updated "
          f"(matched {len(updates)})")
 
    if changed == 0:
        print("[category] WARNING: 0 rows changed despite matches. "
              "market_id values from events do not equal stored market_id. "
              "Check a sample below.")
 
    # 4. Verify by re-reading the distribution.
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(category, 'NULL') AS category, COUNT(*)
            FROM markets
            GROUP BY category
            ORDER BY COUNT(*) DESC
        """)
        print("[category] current distribution:")
        for cat, n in cur.fetchall():
            print(f"    {cat:20s} {n}")
