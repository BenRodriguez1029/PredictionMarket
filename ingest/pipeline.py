from db.connection import get_conn
from db.schema import create_tables
from db.queries import upsert_markets, upsert_outcomes, insert_snapshots
from ingest.polymarket import get_polymarket_data, normalize_polymarket_data

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


if __name__ == "__main__":
    run_ingest(limit=1000)