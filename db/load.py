import pandas as pd
from db.queries import (
    get_markets_by_ids,
    get_markets_by_filter,
    get_snapshots_by_ids,
    get_snapshots_by_filter,
    get_latest_snapshots,
)

"""

This file holds specialized data-loading wrappers. Each function calls a 
core query functions in db.queries with particular arguments.

Wrappers are thin and cheap — add a new one as needed. Query functions stay 
general and unchanged, add specific functions here

"""


# Movement-based loads

def load_movers(min_range: float = 0.02, min_snaps: int = 50,
                columns=None, limit=None) -> pd.DataFrame:
    """
    Snapshots for markets that genuinely moved — price spanned at least
    `min_range` over their life, with at least `min_snaps` usable snapshots.
    Excludes boundary-parked and near-flat markets that have no analyzable
    time-series structure. One DataFrame, ready to group by market_id.
    """
    return get_snapshots_by_filter(
        min_range=min_range,
        min_snaps=min_snaps,
        usable_only=True,
        columns=columns,
        limit=limit
    )


def load_market_history(market_ids: list, columns=None) -> pd.DataFrame:
    """
    Full usable snapshot history for a single market, time-ordered.
    Convenience wrapper for the common single-market case.
    """
    return get_snapshots_by_ids(market_ids, usable_only=True, columns=columns)


def load_resolved_markets(category=None, limit=None) -> pd.DataFrame:
    """
    Cleanly-resolved markets with a usable outcome, plus a resolution_time
    anchor = COALESCE(closed_time, event_time). This is the anchor used to place
    observation windows before resolution (closed_time preferred; event_time is
    only the scheduled end and can be later than actual resolution).

    Returns market metadata with an added resolution_time column.
    """
    df = get_markets_by_filter(
        resolved=True,
        has_outcome=True,
        category=category,
        columns=["market_id", "category", "resolved_outcome",
                 "closed_time", "event_time"],
        limit=limit
    )
    if df.empty:
        return df
    # resolution anchor: prefer actual close time, fall back to scheduled end
    df = df[df["closed_time"].notna() | df["event_time"].notna()].copy()
    df["resolution_time"] = df["closed_time"].fillna(df["event_time"])
    return df


def load_resolved_with_history(category=None, min_snaps: int = 1,
                               columns=None, limit=None) -> pd.DataFrame:
    """
    Snapshots for cleanly-resolved markets. Fetches the snapshot data in one
    query; join against load_resolved_markets() if you also need the outcome
    and resolution_time per market.

    Note: does NOT drop post-resolution snapshots here — that requires the
    per-market resolution_time (see drop_post_resolution in the analysis layer).
    """
    return get_snapshots_by_filter(
        resolved=True,
        has_outcome=True,
        category=category,
        min_snaps=min_snaps,
        usable_only=True,
        columns=columns,
        limit=limit
    )


def load_by_category(category, resolved: bool = None,
                     min_snaps: int = None, columns=None, limit=None) -> pd.DataFrame:
    """
    Snapshots for markets in one category or several. Optionally restrict to
    resolved markets and/or a minimum history length.
    """
    return get_snapshots_by_filter(
        category=category,
        resolved=resolved,
        min_snaps=min_snaps,
        usable_only=True,
        columns=columns,
        limit=limit
    )


def load_recent(days: int, columns=None, limit=None) -> pd.DataFrame:
    """
    All usable snapshots from the last `days` days. Useful for live-signal work
    where only recent data matters.
    """
    from datetime import datetime, timezone, timedelta
    start = datetime.now(timezone.utc) - timedelta(days=days)
    return get_snapshots_by_filter(start=start, usable_only=True, columns=columns, limit=limit)


def load_current_spreads(active_only: bool = True) -> pd.DataFrame:
    """
    Latest bid/ask/mid/spread per market — the most recent snapshot only.
    Defaults to currently active, open markets. Named wrapper for the common
    "current spreads snapshot" intent.
    """
    return get_latest_snapshots(active_only=active_only)