import pandas as pd
from datetime import timedelta
from db.connection import get_conn
from db.queries import get_resolved_markets, get_snapshot_history
import config


def find_observation_snapshot(
    history: pd.DataFrame,
    resolution_time,
    horizon_days: float = None,
    wiggle_hours: float = None,
) -> pd.Series | None:
    """
    From one market's snapshot history, find the snapshot closest to
    horizon_days before resolution_time, within +/- wiggle_hours.

    Returns the matching snapshot row (a Series), or None if no snapshot
    falls inside the tolerance window.

    LOOK-AHEAD GUARD: only considers snapshots strictly before resolution_time.
    """
    if horizon_days is None:
        horizon_days = config.OBSERVATION_HORIZON_DAYS
    if wiggle_hours is None:
        wiggle_hours = config.OBSERVATION_TOLERANCE_HOURS

    if history.empty:
        return None

    target_time = resolution_time - timedelta(days=horizon_days)
    window = timedelta(hours=wiggle_hours)

    # Snapshots within the tolerance window around the target time...
    in_window = history[
        (history["snapshot_at"] >= target_time - window) &
        (history["snapshot_at"] <= target_time + window) &
        (history["snapshot_at"] < resolution_time)   # look-ahead guard
    ]

    if in_window.empty:
        return None

    # return snapshot closest to the target time
    closest_idx = (in_window["snapshot_at"] - target_time).abs().idxmin()
    return in_window.loc[closest_idx]


def build_calibration_data(
    horizon_days: float = None,
    wiggle_hours: float = None,
) -> pd.DataFrame:
    """
    Build the (observation_price, outcome) dataset for calibration analysis.

    For each cleanly-resolved market, finds the snapshot closest to
    horizon_days before resolution (within +/- wiggle_hours) and pairs its
    mid_price with the actual outcome.

    Markets with no snapshot in the tolerance window are dropped.

    Returns a DataFrame: market_id, observation_price, observation_time,
    days_before_resolution, outcome.
    """
    with get_conn() as conn:
        resolved = get_resolved_markets(conn)
        if not resolved:
            print("[calibration] no resolved markets found")
            return pd.DataFrame()

        market_ids = [m["market_id"] for m in resolved]
        history_all = get_snapshot_history(market_ids) 

    # Index resolution info by market_id for lookup
    resolution_by_id = {m["market_id"]: m for m in resolved}

    rows = []
    n_no_window = 0
    for market_id, hist in history_all.groupby("market_id"):
        info = resolution_by_id[market_id]
        snap = find_observation_snapshot(
            hist, info["event_time"], horizon_days, wiggle_hours
        )
        if snap is None:
            n_no_window += 1
            continue

        actual_days_before = (info["event_time"] - snap["snapshot_at"]).total_seconds() / 86400.0
        rows.append({
            "market_id": market_id,
            "observation_price": snap["mid_price"],
            "observation_time": snap["snapshot_at"],
            "days_before_resolution": actual_days_before,
            "outcome": info["resolved_outcome"],
        })

    result = pd.DataFrame(rows)
    print(
        f"[calibration] {len(resolved)} resolved markets, "
        f"{len(result)} with usable observation, "
        f"{n_no_window} dropped (no snapshot in window)"
    )
    return result