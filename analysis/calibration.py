import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        (history["snapshot_at"] < resolution_time) &
        (history["mid_price"].notna()) &
        (history["active"] == True)  # look-ahead guard
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
            hist, info["resolution_time"], horizon_days, wiggle_hours
        )
        if snap is None:
            n_no_window += 1
            continue

        actual_days_before = (info["resolution_time"] - snap["snapshot_at"]).total_seconds() / 86400.0
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


def calibration_curve(calibration_data: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Bin observations by observation_price and compute the empirical YES rate per bin.

    Returns one row per non-empty bin with:
      bin_mid           - nominal bin center (for the diagonal reference)
      mean_price        - actual mean observation price in the bin (better x than bin_mid)
      empirical_yes     - fraction that resolved YES
      n                 - number of markets in the bin
      se                - standard error of empirical_yes (for error bars)
    """
    df = calibration_data.dropna(subset=["observation_price", "outcome"]).copy()

    edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["observation_price"], bins=edges, include_lowest=True)

    grouped = df.groupby("bin", observed=True).agg(
        mean_price=("observation_price", "mean"),
        empirical_yes=("outcome", "mean"),
        n=("outcome", "size"),
    ).reset_index()

    # Bin midpoint from the interval, and binomial standard error of the YES rate
    grouped["bin_mid"] = grouped["bin"].apply(lambda iv: (iv.left + iv.right) / 2)
    grouped["se"] = np.sqrt(
        grouped["empirical_yes"] * (1 - grouped["empirical_yes"]) / grouped["n"]
    )
    return grouped


def plot_calibration(curve: pd.DataFrame, title: str = "Calibration") -> None:
    """
    Plot observed price vs empirical YES rate, with the perfect-calibration diagonal.
    Point size encodes bin count; error bars show binomial standard error.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect calibration")

    # Empirical points — x is the actual mean price in the bin, not the bin center
    ax.errorbar(
        curve["mean_price"], curve["empirical_yes"],
        yerr=curve["se"], fmt="o", capsize=3, color="C0", label="empirical",
    )

    # Size points by sample count so thin bins are visually de-emphasized
    ax.scatter(
        curve["mean_price"], curve["empirical_yes"],
        s=curve["n"] * 8, alpha=0.3, color="C0",
    )

    # Annotate each point with its n, so you can see which bins to distrust
    for _, row in curve.iterrows():
        ax.annotate(f"n={int(row['n'])}",
                    (row["mean_price"], row["empirical_yes"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Observed Market Price")
    ax.set_ylabel("Empirical YES Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()