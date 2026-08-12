import pandas as pd
import numpy as np
from analysis.time_series import to_logit
import config


def drop_invalid_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing or degenerate bid/ask data.

    Drops: null bid/ask, the bid=0/ask=1 placeholder pattern, and any
    row where bid >= ask (a crossed or zero-width quote, never legitimate).
    """
    before = len(df)
    df = df[
        df["best_bid"].notna() &
        df["best_ask"].notna() &
        (df["best_bid"] > 0) &
        (df["best_ask"] < 1) &
        (df["best_bid"] < df["best_ask"])
    ].copy()
    print(f"[filter] dropped {before - len(df)} rows with invalid quotes")
    return df


def drop_post_resolution(df_all: pd.DataFrame,
                         resolution_times: pd.Series = None) -> pd.DataFrame:
    """
    Drop snapshots taken at or after each market's resolution time. For resolved
    markets these rows are junk (boundary-snapped, no real dynamics) AND a
    look-ahead hazard (price already reflects the known outcome).

    resolution_times : Series indexed by market_id giving each market's
        resolution_time (COALESCE(closed_time, event_time)). If None, fetches
        it via load_resolved_markets for the markets present in df_all.

    Markets with no resolution time (still-open markets) keep all their rows.
    """
    from db.load import load_resolved_markets

    if resolution_times is None:
        resolved = load_resolved_markets()
        resolution_times = resolved.set_index("market_id")["resolution_time"]

    df = df_all.copy()
    # map each row's market to its resolution time (NaT if market not resolved)
    df["_res_time"] = df["market_id"].map(resolution_times)

    before = len(df)
    # keep rows that are either from unresolved markets (_res_time is NaT)
    # or strictly before their resolution time
    keep = df["_res_time"].isna() | (df["snapshot_at"] < df["_res_time"])
    df = df[keep].drop(columns=["_res_time"])

    print(f"[post-resolution] dropped {before - len(df)} rows at/after resolution")
    return df


def filter_consistently_active(df: pd.DataFrame, percent_move_threshold: float = 0.50, median_move_threshold: float = 0.01) -> pd.DataFrame:
    """
    Keep only markets that were active for the entire history of the snapshot DataFrame.
    """
    valid_ids = []
    dropped = 0
    for market_id, g in df.groupby("market_id"):
        logit = to_logit(g["mid_price"].values)
        diff = np.diff(logit)
        if np.mean(np.abs(diff) < 1e-9) > percent_move_threshold and np.median(np.abs(diff)) > median_move_threshold:
            valid_ids.append(market_id)
        else:
            dropped += 1
    
    df = df[df["market_id"].isin(valid_ids)].copy()
    print(f"[filter_consistently_active] dropped {dropped} markets not consistently active")
    return df


def filter_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only markets currently active and not closed.
    """
    before = len(df)
    df = df[(df["active"] == True) & (df["closed"] == False)].copy()
    print(f"[filter] dropped {before - len(df)} inactive/closed rows")
    return df


def filter_closed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only markets currently closed.
    """
    before = len(df)
    df = df[(df["active"] == False) & (df["closed"] == True)].copy()
    print(f"[filter] dropped {before - len(df)} inactive/closed rows")
    return df


def filter_min_volume(df: pd.DataFrame, min_volume: float = None) -> pd.DataFrame:
    """
    Drop markets with 24h volume below a minimum threshold.
    Defaults to config.MIN_VOLUME_24H if not specified.
    """
    if min_volume is None:
        min_volume = config.MIN_VOLUME_24H
    before = len(df)
    df = df[df["volume_24h"] >= min_volume].copy()
    print(f"[filter] dropped {before - len(df)} rows below volume threshold ({min_volume})")
    return df


def clean_markets(df: pd.DataFrame, require_active: bool = True, min_volume: float = None) -> pd.DataFrame:
    """
    Standard cleaning pipeline: invalid quotes, optional active filter, optional volume floor.

    This is the single entry point most analysis notebooks should call.
    """
    df = drop_invalid_quotes(df)
    if require_active:
        df = filter_active(df)
    if min_volume is not None:
        df = filter_min_volume(df, min_volume)
    return df


def filter_extreme_prices(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Drop markets priced very near 0 or 1, where tick-size effects dominate
    relative-spread and relative-price-movement metrics (see research log, 1.4).
    """
    before = len(df)
    df = df[(df["mid_price"] >= lower) & (df["mid_price"] <= upper)].copy()
    print(f"[filter] dropped {before - len(df)} rows with extreme prices")
    return df


def filter_min_history(df: pd.DataFrame, min_snapshots: int = 50) -> pd.DataFrame:
    """
    Keep only markets with at least min_snapshots rows of history.
    Needed before any time-series fitting (Module 3) — short series can't
    support ADF tests or AR(1) fits reliably.
    """
    counts = df.groupby("market_id").size()
    valid_ids = counts[counts >= min_snapshots].index
    before = len(df)
    df = df[df["market_id"].isin(valid_ids)].copy()
    print(f"[filter] dropped {before - len(df)} rows from markets with <{min_snapshots} snapshots")
    return df


def filter_stale_quotes(df: pd.DataFrame, max_age_minutes: int = 60) -> pd.DataFrame:
    """
    Drop snapshots whose quote is old relative to when it was recorded —
    a market with no recent trades may show a bid/ask that no longer
    reflects reality (Module 1 pitfall: stale quotes vs. recency of volume).

    Requires a 'snapshot_at' column and assumes you've joined in something
    indicating last trade time, if your schema tracks that separately.
    """
    raise NotImplementedError(
        "Wire this up once you decide how to measure quote staleness "
        "— e.g. comparing snapshot_at to a last-trade timestamp, if you add one."
    )


def filter_max_spread(df: pd.DataFrame, max_relative_spread: float = None) -> pd.DataFrame:
    """
    Drop markets with relative spread above a maximum threshold.
    Defaults to config.LIQUIDITY_TIERS["wide"] (the untradeable cutoff) if not specified.

    Requires 'relative_spread' to already be a column — compute it first.
    """
    if max_relative_spread is None:
        max_relative_spread = config.LIQUIDITY_TIERS["wide"]
    before = len(df)
    df = df[df["relative_spread"] <= max_relative_spread].copy()
    print(f"[filter] dropped {before - len(df)} rows above spread threshold ({max_relative_spread})")
    return df


def filter_by_tier(df: pd.DataFrame, allowed_tiers: list[str]) -> pd.DataFrame:
    """
    Keep only markets whose liquidity_tier is in allowed_tiers.

    Requires 'liquidity_tier' to already be a column — run tag_liquidity_tiers() first.
    Example: filter_by_tier(df, ["tight", "medium"]) keeps only the two best tiers.
    """
    before = len(df)
    df = df[df["liquidity_tier"].isin(allowed_tiers)].copy()
    print(f"[filter] dropped {before - len(df)} rows outside tiers {allowed_tiers}")
    return df