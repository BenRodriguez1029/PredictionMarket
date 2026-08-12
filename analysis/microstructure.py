import pandas as pd
from db.connection import get_engine, get_conn
import config
import numpy as np

def relative_spread(bid, ask):
    """
    Spread as a fraction of mid price. Works on scalars or pandas Series/arrays —
    NumPy/pandas broadcasting handles vectorization automatically.
    """
    mid = (bid + ask) / 2
    return (ask - bid) / mid


def effective_spread_cost(bid, ask):
    """
    Half the spread — the one-way transaction cost of crossing it.
    """
    return (ask - bid) / 2


def liquidity_tier(relative_spread: float) -> str:
    """
    Classify a market's relative spread into a liquidity tier.
    Thresholds come from config.LIQUIDITY_TIERS.
    """
    if pd.isna(relative_spread) or np.isinf(relative_spread):
        return "unknown"
    for tier, upper_bound in config.LIQUIDITY_TIERS.items():
        if relative_spread <= upper_bound:
            return tier
    return "untradeable"

def add_relative_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a relative_spread column = spread / mid_price to a snapshot DataFrame.
    """
    df = df.copy()
    df["relative_spread"] = df["spread"] / df["mid_price"]
    return df


def filter_liquid_markets(df_all: pd.DataFrame,
                          keep_tiers=("tight", "medium")) -> pd.DataFrame:
    """
    Keep only markets whose MEDIAN relative spread falls in an acceptable
    liquidity tier. Returns the flat DataFrame filtered to those markets.
    """
    if "relative_spread" not in df_all:
        df_all = add_relative_spread(df_all)

    keep_tiers = set(keep_tiers)
    med_spread = df_all.groupby("market_id")["relative_spread"].median()
    market_tier = med_spread.apply(liquidity_tier)   # your existing function
    keep_ids = market_tier[market_tier.isin(keep_tiers)].index

    filtered = df_all[df_all["market_id"].isin(keep_ids)].copy()

    n_before = df_all["market_id"].nunique()
    n_after = filtered["market_id"].nunique()
    print(f"[liquidity] {n_before} markets -> {n_after} in tiers "
          f"{sorted(keep_tiers)} ({n_before - n_after} dropped)")
    print(market_tier.value_counts().to_string())
    return filtered
