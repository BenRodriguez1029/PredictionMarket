import pandas as pd
from db.connection import get_conn
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


def get_current_spreads() -> pd.DataFrame:
    """
    Latest bid/ask/mid/spread for every currently active, open market.

    One row per market — the most recent snapshot only, not full history.
    """
    sql = """
        SELECT DISTINCT ON (s.market_id)
            s.market_id, m.title, s.best_bid, s.best_ask,
            s.mid_price, s.spread, s.volume_24h, s.liquidity,
            s.snapshot_at
        FROM market_snapshots s
        JOIN markets m ON m.market_id = s.market_id
        WHERE m.active = true AND m.closed = false
        ORDER BY s.market_id, s.snapshot_at DESC
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    return df


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