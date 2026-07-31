import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from db.queries import get_snapshot_history
import config

"""

This module holds time-series analysis: the log-odds (logit) transform
and its inverse, price-history plotting in both spaces, and stationarity testing.
All modeling works in log-odds space, where the bounded [0,1] price
becomes unbounded and increments behave like the natural unit of evidence.

"""


# Logit transform

def to_logit(p, eps: float = None):
    """
    Probability to log-odds: log(p / (1 - p)).

    Clips p to [eps, 1 - eps] to avoid infinities at the boundaries.
    Vectorized — works on scalars, arrays, or pandas Series.

    Parameters
    ----------
    p : float or array-like
        Probability/price in [0, 1].
    eps : float, optional
        Clip bound. Defaults to config.LOGIT_EPS.
    """
    if eps is None:
        eps = config.LOGIT_EPS
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def from_logit(x):
    """
    Log-odds to probability: 1 / (1 + e^-x). Inverse of to_logit.
    """
    return 1 / (1 + np.exp(-x))


# Data helper

def get_logit_series(market_id: str, min_snaps: int = 1) -> pd.DataFrame:
    """
    Fetch a market's usable snapshot history, sorted by time, with a logit column.

    Returns a DataFrame with at least snapshot_at, mid_price, logit — or an
    empty DataFrame if the market has fewer than min_snaps usable snapshots.

    Parameters
    ----------
    market_id : str
        Market to fetch.
    min_snaps : int
        Minimum usable (non-null mid) snapshots required; else returns empty.
    """
    hist = get_snapshot_history([market_id])
    hist = hist[hist["mid_price"].notna()].sort_values("snapshot_at").copy()
    if len(hist) < min_snaps:
        return hist.iloc[0:0]
    hist["logit"] = to_logit(hist["mid_price"])
    return hist


# Plotting

def plot_price_and_logit(market_id: str) -> None:
    """
    Side-by-side plot of a market's history in price space vs. log-odds space.

    The log-odds panel stretches the extremes: movement near 0 or 1 that looks
    compressed in price space becomes clearly visible, reflecting that those
    moves represent large shifts in odds.
    """
    hist = get_logit_series(market_id)
    if hist.empty:
        print(f"no usable snapshots for {market_id}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(hist["snapshot_at"], hist["mid_price"], color="C0")
    ax1.set_title(f"Price space — {market_id}")
    ax1.set_ylabel("Price (probability)")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8)

    ax2.plot(hist["snapshot_at"], hist["logit"], color="C1")
    ax2.set_title("Log-odds space")
    ax2.set_ylabel("logit = log(p / (1 - p))")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)   # logit 0 = price 0.5

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"{len(hist)} snapshots, price range "
          f"[{hist['mid_price'].min():.3f}, {hist['mid_price'].max():.3f}]")


def find_mover_markets(min_snaps: int = 100, lo: float = 0.3, hi: float = 0.7,
                       limit: int = 10) -> pd.DataFrame:
    """
    Find markets whose price ranged widely (dipped below `lo` and rose above `hi`),
    with at least min_snaps snapshots. Useful for picking demo markets where the
    logit stretch is visually obvious.
    """
    from db.connection import get_conn
    sql = """
        SELECT market_id, 
               MIN(mid_price) AS lo, MAX(mid_price) AS hi,
               COUNT(*) AS n
        FROM market_snapshots
        WHERE mid_price IS NOT NULL
        GROUP BY market_id
        HAVING COUNT(*) >= %s AND MIN(mid_price) < %s AND MAX(mid_price) > %s
        ORDER BY COUNT(*) DESC
        LIMIT %s
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(min_snaps, lo, hi, limit))


# Stationarity

def adf_test(series: np.ndarray) -> dict:
    """
    Run the Augmented Dickey-Fuller test on a 1-D series.

    Returns a dict with the ADF statistic, p-value, lags used, n, and a
    boolean 'stationary' (p < 0.05, rejecting the unit root).

    Parameters
    ----------
    series : array-like
        The series to test. Should have genuine variation — near-constant
        series (e.g. a market parked at the boundary) give unreliable results.
    """
    stat, pval, lags, nobs, crit, _ = adfuller(series, autolag="AIC")
    return {
        "adf_stat": stat,
        "p_value": pval,
        "lags": lags,
        "n": nobs,
        "stationary": pval < 0.05,
    }


def adf_report(market_id: str, min_snaps: int = 50) -> None:
    """
    ADF test on a market's log-odds LEVELS and FIRST DIFFERENCES.

    Confirms the expected prediction-market pattern: levels are non-stationary
    (price drifts toward resolution), differences are stationary (step-to-step
    changes wander around zero). This justifies modeling differenced log-odds
    for the rest of Module 3.
    """
    hist = get_logit_series(market_id, min_snaps=min_snaps)
    if hist.empty:
        print(f"{market_id}: fewer than {min_snaps} usable snapshots, skipping")
        return

    logit = hist["logit"].values
    diff = np.diff(logit)

    def _show(series, label):
        r = adf_test(series)
        verdict = "STATIONARY" if r["stationary"] else "non-stationary"
        print(f"  {label}:")
        print(f"    ADF stat = {r['adf_stat']:.3f}   p = {r['p_value']:.4f}   "
              f"-> {verdict} (5% level)")
        print(f"    lags = {r['lags']}, n = {r['n']}")

    print(f"ADF test — market {market_id}  ({len(hist)} snapshots)")
    _show(logit, "log-odds LEVELS")
    _show(diff, "log-odds FIRST DIFFERENCES")


# Batch stationarity

def adf_summary(market_ids: list[str], min_snaps: int = 50) -> pd.DataFrame:
    """
    Run the levels-vs-differences ADF test across many markets and summarize
    how often each is stationary. Confirms the pattern holds broadly, not just
    for one hand-picked market.

    Returns a DataFrame with one row per market plus the two stationarity flags.
    """
    rows = []
    for mid in market_ids:
        hist = get_logit_series(mid, min_snaps=min_snaps)
        if hist.empty:
            continue
        logit = hist["logit"].values
        diff = np.diff(logit)
        rows.append({
            "market_id": mid,
            "n": len(hist),
            "levels_stationary": adf_test(logit)["stationary"],
            "diffs_stationary": adf_test(diff)["stationary"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"{len(df)} markets tested")
        print(f"  levels stationary:      {df['levels_stationary'].mean():.1%}")
        print(f"  differences stationary: {df['diffs_stationary'].mean():.1%}")
        print("  (expect: levels mostly NON-stationary, differences mostly stationary)")
    return df