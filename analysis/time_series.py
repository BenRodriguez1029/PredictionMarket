import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from db.connection import get_engine, get_conn
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


def logit_diffs(market_id: str, min_snaps: int = 50, min_std: float = 1e-8) -> np.ndarray:
    """
    First differences of a market's log-odds series. Returns an empty array if
    too short OR if the series is degenerate (essentially constant — no price
    movement), since ACF/AR are undefined on a zero-variance series.
    """
    hist = get_logit_series(market_id, min_snaps=min_snaps)
    if hist.empty:
        return np.array([])
    d = np.diff(hist["logit"].values)
    if len(d) < min_snaps:
        return np.array([])
    if np.std(d) < min_std:          # degenerate: price essentially never moved
        return np.array([])
    return d


def has_real_movement(market_id: str, min_snaps: int = 50,
                      min_range: float = 0.02, min_changes: int = 10) -> bool:
    """
    Whether a market moved enough to have analyzable time-series structure.
    Filters out boundary-parked and near-flat markets that produce degenerate
    ACF/AR results.
    """
    hist = get_logit_series(market_id, min_snaps=min_snaps)
    if hist.empty:
        return False
    price = hist["mid_price"].values
    price_range = price.max() - price.min()
    n_changes = int(np.sum(np.diff(price) != 0))
    return price_range >= min_range and n_changes >= min_changes


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

    return pd.read_sql(sql, get_engine(), params=(min_snaps, lo, hi, limit))


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


def plot_acf_pacf(market_id: str, nlags: int = 20, min_snaps: int = 50) -> None:
    """
    Plot ACF and PACF of a market's log-odds changes side by side, with 95%
    significance bands. ACF decaying while PACF cuts off sharply after a few
    lags indicates AR structure; the PACF cutoff suggests the AR order.
    """
    d = logit_diffs(market_id, min_snaps=min_snaps)
    if len(d) < min_snaps:
        print(f"{market_id}: only {len(d)} diffs, too short")
        return
 
    n = len(d)
    band = 2 / np.sqrt(n)   # ~95% band for 'no autocorrelation'
 
    acf_vals = acf(d, nlags=nlags, fft=True)
    pacf_vals = pacf(d, nlags=nlags)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    lags = np.arange(len(acf_vals))
 
    ax1.stem(lags, acf_vals)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(band, color="red", ls="--", lw=0.8)
    ax1.axhline(-band, color="red", ls="--", lw=0.8)
    ax1.set_title(f"ACF of log-odds changes — {market_id}")
    ax1.set_xlabel("lag"); ax1.set_ylabel("autocorrelation")
 
    ax2.stem(np.arange(len(pacf_vals)), pacf_vals)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(band, color="red", ls="--", lw=0.8)
    ax2.axhline(-band, color="red", ls="--", lw=0.8)
    ax2.set_title("PACF (direct effect per lag)")
    ax2.set_xlabel("lag"); ax2.set_ylabel("partial autocorrelation")
 
    plt.tight_layout()
    plt.show()
 
    print(f"n={n} changes, significance band = +/- {band:.3f}")
    print(f"lag-1 ACF = {acf_vals[1]:.3f}   lag-1 PACF = {pacf_vals[1]:.3f}")
 
 
# Autocorrelation — aggregate across markets 
 
def lag1_autocorr_distribution(market_ids: list[str], min_snaps: int = 50) -> pd.DataFrame:
    """
    Compute lag-1 autocorrelation of log-odds changes for many markets and
    summarize the distribution. One market's ACF is noisy; the distribution
    across hundreds reveals whether Polymarket changes tend to momentum
    (positive), mean-revert (negative), or look efficient (centered on zero).
 
    Returns a per-market DataFrame; prints the aggregate summary.
    """
    rows = []
    for mid in market_ids:
        d = logit_diffs(mid, min_snaps=min_snaps)
        if len(d) < min_snaps:
            continue
        a = acf(d, nlags=1, fft=True)[1]
        rows.append({"market_id": mid, "n": len(d), "lag1_acf": a})
 
    df = pd.DataFrame(rows)
    if df.empty:
        print("no markets with enough history")
        return df
 
    mean_a = df["lag1_acf"].mean()
    median_a = df["lag1_acf"].median()
    frac_pos = (df["lag1_acf"] > 0).mean()
    # a rough per-market significance band using the median n
    med_n = df["n"].median()
    band = 2 / np.sqrt(med_n)
    frac_sig_neg = (df["lag1_acf"] < -band).mean()
    frac_sig_pos = (df["lag1_acf"] > band).mean()
 
    print(f"{len(df)} markets")
    print(f"  mean lag-1 ACF   = {mean_a:+.4f}")
    print(f"  median lag-1 ACF = {median_a:+.4f}")
    print(f"  fraction positive = {frac_pos:.1%}  (>50% => momentum tilt, <50% => reversion tilt)")
    print(f"  fraction significantly positive (>+{band:.3f}) = {frac_sig_pos:.1%}  [momentum]")
    print(f"  fraction significantly negative (<-{band:.3f}) = {frac_sig_neg:.1%}  [reversion / bounce]")
    print("  NOTE: short-lag negative autocorr may be bid-ask bounce, not tradeable reversion.")
    return df
 
 
def plot_lag1_distribution(df: pd.DataFrame) -> None:
    """Histogram of per-market lag-1 autocorrelations, with zero marked."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["lag1_acf"], bins=60, color="C0", alpha=0.8)
    ax.axvline(0, color="gray", ls="--", label="zero (efficient)")
    ax.axvline(df["lag1_acf"].mean(), color="red", ls="-", label=f"mean={df['lag1_acf'].mean():+.3f}")
    ax.set_xlabel("lag-1 autocorrelation of log-odds changes")
    ax.set_ylabel("number of markets")
    ax.set_title("Distribution of lag-1 autocorrelation across markets")
    ax.legend()
    plt.tight_layout()
    plt.show()
 
 
# AR(1) fitting — single market
 
def fit_ar1(market_id: str, min_snaps: int = 50) -> dict | None:
    """
    Fit an AR(1) model to a market's log-odds changes.
 
    Returns the AR(1) coefficient (phi), its p-value, and whether it's
    significant. phi > 0 = momentum, phi < 0 = mean reversion, ~0 = efficient.
    Returns None if too short.
    """
    d = logit_diffs(market_id, min_snaps=min_snaps)
    if len(d) < min_snaps:
        return None
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoReg(d, lags=1, old_names=False).fit()
 
    # params: [const, L1]; pvalues aligned
    phi = model.params[1]
    pval = model.pvalues[1]
    return {
        "market_id": market_id,
        "n": len(d),
        "phi": phi,
        "p_value": pval,
        "significant": pval < 0.05,
    }
 
 
def rank_by_predictability(market_ids: list[str], min_snaps: int = 50) -> pd.DataFrame:
    """
    Fit AR(1) across many markets and rank by predictability. Markets with a
    significant AR(1) coefficient have the most exploitable time-series structure.
 
    Returns a DataFrame sorted by |phi| among significant markets first.
    """
    rows = []
    for mid in market_ids:
        r = fit_ar1(mid, min_snaps=min_snaps)
        if r is not None:
            rows.append(r)
 
    df = pd.DataFrame(rows)
    if df.empty:
        print("no markets fit")
        return df
 
    df["abs_phi"] = df["phi"].abs()
    df = df.sort_values(["significant", "abs_phi"], ascending=[False, False])
 
    print(f"{len(df)} markets fit")
    print(f"  significant AR(1): {df['significant'].mean():.1%}")
    print(f"  mean phi (all):        {df['phi'].mean():+.4f}")
    print(f"  mean phi (significant): {df.loc[df['significant'], 'phi'].mean():+.4f}")
    print("  phi>0 momentum, phi<0 mean reversion (or bounce)")
    return df
 
 
# AR order comparison — does higher order earn its keep?
 
def compare_ar_orders(market_id: str, max_order: int = 5, min_snaps: int = 50) -> pd.DataFrame:
    """
    Fit AR(1) through AR(max_order) on a market's log-odds changes and compare
    them by AIC/BIC. The order with the lowest BIC is the parsimony-preferred
    model; if that's 1, AR(1) genuinely suffices for this market.
 
    BIC penalizes extra parameters more heavily than AIC, so it's the stricter
    guard against overfitting higher orders.
    """
    d = logit_diffs(market_id, min_snaps=min_snaps)
    if len(d) < min_snaps:
        print(f"{market_id}: too short")
        return pd.DataFrame()
 
    rows = []
    for p in range(1, max_order + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AutoReg(d, lags=p, old_names=False).fit()
        rows.append({"order": p, "aic": m.aic, "bic": m.bic,
                     "last_lag_pval": m.pvalues[-1]})
 
    df = pd.DataFrame(rows)
    best_bic = df.loc[df["bic"].idxmin(), "order"]
    best_aic = df.loc[df["aic"].idxmin(), "order"]
    print(f"{market_id}: best order by BIC = {best_bic}, by AIC = {best_aic}")
    print("  (best BIC = 1 means AR(1) suffices; last_lag_pval > 0.05 means the "
          "highest lag added nothing)")
    return df
 
 
def ar_order_survey(market_ids: list[str], max_order: int = 5, min_snaps: int = 50) -> pd.DataFrame:
    """
    Across many markets, find each market's BIC-preferred AR order and tabulate
    how often each order wins. Answers empirically: does AR(1) suffice broadly,
    or do many markets genuinely need higher orders?
    """
    best_orders = []
    for mid in market_ids:
        d = logit_diffs(mid, min_snaps=min_snaps)
        if len(d) < min_snaps:
            continue
        bics = []
        for p in range(1, max_order + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = AutoReg(d, lags=p, old_names=False).fit()
                bics.append((p, m.bic))
            except Exception:
                continue
        if bics:
            best_orders.append(min(bics, key=lambda t: t[1])[0])
 
    s = pd.Series(best_orders, name="best_bic_order")
    print(f"{len(s)} markets surveyed")
    print("BIC-preferred AR order distribution:")
    print((s.value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    print("\nIf order 1 dominates, AR(1) is empirically justified. If higher "
          "orders are common, your intuition holds and AR(1) underfits.")
    return s
