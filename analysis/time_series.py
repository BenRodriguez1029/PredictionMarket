import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.ar_model import AutoReg
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


def add_logit(df: pd.DataFrame, price_col: str = "mid_price") -> pd.DataFrame:
    """
    Return a copy of df with a 'logit' column added from price_col.
    Works on single- or multi-market DataFrames (adding a column is
    group-agnostic). Rows with null price get null logit.
    """
    df = df.copy()
    df["logit"] = to_logit(df[price_col].values)
    return df   


# Data helper

def logit_diffs(df: pd.DataFrame, min_len: int = 50, min_std: float = 1e-8) -> np.ndarray:
    """
    First differences of one market's log-odds series (the stationary series
    used for ACF/AR analysis). df is a single market's rows, time-ordered.
 
    Returns an empty array if too short OR degenerate (essentially no price
    movement — a zero-variance series breaks ACF/AR).
    """
    if df.empty:
        return np.array([])
    logit = df["logit"].values if "logit" in df else to_logit(df["mid_price"].values)
    d = np.diff(logit)
    if len(d) < min_len:
        return np.array([])
    if np.std(d) < min_std:
        return np.array([])
    return d


def has_real_movement(df: pd.DataFrame, min_range: float = 0.02,
                      min_changes: int = 10) -> bool:
    """
    Whether one market moved enough to have analyzable time-series structure.
    df is a single market's rows. Filters out boundary-parked / near-flat
    markets that produce degenerate estimates.
    """
    if df.empty:
        return False
    price = df["mid_price"].values
    price_range = price.max() - price.min()
    n_changes = int(np.sum(np.diff(price) != 0))
    return price_range >= min_range and n_changes >= min_changes


# Plotting

def plot_price_and_logit(df: pd.DataFrame) -> None:
    """
    Side-by-side plot of ONE market's history in price space vs. log-odds space.
    df is a single market's time-ordered rows with mid_price (and snapshot_at).
    """
    if df.empty:
        print("empty DataFrame")
        return
    df = add_logit(df) if "logit" not in df else df
    label = df["market_id"].iloc[0] if "market_id" in df else ""
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
 
    ax1.plot(df["snapshot_at"], df["mid_price"], color="C0")
    ax1.set_title(f"Price space — {label}")
    ax1.set_ylabel("Price (probability)")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
 
    ax2.plot(df["snapshot_at"], df["logit"], color="C1")
    ax2.set_title("Log-odds space")
    ax2.set_ylabel("logit = log(p / (1 - p))")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
 
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()
 
    print(f"{len(df)} snapshots, price range "
          f"[{df['mid_price'].min():.3f}, {df['mid_price'].max():.3f}]")


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


def adf_report(df: pd.DataFrame, min_len: int = 50) -> None:
    """
    ADF test on ONE market's log-odds LEVELS and FIRST DIFFERENCES.
    df is a single market's time-ordered rows. Confirms the expected pattern:
    levels non-stationary (drift), differences stationary.
    """
    if df.empty or len(df) < min_len:
        print(f"fewer than {min_len} rows, skipping")
        return
 
    logit = df["logit"].values if "logit" in df else to_logit(df["mid_price"].values)
    diff = np.diff(logit)
    label = df["market_id"].iloc[0] if "market_id" in df else ""
 
    def _show(series, lbl):
        r = adf_test(series)
        verdict = "STATIONARY" if r["stationary"] else "non-stationary"
        print(f"  {lbl}:")
        print(f"    ADF stat = {r['adf_stat']:.3f}   p = {r['p_value']:.4f}   "
              f"-> {verdict} (5% level)")
        print(f"    lags = {r['lags']}, n = {r['n']}")
 
    print(f"ADF test — market {label}  ({len(df)} snapshots)")
    _show(logit, "log-odds LEVELS")
    _show(diff, "log-odds FIRST DIFFERENCES")


# Batch stationarity

def adf_summary(df_all: pd.DataFrame, min_len: int = 50) -> pd.DataFrame:
    """
    Levels-vs-differences ADF test across many markets. df_all is a multi-market
    DataFrame; grouped internally by market_id. Returns one row per market with
    the two stationarity flags and prints the aggregate fractions.
    """
    rows = []
    for market_id, g in df_all.groupby("market_id"):
        if len(g) < min_len:
            continue
        logit = g["logit"].values if "logit" in g else to_logit(g["mid_price"].values)
        diff = np.diff(logit)
        if np.std(diff) < 1e-8:
            continue
        rows.append({
            "market_id": market_id,
            "n": len(g),
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



def plot_acf_pacf(df: pd.DataFrame, nlags: int = 20, min_len: int = 50) -> None:
    """
    ACF and PACF of ONE market's log-odds changes, with 95% significance bands.
    df is a single market's time-ordered rows.
    """
    d = logit_diffs(df, min_len=min_len)
    if len(d) < min_len:
        print(f"only {len(d)} usable diffs, too short/degenerate")
        return
 
    n = len(d)
    band = 2 / np.sqrt(n)
    label = df["market_id"].iloc[0] if "market_id" in df else ""
 
    acf_vals = acf(d, nlags=nlags, fft=True)
    pacf_vals = pacf(d, nlags=nlags)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.stem(np.arange(len(acf_vals)), acf_vals)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(band, color="red", ls="--", lw=0.8)
    ax1.axhline(-band, color="red", ls="--", lw=0.8)
    ax1.set_title(f"ACF of log-odds changes — {label}")
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
 
def lag1_autocorr_distribution(df_all: pd.DataFrame, min_len: int = 50) -> pd.DataFrame:
    """
    Lag-1 autocorrelation of log-odds changes across many markets. df_all is a
    multi-market DataFrame; grouped internally. Returns per-market values and
    prints the aggregate distribution (momentum vs reversion vs efficient).
    """
    rows = []
    for market_id, g in df_all.groupby("market_id"):
        d = logit_diffs(g, min_len=min_len)
        if len(d) < min_len:
            continue
        a = acf(d, nlags=1, fft=True)[1]
        rows.append({"market_id": market_id, "n": len(d), "lag1_acf": a})
 
    df = pd.DataFrame(rows)
    if df.empty:
        print("no markets with enough usable history")
        return df
 
    med_n = df["n"].median()
    band = 2 / np.sqrt(med_n)
    print(f"{len(df)} markets")
    print(f"  mean lag-1 ACF   = {df['lag1_acf'].mean():+.4f}")
    print(f"  median lag-1 ACF = {df['lag1_acf'].median():+.4f}")
    print(f"  fraction positive = {(df['lag1_acf'] > 0).mean():.1%}")
    print(f"  fraction significantly positive (>+{band:.3f}) = "
          f"{(df['lag1_acf'] > band).mean():.1%}  [momentum]")
    print(f"  fraction significantly negative (<-{band:.3f}) = "
          f"{(df['lag1_acf'] < -band).mean():.1%}  [reversion / bounce]")
    print("  NOTE: short-lag negative autocorr may be bid-ask bounce, untradeable.")
    return df

 
 
def plot_lag1_distribution(df: pd.DataFrame) -> None:
    """
    Histogram of per-market lag-1 autocorrelations. df is the output of
    lag1_autocorr_distribution. Unchanged by the refactor (already took a df).
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["lag1_acf"], bins=60, color="C0", alpha=0.8)
    ax.axvline(0, color="gray", ls="--", label="zero (efficient)")
    ax.axvline(df["lag1_acf"].mean(), color="red", ls="-",
               label=f"mean={df['lag1_acf'].mean():+.3f}")
    ax.set_xlabel("lag-1 autocorrelation of log-odds changes")
    ax.set_ylabel("number of markets")
    ax.set_title("Distribution of lag-1 autocorrelation across markets")
    ax.legend()
    plt.tight_layout()
    plt.show()

 
 
# AR(1) fitting — single market
 
def fit_ar1(df: pd.DataFrame, min_len: int = 50) -> dict | None:
    """
    Fit AR(1) to ONE market's log-odds changes. df is a single market's rows.
    Returns phi (coefficient), p-value, significance. phi>0 momentum,
    phi<0 mean reversion, ~0 efficient. None if too short/degenerate.
    """
    d = logit_diffs(df, min_len=min_len)
    if len(d) < min_len:
        return None
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoReg(d, lags=1, old_names=False).fit()
 
    return {
        "n": len(d),
        "phi": model.params[1],
        "p_value": model.pvalues[1],
        "significant": model.pvalues[1] < 0.05,
    }

 
 
def rank_by_predictability(df_all: pd.DataFrame, min_len: int = 50) -> pd.DataFrame:
    """
    Fit AR(1) across many markets and rank by predictability. df_all is a
    multi-market DataFrame; grouped internally. Significant |phi| ranks highest.
    """
    rows = []
    for market_id, g in df_all.groupby("market_id"):
        r = fit_ar1(g, min_len=min_len)
        if r is not None:
            r["market_id"] = market_id
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
    if df['significant'].any():
        print(f"  mean phi (significant): {df.loc[df['significant'], 'phi'].mean():+.4f}")
    print("  phi>0 momentum, phi<0 mean reversion (or bounce)")
    return df

 
 
# AR order comparison — does higher order earn its keep?
 
def compare_ar_orders(df: pd.DataFrame, max_order: int = 5, min_len: int = 50) -> pd.DataFrame:
    """
    Fit AR(1..max_order) on ONE market's log-odds changes; compare by AIC/BIC.
    df is a single market's rows. Lowest-BIC order is parsimony-preferred;
    if 1, AR(1) suffices for this market.
    """
    d = logit_diffs(df, min_len=min_len)
    if len(d) < min_len:
        print("too short/degenerate")
        return pd.DataFrame()
 
    rows = []
    for p in range(1, max_order + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AutoReg(d, lags=p, old_names=False).fit()
        rows.append({"order": p, "aic": m.aic, "bic": m.bic,
                     "last_lag_pval": m.pvalues[-1]})
 
    out = pd.DataFrame(rows)
    label = df["market_id"].iloc[0] if "market_id" in df else ""
    print(f"{label}: best order by BIC = {out.loc[out['bic'].idxmin(), 'order']}, "
          f"by AIC = {out.loc[out['aic'].idxmin(), 'order']}")
    return out

 
 
def ar_order_survey(df_all: pd.DataFrame, max_order: int = 5, min_len: int = 50) -> pd.Series:
    """
    Across many markets, find each market's BIC-preferred AR order and tabulate
    how often each wins. df_all is a multi-market DataFrame; grouped internally.
    Answers empirically whether AR(1) suffices broadly.
    """
    best_orders = []
    for market_id, g in df_all.groupby("market_id"):
        d = logit_diffs(g, min_len=min_len)
        if len(d) < min_len:
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
    if not s.empty:
        print(f"{len(s)} markets surveyed")
        print("BIC-preferred AR order distribution:")
        print((s.value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    return s
