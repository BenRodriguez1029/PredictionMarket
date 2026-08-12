import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from typing import Union, Literal
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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


# Tests 
def resample_market(g: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Resample ONE market's snapshots to a coarser frequency (last value per bucket)."""
    g = g.set_index("snapshot_at").sort_index()
    resampled = g.resample(freq).last()
    # keep only buckets with a full usable quote (mid AND both sides of the book)
    resampled = resampled.dropna(subset=["mid_price", "best_bid", "best_ask"])
    return resampled.reset_index()

def reversion_vs_spread(df_all: pd.DataFrame,
                        min_len: int = 50,
                        horizon: int = 1,
                        std_above: float = 2,
                        acf_threshold: Union[float, Literal["significant"]] = "significant",
                        sampling_freq: str = "1h") -> pd.DataFrame:
    rows = []
    for market_id, g in df_all.groupby("market_id"):
        g = g.sort_values("snapshot_at")
        g = resample_market(g, freq=sampling_freq)

        d = logit_diffs(g, min_len=min_len)
        if len(d) < min_len:
            continue
        thr = 1.96 / np.sqrt(len(d)) if acf_threshold == "significant" else acf_threshold
        a = acf(d, nlags=1, fft=True)[1]
        if a >= -abs(thr):     
            continue

        mid = g["mid_price"].values
        bid = g["best_bid"].values
        ask = g["best_ask"].values

        steps = np.diff(mid)
        move_threshold = np.mean(np.abs(steps)) + std_above * np.std(np.abs(steps))

        captured = []
        for i in range(len(steps) - horizon):
            if abs(steps[i]) <= move_threshold:
                continue
            entry = i + 1              
            exit = entry + horizon
            if exit >= len(mid):
                continue
            if np.isnan(bid[entry]) or np.isnan(ask[entry]) \
               or np.isnan(bid[exit]) or np.isnan(ask[exit]):
                continue

            if steps[i] < 0:
                profit = bid[exit] - ask[entry]
            else:
                profit = bid[entry] - ask[exit]

            captured.append(profit)

        if not captured:
            continue
        captured = np.array(captured)

        rows.append({
            "market_id": market_id,
            "lag1_acf": a,
            "n_events": len(captured),
            "mean_profit": captured.mean(),          # expected profit per trade, after spread
            "median_profit": np.median(captured),
            "frac_profitable": np.mean(captured > 0),
            "median_spread": np.median(g["spread"].values),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no qualifying markets/events")
        return df

    print(f"{len(df)} markets with events")
    print(f"  mean profit per trade (avg over markets): {df['mean_profit'].mean():+.4f}")
    print(f"  fraction of markets with positive mean profit: {(df['mean_profit'] > 0).mean():.1%}")
    return df   


def _lag1_at_frequency(g: pd.DataFrame, freq: str, min_points: int = 20,
                       min_std: float = 1e-8):
    """
    Resample ONE market's logit series to `freq` (e.g. '1h', '4h') using the
    last value in each bin, then return lag-1 autocorrelation of the changes.
    None if too short or degenerate after resampling.
    """
    if g.empty:
        return None
    gg = g.copy()
    if "logit" not in gg:
        gg["logit"] = to_logit(gg["mid_price"].values)
    s = (gg.set_index("snapshot_at")["logit"]
            .resample(freq).last().dropna())
    if len(s) < min_points:
        return None
    d = np.diff(s.values)
    if len(d) < 2 or np.std(d) < min_std:
        return None
    return acf(d, nlags=1, fft=True)[1]
 
 
def autocorr_vs_frequency(df_all: pd.DataFrame,
                          freqs=("15min", "1h", "4h"),
                          min_len: int = 50) -> pd.DataFrame:
    """
    Recompute lag-1 autocorrelation at increasingly coarse sampling frequencies.
    Bid-ask bounce lives at the fine (15-min) frequency and DISSOLVES toward zero
    as you coarsen; genuine economic mean reversion SURVIVES at slower timescales.
 
    df_all is the preloaded flat multi-market DataFrame (needs snapshot_at and
    mid_price/logit). Groups by market_id internally.
 
    Returns a per-market DataFrame with one lag-1 autocorr column per frequency;
    prints the mean at each frequency (the key trend).
    """
    rows = []
    for market_id, g in df_all.groupby("market_id"):
        # require enough fine-grained history to be worth resampling
        if len(g) < min_len:
            continue
        row = {"market_id": market_id}
        for f in freqs:
            row[f] = _lag1_at_frequency(g, f)
        rows.append(row)
 
    df = pd.DataFrame(rows)
    if df.empty:
        print("no markets with enough history")
        return df
 
    print(f"{len(df)} markets — mean lag-1 autocorrelation by sampling frequency:")
    for f in freqs:
        vals = df[f].dropna()
        print(f"  {f:>6}: mean = {vals.mean():+.4f}   (n={len(vals)})")
    print("  -> If it moves toward 0 as frequency coarsens, it's bid-ask bounce.")
    print("  -> If it stays strongly negative at the coarsest, it may be real.")
    return df


# Volatility: 
def rolling_volatility(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Compute rolling volatility (std of log-odds changes) over a sliding window.
 
    df : one market's time-ordered rows, with mid_price (and logit if present).
    window : number of steps in the rolling window (the 'kernel size'). Smaller
             = more responsive to bursts, noisier; larger = smoother regimes.
 
    Returns df with 'logit', 'logit_diff', and 'roll_vol' columns. The first
    `window` rows of roll_vol are NaN (not enough history yet).
    """
    df = df.sort_values("snapshot_at").copy()
    if "logit" not in df:
        from analysis.time_series import to_logit          # reuse your primitive
        df["logit"] = to_logit(df["mid_price"].values)
 
    df["logit_diff"] = df["logit"].diff()
    # rolling std of the changes — the local volatility
    df["roll_vol"] = df["logit_diff"].rolling(window=window).std()
    return df
 
 
def plot_volatility(df: pd.DataFrame, window: int = 15) -> None:
    """
    Two-panel view: price on top, rolling volatility below (shared x-axis).
    Sustained high regions in the lower panel are volatility clusters.
    """
    d = rolling_volatility(df, window=window)
    label = d["market_id"].iloc[0] if "market_id" in d else ""
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
 
    ax1.plot(d["snapshot_at"], d["mid_price"], color="C0")
    ax1.set_ylabel("Price")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Price and rolling volatility (window={window}) — {label}")
 
    ax2.plot(d["snapshot_at"], d["roll_vol"], color="C3")
    ax2.fill_between(d["snapshot_at"], d["roll_vol"], alpha=0.3, color="C3")
    ax2.set_ylabel("Volatility\n(std of log-odds Δ)")
    ax2.set_xlabel("time")
    # mark the market's median volatility so high-vol bursts stand out
    med = d["roll_vol"].median()
    ax2.axhline(med, color="gray", ls="--", lw=0.8, label=f"median={med:.3f}")
    ax2.legend(loc="upper right")
 
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()
 
 
def plot_volatility_colored(df: pd.DataFrame, window: int = 15) -> None:
    """
    The price line itself, colored by local volatility (calm -> hot). Volatility
    clusters appear as colored stretches, showing WHERE the price was choppy in
    the context of its actual movement.
    """
    d = rolling_volatility(df, window=window)
    label = d["market_id"].iloc[0] if "market_id" in d else ""
 
    # matplotlib needs x as a number for the color mapping
    x = np.arange(len(d))
    y = d["mid_price"].values
    vol = d["roll_vol"].values
 
    # build line segments, each colored by the volatility at that point
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
 
    # color scale: clip to a sensible range so a few spikes don't wash it out
    vmax = np.nanpercentile(vol, 95)
    lc = LineCollection(segments, cmap="coolwarm",
                        norm=plt.Normalize(0, vmax if vmax > 0 else 1))
    lc.set_array(vol[:-1])
    lc.set_linewidth(2)
 
    fig, ax = plt.subplots(figsize=(14, 5))
    line = ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Price")
    ax.set_xlabel("snapshot index")
    ax.set_title(f"Price colored by local volatility (window={window}) — {label}")
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("volatility (std of log-odds Δ)")
    plt.tight_layout()
    plt.show()
