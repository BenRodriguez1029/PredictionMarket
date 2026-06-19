import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import config

"""

This module contains functions to fetch and normalize market data from Polymarket's Gamma API.

"""


# Helpers

def _to_float(val) -> float | None:
    """Safely cast any value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
    

def _parse_iso(val) -> datetime | None:
    """Parse an ISO 8601 string to a tz-aware UTC datetime, or return None."""
    if not val or not isinstance(val, str):
        return None
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_json_list(val: str | None, fallback: list | None = None) -> list:
    """
    Safely parse a JSON-encoded list string.
    Returns fallback (default []) if val is None, empty, or malformed.
    """
    if fallback is None:
        fallback = []
    if not val:
        return fallback
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


def _parse_outcomes(market: dict) -> list[dict]:
    """
    Parse the outcomes and outcomePrices fields from a raw market dict
    into a list of {name, price} dicts.
    """
    names  = _parse_json_list(market.get("outcomes"))
    prices = _parse_json_list(market.get("outcomePrices"))

    outcomes = []
    for name, price in zip(names, prices):
        outcomes.append({
            "name":  name,
            "price": _to_float(price),
        })
    return outcomes


def _compute_mid_and_spread(
    best_bid: float | None,
    best_ask: float | None,
) -> tuple[float | None, float | None]:
    """
    Compute mid price and spread from bid/ask.
    Returns (None, None) if either side is missing.
    """
    if best_bid is None or best_ask is None:
        return None, None
    return (best_bid + best_ask) / 2, best_ask - best_bid


# Fetch

def get_polymarket_data(limit: int = 100) -> pd.DataFrame:
    """
    Fetch open, active, non-archived Polymarket events from the Gamma API.

    Parameters
    ----------
    limit : int
        Maximum number of events to fetch. Paginated in batches of 100.

    Returns
    -------
    pd.DataFrame
        One row per event; each row contains a 'markets' column with a list
        of market dicts.
    """
    url = "https://gamma-api.polymarket.com/events"
    print("connecting to Polymarket API...")

    dfs = []
    for offset in range(0, limit, 100):
        params = {
            "limit":    min(100, limit - offset),
            "offset":   offset,
            "closed":   "false",
            "active":   "true",
            "archived": "false",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching Polymarket data at offset {offset}: {e}")
            break

        if not data:
            break

        dfs.append(pd.DataFrame(data))

    if not dfs:
        print("No data fetched from Polymarket API.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"Successfully fetched {df.shape[0]} events from Polymarket API.")
    return df


# Normalize

def normalize_polymarket_data(df: pd.DataFrame) -> list[dict]:
    """
    Normalize a Polymarket events DataFrame into a flat list of market dicts,
    one dict per market (not per event).

    Each dict matches the shared schema used by normalize_kalshi_data(),
    so ingest() accepts output from either source without modification.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from get_polymarket_data(), one row per event.

    Returns
    -------
    list[dict]
        One dict per market, ready to pass to ingest().
    """
    formatted_markets = []

    for _, row in df.iterrows():
        event_id = row.get("id")

        markets = row.get("markets")
        if not isinstance(markets, list):
            continue

        for market in markets:
            # ── Pricing ───────────────────────────────────────────────────────
            best_bid = _to_float(market.get("bestBid"))
            best_ask = _to_float(market.get("bestAsk"))
            mid_price, spread = _compute_mid_and_spread(best_bid, best_ask)

            # ── Resolution ───────────────────────────────────────────────────
            resolved, resolved_outcome = _determine_resolution(market)

            # ── Outcomes ──────────────────────────────────────────────────────
            outcomes = _parse_outcomes(market)

            # ── Assemble ──────────────────────────────────────────────────────
            formatted_markets.append({
                "source":   "polymarket",

                # IDs
                "event_id":  event_id,
                "market_id": market.get("id"),

                # Metadata
                "title":       market.get("question"),
                "description": market.get("description"),
                "event_time":  _parse_iso(market.get("endDate")),

                # Market state
                "active": market.get("active"),
                "closed": market.get("closed"),

                # Liquidity / volume
                "liquidity":  _to_float(market.get("liquidity")),
                "volume":     _to_float(market.get("volume")),
                "volume_24h": _to_float(market.get("volume24hr")),

                # Resolution
                "resolved":         resolved,            # added
                "resolved_outcome": resolved_outcome,    # added

                # Pricing
                "best_bid":  best_bid,
                "best_ask":  best_ask,
                "mid_price": mid_price,
                "spread":    spread,

                # Outcomes
                "outcomes": outcomes,

                # Ingestion timestamp
                "timestamp": datetime.now(timezone.utc),
            })

    return formatted_markets


# Finaize markets


def _determine_resolution(market: dict) -> tuple[bool, int | None]:
    """
    Determine whether a market cleanly and finally resolved, and to which outcome.

    Requires ALL of:
      1. closed == True
      2. closed at least RESOLUTION_SETTLE_BUFFER_HOURS ago (settlement finalized)
      3. outcomePrices is a clean near-{1,0} pair (RESOLUTION_THRESHOLD)

    Returns:
        (True, 1)  -> YES won
        (True, 0)  -> NO won
        (False, None) -> not closed / too recent / [0,0] / near-0.5 / malformed
    """
    if not market.get("closed"):
        return False, None

    closed_time = _parse_iso(market.get("closedTime"))
    if closed_time is None:
        return False, None
    if datetime.now(timezone.utc) - closed_time < timedelta(hours=config.RESOLUTION_SETTLE_BUFFER_HOURS):
        return False, None

    prices = _parse_json_list(market.get("outcomePrices"))
    floats = [_to_float(p) for p in prices]
    if len(floats) != 2 or None in floats:
        return False, None

    yes_price, no_price = floats[0], floats[1]
    hi = config.RESOLUTION_THRESHOLD
    lo = 1 - hi
    if yes_price >= hi and no_price <= lo:
        return True, 1
    if yes_price <= lo and no_price >= hi:
        return True, 0
    return False, None


def fetch_market_by_id(market_id: str) -> dict | None:
    """
    Fetch a single market's current state from the Gamma API by ID.
    Returns the raw market dict, or None on failure.
    """
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[resolution] failed to fetch {market_id}: {e}")
        return None