import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

"""

Position sizing for the trading system — how much of the bankroll to bet given
an edge. Built on the Kelly criterion: the bet fraction that maximizes long-run
expected log growth of wealth (equivalently, the almost-sure exponential growth
rate along the actual path). Covers the closed-form Kelly fraction, the log-
growth objective it maximizes, a numerical cross-check that the two agree, and
the growth-curve plot. Fractional Kelly, estimation-uncertainty shrinkage, and
the wealth simulation live further down (parts 3+).

Convention: binary market, YES share costs p (a probability). Your estimate of
P(YES) is q. Net odds b = (1 - p) / p (profit per unit staked if YES resolves).
An edge exists when q != p; q > p favors YES, q < p favors NO.

"""


# Part 1 — Kelly fraction and the log-growth objective

def kelly_fraction(q: float, p: float) -> float:
    """
    Optimal fraction of bankroll to bet on a binary market.

    Closed form for buying YES: (q - p) / (1 - p). A complete version detects
    which side carries the edge and returns that side's fraction (the NO side is
    the symmetric case p -> 1 - p, q -> 1 - q). Returns 0 when there's no edge.

    Parameters
    ----------
    q : float
        Your probability that YES resolves.
    p : float
        Market price of a YES share (also the market-implied probability).

    Returns
    -------
    float
        Bankroll fraction to bet. Positive = bet the edge side; 0 = no edge.
    """

    if q == p:
        return 0
    elif q > p:
        return (q-p)/(1-p)
    else:
        return (p-q)/(p)


def expected_log_growth(q: float, p: float, f: float) -> float:
    """
    Expected log growth rate per bet at bet fraction f — the objective Kelly
    maximizes.

        G(f) = q * log(1 + f*b) + (1 - q) * log(1 - f),    b = (1 - p) / p

    With probability q you win and wealth multiplies by (1 + f*b); with
    probability (1 - q) you lose and it multiplies by (1 - f). This is the
    expectation of the log of the per-bet wealth multiplier.

    Parameters
    ----------
    q : float
        Your probability that YES resolves.
    p : float
        Market price of a YES share.
    f : float
        Fraction of bankroll bet (0 to 1). At f -> 1 the loss term -> -inf.

    Returns
    -------
    float
        Expected log growth per bet. Undefined/-inf outside valid f; caller
        should keep f in [0, 1).
    """

    b = (1 - p) / p
    return q*np.log(1 + f*b) + (1 - q)*np.log(1 - f)


