import numpy as np

"""

Correlated-market analysis for two related binary markets (e.g. "Will Messi
score?" = A and "Will Argentina win?" = B). From the two market prices it
reconstructs the full joint distribution of the events, measures how correlated
they are, checks whether a set of prices is mutually consistent (no valid joint
=> mispricing), and turns a view on how B depends on A into a fair price for B.

Events A, B are binary with indicators X_A, X_B in {0,1}. p_a = P(A)
and p_b = P(B) are the market prices. p_ab = P(A and B) is the joint. All inputs
are assumed to be valid probabilities.

"""


# Joint distribution

def joint_table(p_a: float, p_b: float, p_ab: float) -> dict:
    """
    Full 2x2 joint distribution from the two marginals and the joint P(A and B).
    Returns the four cell probabilities. Does NOT validate — a cell can come out
    negative if the inputs are inconsistent (use is_consistent for that).

        p11 = P(A & B)      p10 = P(A & not B)
        p01 = P(not A & B)  p00 = P(not A & not B)
    """
    return {
        "p11": p_ab,                        # A and B
        "p10": p_a - p_ab,                  # A, not B
        "p01": p_b - p_ab,                  # not A, B
        "p00": 1 - p_a - p_b + p_ab,        # neither
    }


# Consistency (Frechet bounds)

def frechet_bounds(p_a: float, p_b: float) -> tuple:
    """
    Range of joint probabilities P(A and B) compatible with the marginals. Any
    p_ab outside [lower, upper] forces a negative joint cell, so no valid joint
    distribution exists — a logical mispricing.

        lower = max(0, p_a + p_b - 1),   upper = min(p_a, p_b)
    """
    return max(0.0, p_a + p_b - 1.0), min(p_a, p_b)


def is_consistent(p_a: float, p_b: float, p_ab: float, tol: float = 1e-9) -> bool:
    """Whether p_ab lies within the Frechet bounds (all joint cells nonnegative)."""
    lo, hi = frechet_bounds(p_a, p_b)
    return (lo - tol) <= p_ab <= (hi + tol)


# Correlation

def implied_correlation(p_a: float, p_b: float, p_ab: float) -> float:
    """
    Correlation (phi coefficient) between the two binary indicators:

        phi = (p_ab - p_a*p_b) / sqrt(p_a(1-p_a) p_b(1-p_b))

    0 = independent (p_ab = p_a*p_b); positive = co-occur more than chance.
    """
    denom = np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    return (p_ab - p_a * p_b) / denom


def joint_from_correlation(p_a: float, p_b: float, phi: float) -> float:
    """
    Inverse of implied_correlation: the joint P(A and B) implied by a correlation
    phi and the two marginals.

        p_ab = p_a*p_b + phi*sqrt(p_a(1-p_a) p_b(1-p_b))
    """
    return p_a * p_b + phi * np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))


def correlation_range(p_a: float, p_b: float) -> tuple:
    """
    Most negative and most positive correlation two binaries can have given their
    marginals (phi evaluated at the Frechet bounds). Reaches +1 only when
    p_a == p_b — the marginals cap how correlated the events can be.
    """
    lo, hi = frechet_bounds(p_a, p_b)
    return implied_correlation(p_a, p_b, lo), implied_correlation(p_a, p_b, hi)


# Conditionals and the law of total probability

def conditionals(p_a: float, p_b: float, p_ab: float) -> dict:
    """
    Conditional probabilities of B given A and given not-A:

        P(B|A)     = p_ab / p_a
        P(B|not A) = (p_b - p_ab) / (1 - p_a)
    """
    return {
        "P(B|A)": p_ab / p_a,
        "P(B|notA)": (p_b - p_ab) / (1 - p_a),
    }


def marginal_from_conditionals(p_a: float, p_b_given_a: float,
                               p_b_given_not_a: float) -> dict:
    """
    Build P(B) and the joint from a trusted P(A) and a view on the two
    conditionals, via the law of total probability:

        P(B)       = P(B|A) P(A) + P(B|not A) (1 - P(A))
        P(A and B) = P(B|A) P(A)

    This turns a relationship view (how B depends on A) plus a trusted P(A) into a
    fair price for B — the edge you compare against B's market price.
    """
    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
    p_ab = p_b_given_a * p_a
    return {"p_b": p_b, "p_ab": p_ab}


# Reasonableness report

def reasonableness_report(p_a: float, p_b: float, p_ab: float,
                          expected_sign: str = None) -> dict:
    """
    Run every check on a (p_a, p_b, p_ab) triple and print a readable summary:
    Frechet bounds and consistency, the joint table, implied correlation and its
    feasible range, the conditionals and whether they move monotonically, and the
    comparison to independence. expected_sign ("positive"/"negative") flags a
    correlation whose sign contradicts your prior. Returns the key quantities.
    """
    lo, hi = frechet_bounds(p_a, p_b)
    consistent = is_consistent(p_a, p_b, p_ab)
    table = joint_table(p_a, p_b, p_ab)
    phi = implied_correlation(p_a, p_b, p_ab)
    cmin, cmax = correlation_range(p_a, p_b)
    cond = conditionals(p_a, p_b, p_ab)
    indep = p_a * p_b

    print(f"marginals: P(A)={p_a:.3f}  P(B)={p_b:.3f}")
    print(f"joint:     P(A&B)={p_ab:.3f}   (independence would give {indep:.3f})")
    print(f"Frechet bounds for P(A&B): [{lo:.3f}, {hi:.3f}]  ->  "
          f"{'CONSISTENT' if consistent else 'INCONSISTENT (mispricing)'}")

    if consistent:
        print(f"joint table:  A&B={table['p11']:.3f}   A&notB={table['p10']:.3f}   "
              f"notA&B={table['p01']:.3f}   neither={table['p00']:.3f}")
        print(f"correlation phi = {phi:+.3f}   (feasible range "
              f"[{cmin:+.3f}, {cmax:+.3f}])")
        print(f"conditionals: P(B|A)={cond['P(B|A)']:.3f}   P(B)={p_b:.3f}   "
              f"P(B|notA)={cond['P(B|notA)']:.3f}")
        if cond["P(B|A)"] >= p_b >= cond["P(B|notA)"]:
            print("  monotonic: positive dependence (A makes B more likely)")
        elif cond["P(B|A)"] <= p_b <= cond["P(B|notA)"]:
            print("  monotonic: negative dependence (A makes B less likely)")
        else:
            print("  non-monotonic")
        if expected_sign == "positive" and phi < 0:
            print("  FLAG: correlation is negative but you expected positive")
        if expected_sign == "negative" and phi > 0:
            print("  FLAG: correlation is positive but you expected negative")

    return {"consistent": consistent, "phi": phi, "joint": table,
            "conditionals": cond, "frechet": (lo, hi)}