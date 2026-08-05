"""Bookmaker margin removal.

Raw implied probabilities from decimal odds sum to more than 1; the excess is the
bookmaker's margin (the "overround"). To compare a model against the market you
need the market's *fair* probabilities, which means deciding how the margin is
distributed across selections.

Two methods:

* ``proportional`` divides every implied probability by the booksum. Simple, and it
  assumes the margin is spread evenly in proportion to price.
* ``shin`` models the margin as arising from a proportion ``z`` of insider money and
  solves for it. It takes relatively more margin off longshots, which is closer to
  how books actually price. Favourite-longshot bias means the proportional method
  systematically overstates fair probability on longshots, so ``shin`` usually gives
  a more honest picture at the long end of our 1.50-4.00 odds range.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def implied_probabilities(odds: Sequence[float]) -> np.ndarray:
    """Raw 1/odds. These sum to 1 + margin, not 1."""
    arr = np.asarray(odds, dtype=float)
    if np.any(arr <= 1.0):
        raise ValueError(f"decimal odds must be > 1.0, got {list(odds)}")
    return 1.0 / arr


def booksum(odds: Sequence[float]) -> float:
    """Sum of raw implied probabilities. 1.05 means a 5% overround."""
    return float(implied_probabilities(odds).sum())


def margin(odds: Sequence[float]) -> float:
    """Bookmaker margin as a fraction, e.g. 0.05 for a 5% book."""
    return booksum(odds) - 1.0


def remove_margin_proportional(odds: Sequence[float]) -> np.ndarray:
    p = implied_probabilities(odds)
    return p / p.sum()


def remove_margin_shin(odds: Sequence[float], tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Shin's (1993) method.

    Solves for the insider proportion ``z`` in

        p_i = (sqrt(z^2 + 4(1-z) * pi_i^2 / B) - z) / (2(1-z))

    where ``pi_i`` are the raw implied probabilities and ``B`` their sum, choosing
    ``z`` so that the fair probabilities sum to 1. Bisection on z in [0, 1) is
    plenty fast here and can't diverge.
    """
    pi = implied_probabilities(odds)
    b = pi.sum()

    if b <= 1.0 + 1e-12:
        # No margin (or a negative one). Nothing to strip beyond normalising.
        return pi / b

    def fair(z: float) -> np.ndarray:
        if z <= 0.0:
            return pi / b
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / b) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.99999
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = fair(mid).sum()
        if abs(s - 1.0) < tol:
            break
        # sum(fair) decreases as z increases
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    p = fair(0.5 * (lo + hi))
    return p / p.sum()  # guard against residual float drift


def remove_margin(odds: Sequence[float], method: str = "proportional") -> np.ndarray:
    """Strip the bookmaker margin. Returns probabilities summing to exactly 1."""
    if method == "proportional":
        return remove_margin_proportional(odds)
    if method == "shin":
        return remove_margin_shin(odds)
    raise ValueError(f"unknown margin method: {method!r} (expected 'proportional' or 'shin')")
