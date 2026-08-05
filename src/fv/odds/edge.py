"""Edge calculation and selection qualification."""

from __future__ import annotations

from dataclasses import dataclass


def edge(probability: float, decimal_odds: float) -> float:
    """Expected profit per unit staked.

    ``edge = p * d - 1``. At p = 0.40 and d = 3.00 this is 0.20: a 20% expected
    return on stake. Zero means the price is exactly fair against your probability.
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    return probability * decimal_odds - 1.0


@dataclass(frozen=True)
class Thresholds:
    min_edge: float = 0.04
    min_odds: float = 1.50
    max_odds: float = 4.00


def qualifies(
    probability: float,
    decimal_odds: float,
    thresholds: Thresholds | None = None,
) -> bool:
    """A selection qualifies on edge *and* on being inside the odds range.

    The odds band is not arbitrary: below 1.50 the Kelly stake gets large relative
    to any modelling error, and above 4.00 both the model and the market are poorly
    calibrated, so an apparent edge is more likely to be model error than value.
    """
    t = thresholds or Thresholds()
    if not (t.min_odds <= decimal_odds <= t.max_odds):
        return False
    return edge(probability, decimal_odds) >= t.min_edge
