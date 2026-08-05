"""Stake sizing: fractional Kelly with a hard cap.

Full Kelly maximises long-run log growth *given that your probability is correct*.
It isn't correct — it's a model estimate — and Kelly is brutally sensitive to
overestimated edge: betting 2x the optimal fraction has zero expected log growth,
and more than that is negative. A 25% fraction plus a 2%-of-bankroll cap is what
makes the sizing survive being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass


def kelly_fraction_full(probability: float, decimal_odds: float) -> float:
    """Full-Kelly fraction of bankroll.

    ``f* = (p*d - 1) / (d - 1)`` — the edge divided by the net odds. Returns 0.0
    when there's no edge, so a negative-EV selection never produces a stake.
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    b = decimal_odds - 1.0
    f = (probability * decimal_odds - 1.0) / b
    return max(0.0, f)


@dataclass(frozen=True)
class StakeRules:
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.02  # of current bankroll
    rounding: float = 0.50
    min_stake: float = 1.00


def round_stake(amount: float, rounding: float) -> float:
    """Round down to a placeable amount.

    Rounding *down* keeps the stake at or under the Kelly cap rather than nudging
    past it.
    """
    if rounding <= 0:
        return amount
    return (amount // rounding) * rounding


def stake_for(
    probability: float,
    decimal_odds: float,
    bankroll: float,
    rules: StakeRules | None = None,
) -> float:
    """Recommended stake in currency units. 0.0 means don't bet.

    Order matters: apply the Kelly fraction, then the cap, then rounding, then the
    minimum. A stake that rounds below ``min_stake`` becomes 0.0 rather than being
    rounded up past its own cap.
    """
    r = rules or StakeRules()
    if bankroll <= 0:
        return 0.0

    f_full = kelly_fraction_full(probability, decimal_odds)
    if f_full <= 0.0:
        return 0.0

    f = min(f_full * r.kelly_fraction, r.max_stake_pct)
    raw = f * bankroll
    stake = round_stake(raw, r.rounding)
    if stake < r.min_stake:
        return 0.0
    return round(stake, 2)
