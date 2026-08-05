"""Bet settlement and CLV.

1X2 only in v1. Over/under 2.5 and BTTS arrive in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_SELECTIONS = ("H", "D", "A")


def result_from_goals(home_goals: int, away_goals: int) -> str:
    """Map a final score to an H/D/A result."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


@dataclass(frozen=True)
class Settlement:
    status: str  # won | lost | void
    pnl: float  # profit or loss on the stake, excluding the stake return


def settle_1x2(
    selection: str,
    stake: float,
    decimal_odds: float,
    home_goals: int | None,
    away_goals: int | None,
) -> Settlement:
    """Settle a single 1X2 bet.

    A missing score means the match didn't happen (abandoned, postponed past the
    settlement window), which is a void and returns the stake, not a loss.
    """
    if selection not in VALID_SELECTIONS:
        raise ValueError(f"selection must be one of {VALID_SELECTIONS}, got {selection!r}")
    if stake < 0:
        raise ValueError(f"stake must be non-negative, got {stake}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")

    if home_goals is None or away_goals is None:
        return Settlement(status="void", pnl=0.0)

    actual = result_from_goals(home_goals, away_goals)
    if actual == selection:
        return Settlement(status="won", pnl=round(stake * (decimal_odds - 1.0), 2))
    return Settlement(status="lost", pnl=-round(stake, 2))


def clv(odds_taken: float, closing_odds: float | None) -> float | None:
    """Closing line value as a fraction of the price taken.

    ``+0.03`` means you took a price 3% better than the close. Averaged over enough
    bets this is the leading indicator of a real edge: it becomes statistically
    meaningful far sooner than ROI does, because it doesn't have to wait for match
    outcomes to stop being noise.
    """
    if closing_odds is None or closing_odds <= 1.0 or odds_taken <= 1.0:
        return None
    return odds_taken / closing_odds - 1.0
