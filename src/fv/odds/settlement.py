"""Bet settlement and CLV.

Two markets settle here: 1X2 (home/draw/away) and over/under 2.5 goals. Both are
read off the same final score, so a bet carries the market it was struck on and
:func:`settle_bet` dispatches on it. Anything else is refused rather than guessed —
settling a bet the wrong way silently corrupts the whole performance record.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_SELECTIONS = ("H", "D", "A")
OU_SELECTIONS = ("O", "U")
MARKETS = ("1X2", "OU25")


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


def settle_ou25(
    selection: str,
    stake: float,
    decimal_odds: float,
    home_goals: int | None,
    away_goals: int | None,
    line: float = 2.5,
) -> Settlement:
    """Settle an over/under bet on total goals.

    The line is 2.5 by definition of the market football-data publishes, so it can
    never land exactly on the total and there is no push case to handle. A whole
    line such as 3.0 would need one, which is why the line is a parameter that
    raises rather than a constant that quietly assumes.
    """
    if selection not in OU_SELECTIONS:
        raise ValueError(f"selection must be one of {OU_SELECTIONS}, got {selection!r}")
    if float(line) == int(line):
        raise ValueError(f"whole-number lines can push and are not supported: {line}")
    if stake < 0:
        raise ValueError(f"stake must be non-negative, got {stake}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")

    if home_goals is None or away_goals is None:
        return Settlement(status="void", pnl=0.0)

    total = int(home_goals) + int(away_goals)
    won = total > line if selection == "O" else total < line
    if won:
        return Settlement(status="won", pnl=round(stake * (decimal_odds - 1.0), 2))
    return Settlement(status="lost", pnl=-round(stake, 2))


def settle_bet(
    market: str,
    selection: str,
    stake: float,
    decimal_odds: float,
    home_goals: int | None,
    away_goals: int | None,
) -> Settlement:
    """Settle a bet on whichever market it was struck on."""
    if market == "1X2":
        return settle_1x2(selection, stake, decimal_odds, home_goals, away_goals)
    if market == "OU25":
        return settle_ou25(selection, stake, decimal_odds, home_goals, away_goals)
    raise ValueError(f"market must be one of {MARKETS}, got {market!r}")
