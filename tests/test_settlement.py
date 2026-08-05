"""Settlement and CLV."""

from __future__ import annotations

import pytest

from fv.odds.settlement import clv, result_from_goals, settle_1x2


def test_result_from_goals():
    assert result_from_goals(2, 1) == "H"
    assert result_from_goals(1, 1) == "D"
    assert result_from_goals(0, 3) == "A"


@pytest.mark.parametrize(
    "selection,hg,ag,status",
    [
        ("H", 2, 1, "won"),
        ("H", 1, 1, "lost"),
        ("H", 0, 1, "lost"),
        ("D", 1, 1, "won"),
        ("D", 2, 1, "lost"),
        ("A", 0, 1, "won"),
        ("A", 1, 1, "lost"),
    ],
)
def test_settle_every_outcome_combination(selection, hg, ag, status):
    assert settle_1x2(selection, 10.0, 3.0, hg, ag).status == status


def test_winning_bet_returns_profit_not_total_return():
    """P&L excludes the returned stake: a 10.00 bet at 3.00 profits 20.00."""
    s = settle_1x2("H", 10.0, 3.0, 2, 0)
    assert s.pnl == pytest.approx(20.0)


def test_losing_bet_loses_exactly_the_stake():
    assert settle_1x2("H", 10.0, 3.0, 0, 2).pnl == pytest.approx(-10.0)


def test_missing_score_is_void_not_a_loss():
    """An abandoned or postponed match returns the stake."""
    s = settle_1x2("H", 10.0, 3.0, None, None)
    assert s.status == "void"
    assert s.pnl == 0.0


def test_zero_stake_settles_without_pnl():
    assert settle_1x2("H", 0.0, 3.0, 2, 0).pnl == 0.0


def test_settlement_rejects_bad_input():
    with pytest.raises(ValueError):
        settle_1x2("X", 10.0, 3.0, 1, 0)
    with pytest.raises(ValueError):
        settle_1x2("H", -1.0, 3.0, 1, 0)
    with pytest.raises(ValueError):
        settle_1x2("H", 10.0, 1.0, 1, 0)


# -- CLV --------------------------------------------------------------------

def test_positive_clv_when_you_beat_the_close():
    # Took 2.10, closed at 2.00: 5% better than the close.
    assert clv(2.10, 2.00) == pytest.approx(0.05)


def test_negative_clv_when_the_price_drifts_out():
    assert clv(2.00, 2.10) == pytest.approx(-0.047619, abs=1e-6)


def test_zero_clv_when_the_price_is_unchanged():
    assert clv(2.00, 2.00) == pytest.approx(0.0)


def test_clv_is_none_without_a_closing_price():
    """Seasons before 2019-20 have no closing odds; that must read as unknown,
    not as zero, or the average would be silently diluted toward zero."""
    assert clv(2.00, None) is None
    assert clv(2.00, 0.0) is None
