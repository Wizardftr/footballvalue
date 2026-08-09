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


# ---------------------------------------------------------------------------
# Over/under 2.5 goals
# ---------------------------------------------------------------------------

def test_over_wins_on_three_goals():
    from fv.odds.settlement import settle_ou25

    r = settle_ou25("O", 10.0, 1.90, 2, 1)
    assert r.status == "won"
    assert r.pnl == pytest.approx(9.0)


def test_under_wins_on_two_goals():
    from fv.odds.settlement import settle_ou25

    r = settle_ou25("U", 10.0, 1.90, 1, 1)
    assert r.status == "won"
    assert r.pnl == pytest.approx(9.0)


def test_the_two_sides_are_exactly_complementary():
    """Every scoreline settles one side a winner and the other a loser. A 2.5 line
    cannot push, and a gap here would mean money appearing or vanishing."""
    from fv.odds.settlement import settle_ou25

    for h in range(6):
        for a in range(6):
            over = settle_ou25("O", 10.0, 2.0, h, a)
            under = settle_ou25("U", 10.0, 2.0, h, a)
            assert {over.status, under.status} == {"won", "lost"}


def test_a_whole_line_is_refused_rather_than_guessed():
    from fv.odds.settlement import settle_ou25

    with pytest.raises(ValueError, match="push"):
        settle_ou25("O", 10.0, 1.90, 2, 1, line=3.0)


def test_abandoned_match_voids_a_goals_bet():
    from fv.odds.settlement import settle_ou25

    assert settle_ou25("O", 10.0, 1.90, None, None).status == "void"


def test_settle_bet_dispatches_on_market():
    from fv.odds.settlement import settle_bet

    # 3-1: home wins and there are four goals, so both sides of the slip land.
    assert settle_bet("1X2", "H", 10.0, 2.0, 3, 1).status == "won"
    assert settle_bet("OU25", "O", 10.0, 2.0, 3, 1).status == "won"
    assert settle_bet("OU25", "U", 10.0, 2.0, 3, 1).status == "lost"


def test_an_unknown_market_is_refused():
    """Guessing how to settle a market we do not model would silently corrupt the
    whole performance record."""
    from fv.odds.settlement import settle_bet

    with pytest.raises(ValueError, match="market must be"):
        settle_bet("BTTS", "Y", 10.0, 2.0, 3, 1)


def test_a_1x2_selection_is_refused_on_the_goals_market():
    from fv.odds.settlement import settle_ou25

    with pytest.raises(ValueError):
        settle_ou25("H", 10.0, 1.90, 2, 1)
