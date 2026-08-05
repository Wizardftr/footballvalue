"""Edge and stake sizing.

Kelly errors are asymmetric: understaking costs a little growth, overstaking can
compound to ruin. These tests pin down the caps as much as the formula.
"""

from __future__ import annotations

import pytest

from fv.odds.edge import Thresholds, edge, qualifies
from fv.odds.kelly import StakeRules, kelly_fraction_full, round_stake, stake_for


# -- edge -------------------------------------------------------------------

def test_edge_is_zero_at_a_fair_price():
    assert edge(0.5, 2.0) == pytest.approx(0.0)
    assert edge(0.25, 4.0) == pytest.approx(0.0)


def test_edge_is_expected_profit_per_unit_staked():
    # p=0.40 at 3.00: win 2.00 40% of the time, lose 1.00 60% => +0.20 per unit.
    assert edge(0.40, 3.00) == pytest.approx(0.20)


def test_edge_is_negative_when_the_price_is_short():
    assert edge(0.40, 2.00) == pytest.approx(-0.20)


def test_edge_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        edge(1.5, 2.0)
    with pytest.raises(ValueError):
        edge(0.5, 1.0)


# -- qualification ----------------------------------------------------------

def test_qualifies_needs_both_edge_and_odds_range():
    t = Thresholds(min_edge=0.04, min_odds=1.50, max_odds=4.00)
    assert qualifies(0.60, 1.80, t)  # edge 0.08, in range
    assert not qualifies(0.52, 1.80, t)  # edge -0.064
    assert not qualifies(0.90, 1.20, t)  # big edge but odds too short
    assert not qualifies(0.25, 5.00, t)  # big edge but odds too long


def test_qualification_boundaries_are_inclusive():
    t = Thresholds(min_edge=0.04, min_odds=1.50, max_odds=4.00)
    # Exactly 4% edge at exactly the range limits should qualify.
    assert qualifies(1.04 / 1.50, 1.50, t)
    assert qualifies(1.04 / 4.00, 4.00, t)


# -- Kelly ------------------------------------------------------------------

def test_full_kelly_matches_the_closed_form():
    # p=0.5, d=3.0: f = (0.5*3 - 1)/2 = 0.25
    assert kelly_fraction_full(0.5, 3.0) == pytest.approx(0.25)


def test_full_kelly_is_zero_without_an_edge():
    assert kelly_fraction_full(0.5, 2.0) == 0.0
    assert kelly_fraction_full(0.3, 2.0) == 0.0, "negative edge must never produce a stake"


def test_quarter_kelly_is_a_quarter_of_full():
    rules = StakeRules(kelly_fraction=0.25, max_stake_pct=1.0, rounding=0.0, min_stake=0.0)
    full = kelly_fraction_full(0.5, 3.0)
    assert stake_for(0.5, 3.0, 1000.0, rules) == pytest.approx(0.25 * full * 1000.0)


def test_stake_is_capped_at_max_pct_of_bankroll():
    """The cap is what makes the sizing survive an overestimated probability."""
    rules = StakeRules(kelly_fraction=0.25, max_stake_pct=0.02, rounding=0.0, min_stake=0.0)
    # A huge edge would imply a far larger Kelly stake than 2%.
    stake = stake_for(0.90, 3.00, 1000.0, rules)
    assert stake == pytest.approx(20.0)


def test_stake_scales_with_bankroll():
    rules = StakeRules(kelly_fraction=0.25, max_stake_pct=0.02, rounding=0.0, min_stake=0.0)
    assert stake_for(0.90, 3.00, 500.0, rules) == pytest.approx(10.0)


def test_no_stake_without_an_edge():
    assert stake_for(0.5, 2.0, 1000.0) == 0.0
    assert stake_for(0.1, 2.0, 1000.0) == 0.0


def test_no_stake_on_an_empty_bankroll():
    assert stake_for(0.9, 3.0, 0.0) == 0.0
    assert stake_for(0.9, 3.0, -50.0) == 0.0


def test_rounding_never_rounds_up_past_the_cap():
    """Rounding down keeps the stake inside the Kelly cap."""
    assert round_stake(20.99, 0.50) == pytest.approx(20.50)
    assert round_stake(20.49, 0.50) == pytest.approx(20.00)


def test_stake_below_minimum_becomes_no_bet():
    """A stake too small to place is not rounded up past its own cap."""
    rules = StakeRules(kelly_fraction=0.25, max_stake_pct=0.02, rounding=0.50, min_stake=5.00)
    stake = stake_for(0.55, 2.00, 20.0, rules)  # tiny bankroll => tiny stake
    assert stake == 0.0
