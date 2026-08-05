"""Margin removal must return honest probabilities. Everything downstream depends
on it: a biased fair-probability estimate turns into a phantom edge."""

from __future__ import annotations

import numpy as np
import pytest

from fv.odds.margin import (
    booksum,
    implied_probabilities,
    margin,
    remove_margin,
    remove_margin_proportional,
    remove_margin_shin,
)


def test_implied_probabilities_are_reciprocals():
    assert implied_probabilities([2.0, 4.0]) == pytest.approx([0.5, 0.25])


def test_booksum_and_margin_on_a_real_looking_book():
    odds = [2.10, 3.40, 3.80]
    assert booksum(odds) == pytest.approx(0.4762 + 0.2941 + 0.2632, abs=1e-3)
    assert margin(odds) == pytest.approx(booksum(odds) - 1.0)


def test_fair_book_has_zero_margin():
    # A perfectly fair three-way book: probabilities 0.5/0.25/0.25.
    odds = [2.0, 4.0, 4.0]
    assert margin(odds) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("method", ["proportional", "shin"])
def test_removing_margin_gives_probabilities_summing_to_one(method):
    odds = [1.85, 3.60, 4.50]
    p = remove_margin(odds, method=method)
    assert p.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.all(p > 0)


@pytest.mark.parametrize("method", ["proportional", "shin"])
def test_fair_book_is_unchanged_by_margin_removal(method):
    """With no margin to remove, both methods must be the identity."""
    odds = [2.0, 4.0, 4.0]
    assert remove_margin(odds, method=method) == pytest.approx([0.5, 0.25, 0.25], abs=1e-8)


def test_proportional_scales_every_selection_equally():
    odds = [2.10, 3.40, 3.80]
    raw = implied_probabilities(odds)
    fair = remove_margin_proportional(odds)
    ratios = fair / raw
    assert np.allclose(ratios, ratios[0])


def test_shin_takes_more_margin_off_the_longshot():
    """The distinguishing property of Shin's method.

    Proportional removal shaves the same *fraction* off every selection. Shin
    attributes the margin to insider money, which implies the book pads longshots
    more, so relative to proportional it should push the longshot's fair probability
    down and the favourite's up. Getting this backwards would systematically invent
    edges on longshots, which is where a naive model bleeds money.
    """
    odds = [1.50, 4.50, 7.00]  # clear favourite, clear longshot
    prop = remove_margin_proportional(odds)
    shin = remove_margin_shin(odds)
    assert shin[0] > prop[0], "favourite should get a higher fair probability under Shin"
    assert shin[2] < prop[2], "longshot should get a lower fair probability under Shin"


def test_shin_equals_proportional_when_all_prices_are_equal():
    """With a symmetric book there is no longshot to treat differently."""
    odds = [3.0, 3.0, 3.0]
    assert remove_margin_shin(odds) == pytest.approx(remove_margin_proportional(odds), abs=1e-8)


def test_removing_margin_lowers_every_probability():
    """Stripping margin can only reduce implied probabilities, never raise them."""
    odds = [2.10, 3.40, 3.80]
    raw = implied_probabilities(odds)
    for method in ("proportional", "shin"):
        fair = remove_margin(odds, method=method)
        assert np.all(fair <= raw + 1e-12)


def test_rejects_invalid_odds():
    with pytest.raises(ValueError):
        implied_probabilities([1.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        implied_probabilities([-2.0])


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown margin method"):
        remove_margin([2.0, 4.0, 4.0], method="magic")
