"""Stage combination: pooling, market anchoring, weight tuning.

The tuner tests exist because the first version of ``tune_market_weight`` called
``idxmin()`` on the weight column instead of the log-loss column, so it always
returned the smallest weight in the grid regardless of which one actually scored
best. It produced a plausible-looking number and would have silently disabled the
market anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fv.backtest.stages import (
    anchor_to_market,
    combine,
    log_opinion_pool,
    market_fair_probabilities,
    tune_market_weight,
    tune_pool_weight,
)


def _frame(probs, ftr, prices=(2.0, 4.0, 4.0), n=None):
    probs = np.atleast_2d(probs)
    n = n or len(probs)
    if len(probs) == 1:
        probs = np.repeat(probs, n, axis=0)
    return pd.DataFrame(
        {
            "match_id": range(n),
            "league_code": "TEST",
            "season": "2024-25",
            "kickoff_utc": pd.date_range("2024-08-01", periods=n, freq="D"),
            "week": pd.date_range("2024-08-01", periods=n, freq="D"),
            "home": "A", "away": "B",
            "ftr": ftr if isinstance(ftr, list) else [ftr] * n,
            "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
            "pre_h": prices[0], "pre_d": prices[1], "pre_a": prices[2],
            "close_h": prices[0], "close_d": prices[1], "close_a": prices[2],
        }
    )


# -- pooling ----------------------------------------------------------------

def test_pool_returns_a_distribution():
    p = log_opinion_pool([[0.5, 0.3, 0.2]], [[0.2, 0.3, 0.5]], 0.5)
    assert p.sum(axis=1)[0] == pytest.approx(1.0)
    assert np.all(p > 0)


def test_pool_at_weight_one_is_the_first_model():
    a = np.array([[0.5, 0.3, 0.2]])
    p = log_opinion_pool(a, [[0.2, 0.3, 0.5]], 1.0)
    assert p == pytest.approx(a)


def test_pool_at_weight_zero_is_the_second_model():
    b = np.array([[0.2, 0.3, 0.5]])
    p = log_opinion_pool([[0.5, 0.3, 0.2]], b, 0.0)
    assert p == pytest.approx(b)


def test_pool_of_identical_inputs_is_unchanged():
    a = np.array([[0.5, 0.3, 0.2]])
    assert log_opinion_pool(a, a, 0.37) == pytest.approx(a)


def test_pool_is_geometric_not_arithmetic():
    """The distinguishing property, on a deliberately asymmetric example.

    A symmetric pair makes the two pools coincide, so the numbers here are chosen so
    they genuinely differ. The log pool is the normalised geometric mean, which
    penalises outcomes either model thinks unlikely more than linear averaging does.
    """
    a = np.array([[0.6, 0.3, 0.1]])
    b = np.array([[0.2, 0.3, 0.5]])
    p = log_opinion_pool(a, b, 0.5)

    geo = np.sqrt(a[0] * b[0])
    assert p[0] == pytest.approx(geo / geo.sum())

    linear = (0.5 * a + 0.5 * b)[0]
    assert not np.allclose(p[0], linear), "log pool must differ from a linear average"
    # Both models agree the draw is 0.3, so it survives pooling better than the
    # outcomes they disagree about.
    assert p[0, 1] > linear[1]


def test_pool_survives_a_zero_probability():
    p = log_opinion_pool([[1.0, 0.0, 0.0]], [[0.3, 0.4, 0.3]], 0.5)
    assert np.isfinite(p).all()
    assert p.sum(axis=1)[0] == pytest.approx(1.0)


# -- market anchoring -------------------------------------------------------

def test_market_fair_probabilities_strip_the_margin():
    f = _frame([0.5, 0.3, 0.2], "H", prices=(2.10, 3.40, 3.80), n=1)
    m = market_fair_probabilities(f)
    assert m[0].sum() == pytest.approx(1.0)


def test_market_fair_is_nan_without_a_price():
    f = _frame([0.5, 0.3, 0.2], "H", n=1)
    f.loc[0, "pre_h"] = np.nan
    assert np.isnan(market_fair_probabilities(f)[0]).all()


def test_anchoring_pulls_predictions_toward_the_market():
    """A model that disagrees with the market must end up between the two."""
    f = _frame([0.70, 0.20, 0.10], "H", prices=(2.0, 4.0, 4.0), n=1)
    anchored = anchor_to_market(f, market_weight=0.6)
    # Market implies 0.50 for home; the model says 0.70.
    assert 0.50 < anchored["p_home"].iloc[0] < 0.70


def test_anchoring_at_full_weight_gives_the_market():
    f = _frame([0.70, 0.20, 0.10], "H", prices=(2.0, 4.0, 4.0), n=1)
    anchored = anchor_to_market(f, market_weight=1.0)
    assert anchored["p_home"].iloc[0] == pytest.approx(0.5, abs=1e-9)


def test_anchoring_at_zero_weight_leaves_the_model_alone():
    f = _frame([0.70, 0.20, 0.10], "H", n=1)
    anchored = anchor_to_market(f, market_weight=0.0)
    assert anchored["p_home"].iloc[0] == pytest.approx(0.70)


def test_anchoring_keeps_matches_without_a_price():
    """Dropping them would make stage metrics incomparable."""
    f = _frame([0.70, 0.20, 0.10], "H", n=3)
    f.loc[1, "pre_h"] = np.nan
    anchored = anchor_to_market(f, market_weight=0.6)
    assert len(anchored) == 3
    assert anchored["p_home"].iloc[1] == pytest.approx(0.70)  # untouched


# -- tuning -----------------------------------------------------------------

def test_market_weight_tuner_returns_the_best_scoring_weight():
    """Regression test for the idxmin-on-the-wrong-column bug.

    The model is confidently wrong and the market is right, so anchoring harder must
    score better and the tuner must not return the smallest weight in the grid.
    """
    n = 400
    # Market prices imply home 0.5; home actually wins half the time.
    ftr = ["H", "A"] * (n // 2)
    f = _frame([0.95, 0.03, 0.02], ftr, prices=(2.0, 4.0, 4.0), n=n)
    best, table = tune_market_weight(f, grid=[0.0, 0.2, 0.5, 0.8, 1.0])
    assert best == table.loc[table["log_loss"].idxmin(), "market_weight"]
    assert best > 0.0, "a confidently wrong model must be pulled toward the market"


def test_pool_weight_tuner_returns_the_best_scoring_weight():
    n = 400
    ftr = ["H", "A"] * (n // 2)
    good = _frame([0.5, 0.25, 0.25], ftr, n=n)
    bad = _frame([0.95, 0.03, 0.02], ftr, n=n)
    best, table = tune_pool_weight(good, bad, grid=[0.0, 0.25, 0.5, 0.75, 1.0])
    assert best == table.loc[table["log_loss"].idxmin(), "weight"]
    assert best > 0.5, "the better model should get most of the weight"


def test_combine_keeps_only_matches_present_in_both():
    a = _frame([0.5, 0.3, 0.2], "H", n=5)
    b = _frame([0.4, 0.3, 0.3], "H", n=3)
    out = combine(a, b, 0.5)
    assert len(out) == 3
    assert out[["p_home", "p_draw", "p_away"]].sum(axis=1).round(9).eq(1.0).all()


def test_combine_on_empty_input():
    assert combine(pd.DataFrame(), _frame([0.5, 0.3, 0.2], "H", n=2), 0.5).empty
