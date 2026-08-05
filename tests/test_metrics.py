"""Metrics, with emphasis on the ones that could mislead.

The CLV tests exist because of a real near-miss: the first full backtest returned a
mean CLV of +0.32% and the report called it evidence of an edge. Its 95% interval
was [-0.09%, +0.72%] and only 42.9% of bets beat the close. Mean CLV alone is not a
finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fv.backtest.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    longest_losing_streak,
    max_drawdown,
    roi_confidence_interval,
    summarize_clv,
)


def _preds(p_home, p_draw, p_away, actual):
    return pd.DataFrame(
        {"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "ftr": actual}
    )


# -- scoring ----------------------------------------------------------------

def test_perfect_prediction_scores_zero():
    df = _preds([1.0], [0.0], [0.0], ["H"])
    assert brier_score(df) == pytest.approx(0.0)
    assert log_loss(df) == pytest.approx(0.0)


def test_brier_penalises_a_confident_miss():
    right = _preds([0.9], [0.05], [0.05], ["H"])
    wrong = _preds([0.9], [0.05], [0.05], ["A"])
    assert brier_score(wrong) > brier_score(right)


def test_log_loss_punishes_confident_misses_harder_than_brier():
    """The reason log loss is the tuning objective: Kelly staking is most damaged by
    confident errors, and log loss weights them far more heavily."""
    mild = _preds([0.5, 0.5], [0.3, 0.3], [0.2, 0.2], ["H", "A"])
    harsh = _preds([0.98, 0.98], [0.01, 0.01], [0.01, 0.01], ["H", "A"])
    assert log_loss(harsh) > log_loss(mild)
    assert brier_score(harsh) < log_loss(harsh)


def test_log_loss_is_finite_on_a_zero_probability_outcome():
    """A zero probability on the outcome that happened must not produce infinity."""
    df = _preds([0.0], [0.0], [1.0], ["H"])
    assert np.isfinite(log_loss(df))


def test_calibration_table_recovers_a_known_frequency():
    # 100 predictions at p_home = 0.6, of which exactly 60 are home wins.
    actual = ["H"] * 60 + ["A"] * 40
    df = _preds([0.6] * 100, [0.2] * 100, [0.2] * 100, actual)
    cal = calibration_table(df, bins=10)
    # 0.6 sits exactly on a bin edge, so it lands in [0.6, 0.7). Select by the
    # predicted value rather than guessing which side of the edge it falls on.
    row = cal.iloc[(cal["predicted"] - 0.6).abs().argmin()]
    assert row["predicted"] == pytest.approx(0.6)
    assert row["observed"] == pytest.approx(0.6, abs=1e-9)
    assert row["n"] == 100


# -- drawdown and streaks ---------------------------------------------------

def test_max_drawdown_of_a_rising_curve_is_zero():
    assert max_drawdown([100, 110, 120, 130]) == pytest.approx(0.0)


def test_max_drawdown_measures_peak_to_trough():
    # Peak 200, trough 150 => 25%.
    assert max_drawdown([100, 200, 150, 180]) == pytest.approx(0.25)


def test_longest_losing_streak_ignores_voids():
    assert longest_losing_streak(["L", "L", "void", "L", "W", "L"]) == 3
    assert longest_losing_streak(["W", "W"]) == 0


# -- ROI interval -----------------------------------------------------------

def test_roi_matches_pnl_over_staked():
    pnl = np.array([10.0, -10.0, 20.0])
    staked = np.array([10.0, 10.0, 10.0])
    roi, lo, hi = roi_confidence_interval(pnl, staked)
    assert roi == pytest.approx(20.0 / 30.0)
    assert lo < roi < hi


def test_roi_interval_widens_on_a_small_sample():
    """The property that makes the interval worth reporting at all."""
    rng = np.random.default_rng(0)
    small = rng.choice([1.5, -1.0], size=30)
    large = rng.choice([1.5, -1.0], size=3000)
    _, lo_s, hi_s = roi_confidence_interval(small, np.ones(30))
    _, lo_l, hi_l = roi_confidence_interval(large, np.ones(3000))
    assert (hi_s - lo_s) > (hi_l - lo_l)


# -- CLV --------------------------------------------------------------------

def test_clv_mean_near_zero_is_reported_as_not_significant():
    """The near-miss this test exists for.

    A small positive mean whose interval spans zero must not be reported as an edge.
    """
    rng = np.random.default_rng(1)
    clv = pd.Series(rng.normal(0.003, 0.10, 2371))  # mirrors the real backtest
    s = summarize_clv(clv)
    assert s["mean"] > 0
    assert s["ci_low"] < 0 < s["ci_high"]
    assert s["significant"] is False


def test_clv_is_significant_when_the_interval_clears_zero():
    rng = np.random.default_rng(2)
    clv = pd.Series(rng.normal(0.05, 0.10, 2000))
    s = summarize_clv(clv)
    assert s["significant"] is True
    assert s["ci_low"] > 0


def test_clv_hit_rate_excludes_unmoved_prices():
    """Bets where the price never moved are ties and dilute the hit rate toward 50%."""
    clv = pd.Series([0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.1])
    s = summarize_clv(clv)
    assert s["pct_positive"] == pytest.approx(2 / 7)
    assert s["pct_positive_moved"] == pytest.approx(2 / 3)


def test_clv_ignores_missing_values():
    """Seasons without closing prices must not be counted as zero CLV."""
    s = summarize_clv(pd.Series([0.05, None, 0.05, np.nan]))
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(0.05)


def test_clv_summary_on_empty_input():
    s = summarize_clv(pd.Series([], dtype=float))
    assert s["n"] == 0
    assert s["significant"] is False
