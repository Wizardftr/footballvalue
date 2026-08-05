"""Scoring and performance metrics.

Two families here, and they answer different questions:

* **Probability quality** (Brier, log loss, calibration) asks whether the model's
  numbers are right. It needs no odds and is not affected by betting rules.
* **Betting performance** (ROI, drawdown, CLV) asks whether the numbers are right
  *in the places the market is wrong*, which is a much higher bar.

A model can improve on the first while getting worse on the second, so both are
reported at every stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
EPS = 1e-15


def _prob_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)


def _outcome_matrix(actual: pd.Series) -> np.ndarray:
    a = actual.to_numpy()
    return np.column_stack([(a == o).astype(float) for o in OUTCOMES])


def brier_score(df: pd.DataFrame, actual_col: str = "ftr") -> float:
    """Multiclass Brier score: mean squared error over the three outcomes.

    Lower is better. A model that always predicts the base rates scores about 0.60;
    bet365's closing prices score around 0.56 on these leagues.
    """
    if df.empty:
        return float("nan")
    p = _prob_matrix(df)
    y = _outcome_matrix(df[actual_col])
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def log_loss(df: pd.DataFrame, actual_col: str = "ftr") -> float:
    """Mean negative log probability of the outcome that happened. Lower is better.

    Punishes confident mistakes far harder than Brier does, which is the right
    emphasis when the output feeds a Kelly stake.
    """
    if df.empty:
        return float("nan")
    p = np.clip(_prob_matrix(df), EPS, 1.0)
    y = _outcome_matrix(df[actual_col])
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def calibration_table(df: pd.DataFrame, bins: int = 10, actual_col: str = "ftr") -> pd.DataFrame:
    """Predicted vs observed frequency, pooling all three outcomes.

    A well-calibrated model puts the observed frequency close to the predicted
    probability in every bin. Systematic overprediction at the top end is the
    classic signature of a model that will bleed money on favourites.
    """
    if df.empty:
        return pd.DataFrame(columns=["bin_mid", "predicted", "observed", "n"])
    p = _prob_matrix(df).ravel()
    y = _outcome_matrix(df[actual_col]).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin_mid": (edges[b] + edges[b + 1]) / 2,
                "predicted": float(p[m].mean()),
                "observed": float(y[m].mean()),
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Largest peak-to-trough fall, as a fraction of the peak.

    0.25 means the bankroll at some point sat 25% below its previous high.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
    return float(dd.max())


def longest_losing_streak(results: pd.Series | list[str]) -> int:
    """Longest run of consecutive losses. Voids don't break a streak."""
    longest = current = 0
    for r in results:
        if r == "L":
            current += 1
            longest = max(longest, current)
        elif r == "W":
            current = 0
    return longest


def roi_confidence_interval(
    pnl: np.ndarray, staked: np.ndarray, z: float = 1.96
) -> tuple[float, float, float]:
    """ROI with a normal confidence interval on the mean return per unit staked.

    This exists because the interval is the honest headline, not the point estimate.
    At the volumes this project runs at, a 300-bet window has a standard error of
    roughly 7 percentage points on ROI, so a measured +4% and a true 0% are not
    distinguishable. Reporting ROI without this interval invites exactly the
    misreading the project is trying to avoid.
    """
    pnl = np.asarray(pnl, dtype=float)
    staked = np.asarray(staked, dtype=float)
    total = staked.sum()
    if total <= 0 or len(pnl) < 2:
        return (float("nan"), float("nan"), float("nan"))
    roi = float(pnl.sum() / total)
    # Per-unit-staked returns, so the interval is on the same scale as ROI.
    per_unit = pnl / np.where(staked > 0, staked, np.nan)
    per_unit = per_unit[~np.isnan(per_unit)]
    se = float(np.std(per_unit, ddof=1) / np.sqrt(len(per_unit)))
    return roi, roi - z * se, roi + z * se


def summarize_clv(clv_values: pd.Series) -> dict[str, float]:
    """CLV summary, with the interval and the hit rate alongside the mean.

    CLV is the leading indicator: it resolves on every bet immediately rather than
    waiting for outcomes to average out, so it reaches significance far sooner than
    ROI does. But a positive *mean* on its own proves nothing, and reporting it
    alone is the easiest way to fool yourself on this project.

    CLV is heavily tailed, so a handful of large favourable moves can drag the mean
    positive while the model is on the wrong side of the line more often than not.
    The mean, its confidence interval, and ``pct_positive`` have to be read together:
    a positive mean with ``pct_positive`` below 0.5 is outlier-driven noise, not
    evidence of an edge. ``pct_positive_moved`` excludes bets where the price never
    moved, which are ties and otherwise dilute the hit rate toward 50%.
    """
    s = pd.Series(clv_values).dropna()
    if s.empty:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"),
            "pct_positive": float("nan"), "pct_positive_moved": float("nan"),
            "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "significant": False,
        }
    mean = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else float("nan")
    lo, hi = (mean - 1.96 * se, mean + 1.96 * se) if len(s) > 1 else (float("nan"), float("nan"))
    moved = s[s != 0]
    return {
        "n": int(len(s)),
        "mean": mean,
        "median": float(s.median()),
        "pct_positive": float((s > 0).mean()),
        "pct_positive_moved": float((moved > 0).mean()) if len(moved) else float("nan"),
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        # Significant only when the whole interval sits on one side of zero.
        "significant": bool(len(s) > 1 and (lo > 0 or hi < 0)),
    }
