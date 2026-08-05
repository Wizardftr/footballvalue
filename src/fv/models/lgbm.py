"""LightGBM multiclass stage, with isotonic calibration.

Gradient boosting optimises for discrimination, not for honest probabilities, and
its raw multiclass outputs are usually over-confident in the tails. Those tails are
exactly where the betting rules select from, so uncalibrated output would
manufacture edges at the long end of the odds range. Isotonic regression fixes the
mapping from predicted to observed frequency without assuming a functional form.

The calibrator is fitted on a slice of the training window that the booster never
saw. Calibrating on the booster's own training data would just relearn its
overfitting and leave the real miscalibration untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fv.models.features import FEATURE_COLUMNS

OUTCOMES = ("H", "D", "A")

# Tuned on E0 validation seasons (2019-22). The striking thing is how heavily
# regularised the best configuration is: 4 leaves, 300 matches minimum per leaf.
# With only a few thousand training matches and an outcome that is mostly noise,
# anything deeper overfits — a 15-leaf model scored 1.015 against this one's 0.998.
DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.02,
    "num_leaves": 4,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 30.0,
    "verbose": -1,
    "num_threads": 4,
}
DEFAULT_ROUNDS = 200


@dataclass
class LGBMFit:
    booster: object
    calibrators: list = field(default_factory=list)  # one IsotonicRegression per outcome
    n_train: int = 0
    n_calib: int = 0
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated (n, 3) probabilities in H/D/A order, each row summing to 1."""
        x = frame[self.features]
        raw = np.asarray(self.booster.predict(x))
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if not self.calibrators:
            return raw

        out = np.column_stack(
            [self.calibrators[i].predict(raw[:, i]) for i in range(raw.shape[1])]
        )
        # Isotonic is applied per class, so rows no longer sum to 1. Renormalise;
        # fall back to the raw row if calibration collapsed it to zero.
        totals = out.sum(axis=1, keepdims=True)
        bad = (totals <= 0).ravel()
        out = np.divide(out, np.where(totals > 0, totals, 1.0))
        if bad.any():
            out[bad] = raw[bad]
        return np.clip(out, 1e-6, 1.0)


def fit_lgbm(
    train: pd.DataFrame,
    calibration_fraction: float = 0.2,
    params: dict | None = None,
    num_rounds: int = DEFAULT_ROUNDS,
    seed: int = 42,
) -> LGBMFit | None:
    """Train the booster and its calibrator on a training window.

    The calibration slice is the *most recent* part of the window, not a random
    sample. A random split would let the booster learn from matches played after
    the ones it is calibrated on, which is the same lookahead problem in miniature.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    train = train.dropna(subset=["ftr"]).sort_values("kickoff_utc")
    if len(train) < 500:
        return None

    y = train["ftr"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    x = train[FEATURE_COLUMNS]

    split = int(len(train) * (1.0 - calibration_fraction))
    if split < 300 or len(train) - split < 150:
        split = len(train)  # too small to calibrate; train on everything

    p = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    booster = lgb.train(p, lgb.Dataset(x.iloc[:split], label=y[:split]), num_boost_round=num_rounds)

    calibrators = []
    n_calib = 0
    if split < len(train):
        raw = np.asarray(booster.predict(x.iloc[split:]))
        y_cal = y[split:]
        n_calib = len(y_cal)
        for i in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw[:, i], (y_cal == i).astype(float))
            calibrators.append(iso)

    return LGBMFit(
        booster=booster,
        calibrators=calibrators,
        n_train=split,
        n_calib=n_calib,
    )
