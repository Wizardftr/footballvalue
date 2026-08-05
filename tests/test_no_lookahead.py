"""Proof that the walk-forward backtest cannot see the future.

Lookahead leakage is the single failure mode that would make this whole project
worthless while looking like a triumph: it produces backtests with spectacular ROI
that evaporate in live use. It is tested three ways, because each catches a
different kind of mistake:

1. **Structural** - every training set ends strictly before the week it predicts.
   Catches an off-by-one in the cutoff.
2. **Poisoning** - inserting absurd future results must not change any prediction
   by even a floating-point bit. Catches a training set that is filtered correctly
   on paper but assembled from the wrong frame, and any leakage through a shared
   fit that was warm-started from the future.
3. **Ordering** - a prediction's recorded ``trained_through`` must not exceed its
   own kickoff. Catches leakage that survives into the stored record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fv.backtest.walkforward import generate_predictions, week_start


def _synthetic_matches(n_seasons: int = 4, n_teams: int = 10, seed: int = 7) -> pd.DataFrame:
    """Build a fake league with a fixed weekly schedule and known team strengths."""
    rng = np.random.default_rng(seed)
    teams = [f"Team {i:02d}" for i in range(n_teams)]
    strength = {t: rng.normal(0, 0.35) for t in teams}

    rows = []
    date = pd.Timestamp("2015-08-08 15:00")
    match_id = 0
    for _ in range(n_seasons * 34):  # 34 match-weeks a season
        rng.shuffle(teams)
        for i in range(0, n_teams, 2):
            home, away = teams[i], teams[i + 1]
            lam = np.exp(0.25 + strength[home] - strength[away])
            mu = np.exp(strength[away] - strength[home])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            rows.append(
                {
                    "match_id": match_id,
                    "league_code": "TEST",
                    "season": f"{date.year}-{(date.year + 1) % 100:02d}",
                    "kickoff_utc": date,
                    "home": home,
                    "away": away,
                    "fthg": float(hg),
                    "ftag": float(ag),
                    "ftr": "H" if hg > ag else ("A" if hg < ag else "D"),
                    "pre_h": 2.0, "pre_d": 3.4, "pre_a": 3.8,
                    "close_h": 2.0, "close_d": 3.4, "close_a": 3.8,
                }
            )
            match_id += 1
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return _synthetic_matches()


def test_training_window_never_includes_the_predicted_week(matches):
    """Structural check, applied to every fold."""
    result = generate_predictions("TEST", "2017-01-01", matches=matches, min_train_matches=100)
    preds = result.predictions
    assert not preds.empty

    df = matches.copy()
    df["week"] = week_start(df["kickoff_utc"])

    for cutoff, group in preds.groupby("trained_through"):
        train = df[df["kickoff_utc"] < cutoff]
        assert train["kickoff_utc"].max() < cutoff, "training data reaches the cutoff"
        # Nothing the model trained on may be in the week it is predicting.
        assert group["kickoff_utc"].min() >= cutoff, "predicted a match before the cutoff"
        assert not set(train["match_id"]) & set(group["match_id"])


def test_prediction_is_never_trained_through_its_own_kickoff(matches):
    result = generate_predictions("TEST", "2017-01-01", matches=matches, min_train_matches=100)
    preds = result.predictions
    assert (preds["trained_through"] <= preds["kickoff_utc"]).all()


def test_poisoning_the_future_changes_nothing(matches):
    """The decisive test.

    Rewrite every match after the test window with absurd 9-0 scorelines. If any
    prediction moves, something is reading forward. Predictions must be identical to
    the bit, not merely close.
    """
    baseline = generate_predictions(
        "TEST", "2017-01-01", "2017-12-31", matches=matches, min_train_matches=100
    ).predictions

    poisoned = matches.copy()
    future = poisoned["kickoff_utc"] > pd.Timestamp("2018-01-01")
    assert future.sum() > 0, "test needs matches after the window to poison"
    poisoned.loc[future, "fthg"] = 9.0
    poisoned.loc[future, "ftag"] = 0.0
    poisoned.loc[future, "ftr"] = "H"

    after = generate_predictions(
        "TEST", "2017-01-01", "2017-12-31", matches=poisoned, min_train_matches=100
    ).predictions

    assert len(baseline) == len(after)
    for col in ("p_home", "p_draw", "p_away"):
        np.testing.assert_array_equal(
            baseline[col].to_numpy(), after[col].to_numpy(),
            err_msg=f"{col} moved when future results changed - lookahead leakage",
        )


def test_poisoning_the_past_does_change_predictions(matches):
    """Control for the test above.

    If poisoning the future changes nothing but poisoning the *past* also changes
    nothing, the first test proves only that the code ignores its inputs. This
    confirms the model is genuinely responding to its training data.
    """
    baseline = generate_predictions(
        "TEST", "2017-01-01", "2017-12-31", matches=matches, min_train_matches=100
    ).predictions

    poisoned = matches.copy()
    past = poisoned["kickoff_utc"] < pd.Timestamp("2016-06-01")
    poisoned.loc[past, "fthg"] = 9.0
    poisoned.loc[past, "ftag"] = 0.0

    after = generate_predictions(
        "TEST", "2017-01-01", "2017-12-31", matches=poisoned, min_train_matches=100
    ).predictions

    assert not np.array_equal(baseline["p_home"].to_numpy(), after["p_home"].to_numpy()), (
        "changing past results left predictions untouched - the model is ignoring its input"
    )


def test_week_start_is_monday_midnight():
    ts = pd.Series(pd.to_datetime([
        "2024-08-17 15:00",  # Saturday
        "2024-08-19 20:00",  # Monday
        "2024-08-21 19:45",  # Wednesday
        "2024-08-18 14:00",  # Sunday
    ]))
    ws = week_start(ts)
    assert list(ws.dt.dayofweek.unique()) == [0]
    assert (ws.dt.hour == 0).all()
    # Sat 17th and Sun 18th belong to the week starting Mon 12th;
    # Mon 19th and Wed 21st to the week starting Mon 19th.
    assert ws.iloc[0] == pd.Timestamp("2024-08-12")
    assert ws.iloc[3] == pd.Timestamp("2024-08-12")
    assert ws.iloc[1] == pd.Timestamp("2024-08-19")
    assert ws.iloc[2] == pd.Timestamp("2024-08-19")
