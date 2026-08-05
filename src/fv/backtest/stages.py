"""The Phase 2 model stack, staged so each upgrade has to prove itself.

Five stages, each built on the last:

1. ``dc``       - time-decayed Dixon-Coles on goals (the Phase 1 baseline)
2. ``dc_xg``    - the same model fitted on a goals/xG blend
3. ``lgbm``     - LightGBM on form, Elo, rest and promotion features, calibrated
4. ``ensemble`` - Dixon-Coles and LightGBM combined by a log-opinion pool
5. ``anchored`` - the ensemble blended toward margin-free market probabilities

Stages 4 and 5 are pure functions of prediction frames, so the expensive model
fitting happens once and the combination weights can be tuned in seconds rather
than by refitting thousands of models.

Every weight is tuned on validation seasons that end before the test window. The
market anchor in particular is easy to fool yourself with: tune its weight on the
test set and you will "discover" that the market deserves whatever weight happens
to maximise that period's ROI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fv.backtest.metrics import log_loss
from fv.backtest.walkforward import generate_predictions, load_matches, week_start
from fv.config import Config, load_config
from fv.models.features import build_features
from fv.models.lgbm import OUTCOMES, fit_lgbm

PROB_COLUMNS = ["p_home", "p_draw", "p_away"]


# ---------------------------------------------------------------------------
# Stage 3: LightGBM
# ---------------------------------------------------------------------------

def predict_lgbm(
    league_code: str,
    test_from: str | pd.Timestamp,
    test_to: str | pd.Timestamp | None = None,
    cfg: Config | None = None,
    matches: pd.DataFrame | None = None,
    refit_every_days: int = 120,
    min_train_matches: int = 1000,
    lgbm_params: dict | None = None,
    num_rounds: int | None = None,
) -> pd.DataFrame:
    """Walk-forward LightGBM predictions.

    Features are built once for the whole league — safe, because the builder is
    chronological and each match's features come only from earlier matches. The
    walk-forward part is then purely about *training* windows: the booster is
    refitted every ``refit_every_days`` on everything before the refit date, and
    used for matches until the next refit.

    Refitting every week as Dixon-Coles does would cost hours for no benefit; a
    gradient booster trained on several thousand matches does not meaningfully
    change from one week to the next.
    """
    cfg = cfg or load_config()
    df = matches if matches is not None else load_matches(league_code, cfg)
    if df.empty:
        return pd.DataFrame()

    feats = build_features(df)
    if feats.empty:
        return pd.DataFrame()

    # Carry the prices through so the betting simulation can use this frame directly.
    price_cols = ["match_id", "league_code", "pre_h", "pre_d", "pre_a",
                  "close_h", "close_d", "close_a"]
    feats = feats.merge(df[price_cols], on="match_id", how="left")
    feats["week"] = week_start(feats["kickoff_utc"])

    start = pd.Timestamp(test_from)
    end = pd.Timestamp(test_to) if test_to is not None else feats["kickoff_utc"].max()
    test_mask = (feats["kickoff_utc"] >= start) & (feats["kickoff_utc"] <= end)
    if not test_mask.any():
        return pd.DataFrame()

    out: list[pd.DataFrame] = []
    cutoff = start
    while cutoff <= end:
        next_cutoff = cutoff + pd.Timedelta(days=refit_every_days)
        train = feats[feats["kickoff_utc"] < cutoff]
        block = feats[
            (feats["kickoff_utc"] >= cutoff)
            & (feats["kickoff_utc"] < min(next_cutoff, end + pd.Timedelta(days=1)))
        ]
        if block.empty:
            cutoff = next_cutoff
            continue
        if len(train) < min_train_matches:
            cutoff = next_cutoff
            continue

        fit = fit_lgbm(
            train,
            params=lgbm_params,
            **({"num_rounds": num_rounds} if num_rounds is not None else {}),
        )
        if fit is None:
            cutoff = next_cutoff
            continue

        probs = fit.predict(block)
        chunk = block[["match_id", "league_code", "season", "kickoff_utc", "week",
                       "home", "away", "ftr",
                       "pre_h", "pre_d", "pre_a", "close_h", "close_d", "close_a"]].copy()
        # Same column names the Dixon-Coles frame uses, so the betting simulation's
        # "enough history to bet on" guard works identically across stages.
        chunk["n_home_matches"] = block["home_matches_played"].to_numpy()
        chunk["n_away_matches"] = block["away_matches_played"].to_numpy()
        chunk[PROB_COLUMNS] = probs
        chunk["trained_through"] = cutoff
        out.append(chunk)
        cutoff = next_cutoff

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Stage 4/5: combination
# ---------------------------------------------------------------------------

def log_opinion_pool(p1: np.ndarray, p2: np.ndarray, weight: float) -> np.ndarray:
    """Weighted geometric combination: ``p ∝ p1^weight * p2^(1-weight)``.

    A log pool rather than a linear average because it is the combination that is
    consistent with treating each model as independent evidence: it multiplies
    likelihood ratios instead of averaging them. In practice it is also less
    forgiving of a model that is confidently wrong, which is the behaviour we want
    when the output feeds a Kelly stake.
    """
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-12, 1.0)
    p2 = np.clip(np.asarray(p2, dtype=float), 1e-12, 1.0)
    pooled = np.exp(weight * np.log(p1) + (1.0 - weight) * np.log(p2))
    totals = pooled.sum(axis=1, keepdims=True)
    return pooled / totals


def market_fair_probabilities(frame: pd.DataFrame, method: str = "proportional") -> np.ndarray:
    """Margin-free market probabilities, NaN where no price exists."""
    from fv.odds.margin import remove_margin

    out = np.full((len(frame), 3), np.nan)
    for i, row in enumerate(frame.itertuples(index=False)):
        trio = (row.pre_h, row.pre_d, row.pre_a)
        if any(o is None or pd.isna(o) or float(o) <= 1.0 for o in trio):
            continue
        out[i] = remove_margin([float(o) for o in trio], method=method)
    return out


def combine(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    weight: float,
    suffixes: tuple[str, str] = ("_a", "_b"),
) -> pd.DataFrame:
    """Pool two prediction frames on match_id. Only matches present in both survive."""
    if frame_a.empty or frame_b.empty:
        return pd.DataFrame()
    keep = [c for c in frame_a.columns if c not in PROB_COLUMNS]
    merged = frame_a.merge(
        frame_b[["match_id"] + PROB_COLUMNS], on="match_id", suffixes=suffixes, how="inner"
    )
    if merged.empty:
        return pd.DataFrame()
    pa = merged[[f"{c}{suffixes[0]}" for c in PROB_COLUMNS]].to_numpy()
    pb = merged[[f"{c}{suffixes[1]}" for c in PROB_COLUMNS]].to_numpy()
    pooled = log_opinion_pool(pa, pb, weight)
    out = merged[[c for c in keep if c in merged.columns]].copy()
    out[PROB_COLUMNS] = pooled
    return out


def anchor_to_market(
    frame: pd.DataFrame,
    market_weight: float,
    method: str = "proportional",
) -> pd.DataFrame:
    """Blend model probabilities toward the margin-free market price.

    This is the layer that filters out bets driven by things the model cannot see -
    injuries, suspensions, cup rotation, dead rubbers. The market has all of that
    priced in; the model has none of it. Anchoring means an edge only survives when
    the model disagrees with the market by more than the anchor weight discounts,
    which is precisely the point: most model-market disagreement is model error.

    Matches without a price keep their model probabilities rather than being
    dropped, so probability-quality metrics stay comparable across stages.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    model = out[PROB_COLUMNS].to_numpy()
    market = market_fair_probabilities(out, method=method)
    have = np.isfinite(market).all(axis=1)
    if have.any():
        blended = log_opinion_pool(market[have], model[have], market_weight)
        model = model.copy()
        model[have] = blended
    out[PROB_COLUMNS] = model
    return out


# ---------------------------------------------------------------------------
# Weight tuning
# ---------------------------------------------------------------------------

def tune_pool_weight(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    grid: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Choose the ensemble weight on ``frame_a`` by log loss. Weight 1.0 = all A."""
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
    rows = []
    for w in grid:
        combined = combine(frame_a, frame_b, w)
        if combined.empty:
            continue
        rows.append({"weight": w, "n": len(combined), "log_loss": log_loss(combined)})
    table = pd.DataFrame(rows)
    if table.empty:
        return 0.5, table
    return float(table.loc[table["log_loss"].idxmin(), "weight"]), table


def tune_market_weight(
    frame: pd.DataFrame,
    grid: list[float] | None = None,
    method: str = "proportional",
) -> tuple[float, pd.DataFrame]:
    """Choose how much weight the market deserves, by log loss on validation."""
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
    rows = []
    for w in grid:
        anchored = anchor_to_market(frame, w, method=method)
        if anchored.empty:
            continue
        rows.append({"market_weight": w, "n": len(anchored), "log_loss": log_loss(anchored)})
    table = pd.DataFrame(rows)
    if table.empty:
        return 0.6, table
    return float(table.loc[table["log_loss"].idxmin(), "market_weight"]), table
