"""Over/under 2.5 and BTTS.

Both come from the same fitted Dixon-Coles score matrix as 1X2 — they are different
aggregations of one joint distribution over scorelines, not separate models. Beyond
being simpler, that guarantees the three markets can never contradict one another,
which independently fitted models could.

The two markets are in very different evidential positions, and the code keeps them
apart for that reason:

* **Over/under 2.5** has bet365 prices in football-data from 2019-20, pre and
  closing. It can be backtested exactly like 1X2, and it is.
* **BTTS** has no historical prices in any free source. The model can produce a
  probability, but there is nothing to measure an edge or CLV against. Under the
  project's own rule — an upgrade ships only if it beats the previous stage
  out-of-sample — BTTS is therefore **not validated** and must be labelled that way
  wherever it appears. Prices for it only start accumulating once The Odds API
  snapshots begin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text

from fv.backtest.walkforward import week_start
from fv.config import Config, load_config
from fv.db.session import get_engine
from fv.models.dixon_coles import fit_dixon_coles

MARKET_QUERY = """
SELECT
    m.id AS match_id, m.league_code, m.season, m.kickoff_utc,
    th.canonical_name AS home, ta.canonical_name AS away,
    m.fthg, m.ftag,
    x.home_xg, x.away_xg,
    MAX(CASE WHEN o.market='OU25' AND o.odds_type='pre'     AND o.selection='O'
        THEN o.decimal_odds END) AS pre_over,
    MAX(CASE WHEN o.market='OU25' AND o.odds_type='pre'     AND o.selection='U'
        THEN o.decimal_odds END) AS pre_under,
    MAX(CASE WHEN o.market='OU25' AND o.odds_type='closing' AND o.selection='O'
        THEN o.decimal_odds END) AS close_over,
    MAX(CASE WHEN o.market='OU25' AND o.odds_type='closing' AND o.selection='U'
        THEN o.decimal_odds END) AS close_under
FROM matches m
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
LEFT JOIN match_xg x ON x.match_id = m.id
LEFT JOIN odds o ON o.match_id = m.id AND o.bookmaker='B365'
WHERE m.league_code = :league AND m.status='played'
GROUP BY m.id
ORDER BY m.kickoff_utc
"""


def load_market_matches(league_code: str, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(text(MARKET_QUERY), conn, params={"league": league_code})
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    return df


def generate_ou_predictions(
    league_code: str,
    test_from: str | pd.Timestamp,
    test_to: str | pd.Timestamp | None = None,
    xi: float = 0.0015,
    xg_weight: float = 0.0,
    cfg: Config | None = None,
    matches: pd.DataFrame | None = None,
    line: float = 2.5,
    min_train_matches: int = 200,
) -> pd.DataFrame:
    """Walk-forward over/under predictions, on the same cutoff rule as 1X2.

    Training stops strictly before the Monday of the week being predicted, exactly as
    in :mod:`fv.backtest.walkforward`; the only difference is which aggregation of
    the score matrix comes out.
    """
    cfg = cfg or load_config()
    df = matches if matches is not None else load_market_matches(league_code, cfg)
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["fthg", "ftag"]).copy()
    df["week"] = week_start(df["kickoff_utc"])

    start = pd.Timestamp(test_from)
    end = pd.Timestamp(test_to) if test_to is not None else df["kickoff_utc"].max()
    weeks = sorted(w for w in df["week"].unique() if start <= w <= end)

    rows: list[dict] = []
    previous = None

    for wk in weeks:
        train = df[df["kickoff_utc"] < wk]
        if len(train) < min_train_matches:
            continue
        days = (wk - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400.0
        try:
            fit = fit_dixon_coles(
                train["home"].to_numpy(), train["away"].to_numpy(),
                train["fthg"].to_numpy(), train["ftag"].to_numpy(), days,
                xi=xi, init=previous,
                home_xg=train["home_xg"].to_numpy(), away_xg=train["away_xg"].to_numpy(),
                xg_weight=xg_weight,
            )
        except ValueError:
            continue
        previous = fit

        for r in df[df["week"] == wk].itertuples(index=False):
            if r.home not in fit.index or r.away not in fit.index:
                continue
            p_over = fit.prob_over(r.home, r.away, line=line)
            total = float(r.fthg) + float(r.ftag)
            rows.append({
                "match_id": r.match_id, "league_code": r.league_code, "season": r.season,
                "kickoff_utc": r.kickoff_utc, "week": wk, "trained_through": wk,
                "home": r.home, "away": r.away,
                "p_over": p_over, "p_under": 1.0 - p_over,
                "actual_over": total > line,
                "n_home_matches": fit.n_matches.get(r.home, 0),
                "n_away_matches": fit.n_matches.get(r.away, 0),
                "pre_over": r.pre_over, "pre_under": r.pre_under,
                "close_over": r.close_over, "close_under": r.close_under,
            })

    return pd.DataFrame(rows)


def generate_btts_predictions(
    league_code: str,
    test_from: str | pd.Timestamp,
    test_to: str | pd.Timestamp | None = None,
    xi: float = 0.0015,
    xg_weight: float = 0.0,
    cfg: Config | None = None,
    matches: pd.DataFrame | None = None,
    min_train_matches: int = 200,
) -> pd.DataFrame:
    """Walk-forward BTTS probabilities.

    Produces predictions and scores their *calibration*, which is all that can
    honestly be done: there are no historical BTTS prices, so no edge, ROI or CLV is
    computable. Calibration still answers a real question — whether the model's BTTS
    numbers are trustworthy at all — and that has to be established before any price
    arriving from The Odds API could be acted on.
    """
    cfg = cfg or load_config()
    df = matches if matches is not None else load_market_matches(league_code, cfg)
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["fthg", "ftag"]).copy()
    df["week"] = week_start(df["kickoff_utc"])
    start = pd.Timestamp(test_from)
    end = pd.Timestamp(test_to) if test_to is not None else df["kickoff_utc"].max()
    weeks = sorted(w for w in df["week"].unique() if start <= w <= end)

    rows: list[dict] = []
    previous = None
    for wk in weeks:
        train = df[df["kickoff_utc"] < wk]
        if len(train) < min_train_matches:
            continue
        days = (wk - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400.0
        try:
            fit = fit_dixon_coles(
                train["home"].to_numpy(), train["away"].to_numpy(),
                train["fthg"].to_numpy(), train["ftag"].to_numpy(), days,
                xi=xi, init=previous,
                home_xg=train["home_xg"].to_numpy(), away_xg=train["away_xg"].to_numpy(),
                xg_weight=xg_weight,
            )
        except ValueError:
            continue
        previous = fit
        for r in df[df["week"] == wk].itertuples(index=False):
            if r.home not in fit.index or r.away not in fit.index:
                continue
            rows.append({
                "match_id": r.match_id, "league_code": r.league_code, "season": r.season,
                "kickoff_utc": r.kickoff_utc, "home": r.home, "away": r.away,
                "p_btts": fit.prob_btts(r.home, r.away),
                "actual_btts": (float(r.fthg) >= 1) and (float(r.ftag) >= 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def binary_calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"predicted": float(p[m].mean()), "observed": float(y[m].mean()),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


@dataclass
class MarketBacktest:
    predictions: pd.DataFrame
    bets: pd.DataFrame
    log_loss: float
    brier: float
    market_log_loss: float | None
    market_brier: float | None


def backtest_ou(
    predictions: pd.DataFrame,
    min_edge: float = 0.04,
    min_odds: float = 1.50,
    max_odds: float = 4.00,
    stake: float = 10.0,
    margin_method: str = "proportional",
) -> MarketBacktest:
    """Score over/under predictions and simulate flat-stake betting on them."""
    from fv.odds.margin import remove_margin
    from fv.odds.settlement import clv as clv_of

    if predictions.empty:
        return MarketBacktest(predictions, pd.DataFrame(), float("nan"), float("nan"), None, None)

    y = predictions["actual_over"].to_numpy(dtype=float)
    p = predictions["p_over"].to_numpy(dtype=float)
    ll, br = binary_log_loss(p, y), binary_brier(p, y)

    priced = predictions.dropna(subset=["pre_over", "pre_under"])
    market_ll = market_br = None
    bets: list[dict] = []

    if not priced.empty:
        fair_over = []
        for r in priced.itertuples(index=False):
            fair = remove_margin([float(r.pre_over), float(r.pre_under)], method=margin_method)
            fair_over.append(fair[0])
        fair_over = np.asarray(fair_over)
        y_p = priced["actual_over"].to_numpy(dtype=float)
        market_ll = binary_log_loss(fair_over, y_p)
        market_br = binary_brier(fair_over, y_p)

        for i, r in enumerate(priced.itertuples(index=False)):
            for sel, model_p, price, close in (
                ("O", float(r.p_over), float(r.pre_over), r.close_over),
                ("U", float(r.p_under), float(r.pre_under), r.close_under),
            ):
                if not (min_odds <= price <= max_odds):
                    continue
                edge = model_p * price - 1.0
                if edge < min_edge:
                    continue
                won = bool(r.actual_over) if sel == "O" else not bool(r.actual_over)
                bets.append({
                    "match_id": r.match_id, "league_code": r.league_code, "season": r.season,
                    "kickoff_utc": r.kickoff_utc, "market": "OU25", "selection": sel,
                    "model_prob": model_p,
                    "market_prob_fair": fair_over[i] if sel == "O" else 1 - fair_over[i],
                    "odds_taken": price,
                    "closing_odds": None if close is None or pd.isna(close) else float(close),
                    "edge": edge, "stake": stake,
                    "result": "W" if won else "L",
                    "pnl": stake * (price - 1.0) if won else -stake,
                    "clv": clv_of(price, None if close is None or pd.isna(close) else float(close)),
                })

    return MarketBacktest(predictions, pd.DataFrame(bets), ll, br, market_ll, market_br)
