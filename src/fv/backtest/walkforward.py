"""Walk-forward backtest.

The design separates two concerns that are usually tangled together:

1. :func:`generate_predictions` walks forward through history, refitting the model
   before each match-week and predicting that week. This is the expensive part and
   the part where lookahead leakage would hide.
2. :func:`simulate_betting` replays those predictions against the recorded prices
   under a set of betting rules, tracking one shared bankroll.

Keeping them apart means changing the edge threshold or Kelly fraction re-runs in
under a second instead of refitting thousands of models, and it makes the leakage
question purely about step 1.

**The leakage rule**: predictions for a match-week are produced by a model trained
only on matches that kicked off strictly before that week's Monday 00:00 UTC. The
cutoff is recorded on every prediction as ``trained_through`` so the property can be
verified after the fact, not merely asserted. ``tests/test_no_lookahead.py`` checks
it both structurally and by poisoning the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text

from fv.config import Config, load_config
from fv.db.session import get_engine
from fv.models.dixon_coles import DixonColesFit, fit_dixon_coles
from fv.odds.edge import Thresholds, edge
from fv.odds.kelly import StakeRules, stake_for
from fv.odds.margin import remove_margin
from fv.odds.settlement import clv as clv_of

MATCH_QUERY = """
SELECT
    m.id            AS match_id,
    m.league_code   AS league_code,
    m.season        AS season,
    m.kickoff_utc   AS kickoff_utc,
    th.canonical_name AS home,
    ta.canonical_name AS away,
    m.fthg AS fthg, m.ftag AS ftag, m.ftr AS ftr,
    MAX(CASE WHEN o.odds_type='pre'     AND o.selection='H' THEN o.decimal_odds END) AS pre_h,
    MAX(CASE WHEN o.odds_type='pre'     AND o.selection='D' THEN o.decimal_odds END) AS pre_d,
    MAX(CASE WHEN o.odds_type='pre'     AND o.selection='A' THEN o.decimal_odds END) AS pre_a,
    MAX(CASE WHEN o.odds_type='closing' AND o.selection='H' THEN o.decimal_odds END) AS close_h,
    MAX(CASE WHEN o.odds_type='closing' AND o.selection='D' THEN o.decimal_odds END) AS close_d,
    MAX(CASE WHEN o.odds_type='closing' AND o.selection='A' THEN o.decimal_odds END) AS close_a
FROM matches m
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
LEFT JOIN odds o ON o.match_id = m.id AND o.bookmaker = 'B365' AND o.market = '1X2'
WHERE m.league_code = :league AND m.status = 'played'
GROUP BY m.id
ORDER BY m.kickoff_utc
"""


def load_matches(league_code: str, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(text(MATCH_QUERY), conn, params={"league": league_code})
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    return df


def week_start(ts: pd.Series) -> pd.Series:
    """Monday 00:00 UTC of the week each timestamp falls in.

    Betting weeks, not calendar weeks: the slip is built once at the start of the
    week, so the model must not see any of that week's results, including midweek
    fixtures played before a weekend match.
    """
    return (ts - pd.to_timedelta(ts.dt.dayofweek, unit="D")).dt.normalize()


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    fits: dict[pd.Timestamp, DixonColesFit] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


def generate_predictions(
    league_code: str,
    test_from: str | pd.Timestamp,
    test_to: str | pd.Timestamp | None = None,
    xi: float = 0.0018,
    cfg: Config | None = None,
    matches: pd.DataFrame | None = None,
    min_train_matches: int = 200,
    max_goals: int = 10,
    keep_fits: bool = False,
) -> PredictionResult:
    """Walk forward week by week, training only on the past.

    ``test_from`` / ``test_to`` are inclusive bounds on the match date.
    """
    cfg = cfg or load_config()
    df = matches if matches is not None else load_matches(league_code, cfg)
    if df.empty:
        return PredictionResult(pd.DataFrame())

    df = df.dropna(subset=["fthg", "ftag"]).copy()
    df["week"] = week_start(df["kickoff_utc"])

    start = pd.Timestamp(test_from)
    end = pd.Timestamp(test_to) if test_to is not None else df["kickoff_utc"].max()

    target_weeks = sorted(w for w in df["week"].unique() if start <= w <= end)

    rows: list[dict] = []
    diags: list[dict] = []
    fits: dict[pd.Timestamp, DixonColesFit] = {}
    previous: DixonColesFit | None = None

    for week in target_weeks:
        cutoff = week  # Monday 00:00 UTC; nothing at or after this is visible.
        train = df[df["kickoff_utc"] < cutoff]
        if len(train) < min_train_matches:
            continue

        days_before = (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400.0
        try:
            fit = fit_dixon_coles(
                train["home"].to_numpy(),
                train["away"].to_numpy(),
                train["fthg"].to_numpy(),
                train["ftag"].to_numpy(),
                days_before,
                xi=xi,
                init=previous,
            )
        except ValueError:
            continue
        previous = fit
        if keep_fits:
            fits[week] = fit

        diags.append(
            {
                "league_code": league_code,
                "week": week,
                "n_train": len(train),
                "effective_sample": fit.effective_sample,
                "n_teams": len(fit.teams),
                "home_adv": fit.home_adv,
                "rho": fit.rho,
                "converged": fit.converged,
            }
        )

        target = df[df["week"] == week]
        for row in target.itertuples(index=False):
            if row.home not in fit.index or row.away not in fit.index:
                continue
            p_home, p_draw, p_away = fit.predict(row.home, row.away, max_goals=max_goals)
            rows.append(
                {
                    "match_id": row.match_id,
                    "league_code": row.league_code,
                    "season": row.season,
                    "kickoff_utc": row.kickoff_utc,
                    "week": week,
                    "trained_through": cutoff,
                    "home": row.home,
                    "away": row.away,
                    "ftr": row.ftr,
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                    "n_home_matches": fit.n_matches.get(row.home, 0),
                    "n_away_matches": fit.n_matches.get(row.away, 0),
                    "pre_h": row.pre_h,
                    "pre_d": row.pre_d,
                    "pre_a": row.pre_a,
                    "close_h": row.close_h,
                    "close_d": row.close_d,
                    "close_a": row.close_a,
                }
            )

    return PredictionResult(
        predictions=pd.DataFrame(rows),
        fits=fits,
        diagnostics=pd.DataFrame(diags),
    )


def tune_xi(
    league_code: str,
    validation_from: str | pd.Timestamp,
    validation_to: str | pd.Timestamp,
    grid: list[float],
    cfg: Config | None = None,
    matches: pd.DataFrame | None = None,
) -> tuple[float, pd.DataFrame]:
    """Pick the time-decay rate by walk-forward log loss on a validation window.

    The window must end before the test period starts, so the choice of ``xi`` never
    sees test data — tuning a hyperparameter on the test set is leakage just as much
    as training on it is, and it is the easier of the two to do by accident.

    Log loss rather than Brier or ROI: it is the proper scoring rule most sensitive
    to the confident-but-wrong predictions that Kelly staking punishes hardest, and
    unlike ROI it doesn't depend on the betting rules we're about to tune separately.
    """
    from fv.backtest.metrics import brier_score, log_loss

    cfg = cfg or load_config()
    df = matches if matches is not None else load_matches(league_code, cfg)

    rows = []
    for xi in grid:
        result = generate_predictions(
            league_code, validation_from, validation_to, xi=xi, cfg=cfg, matches=df
        )
        preds = result.predictions
        if preds.empty:
            continue
        rows.append(
            {
                "xi": xi,
                "half_life_days": float("inf") if xi == 0 else float(np.log(2) / xi),
                "n": len(preds),
                "log_loss": log_loss(preds),
                "brier": brier_score(preds),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return grid[0], table
    best = float(table.loc[table["log_loss"].idxmin(), "xi"])
    return best, table


# --------------------------------------------------------------------------
# Betting simulation
# --------------------------------------------------------------------------

SELECTION_COLUMNS = {
    "H": ("p_home", "pre_h", "close_h"),
    "D": ("p_draw", "pre_d", "close_d"),
    "A": ("p_away", "pre_a", "close_a"),
}


@dataclass
class BettingParams:
    starting_bankroll: float = 1000.0
    thresholds: Thresholds = field(default_factory=Thresholds)
    stake_rules: StakeRules = field(default_factory=StakeRules)
    max_bets_per_week: int = 8
    min_team_matches: int = 6
    margin_method: str = "proportional"
    weekly_stop_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.25
    # In live use the drawdown pause stops betting until it is manually reset. In a
    # backtest that would end the evaluation at the first bad run and hide
    # everything after it, so breaches are recorded and the simulation continues.
    # Set True to see where a live account would actually have stopped.
    halt_on_max_drawdown: bool = False


@dataclass
class SimulationResult:
    bets: pd.DataFrame
    equity: pd.DataFrame
    halted_at: pd.Timestamp | None = None
    drawdown_breaches: list = field(default_factory=list)
    candidates_considered: int = 0
    skipped_no_odds: int = 0
    skipped_new_team: int = 0


def find_candidates(predictions: pd.DataFrame, params: BettingParams) -> pd.DataFrame:
    """Score every selection and keep the ones that clear the thresholds.

    Margin is stripped from the market price so ``market_prob_fair`` is comparable
    with the model, but the edge is computed against the *actual* price on offer,
    because that is what determines the payout.
    """
    if predictions.empty:
        return pd.DataFrame()

    out: list[dict] = []
    for row in predictions.itertuples(index=False):
        odds_triplet = (row.pre_h, row.pre_d, row.pre_a)
        if any(o is None or pd.isna(o) or float(o) <= 1.0 for o in odds_triplet):
            continue
        fair = remove_margin([float(o) for o in odds_triplet], method=params.margin_method)
        raw = np.array([1.0 / float(o) for o in odds_triplet])

        enough_history = (
            row.n_home_matches >= params.min_team_matches
            and row.n_away_matches >= params.min_team_matches
        )

        for i, sel in enumerate(("H", "D", "A")):
            p_col, pre_col, close_col = SELECTION_COLUMNS[sel]
            model_p = float(getattr(row, p_col))
            price = float(getattr(row, pre_col))
            if not (params.thresholds.min_odds <= price <= params.thresholds.max_odds):
                continue
            e = edge(model_p, price)
            if e < params.thresholds.min_edge:
                continue
            close = getattr(row, close_col)
            out.append(
                {
                    "match_id": row.match_id,
                    "league_code": row.league_code,
                    "season": row.season,
                    "kickoff_utc": row.kickoff_utc,
                    "week": row.week,
                    "home": row.home,
                    "away": row.away,
                    "selection": sel,
                    "model_prob": model_p,
                    "market_prob_raw": float(raw[i]),
                    "market_prob_fair": float(fair[i]),
                    "odds_taken": price,
                    "closing_odds": None if close is None or pd.isna(close) else float(close),
                    "edge": e,
                    "ftr": row.ftr,
                    "enough_history": enough_history,
                }
            )
    return pd.DataFrame(out)


def simulate_betting(
    predictions: pd.DataFrame,
    params: BettingParams | None = None,
) -> SimulationResult:
    """Replay the candidate bets chronologically against one shared bankroll.

    Bankroll, weekly cap, stop-loss and drawdown pause are all global rather than
    per-league, because that is how the money actually behaves.
    """
    params = params or BettingParams()
    candidates = find_candidates(predictions, params)
    if candidates.empty:
        return SimulationResult(pd.DataFrame(), pd.DataFrame())

    skipped_new_team = int((~candidates["enough_history"]).sum())
    candidates = candidates[candidates["enough_history"]]

    bankroll = params.starting_bankroll
    peak = bankroll
    halted_at: pd.Timestamp | None = None
    breaches: list[dict] = []
    in_breach = False
    bet_rows: list[dict] = []
    equity_rows: list[dict] = []

    for week, group in candidates.groupby("week", sort=True):
        if halted_at is not None:
            break

        week_start_bankroll = bankroll
        # Rank by edge, take the best, cap the count.
        chosen = group.sort_values("edge", ascending=False).head(params.max_bets_per_week)
        # Settle in kickoff order so the bankroll path is chronological.
        chosen = chosen.sort_values("kickoff_utc")

        week_loss = 0.0
        for row in chosen.itertuples(index=False):
            stake = stake_for(row.model_prob, row.odds_taken, bankroll, params.stake_rules)
            if stake <= 0:
                continue

            # Weekly stop-loss: stop taking new bets once the week is far enough down.
            if week_loss >= params.weekly_stop_loss_pct * week_start_bankroll:
                break

            won = row.ftr == row.selection
            pnl = stake * (row.odds_taken - 1.0) if won else -stake
            bankroll += pnl
            peak = max(peak, bankroll)
            if pnl < 0:
                week_loss += -pnl

            bet_rows.append(
                {
                    "match_id": row.match_id,
                    "league_code": row.league_code,
                    "season": row.season,
                    "kickoff_utc": row.kickoff_utc,
                    "week": week,
                    "home": row.home,
                    "away": row.away,
                    "selection": row.selection,
                    "model_prob": row.model_prob,
                    "market_prob_raw": row.market_prob_raw,
                    "market_prob_fair": row.market_prob_fair,
                    "odds_taken": row.odds_taken,
                    "closing_odds": row.closing_odds,
                    "edge": row.edge,
                    "stake": stake,
                    "result": "W" if won else "L",
                    "pnl": pnl,
                    "bankroll_after": bankroll,
                    "clv": clv_of(row.odds_taken, row.closing_odds),
                }
            )

            drawdown = (peak - bankroll) / peak if peak > 0 else 0.0
            if drawdown >= params.max_drawdown_pct:
                # Record the transition into breach, not every bet while in one.
                if not in_breach:
                    breaches.append(
                        {
                            "kickoff_utc": row.kickoff_utc,
                            "bankroll": bankroll,
                            "peak": peak,
                            "drawdown": drawdown,
                        }
                    )
                    in_breach = True
                if params.halt_on_max_drawdown:
                    halted_at = row.kickoff_utc
                    break
            elif drawdown < params.max_drawdown_pct * 0.5:
                # Recovered well clear of the threshold; a later breach is a new event.
                in_breach = False

        equity_rows.append({"week": week, "bankroll": bankroll, "peak": peak})

    return SimulationResult(
        bets=pd.DataFrame(bet_rows),
        equity=pd.DataFrame(equity_rows),
        halted_at=halted_at,
        drawdown_breaches=breaches,
        candidates_considered=len(candidates),
        skipped_new_team=skipped_new_team,
    )
