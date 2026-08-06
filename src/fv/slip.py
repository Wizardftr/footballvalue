"""Weekly slip generation.

Fits the current model on everything played to date, prices the upcoming fixtures,
and ranks whatever clears the thresholds. The slip is a recommendation to place
manually on bet365 — nothing here talks to a bookmaker.

The per-league weights (decay rate, xG blend, ensemble pool, market anchor) come
from the last ``fv stages`` run. If that has never been run, the slip falls back to
config defaults and says so, because a slip built on untuned weights is a different
thing from one built on validated ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from fv.backtest.stages import PROB_COLUMNS, anchor_to_market, combine, log_opinion_pool
from fv.backtest.walkforward import load_matches, week_start
from fv.config import PROJECT_ROOT, Config, load_config
from fv.db.session import get_engine
from fv.models.dixon_coles import fit_dixon_coles
from fv.models.features import build_features
from fv.models.lgbm import fit_lgbm
from fv.odds.edge import Thresholds, edge
from fv.odds.kelly import StakeRules, stake_for
from fv.odds.margin import remove_margin

WEIGHTS_PATH = PROJECT_ROOT / "data" / "tuned_weights.json"

FIXTURE_QUERY = """
SELECT
    m.id AS match_id, m.league_code, m.season, m.kickoff_utc,
    th.canonical_name AS home, ta.canonical_name AS away,
    MAX(CASE WHEN o.selection='H' THEN o.decimal_odds END) AS pre_h,
    MAX(CASE WHEN o.selection='D' THEN o.decimal_odds END) AS pre_d,
    MAX(CASE WHEN o.selection='A' THEN o.decimal_odds END) AS pre_a
FROM matches m
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
LEFT JOIN odds o ON o.match_id = m.id AND o.bookmaker='B365'
     AND o.market='1X2' AND o.odds_type='pre'
WHERE m.status = 'scheduled' AND m.kickoff_utc >= :now
GROUP BY m.id
ORDER BY m.kickoff_utc
"""


def load_tuned_weights(path: Path | None = None) -> dict[str, dict]:
    """Per-league weights from the last ``fv stages`` run, keyed by league code."""
    path = path or WEIGHTS_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def save_tuned_weights(rows: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist tuned weights so the slip uses the same numbers the backtest validated."""
    path = path or WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        r.league: {
            "xi": float(r.xi),
            "xg_weight": float(r.xg_weight),
            "dc_weight": float(r.dc_weight_in_ensemble),
            "market_weight": float(r.market_weight),
        }
        for r in rows.itertuples(index=False)
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_upcoming(cfg: Config | None = None, now: datetime | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    now = now or datetime.utcnow()
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(text(FIXTURE_QUERY), conn, params={"now": now.isoformat(sep=" ")})
    if df.empty:
        return df
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    enabled = {lg.code for lg in cfg.enabled_leagues}
    return df[df["league_code"].isin(enabled)].reset_index(drop=True)


def predict_fixtures(
    league_code: str,
    fixtures: pd.DataFrame,
    cfg: Config | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Price one league's upcoming fixtures with the full stage stack."""
    cfg = cfg or load_config()
    w = weights or {}
    xi = w.get("xi", cfg.dixon_coles.get("xi") or 0.0015)
    xg_weight = w.get("xg_weight", 0.0)
    dc_weight = w.get("dc_weight", 1.0)
    market_weight = w.get("market_weight", 0.6)

    history = load_matches(league_code, cfg)
    played = history.dropna(subset=["fthg", "ftag"])
    if played.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp(fixtures["kickoff_utc"].min()).normalize()
    train = played[played["kickoff_utc"] < cutoff]
    if len(train) < 200:
        return pd.DataFrame()

    days_before = (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400.0
    fit = fit_dixon_coles(
        train["home"].to_numpy(),
        train["away"].to_numpy(),
        train["fthg"].to_numpy(),
        train["ftag"].to_numpy(),
        days_before,
        xi=xi,
        home_xg=train["home_xg"].to_numpy(),
        away_xg=train["away_xg"].to_numpy(),
        xg_weight=xg_weight,
    )

    rows = []
    for f in fixtures.itertuples(index=False):
        if f.home not in fit.index or f.away not in fit.index:
            rows.append({**f._asdict(), "p_home": np.nan, "p_draw": np.nan, "p_away": np.nan,
                         "n_home_matches": 0, "n_away_matches": 0, "unknown_team": True})
            continue
        p_home, p_draw, p_away = fit.predict(f.home, f.away)
        rows.append({
            **f._asdict(),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "n_home_matches": fit.n_matches.get(f.home, 0),
            "n_away_matches": fit.n_matches.get(f.away, 0),
            "unknown_team": False,
        })
    dc = pd.DataFrame(rows)
    dc = dc[~dc["unknown_team"]].drop(columns=["unknown_team"])
    if dc.empty:
        return dc

    # LightGBM on the same history, then pool and anchor.
    if dc_weight < 1.0:
        combined_history = pd.concat([played, fixtures.assign(fthg=np.nan, ftag=np.nan, ftr=None)],
                                     ignore_index=True)
        feats = build_features(combined_history)
        lg_fit = fit_lgbm(feats[feats["match_id"].isin(played["match_id"])])
        if lg_fit is not None:
            target = feats[feats["match_id"].isin(dc["match_id"])]
            if not target.empty:
                probs = lg_fit.predict(target)
                lg = target[["match_id"]].copy()
                lg[PROB_COLUMNS] = probs
                merged = dc.merge(lg, on="match_id", suffixes=("", "_lg"), how="left")
                have = merged[[f"{c}_lg" for c in PROB_COLUMNS]].notna().all(axis=1)
                if have.any():
                    pooled = log_opinion_pool(
                        merged.loc[have, PROB_COLUMNS].to_numpy(),
                        merged.loc[have, [f"{c}_lg" for c in PROB_COLUMNS]].to_numpy(),
                        dc_weight,
                    )
                    merged.loc[have, PROB_COLUMNS] = pooled
                dc = merged.drop(columns=[f"{c}_lg" for c in PROB_COLUMNS])

    dc = anchor_to_market(dc, market_weight, method=cfg.odds.get("margin_method", "proportional"))
    dc["market_weight_used"] = market_weight
    return dc


@dataclass
class Slip:
    selections: pd.DataFrame
    all_candidates: pd.DataFrame
    bankroll: float
    generated_at: datetime = field(default_factory=datetime.utcnow)
    untuned_leagues: list[str] = field(default_factory=list)

    @property
    def total_stake(self) -> float:
        return float(self.selections["stake"].sum()) if not self.selections.empty else 0.0


def generate_slip(
    cfg: Config | None = None,
    bankroll: float | None = None,
    now: datetime | None = None,
    weights_path: Path | None = None,
) -> Slip:
    """Build this week's recommended slip."""
    cfg = cfg or load_config()
    b = cfg.betting
    bankroll = bankroll if bankroll is not None else b.get("starting_bankroll", 1000.0)
    thresholds = Thresholds(
        min_edge=b.get("min_edge", 0.04),
        min_odds=b.get("min_odds", 1.50),
        max_odds=b.get("max_odds", 4.00),
    )
    rules = StakeRules(
        kelly_fraction=b.get("kelly_fraction", 0.25),
        max_stake_pct=b.get("max_stake_pct", 0.02),
        rounding=b.get("stake_rounding", 0.50),
        min_stake=b.get("min_stake", 1.00),
    )
    min_team_matches = cfg.dixon_coles.get("min_team_matches", 6)

    upcoming = load_upcoming(cfg, now=now)
    if upcoming.empty:
        return Slip(pd.DataFrame(), pd.DataFrame(), bankroll)

    tuned = load_tuned_weights(weights_path)
    untuned = []
    priced = []
    for code, group in upcoming.groupby("league_code"):
        if code not in tuned:
            untuned.append(code)
        frame = predict_fixtures(code, group.reset_index(drop=True), cfg, tuned.get(code))
        if not frame.empty:
            priced.append(frame)

    if not priced:
        return Slip(pd.DataFrame(), pd.DataFrame(), bankroll, untuned_leagues=untuned)

    priced_df = pd.concat(priced, ignore_index=True)

    candidates = []
    for row in priced_df.itertuples(index=False):
        trio = (row.pre_h, row.pre_d, row.pre_a)
        if any(o is None or pd.isna(o) or float(o) <= 1.0 for o in trio):
            continue
        fair = remove_margin([float(o) for o in trio],
                             method=cfg.odds.get("margin_method", "proportional"))
        enough = (row.n_home_matches >= min_team_matches
                  and row.n_away_matches >= min_team_matches)
        for i, (sel, p_col, o_col) in enumerate(
            [("H", "p_home", "pre_h"), ("D", "p_draw", "pre_d"), ("A", "p_away", "pre_a")]
        ):
            price = float(getattr(row, o_col))
            model_p = float(getattr(row, p_col))
            e = edge(model_p, price)
            candidates.append({
                "match_id": row.match_id,
                "league_code": row.league_code,
                "kickoff_utc": row.kickoff_utc,
                "home": row.home,
                "away": row.away,
                "selection": sel,
                "model_prob": model_p,
                "market_prob_fair": float(fair[i]),
                "odds": price,
                "edge": e,
                "in_odds_range": thresholds.min_odds <= price <= thresholds.max_odds,
                "clears_edge": e >= thresholds.min_edge,
                "enough_history": enough,
            })

    all_candidates = pd.DataFrame(candidates)
    if all_candidates.empty:
        return Slip(pd.DataFrame(), all_candidates, bankroll, untuned_leagues=untuned)

    qualifying = all_candidates[
        all_candidates["in_odds_range"]
        & all_candidates["clears_edge"]
        & all_candidates["enough_history"]
    ].copy()

    if qualifying.empty:
        return Slip(pd.DataFrame(), all_candidates, bankroll, untuned_leagues=untuned)

    qualifying = qualifying.sort_values("edge", ascending=False).head(
        b.get("max_bets_per_week", 8)
    )
    qualifying["stake"] = [
        stake_for(r.model_prob, r.odds, bankroll, rules) for r in qualifying.itertuples(index=False)
    ]
    qualifying = qualifying[qualifying["stake"] > 0]
    qualifying = qualifying.sort_values("kickoff_utc").reset_index(drop=True)

    return Slip(qualifying, all_candidates, bankroll, untuned_leagues=untuned)


SELECTION_WORDS = {"H": "Home", "D": "Draw", "A": "Away"}


def slip_to_text(slip: Slip) -> str:
    """Plain-text slip, formatted to be easy to work through on bet365 by hand."""
    lines = [
        "FOOTBALLVALUE - WEEKLY SLIP",
        f"Generated {slip.generated_at:%Y-%m-%d %H:%M} UTC",
        f"Bankroll {slip.bankroll:,.2f}",
        "",
    ]
    if slip.selections.empty:
        lines.append("No qualifying selections this week.")
        lines.append("")
        lines.append("This is a normal outcome, not a failure. The thresholds exist")
        lines.append("to say no; a week with nothing worth backing is the system working.")
        return "\n".join(lines)

    lines.append(f"{len(slip.selections)} singles, total stake {slip.total_stake:,.2f}")
    lines.append("SINGLES ONLY - do not combine these into an accumulator.")
    lines.append("")
    for i, r in enumerate(slip.selections.itertuples(index=False), start=1):
        lines.append(f"{i}. {r.kickoff_utc:%a %d %b %H:%M}  [{r.league_code}]")
        lines.append(f"   {r.home} v {r.away}")
        lines.append(f"   {SELECTION_WORDS[r.selection]} @ {r.odds:.2f}   stake {r.stake:,.2f}")
        lines.append(f"   model {r.model_prob:.1%} vs market {r.market_prob_fair:.1%}"
                     f"   edge {r.edge:+.1%}")
        lines.append("")
    lines.append("Prices move. If the price has shortened past the listed odds,")
    lines.append("the edge may be gone - skip it rather than taking a worse number.")
    return "\n".join(lines)


def slip_to_csv(slip: Slip) -> str:
    if slip.selections.empty:
        return "kickoff_utc,league,home,away,selection,odds,stake,model_prob,edge\n"
    out = slip.selections[[
        "kickoff_utc", "league_code", "home", "away", "selection",
        "odds", "stake", "model_prob", "market_prob_fair", "edge",
    ]].copy()
    return out.to_csv(index=False)
