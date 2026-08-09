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
from fv.odds.kelly import StakeRules, round_stake, stake_for
from fv.odds.margin import remove_margin

WEIGHTS_PATH = PROJECT_ROOT / "data" / "tuned_weights.json"

FIXTURE_QUERY = """
SELECT
    m.id AS match_id, m.league_code, m.season, m.kickoff_utc,
    th.canonical_name AS home, ta.canonical_name AS away,
    MAX(CASE WHEN o.market='1X2' AND o.selection='H' THEN o.decimal_odds END) AS pre_h,
    MAX(CASE WHEN o.market='1X2' AND o.selection='D' THEN o.decimal_odds END) AS pre_d,
    MAX(CASE WHEN o.market='1X2' AND o.selection='A' THEN o.decimal_odds END) AS pre_a,
    MAX(CASE WHEN o.market='OU25' AND o.selection='O' THEN o.decimal_odds END) AS pre_o25,
    MAX(CASE WHEN o.market='OU25' AND o.selection='U' THEN o.decimal_odds END) AS pre_u25
FROM matches m
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
LEFT JOIN odds o ON o.match_id = m.id AND o.bookmaker='B365' AND o.odds_type='pre'
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
                         "p_over": np.nan, "n_home_matches": 0, "n_away_matches": 0,
                         "unknown_team": True})
            continue
        p_home, p_draw, p_away = fit.predict(f.home, f.away)
        rows.append({
            **f._asdict(),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            # Over/under comes off the same fitted score matrix as the 1X2 numbers,
            # so the two markets can never contradict each other.
            "p_over": fit.prob_over(f.home, f.away, 2.5),
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

    method = cfg.odds.get("margin_method", "proportional")
    dc = anchor_to_market(dc, market_weight, method=method)
    dc = anchor_over_under(dc, market_weight, method=method)
    dc["market_weight_used"] = market_weight
    return dc


def anchor_over_under(
    frame: pd.DataFrame, market_weight: float, method: str = "proportional"
) -> pd.DataFrame:
    """The same anchoring as 1X2, applied to the two-outcome goals market.

    The pooling helper is shape-agnostic, so this is the identical operation on two
    columns rather than three: blend the model's P(over 2.5) toward the margin-free
    price by the league's tuned weight. Using the same weight is deliberate — it
    answers the same question (how far to trust this model against this market in
    this league), and inventing a second untuned weight would be a number nothing
    had validated.
    """
    if frame.empty or "p_over" not in frame.columns:
        return frame
    out = frame.copy()
    model = np.column_stack([out["p_over"].to_numpy(dtype=float),
                             1.0 - out["p_over"].to_numpy(dtype=float)])
    market = np.full_like(model, np.nan)
    for i, row in enumerate(out.itertuples(index=False)):
        pair = (getattr(row, "pre_o25", None), getattr(row, "pre_u25", None))
        if any(o is None or pd.isna(o) or float(o) <= 1.0 for o in pair):
            continue
        market[i] = remove_margin([float(o) for o in pair], method=method)
    have = np.isfinite(market).all(axis=1)
    if have.any():
        model[have] = log_opinion_pool(market[have], model[have], market_weight)
    out["p_over"] = model[:, 0]
    out["market_p_over"] = market[:, 0]
    return out


@dataclass
class Slip:
    selections: pd.DataFrame
    all_candidates: pd.DataFrame
    bankroll: float
    generated_at: datetime = field(default_factory=datetime.utcnow)
    untuned_leagues: list[str] = field(default_factory=list)
    # Leagues whose market anchor tuned to 1.0. There, the anchored probability *is*
    # the market's own margin-free probability, so every edge equals minus the
    # bookmaker's margin and no selection can ever clear the threshold. That is a
    # structural fact about those leagues, not a quiet week, and saying "nothing
    # qualified" without saying so would imply next week might differ.
    no_edge_possible_leagues: list[str] = field(default_factory=list)
    # "value" = only selections clearing the edge threshold. "filled" = the best N by
    # edge regardless of threshold, so the slip always has something on it.
    mode: str = "value"
    # "value" ranks by disagreement with the price, "likely" by win probability.
    rank_by: str = "value"

    @property
    def total_stake(self) -> float:
        return float(self.selections["stake"].sum()) if not self.selections.empty else 0.0

    @property
    def expected_return(self) -> float:
        """Expected profit on the whole slip, by the model's own probabilities.

        Negative means the model itself expects to lose money on these bets. On a
        filled slip that is the normal case and the number worth looking at.
        """
        if self.selections.empty:
            return 0.0
        return float((self.selections["stake"] * self.selections["edge"]).sum())


def generate_slip(
    cfg: Config | None = None,
    bankroll: float | None = None,
    now: datetime | None = None,
    weights_path: Path | None = None,
    fill_to: int | None = None,
    rank_by: str = "value",
) -> Slip:
    """Build this week's slip.

    By default only selections clearing the edge threshold appear, so a week with
    no value produces an empty slip. Passing ``fill_to`` instead takes the best N
    selections by edge whatever their edge is, which guarantees a slip to place.

    Two markets are considered: the winner (home/draw/away) and total goals over or
    under 2.5. They come from the same fitted model, so they cannot contradict each
    other, and having both means a match whose winner is a coin toss can still say
    something useful about the goals.

    ``rank_by`` defaults to ``"value"``: selections are ordered by how far the model
    disagrees with the price in our favour. ``"likely"`` orders by win probability
    instead. That is not a better bet and the docstring should not pretend
    otherwise - at a bookmaker's margin, backing the most likely outcome every time
    is a reliable way to lose money slowly. It exists because "I want more of them
    to come in" is a real preference, and a slip full of 1.50 shots does win more
    often than one full of 3.50 shots while returning less. The expected return is
    reported either way so the trade is visible rather than implied.
    """
    if rank_by not in ("value", "likely"):
        raise ValueError(f"rank_by must be 'value' or 'likely', got {rank_by!r}")
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
    no_edge = []
    priced = []
    for code, group in upcoming.groupby("league_code"):
        if code not in tuned:
            untuned.append(code)
        elif tuned[code].get("market_weight", 0.0) >= 1.0:
            no_edge.append(code)
        frame = predict_fixtures(code, group.reset_index(drop=True), cfg, tuned.get(code))
        if not frame.empty:
            priced.append(frame)

    if not priced:
        return Slip(pd.DataFrame(), pd.DataFrame(), bankroll, untuned_leagues=untuned,
                    no_edge_possible_leagues=no_edge)

    priced_df = pd.concat(priced, ignore_index=True)

    method = cfg.odds.get("margin_method", "proportional")
    candidates = []
    for row in priced_df.itertuples(index=False):
        enough = (row.n_home_matches >= min_team_matches
                  and row.n_away_matches >= min_team_matches)
        base = {
            "match_id": row.match_id,
            "league_code": row.league_code,
            "kickoff_utc": row.kickoff_utc,
            "home": row.home,
            "away": row.away,
            "enough_history": enough,
        }

        def add(market, sel, model_p, price, fair_p):
            e = edge(model_p, price)
            candidates.append({
                **base,
                "market": market,
                "selection": sel,
                "model_prob": model_p,
                "market_prob_fair": fair_p,
                "odds": price,
                "edge": e,
                "in_odds_range": thresholds.min_odds <= price <= thresholds.max_odds,
                "clears_edge": e >= thresholds.min_edge,
            })

        trio = (row.pre_h, row.pre_d, row.pre_a)
        if all(o is not None and not pd.isna(o) and float(o) > 1.0 for o in trio):
            fair = remove_margin([float(o) for o in trio], method=method)
            for i, (sel, p_col, o_col) in enumerate(
                [("H", "p_home", "pre_h"), ("D", "p_draw", "pre_d"), ("A", "p_away", "pre_a")]
            ):
                add("1X2", sel, float(getattr(row, p_col)), float(getattr(row, o_col)),
                    float(fair[i]))

        # Over/under 2.5 goals, priced from the same fitted model. It is a genuinely
        # different question about the same match - how many goals, rather than who
        # wins - so on a week where the winner looks like a coin toss there can still
        # be something worth saying about the goals.
        pair = (getattr(row, "pre_o25", None), getattr(row, "pre_u25", None))
        p_over = getattr(row, "p_over", None)
        if (p_over is not None and not pd.isna(p_over)
                and all(o is not None and not pd.isna(o) and float(o) > 1.0 for o in pair)):
            fair_ou = remove_margin([float(pair[0]), float(pair[1])], method=method)
            add("OU25", "O", float(p_over), float(pair[0]), float(fair_ou[0]))
            add("OU25", "U", 1.0 - float(p_over), float(pair[1]), float(fair_ou[1]))

    all_candidates = pd.DataFrame(candidates)
    if all_candidates.empty:
        return Slip(pd.DataFrame(), all_candidates, bankroll, untuned_leagues=untuned,
                    no_edge_possible_leagues=no_edge)

    # The odds range and team-history guards always apply: they are about whether
    # the model has any business having an opinion, not about whether the price is
    # good. Only the edge threshold is relaxed when filling.
    eligible = all_candidates[
        all_candidates["in_odds_range"] & all_candidates["enough_history"]
    ].copy()

    # One bet per match, whichever market it comes from. The three 1X2 outcomes are
    # alternatives to each other, and a winner bet and a goals bet on the same match
    # are not independent either - the same red card moves both. Keeping it to one
    # means eight picks are eight matches, which is what the stake caps assume.
    rank_column = "model_prob" if rank_by == "likely" else "edge"
    eligible = (
        eligible.sort_values(rank_column, ascending=False)
        .drop_duplicates(subset="match_id", keep="first")
    )

    if fill_to:
        qualifying = eligible.head(fill_to).copy()
        mode = "filled"
    else:
        qualifying = eligible[eligible["clears_edge"]].copy()
        qualifying = qualifying.sort_values(rank_column, ascending=False).head(
            b.get("max_bets_per_week", 8)
        )
        mode = "value"

    if qualifying.empty:
        return Slip(pd.DataFrame(), all_candidates, bankroll, untuned_leagues=untuned,
                    no_edge_possible_leagues=no_edge, mode=mode)
    if fill_to:
        # Kelly correctly stakes nothing on a negative edge, so a filled slip has to
        # use a flat stake. It is capped at the same fraction of bankroll a Kelly bet
        # would be, so a filled slip can never risk more than a value slip would.
        flat = round_stake(bankroll * rules.max_stake_pct, rules.rounding)
        qualifying["stake"] = max(flat, rules.min_stake)
    else:
        qualifying["stake"] = [
            stake_for(r.model_prob, r.odds, bankroll, rules)
            for r in qualifying.itertuples(index=False)
        ]
    qualifying = qualifying[qualifying["stake"] > 0]
    qualifying = qualifying.sort_values("kickoff_utc").reset_index(drop=True)

    return Slip(qualifying, all_candidates, bankroll, untuned_leagues=untuned,
                no_edge_possible_leagues=no_edge, mode=mode, rank_by=rank_by)


SELECTION_WORDS = {"H": "Home", "D": "Draw", "A": "Away", "O": "Over 2.5 goals",
                   "U": "Under 2.5 goals"}


def describe(market: str, selection: str) -> str:
    """How a pick reads on a betting slip."""
    return SELECTION_WORDS.get(selection, selection)


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
        if slip.no_edge_possible_leagues:
            leagues = ", ".join(slip.no_edge_possible_leagues)
            lines.append(f"NOTE - {leagues}: no selection can qualify in these leagues.")
            lines.append("Validation gave the market a weight of 1.0 there, meaning the")
            lines.append("model added nothing to the price. Every edge therefore equals")
            lines.append("minus the bookmaker's margin. This is structural, not a quiet")
            lines.append("week: these leagues will not produce a bet until the model")
            lines.append("earns some weight back against the market.")
            lines.append("")
        lines.append("A week with nothing worth backing is the thresholds working.")
        return "\n".join(lines)

    lines.append(f"{len(slip.selections)} singles, total stake {slip.total_stake:,.2f}")
    lines.append("SINGLES ONLY - do not combine these into an accumulator.")
    lines.append("")
    if slip.mode == "filled":
        ev = slip.expected_return
        lines.append(f"FILLED SLIP - ranked by edge, edge threshold ignored.")
        lines.append(f"Expected return by the model's own numbers: {ev:+,.2f}")
        if ev < 0:
            pct = ev / slip.total_stake if slip.total_stake else 0.0
            lines.append(f"That is {pct:+.1%} of stake. The model does not think these")
            lines.append("are good bets - it thinks they are the least bad ones available.")
            lines.append("Paper mode is the right place for this until CLV says otherwise.")
        lines.append("")
    for i, r in enumerate(slip.selections.itertuples(index=False), start=1):
        lines.append(f"{i}. {r.kickoff_utc:%a %d %b %H:%M}  [{r.league_code}]")
        lines.append(f"   {r.home} v {r.away}")
        market = getattr(r, "market", "1X2")
        lines.append(f"   {describe(market, r.selection)} @ {r.odds:.2f}"
                     f"   stake {r.stake:,.2f}")
        lines.append(f"   our chance {r.model_prob:.0%} vs their price {r.market_prob_fair:.0%}"
                     f"   value {r.edge:+.1%}")
        lines.append("")
    lines.append("Prices move. If the price has shortened past the listed odds,")
    lines.append("the edge may be gone - skip it rather than taking a worse number.")
    return "\n".join(lines)


def slip_to_csv(slip: Slip) -> str:
    if slip.selections.empty:
        return "kickoff_utc,league,home,away,market,selection,bet,odds,stake,model_prob,edge\n"
    out = slip.selections[[
        "kickoff_utc", "league_code", "home", "away", "market", "selection",
        "odds", "stake", "model_prob", "market_prob_fair", "edge",
    ]].copy()
    out["bet"] = [describe(m, s) for m, s in zip(out["market"], out["selection"], strict=True)]
    return out.to_csv(index=False)
