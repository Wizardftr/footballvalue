"""Feature engineering for the LightGBM stage.

Every feature for a match is computed from matches that kicked off strictly before
it. That is enforced structurally: the builder walks the league in chronological
order and writes each match's features from the state accumulated so far, *then*
folds that match's result into the state. There is no windowing function that could
accidentally see forward, and no way to compute a feature for match ``k`` from match
``k+1`` without rewriting the loop.

This matters more here than in the Dixon-Coles stage. Rolling-form features are the
classic place lookahead creeps into football models — a `groupby().rolling()` that
includes the current row leaks the result you are trying to predict, and the model
looks brilliant right up until it meets real money.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Elo settings. K controls how fast ratings move; 20 is the usual football choice.
# HOME_ELO_ADVANTAGE is in rating points and roughly matches the observed home edge.
ELO_K = 20.0
ELO_START = 1500.0
ELO_HOME_ADVANTAGE = 65.0
ELO_SCALE = 400.0

FORM_WINDOWS = (5, 10)


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for A against B, on the standard logistic Elo curve."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def elo_update(home: float, away: float, result: str, k: float = ELO_K) -> tuple[float, float]:
    """Return updated (home, away) ratings after a result.

    A draw scores 0.5 for both sides, which is what keeps Elo sensible in a sport
    where roughly a quarter of matches are drawn.
    """
    expected_home = elo_expected(home + ELO_HOME_ADVANTAGE, away)
    actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
    delta = k * (actual_home - expected_home)
    return home + delta, away - delta


@dataclass
class _TeamState:
    """Rolling history for one team. Deques cap memory and keep the windows honest."""

    elo: float = ELO_START
    last_kickoff: pd.Timestamp | None = None
    goals_for: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    xg_for: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    xg_against: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    points: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    matches_played: int = 0
    seasons_seen: set = field(default_factory=set)


def _window_mean(values: deque, window: int) -> float:
    """Mean of the last ``window`` entries. NaN when there is no history yet.

    NaN rather than 0.0 deliberately: LightGBM handles missing values natively, and
    a promoted side with no history is genuinely unknown, not genuinely zero.
    """
    if not values:
        return np.nan
    recent = list(values)[-window:]
    if not recent:
        return np.nan
    return float(np.mean(recent))


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build the feature frame for one league, in chronological order.

    ``matches`` needs: match_id, season, kickoff_utc, home, away, fthg, ftag, ftr,
    and optionally home_xg / away_xg.
    """
    df = matches.dropna(subset=["fthg", "ftag"]).sort_values("kickoff_utc").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    state: dict[str, _TeamState] = defaultdict(_TeamState)
    league_goals: deque = deque(maxlen=200)
    # Which teams appeared in each season, so promotion can be detected.
    season_teams: dict[str, set] = defaultdict(set)

    rows: list[dict] = []

    for m in df.itertuples(index=False):
        home, away = state[m.home], state[m.away]
        kickoff = m.kickoff_utc

        # --- features, from history only ---------------------------------
        feat = {
            "match_id": m.match_id,
            "season": m.season,
            "kickoff_utc": kickoff,
            "home": m.home,
            "away": m.away,
            "ftr": m.ftr,
            "elo_home": home.elo,
            "elo_away": away.elo,
            "elo_diff": home.elo - away.elo,
            "elo_expected_home": elo_expected(home.elo + ELO_HOME_ADVANTAGE, away.elo),
            "home_matches_played": home.matches_played,
            "away_matches_played": away.matches_played,
            "league_goal_env": float(np.mean(league_goals)) if league_goals else np.nan,
        }

        for w in FORM_WINDOWS:
            feat[f"home_gf_{w}"] = _window_mean(home.goals_for, w)
            feat[f"home_ga_{w}"] = _window_mean(home.goals_against, w)
            feat[f"away_gf_{w}"] = _window_mean(away.goals_for, w)
            feat[f"away_ga_{w}"] = _window_mean(away.goals_against, w)
            feat[f"home_xgf_{w}"] = _window_mean(home.xg_for, w)
            feat[f"home_xga_{w}"] = _window_mean(home.xg_against, w)
            feat[f"away_xgf_{w}"] = _window_mean(away.xg_for, w)
            feat[f"away_xga_{w}"] = _window_mean(away.xg_against, w)
            feat[f"home_pts_{w}"] = _window_mean(home.points, w)
            feat[f"away_pts_{w}"] = _window_mean(away.points, w)
            # Attack-minus-defence differentials carry most of the signal and save
            # the trees from having to discover the subtraction themselves.
            feat[f"gf_diff_{w}"] = feat[f"home_gf_{w}"] - feat[f"away_gf_{w}"]
            feat[f"xg_diff_{w}"] = feat[f"home_xgf_{w}"] - feat[f"away_xgf_{w}"]

        feat["home_rest_days"] = (
            (kickoff - home.last_kickoff).total_seconds() / 86400.0
            if home.last_kickoff is not None
            else np.nan
        )
        feat["away_rest_days"] = (
            (kickoff - away.last_kickoff).total_seconds() / 86400.0
            if away.last_kickoff is not None
            else np.nan
        )
        feat["rest_diff"] = feat["home_rest_days"] - feat["away_rest_days"]

        # Newly promoted: in this league this season, but not the previous one.
        feat["home_promoted"] = int(_is_new_this_season(m.home, m.season, season_teams))
        feat["away_promoted"] = int(_is_new_this_season(m.away, m.season, season_teams))

        rows.append(feat)

        # --- fold this match into the state, after the features are written ---
        hg, ag = float(m.fthg), float(m.ftag)
        hxg = getattr(m, "home_xg", np.nan)
        axg = getattr(m, "away_xg", np.nan)

        home.goals_for.append(hg)
        home.goals_against.append(ag)
        away.goals_for.append(ag)
        away.goals_against.append(hg)
        if pd.notna(hxg) and pd.notna(axg):
            home.xg_for.append(float(hxg))
            home.xg_against.append(float(axg))
            away.xg_for.append(float(axg))
            away.xg_against.append(float(hxg))

        home.points.append({"H": 3.0, "D": 1.0, "A": 0.0}[m.ftr])
        away.points.append({"A": 3.0, "D": 1.0, "H": 0.0}[m.ftr])

        home.elo, away.elo = elo_update(home.elo, away.elo, m.ftr)
        home.last_kickoff = away.last_kickoff = kickoff
        home.matches_played += 1
        away.matches_played += 1
        home.seasons_seen.add(m.season)
        away.seasons_seen.add(m.season)
        season_teams[m.season].add(m.home)
        season_teams[m.season].add(m.away)
        league_goals.append(hg + ag)

    return pd.DataFrame(rows)


def _is_new_this_season(team: str, season: str, season_teams: dict[str, set]) -> bool:
    """True when the team did not appear in the previous season of this league.

    Uses only seasons already seen, so it never consults the future.
    """
    prior = sorted(s for s in season_teams if s < season)
    if not prior:
        return False
    return team not in season_teams[prior[-1]]


FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff", "elo_expected_home",
    "home_matches_played", "away_matches_played", "league_goal_env",
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_promoted", "away_promoted",
] + [
    f"{prefix}_{w}"
    for w in FORM_WINDOWS
    for prefix in (
        "home_gf", "home_ga", "away_gf", "away_ga",
        "home_xgf", "home_xga", "away_xgf", "away_xga",
        "home_pts", "away_pts", "gf_diff", "xg_diff",
    )
]
