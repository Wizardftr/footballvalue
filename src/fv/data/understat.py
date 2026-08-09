"""Understat xG ingestion.

Covers the big-five top divisions, which is exactly the set flagged ``has_xg`` in
config.yaml. The other six leagues have no reliable free xG source and run on goals
only, as the spec allows.

Two notes on how this is done:

**Not via soccerdata.** Its Understat reader depends on ``tls_requests``, which
downloads a native TLS library from GitHub releases at import time; that download is
blocked here. Understat's own JSON endpoint is reachable, so this talks to it
directly, which also gives us the local caching and rate limiting the spec asks for.

**Team names are resolved by fixture alignment, not by string similarity.** Understat
says "Manchester United" where football-data says "Man United"; fuzzy matching that
pair is a coin toss, and a wrong match silently attaches one club's xG to another.
Instead, matches are aligned on date and scoreline — nearly unique within a
league-season — and the name mapping is *derived* from the alignment, then required
to be consistent across the season before it is accepted. A mapping that can't be
established that way is reported, not guessed.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import select

from fv.config import Config, load_config
from fv.db.models import Match, MatchXG, Team, TeamAlias
from fv.db.session import session_scope

BASE_URL = "https://understat.com/getLeagueData"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}
# Understat's coverage starts with 2014-15.
FIRST_SEASON = "2014-15"
REQUEST_DELAY_SECONDS = 3.0

# fv league code -> Understat league slug
LEAGUE_SLUGS = {
    "E0": "EPL",
    "SP1": "La_liga",
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}


def season_to_understat(season: str) -> str:
    """"2024-25" -> "2024" (Understat keys a season by its starting year)."""
    return season.split("-")[0]


def fetch_league_season(
    league_code: str,
    season: str,
    cache_dir: Path,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict | None:
    """Fetch one league-season from Understat, caching the raw JSON on disk.

    Returns None when the league has no Understat coverage or the season is absent.
    """
    slug = LEAGUE_SLUGS.get(league_code)
    if slug is None:
        return None

    year = season_to_understat(season)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"understat_{slug}_{year}.json"

    if path.exists() and not force:
        return json.loads(path.read_text())

    url = f"{BASE_URL}/{slug}/{year}"
    http = session or requests
    try:
        resp = http.get(
            url,
            headers={**HEADERS, "Referer": f"https://understat.com/league/{slug}/{year}"},
            timeout=60,
        )
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or "dates" not in data:
        return None

    path.write_text(json.dumps(data))
    time.sleep(REQUEST_DELAY_SECONDS)  # be a considerate client
    return data


def parse_matches(data: dict) -> pd.DataFrame:
    """Understat league payload -> one row per played match."""
    rows = []
    for m in data.get("dates", []):
        if not m.get("isResult"):
            continue
        try:
            rows.append(
                {
                    "understat_id": m["id"],
                    "kickoff": pd.Timestamp(m["datetime"]),
                    "home_name": m["h"]["title"],
                    "away_name": m["a"]["title"],
                    "home_goals": int(m["goals"]["h"]),
                    "away_goals": int(m["goals"]["a"]),
                    "home_xg": float(m["xG"]["h"]),
                    "away_xg": float(m["xG"]["a"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


@dataclass
class AlignmentResult:
    """Outcome of aligning Understat matches to ours for one league-season."""

    name_map: dict[str, int] = field(default_factory=dict)  # understat name -> team_id
    matched: list[tuple[int, float, float]] = field(default_factory=list)  # (match_id, hxg, axg)
    unmatched_names: set[str] = field(default_factory=set)
    n_aligned: int = 0
    n_total: int = 0


def align_by_fixture(
    understat: pd.DataFrame,
    ours: pd.DataFrame,
    date_tolerance_days: int = 1,
    min_votes: int = 3,
    min_agreement: float = 0.9,
) -> AlignmentResult:
    """Derive the Understat -> canonical team mapping from fixture agreement.

    A match is a candidate pairing when the date agrees within a day (Understat
    stores local kickoff time, football-data stores the match date) and the
    scoreline is identical. Where exactly one of our matches fits, both teams cast
    a vote for their name mapping.

    A name is only accepted when it has at least ``min_votes`` and at least
    ``min_agreement`` of those votes point at the same team. Requiring agreement
    across the season is what makes this safe: a coincidental date-and-score
    collision produces one stray vote, not a consistent majority.
    """
    result = AlignmentResult(n_total=len(understat))
    if understat.empty or ours.empty:
        result.unmatched_names = set(understat.get("home_name", []))
        return result

    votes: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ours = ours.copy()
    ours["date"] = pd.to_datetime(ours["match_date"])

    for row in understat.itertuples(index=False):
        lo = row.kickoff.normalize() - timedelta(days=date_tolerance_days)
        hi = row.kickoff.normalize() + timedelta(days=date_tolerance_days)
        cand = ours[
            (ours["date"] >= lo)
            & (ours["date"] <= hi)
            & (ours["fthg"] == row.home_goals)
            & (ours["ftag"] == row.away_goals)
        ]
        if len(cand) != 1:
            continue
        c = cand.iloc[0]
        votes[row.home_name][int(c["home_team_id"])] += 1
        votes[row.away_name][int(c["away_team_id"])] += 1

    for name, counts in votes.items():
        total = sum(counts.values())
        best_id, best_n = max(counts.items(), key=lambda kv: kv[1])
        if total >= min_votes and best_n / total >= min_agreement:
            result.name_map[name] = best_id
        else:
            result.unmatched_names.add(name)

    all_names = set(understat["home_name"]) | set(understat["away_name"])
    result.unmatched_names |= all_names - set(result.name_map)

    # With the mapping settled, attach xG to every match it can identify.
    key = {
        (int(r.home_team_id), int(r.away_team_id)): int(r.match_id)
        for r in ours.itertuples(index=False)
    }
    for row in understat.itertuples(index=False):
        h = result.name_map.get(row.home_name)
        a = result.name_map.get(row.away_name)
        if h is None or a is None:
            continue
        match_id = key.get((h, a))
        if match_id is None:
            continue
        result.matched.append((match_id, row.home_xg, row.away_xg))
    result.n_aligned = len(result.matched)
    return result


@dataclass
class XGLoadStats:
    leagues: int = 0
    seasons_fetched: int = 0
    seasons_missing: int = 0
    matches_loaded: int = 0
    unmatched_names: dict[str, set] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"xG: {self.matches_loaded:,} matches loaded across {self.leagues} leagues, "
            f"{self.seasons_fetched} seasons fetched, {self.seasons_missing} unavailable"
        )


def download_and_load_xg(
    cfg: Config | None = None,
    leagues: list[str] | None = None,
    first_season: str = FIRST_SEASON,
    force: bool = False,
    progress=None,
) -> XGLoadStats:
    """Fetch Understat xG for the covered leagues and load it into ``match_xg``."""
    from fv.data.football_data import season_range

    cfg = cfg or load_config()
    stats = XGLoadStats()
    cache = cfg.raw_cache
    http = requests.Session()

    codes = [
        lg.code
        for lg in cfg.enabled_leagues
        if lg.code in LEAGUE_SLUGS and (leagues is None or lg.code in leagues)
    ]
    seasons = season_range(first_season)

    for code in codes:
        stats.leagues += 1
        unresolved: set[str] = set()

        for season in seasons:
            data = fetch_league_season(code, season, cache, force=force, session=http)
            if progress:
                progress(code, season)
            if data is None:
                stats.seasons_missing += 1
                continue
            stats.seasons_fetched += 1

            us = parse_matches(data)
            if us.empty:
                continue

            with session_scope(cfg) as s:
                ours = pd.read_sql(
                    select(
                        Match.id.label("match_id"),
                        Match.match_date,
                        Match.home_team_id,
                        Match.away_team_id,
                        Match.fthg,
                        Match.ftag,
                    ).where(
                        Match.league_code == code,
                        Match.season == season,
                        Match.status == "played",
                    ),
                    s.connection(),
                )
                if ours.empty:
                    continue

                alignment = align_by_fixture(us, ours)
                unresolved |= alignment.unmatched_names

                for match_id, hxg, axg in alignment.matched:
                    existing = s.get(MatchXG, match_id)
                    if existing is None:
                        s.add(
                            MatchXG(
                                match_id=match_id,
                                home_xg=hxg,
                                away_xg=axg,
                                source="understat",
                            )
                        )
                        stats.matches_loaded += 1
                    else:
                        existing.home_xg, existing.away_xg = hxg, axg

                # Persist the derived mapping so it is inspectable and reusable.
                for name, team_id in alignment.name_map.items():
                    exists = s.scalar(
                        select(TeamAlias).where(
                            TeamAlias.alias == name, TeamAlias.source == "understat"
                        )
                    )
                    if exists is None:
                        s.add(
                            TeamAlias(
                                alias=name,
                                source="understat",
                                team_id=team_id,
                                confidence=1.0,
                                verified=True,  # derived from fixture agreement, not a guess
                            )
                        )

        if unresolved:
            stats.unmatched_names[code] = unresolved

    return stats
