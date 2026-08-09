"""Upcoming fixtures and their opening bet365 prices.

football-data.co.uk publishes a single ``fixtures.csv`` covering the next week or so
across every division it tracks, including bet365 1X2 prices. That is enough for the
weekly slip, so Phase 3 needs no paid odds feed; The Odds API arrives in Phase 4 for
live re-pricing and proper CLV capture.

Fixtures are stored as ordinary ``matches`` rows with ``status='scheduled'`` and no
score. When results load the following week the same natural key upserts them to
``status='played'``, so a fixture and its result are the same row throughout and
settlement has something stable to join on.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import requests
from sqlalchemy import select

from fv.config import Config, load_config
from fv.data.football_data import USER_AGENT, looks_like_html
from fv.data.teams import TeamResolver
from fv.db.models import IngestLog, Match, Odds
from fv.db.session import session_scope

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def current_season(today: datetime | None = None) -> str:
    """Season label for a date. A new season is taken to start in July."""
    today = today or datetime.utcnow()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


@dataclass
class FixtureStats:
    fetched: int = 0
    in_scope: int = 0
    inserted: int = 0
    updated: int = 0
    odds_rows: int = 0
    unresolved: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.fetched} fixtures published, {self.in_scope} in enabled leagues, "
            f"{self.inserted} new, {self.updated} updated, {self.odds_rows} prices"
        )


def parse_fixtures(payload: bytes, league_codes: set[str]) -> pd.DataFrame:
    """Parse fixtures.csv, keeping only the leagues we care about."""
    df = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", on_bad_lines="skip")
    df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()]
    df = df[df["Div"].isin(league_codes)]
    if df.empty:
        return pd.DataFrame()

    dates = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    times = df["Time"].astype(str).str.strip() if "Time" in df.columns else None
    if times is not None:
        kickoff = pd.to_datetime(
            dates.dt.strftime("%Y-%m-%d") + " " + times,
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        ).fillna(dates + pd.Timedelta(hours=12))
    else:
        kickoff = dates + pd.Timedelta(hours=12)

    out = pd.DataFrame(
        {
            "league_code": df["Div"].astype(str),
            "match_date": dates,
            "kickoff_utc": kickoff,
            "home": df["HomeTeam"].astype(str).str.strip(),
            "away": df["AwayTeam"].astype(str).str.strip(),
            "b365_h": pd.to_numeric(df.get("B365H"), errors="coerce"),
            "b365_d": pd.to_numeric(df.get("B365D"), errors="coerce"),
            "b365_a": pd.to_numeric(df.get("B365A"), errors="coerce"),
            # fixtures.csv carries the over/under 2.5 goals prices in the same row as
            # the 1X2 ones. Reading only the 1X2 columns is why the slip could offer
            # nothing but home/draw/away.
            "b365_o25": pd.to_numeric(df.get("B365>2.5"), errors="coerce"),
            "b365_u25": pd.to_numeric(df.get("B365<2.5"), errors="coerce"),
        }
    )
    return out[out["match_date"].notna()].reset_index(drop=True)


def download_fixtures(cfg: Config | None = None) -> FixtureStats:
    """Fetch and store upcoming fixtures with their bet365 prices."""
    cfg = cfg or load_config()
    stats = FixtureStats()
    enabled = {lg.code: lg for lg in cfg.enabled_leagues}

    resp = requests.get(FIXTURES_URL, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    if looks_like_html(resp.content):
        return stats

    frame = parse_fixtures(resp.content, set(enabled))
    total = len(pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig", on_bad_lines="skip"))
    stats.fetched = total
    stats.in_scope = len(frame)
    if frame.empty:
        return stats

    season = current_season()

    with session_scope(cfg) as s:
        resolver = TeamResolver(s, source="football_data")
        for row in frame.itertuples(index=False):
            league = enabled[row.league_code]
            home_id = resolver.get_or_create(row.home, league.country, season)
            away_id = resolver.get_or_create(row.away, league.country, season)
            if home_id == away_id:
                continue

            existing = s.scalar(
                select(Match).where(
                    Match.league_code == row.league_code,
                    Match.season == season,
                    Match.home_team_id == home_id,
                    Match.away_team_id == away_id,
                )
            )
            if existing is None:
                match = Match(
                    league_code=row.league_code,
                    season=season,
                    match_date=row.match_date.date(),
                    kickoff_utc=row.kickoff_utc.to_pydatetime(),
                    home_team_id=home_id,
                    away_team_id=away_id,
                    status="scheduled",
                    source="football_data_fixtures",
                )
                s.add(match)
                s.flush()
                stats.inserted += 1
            else:
                # Kickoffs move. Never overwrite a played result with a fixture row.
                if existing.status == "scheduled":
                    existing.match_date = row.match_date.date()
                    existing.kickoff_utc = row.kickoff_utc.to_pydatetime()
                    stats.updated += 1
                match = existing

            prices = (
                ("1X2", "H", row.b365_h),
                ("1X2", "D", row.b365_d),
                ("1X2", "A", row.b365_a),
                ("OU25", "O", row.b365_o25),
                ("OU25", "U", row.b365_u25),
            )
            for market, sel, value in prices:
                if value is None or pd.isna(value) or float(value) <= 1.0:
                    continue
                found = s.scalar(
                    select(Odds).where(
                        Odds.match_id == match.id,
                        Odds.bookmaker == "B365",
                        Odds.market == market,
                        Odds.selection == sel,
                        Odds.odds_type == "pre",
                    )
                )
                if found is None:
                    s.add(
                        Odds(
                            match_id=match.id,
                            bookmaker="B365",
                            market=market,
                            selection=sel,
                            decimal_odds=float(value),
                            odds_type="pre",
                            captured_at=datetime.utcnow(),
                            source="football_data_fixtures",
                        )
                    )
                    stats.odds_rows += 1
                else:
                    found.decimal_odds = float(value)

        s.add(
            IngestLog(
                source="football_data_fixtures",
                url=FIXTURES_URL,
                rows_parsed=len(frame),
                rows_upserted=stats.inserted + stats.updated,
                status="ok",
            )
        )

    stats.unresolved = [u.raw for u in resolver.unresolved]
    return stats
