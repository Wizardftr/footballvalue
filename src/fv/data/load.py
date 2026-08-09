"""Download + load orchestration. Safe to re-run weekly."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sqlalchemy import select

from fv.config import Config, load_config
from fv.data.football_data import (
    ODDS_FIELDS,
    WrongLeagueError,
    FetchResult,
    fetch_season,
    parse_csv,
    season_range,
)
from fv.data.teams import TeamResolver
from fv.db.models import IngestLog, Match, Odds
from fv.db.session import session_scope

console = Console()


@dataclass
class LoadStats:
    files_ok: int = 0
    files_missing: int = 0
    files_unchanged: int = 0
    files_error: int = 0
    matches_inserted: int = 0
    matches_updated: int = 0
    odds_inserted: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"files: {self.files_ok} loaded, {self.files_unchanged} unchanged, "
            f"{self.files_missing} missing, {self.files_error} errored | "
            f"matches: +{self.matches_inserted} new, {self.matches_updated} updated | "
            f"odds: +{self.odds_inserted}"
        )


def _upsert_match(session, resolver: TeamResolver, row, country: str) -> tuple[int | None, str]:
    """Insert or update one match. Returns (match_id, 'inserted'|'updated'|'skipped')."""
    home_id = resolver.get_or_create(row.home, country, row.season)
    away_id = resolver.get_or_create(row.away, country, row.season)
    if home_id == away_id:
        return None, "skipped"  # data error: a team cannot play itself

    existing = session.scalar(
        select(Match).where(
            Match.league_code == row.league_code,
            Match.season == row.season,
            Match.home_team_id == home_id,
            Match.away_team_id == away_id,
        )
    )

    fthg = None if pd.isna(row.fthg) else int(row.fthg)
    ftag = None if pd.isna(row.ftag) else int(row.ftag)
    hthg = None if pd.isna(row.hthg) else int(row.hthg)
    htag = None if pd.isna(row.htag) else int(row.htag)
    ftr = None if pd.isna(row.ftr) else str(row.ftr)

    if existing is None:
        match = Match(
            league_code=row.league_code,
            season=row.season,
            match_date=row.match_date.date(),
            kickoff_utc=row.kickoff_utc.to_pydatetime(),
            home_team_id=home_id,
            away_team_id=away_id,
            fthg=fthg,
            ftag=ftag,
            ftr=ftr,
            hthg=hthg,
            htag=htag,
            status=row.status,
        )
        session.add(match)
        session.flush()
        return match.id, "inserted"

    # Results can arrive after the fixture (weekly refresh), so keep them current.
    changed = (
        existing.fthg != fthg
        or existing.ftag != ftag
        or existing.status != row.status
        or existing.match_date != row.match_date.date()
    )
    existing.match_date = row.match_date.date()
    existing.kickoff_utc = row.kickoff_utc.to_pydatetime()
    existing.fthg, existing.ftag, existing.ftr = fthg, ftag, ftr
    existing.hthg, existing.htag = hthg, htag
    existing.status = row.status
    return existing.id, "updated" if changed else "skipped"


def _upsert_odds(session, match_id: int, row) -> int:
    """Insert bet365 prices for a match, across every market present.

    Returns the number of rows inserted.
    """
    inserted = 0
    for key, (market, odds_type, sel) in ODDS_FIELDS.items():
        value = getattr(row, key, None)
        if value is None or pd.isna(value) or float(value) <= 1.0:
            continue
        exists = session.scalar(
            select(Odds).where(
                Odds.match_id == match_id,
                Odds.bookmaker == "B365",
                Odds.market == market,
                Odds.selection == sel,
                Odds.odds_type == odds_type,
            )
        )
        if exists is not None:
            exists.decimal_odds = float(value)
            continue
        session.add(
            Odds(
                match_id=match_id,
                bookmaker="B365",
                market=market,
                selection=sel,
                decimal_odds=float(value),
                odds_type=odds_type,
                source="football_data",
            )
        )
        inserted += 1
    return inserted


def download_and_load(
    cfg: Config | None = None,
    leagues: list[str] | None = None,
    first_season: str | None = None,
    force: bool = False,
) -> LoadStats:
    """Fetch every enabled league-season and load it into SQLite.

    Idempotent: unchanged files are skipped by content hash, and matches are
    upserted on their natural key, so a weekly re-run only picks up new results.
    """
    cfg = cfg or load_config()
    stats = LoadStats()

    selected = [lg for lg in cfg.enabled_leagues if leagues is None or lg.code in leagues]
    seasons = season_range(first_season or cfg.data.get("first_season", "2000-01"))
    base_url = cfg.data.get("base_url", "https://www.football-data.co.uk/mmz4281")
    cache = cfg.raw_cache

    jobs = [(lg, season) for lg in selected for season in seasons]
    http = requests.Session()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("downloading", total=len(jobs))

        for lg, season in jobs:
            progress.update(task, description=f"{lg.code} {season}")
            result = fetch_season(lg.code, season, cache, base_url, force=force, session=http)
            _record_and_load(cfg, result, lg, season, stats, force=force)
            progress.advance(task)

    return stats


def _record_and_load(cfg, result: FetchResult, lg, season: str, stats: LoadStats, force: bool):
    if result.status == "missing":
        stats.files_missing += 1
        return
    if result.status == "error":
        stats.files_error += 1
        stats.errors.append(f"{lg.code} {season}: {result.error}")
        return
    if result.status == "unchanged" and not force:
        stats.files_unchanged += 1
        return

    try:
        frame = parse_csv(result.payload, lg.code, season)
    except WrongLeagueError as exc:
        # football-data has served another division's file at a league's URL. Skip it
        # loudly rather than loading one country's clubs as another's.
        stats.files_error += 1
        stats.errors.append(str(exc))
        with session_scope(cfg) as session:
            session.add(IngestLog(source="football_data", url=result.url,
                                  league_code=lg.code, season=season,
                                  sha256=result.sha256, status="wrong_league"))
        return
    except Exception as exc:
        stats.files_error += 1
        stats.errors.append(f"{lg.code} {season}: parse failed: {exc}")
        return

    with session_scope(cfg) as session:
        resolver = TeamResolver(session, source="football_data")
        inserted = updated = odds_added = 0
        for row in frame.itertuples(index=False):
            match_id, action = _upsert_match(session, resolver, row, lg.country)
            if match_id is None:
                continue
            if action == "inserted":
                inserted += 1
            elif action == "updated":
                updated += 1
            odds_added += _upsert_odds(session, match_id, row)

        session.add(
            IngestLog(
                source="football_data",
                url=result.url,
                league_code=lg.code,
                season=season,
                rows_parsed=len(frame),
                rows_upserted=inserted + updated,
                sha256=result.sha256,
                status="ok",
            )
        )

    stats.files_ok += 1
    stats.matches_inserted += inserted
    stats.matches_updated += updated
    stats.odds_inserted += odds_added
