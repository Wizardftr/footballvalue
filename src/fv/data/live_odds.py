"""Store Odds API snapshots and derive closing prices from them.

A "snapshot" is a price observed at a moment. The last snapshot taken before kickoff
is the closing price, which is what closing-line value is measured against.

Deriving the close from snapshots rather than trusting any single fetch matters: a
weekly refresh might capture a price four days out, and calling that "the close"
would flatter CLV enormously. :func:`promote_closing_odds` only promotes a snapshot
that was taken within a window of kickoff, and records how far out it actually was,
so a CLV figure can never quietly rest on a stale price.

Team names come from a third naming scheme (after football-data and Understat), so
they go through the alias table. Names that cannot be resolved are reported rather
than guessed — attaching a price to the wrong fixture is worse than missing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select

from fv.config import Config, load_config
from fv.data.odds_api import OddsApiClient, Quote, best_quotes, parse_events
from fv.data.teams import TeamResolver
from fv.db.models import Match, Odds, Team
from fv.db.session import session_scope

# A snapshot older than this before kickoff is not a closing price.
CLOSING_WINDOW_HOURS = 12


@dataclass
class SnapshotStats:
    leagues: int = 0
    events: int = 0
    quotes: int = 0
    stored: int = 0
    unmatched_events: list[str] = field(default_factory=list)
    unresolved_names: set[str] = field(default_factory=set)
    quota: str = "unknown"

    def summary(self) -> str:
        return (
            f"{self.events} events across {self.leagues} leagues, {self.quotes} quotes, "
            f"{self.stored} stored | {self.quota}"
        )


def _match_for_event(session, resolver: TeamResolver, quote: Quote, country: str,
                     league_code: str) -> int | None:
    """Find our match id for an Odds API event, by resolved teams and kickoff date."""
    home_id = resolver.resolve(quote.home_name, country)
    away_id = resolver.resolve(quote.away_name, country)
    if home_id is None or away_id is None:
        return None

    # Kickoff can drift by a few hours between sources; the date plus both teams is
    # unique within a league season.
    lo = quote.commence_time - timedelta(days=1)
    hi = quote.commence_time + timedelta(days=1)
    match = session.scalar(
        select(Match).where(
            and_(
                Match.league_code == league_code,
                Match.home_team_id == home_id,
                Match.away_team_id == away_id,
                Match.kickoff_utc >= lo.replace(tzinfo=None),
                Match.kickoff_utc <= hi.replace(tzinfo=None),
            )
        )
    )
    return match.id if match else None


def store_snapshots(
    cfg: Config | None = None,
    leagues: list[str] | None = None,
    client: OddsApiClient | None = None,
    force: bool = False,
) -> SnapshotStats:
    """Fetch current prices for every enabled league and store them as snapshots."""
    from fv.data.odds_api import SPORT_KEYS

    cfg = cfg or load_config()
    stats = SnapshotStats()
    client = client or OddsApiClient(cfg.odds_api_key, cache_dir=cfg.raw_cache)

    codes = [
        lg for lg in cfg.enabled_leagues
        if lg.code in SPORT_KEYS and (leagues is None or lg.code in leagues)
    ]

    for league in codes:
        payload = client.fetch_odds(league.code, force=force)
        if not payload:
            continue
        stats.leagues += 1
        stats.events += len(payload)

        quotes = best_quotes(parse_events(payload))
        stats.quotes += len(quotes)
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        with session_scope(cfg) as s:
            resolver = TeamResolver(s, source="odds_api")
            seen_events: dict[str, int | None] = {}

            for q in quotes:
                if q.event_id not in seen_events:
                    seen_events[q.event_id] = _match_for_event(
                        s, resolver, q, league.country, league.code
                    )
                match_id = seen_events[q.event_id]
                if match_id is None:
                    continue

                s.add(
                    Odds(
                        match_id=match_id,
                        bookmaker=q.bookmaker,
                        market=q.market,
                        selection=q.selection,
                        decimal_odds=q.decimal_odds,
                        odds_type="snapshot",
                        captured_at=now,
                        source="odds_api",
                    )
                )
                stats.stored += 1

            stats.unresolved_names |= {u.raw for u in resolver.unresolved}
            stats.unmatched_events += [
                f"{q.home_name} v {q.away_name}"
                for q in quotes
                if seen_events.get(q.event_id) is None
            ][:10]

    stats.quota = client.quota.describe()
    return stats


@dataclass
class ClosingStats:
    promoted: int = 0
    too_early: int = 0
    already_had_closing: int = 0

    def summary(self) -> str:
        return (
            f"promoted {self.promoted} closing prices, skipped {self.too_early} snapshots "
            f"taken more than {CLOSING_WINDOW_HOURS}h before kickoff, "
            f"{self.already_had_closing} already had one"
        )


def promote_closing_odds(
    cfg: Config | None = None,
    window_hours: int = CLOSING_WINDOW_HOURS,
    now: datetime | None = None,
) -> ClosingStats:
    """Promote the last pre-kickoff snapshot of each selection to a closing price.

    Only snapshots inside ``window_hours`` of kickoff qualify. A price captured four
    days out is not a close, and treating it as one would inflate CLV — the metric
    the whole project leans on.
    """
    cfg = cfg or load_config()
    stats = ClosingStats()
    now = now or datetime.utcnow()

    with session_scope(cfg) as s:
        # Matches that have kicked off and carry at least one snapshot.
        rows = s.execute(
            select(Odds.match_id, Odds.bookmaker, Odds.market, Odds.selection)
            .join(Match, Match.id == Odds.match_id)
            .where(Odds.odds_type == "snapshot", Match.kickoff_utc <= now)
            .distinct()
        ).all()

        for match_id, bookmaker, market, selection in rows:
            match = s.get(Match, match_id)
            if match is None:
                continue

            existing = s.scalar(
                select(Odds).where(
                    Odds.match_id == match_id,
                    Odds.bookmaker == bookmaker,
                    Odds.market == market,
                    Odds.selection == selection,
                    Odds.odds_type == "closing",
                )
            )
            if existing is not None:
                stats.already_had_closing += 1
                continue

            latest = s.scalar(
                select(Odds)
                .where(
                    Odds.match_id == match_id,
                    Odds.bookmaker == bookmaker,
                    Odds.market == market,
                    Odds.selection == selection,
                    Odds.odds_type == "snapshot",
                    Odds.captured_at <= match.kickoff_utc,
                )
                .order_by(Odds.captured_at.desc())
                .limit(1)
            )
            if latest is None or latest.captured_at is None:
                continue

            lead = match.kickoff_utc - latest.captured_at
            if lead > timedelta(hours=window_hours):
                stats.too_early += 1
                continue

            s.add(
                Odds(
                    match_id=match_id,
                    bookmaker=bookmaker,
                    market=market,
                    selection=selection,
                    decimal_odds=latest.decimal_odds,
                    odds_type="closing",
                    captured_at=latest.captured_at,
                    source="odds_api",
                )
            )
            stats.promoted += 1

    return stats


def backfill_clv(cfg: Config | None = None) -> int:
    """Fill in CLV on settled bets whose closing price has since arrived."""
    from fv.db.models import Bet
    from fv.odds.settlement import clv as clv_of

    cfg = cfg or load_config()
    filled = 0
    with session_scope(cfg) as s:
        bets = s.scalars(
            select(Bet).where(Bet.clv.is_(None), Bet.status.in_(("won", "lost")))
        ).all()
        for bet in bets:
            closing = s.scalar(
                select(Odds)
                .where(
                    Odds.match_id == bet.match_id,
                    Odds.market == "1X2",
                    Odds.selection == bet.selection,
                    Odds.odds_type == "closing",
                )
                .order_by(Odds.bookmaker == "B365")
                .limit(1)
            )
            if closing is None:
                continue
            bet.closing_odds = closing.decimal_odds
            bet.clv = clv_of(bet.odds_taken, closing.decimal_odds)
            filled += 1
    return filled
