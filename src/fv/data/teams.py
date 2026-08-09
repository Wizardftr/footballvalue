"""Team name normalization.

football-data.co.uk turns out to be internally consistent: Premier League files use
46 distinct home-team strings across 26 seasons, exactly the number of clubs that
have played in the division, with no spelling drift. So within that source an exact
(country, name) match is the right resolution rule, and a team relegated from E0 to
E1 keeps its id because both files spell it the same way.

The alias table exists for the sources that *aren't* consistent — The Odds API
("Manchester United"), understat, FBref — which arrive in Phases 2 and 4.

The rule that matters: a name we cannot resolve is an error, never a silently
created new team. Splitting one club into two ids would corrupt its ratings
invisibly, which is far worse than a loud failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from fv.db.models import Team, TeamAlias

# Below this similarity we refuse to guess.
FUZZY_ACCEPT = 92.0
FUZZY_SUGGEST = 75.0


def normalize_raw(name: str) -> str:
    """Light cleanup only: trim and collapse whitespace.

    Deliberately does not strip punctuation or case-fold, because "Nott'm Forest"
    and "M'gladbach" are the canonical football-data spellings and mangling them
    would only create more names to reconcile.
    """
    return " ".join(str(name).split()).strip()


@dataclass
class UnresolvedName:
    raw: str
    source: str
    country: str
    suggestion: str | None = None
    score: float = 0.0


class TeamResolver:
    """Resolves raw team strings to canonical team ids, scoped by country."""

    def __init__(self, session: Session, source: str = "football_data"):
        self.session = session
        self.source = source
        self._cache: dict[tuple[str, str], int] = {}
        self.unresolved: list[UnresolvedName] = []
        self._load()

    def _load(self) -> None:
        for team in self.session.scalars(select(Team)).all():
            self._cache[(team.country, team.canonical_name)] = team.id
        stmt = select(TeamAlias, Team).join(Team, TeamAlias.team_id == Team.id)
        for alias, team in self.session.execute(stmt).all():
            self._cache[(team.country, alias.alias)] = team.id

    def get_or_create(self, raw: str, country: str, season: str | None = None) -> int:
        """Resolve a name from the authoritative source, creating the team if new.

        Only use this for football-data, which defines the canonical spelling. Other
        sources must go through :meth:`resolve`, which never creates.
        """
        name = normalize_raw(raw)
        key = (country, name)
        if key in self._cache:
            team_id = self._cache[key]
            if season:
                self._touch_season(team_id, season)
            return team_id

        team = Team(
            canonical_name=name,
            country=country,
            first_seen_season=season,
            last_seen_season=season,
        )
        self.session.add(team)
        self.session.flush()
        self._cache[key] = team.id
        return team.id

    def resolve(self, raw: str, country: str) -> int | None:
        """Resolve a name from a secondary source. Never creates a team.

        Returns None and records the name in ``self.unresolved`` when there is no
        confident match, so the caller can fail loudly and a human can add the alias.
        """
        name = normalize_raw(raw)
        key = (country, name)
        if key in self._cache:
            return self._cache[key]

        candidates = [n for (c, n) in self._cache if c == country]
        if not candidates:
            self.unresolved.append(UnresolvedName(raw=name, source=self.source, country=country))
            return None

        match = process.extractOne(name, candidates, scorer=fuzz.WRatio)
        best, score = (match[0], match[1]) if match else (None, 0.0)

        if score >= FUZZY_ACCEPT:
            team_id = self._cache[(country, best)]
            self._record_alias(name, team_id, confidence=score / 100.0)
            self._cache[(country, name)] = team_id
            return team_id

        self.unresolved.append(
            UnresolvedName(
                raw=name,
                source=self.source,
                country=country,
                suggestion=best if score >= FUZZY_SUGGEST else None,
                score=score,
            )
        )
        return None

    def _record_alias(self, alias: str, team_id: int, confidence: float) -> None:
        exists = self.session.scalar(
            select(TeamAlias).where(TeamAlias.alias == alias, TeamAlias.source == self.source)
        )
        if exists is None:
            self.session.add(
                TeamAlias(
                    alias=alias,
                    source=self.source,
                    team_id=team_id,
                    confidence=confidence,
                    verified=False,
                )
            )

    def _touch_season(self, team_id: int, season: str) -> None:
        team = self.session.get(Team, team_id)
        if team is None:
            return
        if team.first_seen_season is None or season < team.first_seen_season:
            team.first_seen_season = season
        if team.last_seen_season is None or season > team.last_seen_season:
            team.last_seen_season = season
