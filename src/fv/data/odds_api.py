"""The Odds API client (the-odds-api.com).

Used for two things football-data cannot provide: prices for markets it doesn't
publish (BTTS), and repeated snapshots close to kickoff so closing-line value can be
measured on bets we actually placed rather than only in backtests.

Three things shape this module:

**Quota is the binding constraint.** The free tier allows 500 requests a month, and
one request costs `regions x markets` credits, not one. Fetching three markets in one
region for one league is 3 credits. At a weekly refresh across eleven leagues that
adds up fast, so requests are batched per league, cached on disk, and the remaining
quota is read back from the response headers and surfaced.

**bet365 may be absent.** The API returns whichever bookmakers it has for a region.
The spec asks to prefer bet365 and fall back to the best available price, so every
quote records which bookmaker it came from — mixing books silently would corrupt CLV,
because a "closing price" from a different book is not the close you bet into.

**No key is a normal state, not an error.** The key lives in .env and is never
committed, so the client reports the situation clearly instead of raising.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.the-odds-api.com/v4"

# fv league code -> The Odds API sport key.
SPORT_KEYS = {
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division",
    "I1": "soccer_italy_serie_a",
    "I2": "soccer_italy_serie_b",
    "D1": "soccer_germany_bundesliga",
    "D2": "soccer_germany_bundesliga2",
    "F1": "soccer_france_ligue_one",
    "F2": "soccer_france_ligue_two",
    "N1": "soccer_netherlands_eredivisie",
}

# The Odds API market key -> (our market, {api outcome name -> our selection})
MARKET_MAP = {
    "h2h": ("1X2", None),  # outcomes are team names; resolved per event
    "totals": ("OU25", {"Over": "O", "Under": "U"}),
    "btts": ("BTTS", {"Yes": "Y", "No": "N"}),
}

PREFERRED_BOOKMAKER = "bet365"
DEFAULT_MARKETS = ("h2h", "totals", "btts")
REQUEST_DELAY_SECONDS = 1.0


class OddsApiError(RuntimeError):
    pass


class MissingApiKey(OddsApiError):
    """Raised only where a caller has explicitly asked for a live fetch."""


@dataclass
class Quota:
    """Remaining request allowance, read from the response headers."""

    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None

    def describe(self) -> str:
        if self.remaining is None:
            return "quota unknown"
        return f"{self.remaining} requests remaining (used {self.used}, last cost {self.last_cost})"


@dataclass
class Quote:
    """One price for one selection, from one bookmaker, at one moment."""

    event_id: str
    commence_time: datetime
    home_name: str
    away_name: str
    market: str
    selection: str
    decimal_odds: float
    bookmaker: str
    point: float | None = None  # the line, for totals
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_events(payload: list[dict], line: float = 2.5) -> list[Quote]:
    """Turn an Odds API ``/odds`` payload into flat quotes.

    Only the requested totals line is kept. The API returns whatever lines a book
    offers, and a 3.5 total priced as if it were 2.5 would be a silent, expensive
    mistake.
    """
    quotes: list[Quote] = []
    for event in payload or []:
        try:
            event_id = event["id"]
            commence = _parse_time(event["commence_time"])
            home, away = event["home_team"], event["away_team"]
        except (KeyError, TypeError, ValueError):
            continue

        for book in event.get("bookmakers", []) or []:
            book_key = book.get("key", "unknown")
            for market in book.get("markets", []) or []:
                api_market = market.get("key")
                if api_market not in MARKET_MAP:
                    continue
                our_market, outcome_map = MARKET_MAP[api_market]

                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if price is None or float(price) <= 1.0:
                        continue

                    if our_market == "1X2":
                        if name == home:
                            sel = "H"
                        elif name == away:
                            sel = "A"
                        elif name == "Draw":
                            sel = "D"
                        else:
                            continue
                    else:
                        if point is not None and float(point) != line:
                            continue  # a different total; not ours
                        sel = (outcome_map or {}).get(name)
                        if sel is None:
                            continue

                    quotes.append(
                        Quote(
                            event_id=event_id,
                            commence_time=commence,
                            home_name=home,
                            away_name=away,
                            market=our_market,
                            selection=sel,
                            decimal_odds=float(price),
                            bookmaker=book_key,
                            point=float(point) if point is not None else None,
                        )
                    )
    return quotes


def best_quotes(quotes: list[Quote], prefer: str = PREFERRED_BOOKMAKER) -> list[Quote]:
    """One quote per (event, market, selection): bet365 if present, else best price.

    Preferring bet365 even when another book is offering more is deliberate — the
    slip is placed on bet365, so a price we cannot actually take is not the relevant
    number. The fallback exists so a missing bet365 quote doesn't drop the selection
    entirely, and the bookmaker is recorded either way.
    """
    grouped: dict[tuple[str, str, str], list[Quote]] = {}
    for q in quotes:
        grouped.setdefault((q.event_id, q.market, q.selection), []).append(q)

    out = []
    for group in grouped.values():
        preferred = [q for q in group if q.bookmaker == prefer]
        out.append(preferred[0] if preferred else max(group, key=lambda q: q.decimal_odds))
    return out


class OddsApiClient:
    def __init__(
        self,
        api_key: str | None,
        cache_dir: Path | None = None,
        region: str = "uk",
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.region = region
        self.session = session or requests.Session()
        self.quota = Quota()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _record_quota(self, resp: requests.Response) -> None:
        def as_int(header: str) -> int | None:
            value = resp.headers.get(header)
            try:
                return int(value) if value is not None else None
            except ValueError:
                return None

        self.quota = Quota(
            remaining=as_int("x-requests-remaining"),
            used=as_int("x-requests-used"),
            last_cost=as_int("x-requests-last"),
        )

    def fetch_odds(
        self,
        league_code: str,
        markets: tuple[str, ...] = DEFAULT_MARKETS,
        force: bool = False,
        cache_max_age_minutes: int = 60,
    ) -> list[dict]:
        """Fetch current odds for one league. Returns the raw payload.

        One call costs ``len(markets)`` credits per region, so markets are requested
        together rather than in separate calls.
        """
        sport = SPORT_KEYS.get(league_code)
        if sport is None:
            return []
        if not self.configured:
            raise MissingApiKey(
                "ODDS_API_KEY is not set. Copy .env.example to .env and add a key from "
                "https://the-odds-api.com — the free tier is enough for weekly use."
            )

        cached = self._read_cache(league_code, cache_max_age_minutes) if not force else None
        if cached is not None:
            return cached

        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        resp = self.session.get(url, params=params, timeout=60)
        self._record_quota(resp)

        if resp.status_code == 401:
            raise OddsApiError("The Odds API rejected the key (401).")
        if resp.status_code == 429:
            raise OddsApiError(f"Quota exhausted (429). {self.quota.describe()}")
        if resp.status_code != 200:
            raise OddsApiError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        self._write_cache(league_code, payload)
        time.sleep(REQUEST_DELAY_SECONDS)
        return payload

    def _cache_path(self, league_code: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"oddsapi_{league_code}.json"

    def _read_cache(self, league_code: str, max_age_minutes: int) -> list[dict] | None:
        path = self._cache_path(league_code)
        if path is None or not path.exists():
            return None
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        )
        if age > timedelta(minutes=max_age_minutes):
            return None
        try:
            return json.loads(path.read_text())
        except ValueError:
            return None

    def _write_cache(self, league_code: str, payload) -> None:
        path = self._cache_path(league_code)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
