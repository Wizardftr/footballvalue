"""football-data.co.uk downloader and parser.

Quirks in these files that the parser has to survive, all confirmed by inspection
rather than assumed:

* A season file that doesn't exist yet (a season not started) returns an HTML page
  with HTTP 200, not a 404. Parsing that as CSV would inject garbage rows.
* Dates are ``dd/mm/yy`` before roughly 2019 and ``dd/mm/yyyy`` after.
* Column layouts shift between eras: older files have ``Attendance`` and no
  ``Time``; newer ones have ``Time``. Everything is parsed by header name, never by
  position.
* bet365 columns (``B365H/D/A``) start in 2002-03. Closing columns
  (``B365CH/CD/CA``) start in 2019-20. Both are absent before then, so the row is
  loaded for its goals and simply carries no price.
* Files contain trailing blank rows and, in a few seasons, stray unnamed columns.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = "footballvalue/0.1 (personal research tool)"
REQUEST_TIMEOUT = 60


def season_to_code(season: str) -> str:
    """"2024-25" -> "2425" (the path segment football-data uses)."""
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


def code_to_season(code: str) -> str:
    """"2425" -> "2024-25". Assumes 1990s-2080s, which outlives this project."""
    start2, end2 = int(code[:2]), int(code[2:])
    century = 1900 if start2 >= 90 else 2000
    return f"{century + start2}-{end2:02d}"


def season_range(first_season: str, today: datetime | None = None) -> list[str]:
    """All season labels from ``first_season`` to the current one, inclusive.

    A new season is considered to have started in July, which is ahead of every
    league's first fixture. If the file isn't published yet the download just
    records it as missing.
    """
    today = today or datetime.utcnow()
    first_year = int(first_season.split("-")[0])
    current_start = today.year if today.month >= 7 else today.year - 1
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(first_year, current_start + 1)]


def looks_like_html(payload: bytes) -> bool:
    head = payload[:512].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<html" in head


@dataclass
class FetchResult:
    league_code: str
    season: str
    url: str
    status: str  # ok | missing | unchanged | error
    payload: bytes | None = None
    sha256: str | None = None
    path: Path | None = None
    error: str | None = None


def fetch_season(
    league_code: str,
    season: str,
    cache_dir: Path,
    base_url: str,
    force: bool = False,
    session: requests.Session | None = None,
) -> FetchResult:
    """Download one league-season CSV, caching the raw bytes on disk."""
    code = season_to_code(season)
    url = f"{base_url}/{code}/{league_code}.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{league_code}_{code}.csv"

    http = session or requests
    try:
        resp = http.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    except Exception as exc:  # network problems shouldn't kill a 250-file run
        return FetchResult(league_code, season, url, "error", error=str(exc))

    if resp.status_code == 404:
        return FetchResult(league_code, season, url, "missing")
    if resp.status_code != 200:
        return FetchResult(
            league_code, season, url, "error", error=f"HTTP {resp.status_code}"
        )

    payload = resp.content
    if looks_like_html(payload):
        # Season not published yet. Not an error, just nothing to load.
        return FetchResult(league_code, season, url, "missing")

    digest = hashlib.sha256(payload).hexdigest()
    previously = path.read_bytes() if path.exists() else None
    if not force and previously is not None and hashlib.sha256(previously).hexdigest() == digest:
        return FetchResult(
            league_code, season, url, "unchanged", payload=payload, sha256=digest, path=path
        )

    path.write_bytes(payload)
    return FetchResult(league_code, season, url, "ok", payload=payload, sha256=digest, path=path)


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse dd/mm/yy and dd/mm/yyyy, both of which appear in these files."""
    s = raw.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(fallback)


def _parse_kickoff(dates: pd.Series, times: pd.Series | None) -> pd.Series:
    """Combine date and time. Missing times default to 12:00, which keeps ordering
    stable without inventing precision we don't have."""
    if times is None:
        return dates + pd.Timedelta(hours=12)
    t = times.astype(str).str.strip()
    combined = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + t, format="%Y-%m-%d %H:%M", errors="coerce"
    )
    return combined.fillna(dates + pd.Timedelta(hours=12))


# market -> odds_type -> selection -> column name in the CSV.
#
# Over/under 2.5 coverage is patchier than 1X2: bet365 columns appear in 2002-05,
# vanish for 2005-19 (only Betbrain aggregates are published then), and return from
# 2019-20 with closing prices alongside. So the O/U backtest window is effectively
# 2019-20 onward. There are no BTTS prices in this source at all.
MARKET_COLUMNS = {
    "1X2": {
        "pre": {"H": "B365H", "D": "B365D", "A": "B365A"},
        "closing": {"H": "B365CH", "D": "B365CD", "A": "B365CA"},
    },
    "OU25": {
        "pre": {"O": "B365>2.5", "U": "B365<2.5"},
        "closing": {"O": "B365C>2.5", "U": "B365C<2.5"},
    },
}

# Flattened column key -> (market, odds_type, selection), used by the loader.
ODDS_FIELDS = {
    f"b365_{market}_{odds_type}_{sel}".lower(): (market, odds_type, sel)
    for market, by_type in MARKET_COLUMNS.items()
    for odds_type, by_sel in by_type.items()
    for sel in by_sel
}


class WrongLeagueError(ValueError):
    """The file served at a league's URL contains a different league's data."""


def parse_csv(payload: bytes, league_code: str, season: str) -> pd.DataFrame:
    """Parse raw CSV bytes into a tidy frame, one row per match.

    Returns columns: league_code, season, match_date, kickoff_utc, home, away,
    fthg, ftag, ftr, hthg, htag, and whichever b365 market columns exist.

    Raises :class:`WrongLeagueError` when the file's own ``Div`` column disagrees
    with the league we asked for. This is not hypothetical: football-data served
    Scottish Division 1 and 2 data at the La Liga and Segunda URLs for the
    unstarted 2026-27 season, which silently created ten Scottish clubs as Spanish
    teams. Trusting the URL over the file's own contents is exactly the kind of
    silent corruption the team-name handling is careful about everywhere else.
    """
    # These files carry a UTF-8 BOM, which turns the first column into "﻿Div"
    # under latin-1 and hides it from any lookup by name.
    df = pd.read_csv(
        io.BytesIO(payload),
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
    )
    df.columns = [str(c).lstrip("﻿").lstrip("ï»¿").strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    if "Div" in df.columns:
        divisions = {str(d).strip() for d in df["Div"].dropna().unique()}
        if divisions and divisions != {league_code}:
            raise WrongLeagueError(
                f"{league_code} {season}: file contains {sorted(divisions)}, not {league_code}"
            )

    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{league_code} {season}: missing required columns {sorted(missing)}")

    # Drop the trailing blank rows these files carry.
    df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()]
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]

    dates = _parse_dates(df["Date"])
    kickoff = _parse_kickoff(dates, df["Time"] if "Time" in df.columns else None)

    out = pd.DataFrame(
        {
            "league_code": league_code,
            "season": season,
            "match_date": dates,
            "kickoff_utc": kickoff,
            "home": df["HomeTeam"].astype(str).str.strip(),
            "away": df["AwayTeam"].astype(str).str.strip(),
            "fthg": pd.to_numeric(df["FTHG"], errors="coerce"),
            "ftag": pd.to_numeric(df["FTAG"], errors="coerce"),
            "hthg": pd.to_numeric(df.get("HTHG"), errors="coerce")
            if "HTHG" in df.columns
            else pd.NA,
            "htag": pd.to_numeric(df.get("HTAG"), errors="coerce")
            if "HTAG" in df.columns
            else pd.NA,
        }
    )

    for market, by_type in MARKET_COLUMNS.items():
        for odds_type, mapping in by_type.items():
            for sel, col in mapping.items():
                key = f"b365_{market}_{odds_type}_{sel}".lower()
                out[key] = (
                    pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA
                )

    # A row without a parseable date can't be ordered, so it can't be used.
    out = out[out["match_date"].notna()]

    # Derive the result rather than trusting FTR, which is occasionally blank.
    played = out["fthg"].notna() & out["ftag"].notna()
    out["ftr"] = pd.NA
    out.loc[played, "ftr"] = [
        "H" if h > a else ("A" if h < a else "D")
        for h, a in zip(out.loc[played, "fthg"], out.loc[played, "ftag"], strict=True)
    ]
    out["status"] = played.map({True: "played", False: "scheduled"})

    return out.reset_index(drop=True)
