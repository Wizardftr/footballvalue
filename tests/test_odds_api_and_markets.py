"""The Odds API client, closing-price promotion, and the new markets.

The API tests run against recorded payloads rather than the live service: the key
lives in .env and is never committed, and a test suite that needs a paid API and a
network is a test suite that stops being run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fv.data.odds_api import (
    MARKET_MAP,
    SPORT_KEYS,
    MissingApiKey,
    OddsApiClient,
    best_quotes,
    parse_events,
)
from fv.models.dixon_coles import fit_dixon_coles


def _event(home="Arsenal", away="Chelsea", event_id="evt1", books=None):
    return {
        "id": event_id,
        "commence_time": "2026-08-15T14:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": books
        or [
            {
                "key": "bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": 2.10},
                            {"name": away, "price": 3.60},
                            {"name": "Draw", "price": 3.40},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.90, "point": 2.5},
                            {"name": "Under", "price": 1.95, "point": 2.5},
                        ],
                    },
                    {
                        "key": "btts",
                        "outcomes": [
                            {"name": "Yes", "price": 1.80},
                            {"name": "No", "price": 2.00},
                        ],
                    },
                ],
            }
        ],
    }


# -- parsing ----------------------------------------------------------------

def test_parses_all_three_markets():
    quotes = parse_events([_event()])
    markets = {q.market for q in quotes}
    assert markets == {"1X2", "OU25", "BTTS"}
    assert len(quotes) == 7  # 3 + 2 + 2


def test_home_and_away_are_resolved_by_team_name():
    """The API labels 1X2 outcomes with team names, not H/D/A."""
    quotes = parse_events([_event(home="Arsenal", away="Chelsea")])
    by_sel = {q.selection: q.decimal_odds for q in quotes if q.market == "1X2"}
    assert by_sel == {"H": 2.10, "D": 3.40, "A": 3.60}


def test_a_different_totals_line_is_discarded():
    """A 3.5 total priced as if it were 2.5 would be an expensive silent error."""
    ev = _event()
    ev["bookmakers"][0]["markets"][1]["outcomes"] = [
        {"name": "Over", "price": 2.50, "point": 3.5},
        {"name": "Under", "price": 1.50, "point": 3.5},
    ]
    assert [q for q in parse_events([ev], line=2.5) if q.market == "OU25"] == []


def test_the_requested_line_is_kept_when_several_are_offered():
    ev = _event()
    ev["bookmakers"][0]["markets"][1]["outcomes"] = [
        {"name": "Over", "price": 2.50, "point": 3.5},
        {"name": "Over", "price": 1.90, "point": 2.5},
        {"name": "Under", "price": 1.95, "point": 2.5},
    ]
    ou = [q for q in parse_events([ev], line=2.5) if q.market == "OU25"]
    assert {q.selection: q.decimal_odds for q in ou} == {"O": 1.90, "U": 1.95}


def test_unknown_markets_and_outcomes_are_ignored():
    ev = _event()
    ev["bookmakers"][0]["markets"].append(
        {"key": "spreads", "outcomes": [{"name": "Arsenal", "price": 1.9, "point": -1.5}]}
    )
    assert all(q.market in {"1X2", "OU25", "BTTS"} for q in parse_events([ev]))


def test_malformed_events_are_skipped_not_fatal():
    assert parse_events([{"id": "x"}, _event()]) != []
    assert parse_events([]) == []
    assert parse_events(None) == []


def test_invalid_prices_are_dropped():
    ev = _event()
    ev["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 1.0
    assert not any(q.market == "1X2" and q.selection == "H" for q in parse_events([ev]))


# -- bookmaker preference ---------------------------------------------------

def test_bet365_is_preferred_even_when_another_book_pays_more():
    """The slip is placed on bet365, so a price we cannot take is not the relevant one."""
    ev = _event(books=[
        {"key": "bet365", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.10}, {"name": "Chelsea", "price": 3.60},
            {"name": "Draw", "price": 3.40}]}]},
        {"key": "williamhill", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.50}, {"name": "Chelsea", "price": 3.60},
            {"name": "Draw", "price": 3.40}]}]},
    ])
    best = best_quotes(parse_events([ev]))
    home = [q for q in best if q.market == "1X2" and q.selection == "H"][0]
    assert home.bookmaker == "bet365"
    assert home.decimal_odds == pytest.approx(2.10)


def test_falls_back_to_the_best_price_when_bet365_is_absent():
    ev = _event(books=[
        {"key": "williamhill", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.20}, {"name": "Chelsea", "price": 3.60},
            {"name": "Draw", "price": 3.40}]}]},
        {"key": "paddypower", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.40}, {"name": "Chelsea", "price": 3.60},
            {"name": "Draw", "price": 3.40}]}]},
    ])
    home = [q for q in best_quotes(parse_events([ev]))
            if q.market == "1X2" and q.selection == "H"][0]
    assert home.bookmaker == "paddypower"
    assert home.decimal_odds == pytest.approx(2.40)


def test_best_quotes_returns_one_price_per_selection():
    ev = _event()
    assert len(best_quotes(parse_events([ev, _event(event_id="evt2")]))) == 14


# -- client behaviour -------------------------------------------------------

def test_missing_key_is_reported_clearly_not_silently():
    client = OddsApiClient(api_key=None)
    assert client.configured is False
    with pytest.raises(MissingApiKey, match="ODDS_API_KEY"):
        client.fetch_odds("E0")


def test_unknown_league_returns_nothing_without_spending_quota():
    client = OddsApiClient(api_key=None)
    assert client.fetch_odds("SC0") == []  # no key needed: never reaches the network


def test_every_configured_league_has_a_sport_key():
    from fv.config import load_config

    unmapped = [lg.code for lg in load_config().leagues if lg.code not in SPORT_KEYS]
    assert unmapped == [], f"leagues without an Odds API sport key: {unmapped}"


def test_quota_is_read_from_response_headers():
    class FakeResponse:
        status_code = 200
        headers = {"x-requests-remaining": "437", "x-requests-used": "63",
                   "x-requests-last": "3"}

        def json(self):
            return [_event()]

    class FakeSession:
        def get(self, *a, **k):
            return FakeResponse()

    client = OddsApiClient(api_key="k", session=FakeSession())
    client.fetch_odds("E0")
    assert (client.quota.remaining, client.quota.used, client.quota.last_cost) == (437, 63, 3)
    assert "437 requests remaining" in client.quota.describe()


def test_quota_exhaustion_is_reported_with_the_numbers():
    class FakeResponse:
        status_code = 429
        headers = {"x-requests-remaining": "0", "x-requests-used": "500"}
        text = "quota"

        def json(self):
            return []

    class FakeSession:
        def get(self, *a, **k):
            return FakeResponse()

    from fv.data.odds_api import OddsApiError

    client = OddsApiClient(api_key="k", session=FakeSession())
    with pytest.raises(OddsApiError, match="Quota exhausted"):
        client.fetch_odds("E0")


def test_market_map_covers_the_markets_we_bet():
    assert {m for m, _ in MARKET_MAP.values()} == {"1X2", "OU25", "BTTS"}


# -- market probabilities from the score matrix -----------------------------

def _fit():
    rng = np.random.default_rng(4)
    k = 900
    teams = np.array([f"T{i}" for i in range(8)])
    h = teams[rng.integers(0, 8, k)]
    a = teams[(np.searchsorted(teams, h) + rng.integers(1, 8, k)) % 8]
    return fit_dixon_coles(h, a, rng.poisson(1.5, k).astype(float),
                           rng.poisson(1.2, k).astype(float), rng.uniform(1, 700, k), xi=0.0)


def test_over_and_under_sum_to_one():
    fit = _fit()
    p = fit.prob_over(fit.teams[0], fit.teams[1], line=2.5)
    assert 0.0 < p < 1.0
    assert p + (1.0 - p) == pytest.approx(1.0)


def test_a_higher_line_is_less_likely_to_be_exceeded():
    fit = _fit()
    h, a = fit.teams[0], fit.teams[1]
    assert fit.prob_over(h, a, 0.5) > fit.prob_over(h, a, 2.5) > fit.prob_over(h, a, 4.5)


def test_btts_is_a_probability():
    fit = _fit()
    assert 0.0 < fit.prob_btts(fit.teams[0], fit.teams[1]) < 1.0


def test_markets_are_consistent_with_the_same_score_matrix():
    """1X2, over/under and BTTS are aggregations of one distribution, so they must
    agree. BTTS implies at least two goals, so it can never exceed P(over 1.5)."""
    fit = _fit()
    h, a = fit.teams[0], fit.teams[2]
    assert fit.prob_btts(h, a) <= fit.prob_over(h, a, 1.5) + 1e-9
    assert sum(fit.predict(h, a)) == pytest.approx(1.0)


def test_over_probability_rises_with_stronger_attacks():
    """A high-scoring pairing must give a higher over 2.5 than a low-scoring one."""
    n = 400
    h = np.array(["Att", "Def"] * n)
    a = np.array(["Def", "Att"] * n)
    hg = np.array([4.0, 0.0] * n)
    ag = np.array([3.0, 0.0] * n)
    days = np.linspace(300, 1, 2 * n)
    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)
    assert fit.prob_over("Att", "Def", 2.5) > 0.5


# -- closing-price promotion ------------------------------------------------

def test_a_stale_snapshot_is_not_treated_as_a_close(tmp_path, monkeypatch):
    """A price captured four days out is not a closing price, and calling it one
    would inflate CLV — the metric the whole project leans on."""
    from fv.config import Config, load_config
    from fv.data.live_odds import promote_closing_odds
    from fv.db.models import LeagueRow, Match, Odds, Team
    from fv.db.session import init_db, session_scope

    base = load_config()
    raw = dict(base.raw)
    raw["data"] = {**raw.get("data", {}), "db_path": str(tmp_path / "t.db")}
    cfg = Config(raw=raw, path=base.path)
    monkeypatch.delenv("FV_DB_PATH", raising=False)
    init_db(cfg)

    kickoff = datetime.utcnow() - timedelta(hours=1)
    with session_scope(cfg) as s:
        if s.get(LeagueRow, "E0") is None:
            s.add(LeagueRow(code="E0", name="PL", country="England", tier=1))
        home, away = Team(canonical_name="A", country="England"), Team(
            canonical_name="B", country="England")
        s.add_all([home, away])
        s.flush()
        m = Match(league_code="E0", season="2026-27", match_date=kickoff.date(),
                  kickoff_utc=kickoff, home_team_id=home.id, away_team_id=away.id,
                  status="played", fthg=1, ftag=0, ftr="H")
        s.add(m)
        s.flush()
        # Captured four days before kickoff: far too early to be a close.
        s.add(Odds(match_id=m.id, bookmaker="bet365", market="1X2", selection="H",
                   decimal_odds=2.10, odds_type="snapshot",
                   captured_at=kickoff - timedelta(days=4)))

    stats = promote_closing_odds(cfg, window_hours=12)
    assert stats.promoted == 0
    assert stats.too_early == 1


def test_a_recent_snapshot_becomes_the_close(tmp_path, monkeypatch):
    from fv.config import Config, load_config
    from fv.data.live_odds import promote_closing_odds
    from fv.db.models import LeagueRow, Match, Odds, Team
    from fv.db.session import init_db, session_scope

    base = load_config()
    raw = dict(base.raw)
    raw["data"] = {**raw.get("data", {}), "db_path": str(tmp_path / "t2.db")}
    cfg = Config(raw=raw, path=base.path)
    monkeypatch.delenv("FV_DB_PATH", raising=False)
    init_db(cfg)

    kickoff = datetime.utcnow() - timedelta(hours=1)
    with session_scope(cfg) as s:
        if s.get(LeagueRow, "E0") is None:
            s.add(LeagueRow(code="E0", name="PL", country="England", tier=1))
        home, away = Team(canonical_name="A", country="England"), Team(
            canonical_name="B", country="England")
        s.add_all([home, away])
        s.flush()
        m = Match(league_code="E0", season="2026-27", match_date=kickoff.date(),
                  kickoff_utc=kickoff, home_team_id=home.id, away_team_id=away.id,
                  status="played", fthg=1, ftag=0, ftr="H")
        s.add(m)
        s.flush()
        # Two snapshots; the later one is the close.
        s.add(Odds(match_id=m.id, bookmaker="bet365", market="1X2", selection="H",
                   decimal_odds=2.10, odds_type="snapshot",
                   captured_at=kickoff - timedelta(hours=6)))
        s.add(Odds(match_id=m.id, bookmaker="bet365", market="1X2", selection="H",
                   decimal_odds=1.95, odds_type="snapshot",
                   captured_at=kickoff - timedelta(hours=1)))

    stats = promote_closing_odds(cfg, window_hours=12)
    assert stats.promoted == 1
    with session_scope(cfg) as s:
        closing = s.query(Odds).filter_by(odds_type="closing").one()
        assert closing.decimal_odds == pytest.approx(1.95), "must take the latest snapshot"


# -- market scoring ---------------------------------------------------------

def test_binary_log_loss_rewards_a_confident_correct_call():
    from fv.backtest.markets import binary_log_loss

    good = binary_log_loss(np.array([0.9]), np.array([1.0]))
    bad = binary_log_loss(np.array([0.1]), np.array([1.0]))
    assert good < bad


def test_binary_log_loss_is_finite_at_the_extremes():
    from fv.backtest.markets import binary_log_loss

    assert np.isfinite(binary_log_loss(np.array([0.0, 1.0]), np.array([1.0, 0.0])))


def test_ou_backtest_settles_correctly():
    from fv.backtest.markets import backtest_ou

    preds = pd.DataFrame([
        {"match_id": 1, "league_code": "E0", "season": "2024-25",
         "kickoff_utc": pd.Timestamp("2024-08-17"), "p_over": 0.70, "p_under": 0.30,
         "actual_over": True, "pre_over": 2.00, "pre_under": 1.90,
         "close_over": 1.90, "close_under": 2.00},
    ])
    res = backtest_ou(preds, min_edge=0.04, stake=10.0)
    assert len(res.bets) == 1
    bet = res.bets.iloc[0]
    assert bet["selection"] == "O"
    assert bet["result"] == "W"
    assert bet["pnl"] == pytest.approx(10.0)  # 10 * (2.00 - 1)
    assert bet["clv"] == pytest.approx(2.00 / 1.90 - 1)


def test_ou_backtest_settles_an_under_bet():
    from fv.backtest.markets import backtest_ou

    preds = pd.DataFrame([
        {"match_id": 1, "league_code": "E0", "season": "2024-25",
         "kickoff_utc": pd.Timestamp("2024-08-17"), "p_over": 0.30, "p_under": 0.70,
         "actual_over": False, "pre_over": 1.90, "pre_under": 2.00,
         "close_over": None, "close_under": None},
    ])
    bet = backtest_ou(preds, stake=10.0).bets.iloc[0]
    assert (bet["selection"], bet["result"]) == ("U", "W")
    assert bet["clv"] is None
