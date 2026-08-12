"""Combined bets.

A combination is one stake with one outcome. The whole reason it lives in its own
tables is that treating its legs as ordinary bets would report a profit on the legs
that won when the bet itself returned nothing — which is exactly the lie this
project exists to avoid.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fv.bets import combo_log, current_bankroll, log_combo, settle_all, settle_combos
from fv.config import Config, load_config
from fv.db.models import LeagueRow, Match, Team
from fv.db.session import init_db, session_scope
from fv.settings_store import set_setting


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Config:
    base = load_config()
    raw = dict(base.raw)
    raw["data"] = {**raw.get("data", {}), "db_path": str(tmp_path / "test.db")}
    c = Config(raw=raw, path=base.path)
    monkeypatch.delenv("FV_DB_PATH", raising=False)
    init_db(c)
    return c


def _match(cfg, home, away, score=None, played=True):
    with session_scope(cfg) as s:
        if s.get(LeagueRow, "E0") is None:
            s.add(LeagueRow(code="E0", name="PL", country="England", tier=1))
        ids = {}
        for name in (home, away):
            team = s.query(Team).filter_by(canonical_name=name, country="England").one_or_none()
            if team is None:
                team = Team(canonical_name=name, country="England")
                s.add(team)
                s.flush()
            ids[name] = team.id
        kickoff = datetime.utcnow() - timedelta(hours=3)
        m = Match(league_code="E0", season="2026-27", match_date=kickoff.date(),
                  kickoff_utc=kickoff, home_team_id=ids[home], away_team_id=ids[away],
                  status="played" if played else "scheduled",
                  fthg=score[0] if score else None, ftag=score[1] if score else None)
        s.add(m)
        s.flush()
        return m.id


def _legs(rows):
    return pd.DataFrame(rows)


def test_one_losing_leg_loses_the_whole_stake(cfg):
    """Three winners and one loser returns nothing. Recording the three winners as
    separate bets would have shown a profit."""
    won1 = _match(cfg, "A1", "A2", (3, 0))   # over 2.5 wins
    won2 = _match(cfg, "B1", "B2", (2, 1))   # over 2.5 wins
    won3 = _match(cfg, "C1", "C2", (0, 2))   # away wins
    lost = _match(cfg, "D1", "D2", (1, 0))   # away loses

    set_setting("starting_bankroll", 5.0, cfg)
    combo = log_combo(_legs([
        {"match_id": won1, "market": "OU25", "selection": "O", "odds": 1.70},
        {"match_id": won2, "market": "OU25", "selection": "O", "odds": 1.80},
        {"match_id": won3, "market": "1X2", "selection": "A", "odds": 1.50},
        {"match_id": lost, "market": "1X2", "selection": "A", "odds": 2.20},
    ]), stake=5.0, combined_odds=10.10, mode="real", cfg=cfg)

    settle_combos(cfg)
    row = combo_log(cfg).iloc[0]
    assert row["id"] == combo
    assert row["status"] == "lost"
    assert row["pnl"] == pytest.approx(-5.0)
    assert row["legs"] == 4
    assert row["legs_won"] == 3
    assert current_bankroll(cfg) == pytest.approx(0.0)


def test_every_leg_landing_pays_the_bookmakers_price(cfg):
    a = _match(cfg, "A1", "A2", (3, 0))
    b = _match(cfg, "B1", "B2", (0, 2))
    set_setting("starting_bankroll", 10.0, cfg)
    log_combo(_legs([
        {"match_id": a, "market": "OU25", "selection": "O", "odds": 1.70},
        {"match_id": b, "market": "1X2", "selection": "A", "odds": 1.50},
    ]), stake=10.0, combined_odds=2.55, cfg=cfg)

    settle_combos(cfg)
    row = combo_log(cfg).iloc[0]
    assert row["status"] == "won"
    # The stored price, not the recomputed product: bookmakers round.
    assert row["pnl"] == pytest.approx(15.5)
    assert current_bankroll(cfg) == pytest.approx(25.5)


def test_a_voided_leg_drops_out_and_shortens_the_price(cfg):
    """What bookmakers actually do. Treating a void as a loss would take money the
    bet never risked."""
    won = _match(cfg, "A1", "A2", (3, 0))
    abandoned = _match(cfg, "B1", "B2", None, played=True)
    set_setting("starting_bankroll", 10.0, cfg)
    log_combo(_legs([
        {"match_id": won, "market": "OU25", "selection": "O", "odds": 2.00},
        {"match_id": abandoned, "market": "1X2", "selection": "H", "odds": 3.00},
    ]), stake=10.0, combined_odds=6.00, cfg=cfg)

    settle_combos(cfg)
    row = combo_log(cfg).iloc[0]
    assert row["status"] == "won"
    # Paid at 2.00, the surviving leg — not at the 6.00 originally struck.
    assert row["pnl"] == pytest.approx(10.0)


def test_all_legs_void_returns_the_stake(cfg):
    a = _match(cfg, "A1", "A2", None, played=True)
    b = _match(cfg, "B1", "B2", None, played=True)
    set_setting("starting_bankroll", 10.0, cfg)
    log_combo(_legs([
        {"match_id": a, "market": "1X2", "selection": "H", "odds": 2.0},
        {"match_id": b, "market": "1X2", "selection": "H", "odds": 2.0},
    ]), stake=10.0, cfg=cfg)

    settle_combos(cfg)
    assert combo_log(cfg).iloc[0]["status"] == "void"
    assert current_bankroll(cfg) == pytest.approx(10.0)


def test_an_unfinished_leg_keeps_the_whole_bet_pending(cfg):
    done = _match(cfg, "A1", "A2", (3, 0))
    later = _match(cfg, "B1", "B2", None, played=False)
    log_combo(_legs([
        {"match_id": done, "market": "OU25", "selection": "O", "odds": 1.7},
        {"match_id": later, "market": "1X2", "selection": "H", "odds": 2.0},
    ]), stake=5.0, cfg=cfg)

    stats = settle_combos(cfg)
    assert stats.settled == 0
    assert stats.still_pending == 1
    assert combo_log(cfg).iloc[0]["status"] == "pending"


def test_settling_twice_does_not_pay_twice(cfg):
    a = _match(cfg, "A1", "A2", (3, 0))
    set_setting("starting_bankroll", 10.0, cfg)
    log_combo(_legs([{"match_id": a, "market": "OU25", "selection": "O", "odds": 2.0}]),
              stake=10.0, cfg=cfg)
    settle_combos(cfg)
    after = current_bankroll(cfg)
    settle_combos(cfg)
    assert current_bankroll(cfg) == pytest.approx(after)


def test_combined_odds_default_to_the_product_of_the_legs(cfg):
    a = _match(cfg, "A1", "A2", (3, 0))
    b = _match(cfg, "B1", "B2", (0, 2))
    log_combo(_legs([
        {"match_id": a, "market": "OU25", "selection": "O", "odds": 2.0},
        {"match_id": b, "market": "1X2", "selection": "A", "odds": 1.5},
    ]), stake=5.0, cfg=cfg)
    assert combo_log(cfg).iloc[0]["combined_odds"] == pytest.approx(3.0)


def test_a_combo_needs_legs_and_a_real_stake(cfg):
    with pytest.raises(ValueError, match="at least one leg"):
        log_combo(pd.DataFrame(), stake=5.0, cfg=cfg)
    with pytest.raises(ValueError, match="stake must be positive"):
        log_combo(_legs([{"match_id": 1, "selection": "H", "odds": 2.0}]), stake=0, cfg=cfg)


def test_settle_all_covers_singles_and_combos(cfg):
    from fv.bets import log_slip

    single = _match(cfg, "S1", "S2", (2, 0))
    combo_match = _match(cfg, "K1", "K2", (3, 0))
    log_slip(pd.DataFrame([{"match_id": single, "market": "1X2", "selection": "H",
                            "odds": 2.0, "stake": 1.0}]), cfg=cfg)
    log_combo(_legs([{"match_id": combo_match, "market": "OU25", "selection": "O",
                      "odds": 2.0}]), stake=1.0, cfg=cfg)

    stats = settle_all(cfg)
    assert stats.settled == 2
    assert stats.won == 2


def test_the_balance_starts_from_the_users_own_figure(cfg):
    """The Settings page lets you set a starting balance; the ledger has to begin
    there, not at config.yaml's default."""
    set_setting("starting_bankroll", 5.0, cfg)
    assert current_bankroll(cfg) == pytest.approx(5.0)
