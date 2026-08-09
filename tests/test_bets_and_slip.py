"""Bet logging, settlement, the bankroll ledger, and the real-money gate.

These run against a temporary database built per test, so they exercise the real
SQLAlchemy paths rather than mocks — settlement bugs tend to live in the join
between bets and matches, which a mock would hide.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fv.config import Config, load_config
from fv.db.models import LeagueRow, Match, Odds, Team
from fv.db.session import init_db, session_scope


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Config:
    """A config pointed at an empty temporary database."""
    base = load_config()
    raw = dict(base.raw)
    raw["data"] = {**raw.get("data", {}), "db_path": str(tmp_path / "test.db")}
    c = Config(raw=raw, path=base.path)
    monkeypatch.delenv("FV_DB_PATH", raising=False)
    init_db(c)
    return c


def _add_match(cfg, home="Alpha", away="Beta", days_ahead=3, odds=(2.10, 3.40, 3.80),
               played=False, score=(2, 1)):
    with session_scope(cfg) as s:
        if s.get(LeagueRow, "E0") is None:
            s.add(LeagueRow(code="E0", name="PL", country="England", tier=1))
            s.flush()
        ids = {}
        for name in (home, away):
            team = s.query(Team).filter_by(canonical_name=name, country="England").one_or_none()
            if team is None:
                team = Team(canonical_name=name, country="England")
                s.add(team)
                s.flush()
            ids[name] = team.id
        kickoff = datetime.utcnow() + timedelta(days=days_ahead)
        m = Match(
            league_code="E0", season="2026-27", match_date=kickoff.date(),
            kickoff_utc=kickoff, home_team_id=ids[home], away_team_id=ids[away],
            status="played" if played else "scheduled",
            fthg=score[0] if played else None, ftag=score[1] if played else None,
            ftr=("H" if score[0] > score[1] else "A" if score[0] < score[1] else "D")
            if played else None,
        )
        s.add(m)
        s.flush()
        for sel, o in zip(("H", "D", "A"), odds, strict=True):
            s.add(Odds(match_id=m.id, bookmaker="B365", market="1X2", selection=sel,
                       decimal_odds=o, odds_type="pre"))
        return m.id


def _selections(match_id, selection="H", odds=2.10, stake=10.0):
    return pd.DataFrame([{
        "match_id": match_id, "selection": selection, "odds": odds, "stake": stake,
    }])


# -- bankroll ledger --------------------------------------------------------

def test_bankroll_starts_from_config(cfg):
    from fv.bets import current_bankroll

    assert current_bankroll(cfg) == pytest.approx(cfg.betting.get("starting_bankroll", 1000.0))


def test_ledger_events_accumulate(cfg):
    from fv.bets import current_bankroll, record_bankroll_event

    start = current_bankroll(cfg)
    record_bankroll_event("deposit", 500.0, cfg=cfg)
    assert current_bankroll(cfg) == pytest.approx(start + 500.0)
    record_bankroll_event("withdrawal", -200.0, cfg=cfg)
    assert current_bankroll(cfg) == pytest.approx(start + 300.0)


# -- logging ----------------------------------------------------------------

def test_logging_a_slip_creates_pending_bets(cfg):
    from fv.bets import bet_log, log_slip

    mid = _add_match(cfg)
    slip_id, n = log_slip(_selections(mid), mode="paper", cfg=cfg)
    assert n == 1
    log = bet_log(cfg)
    assert len(log) == 1
    assert log.iloc[0]["status"] == "pending"
    assert log.iloc[0]["slip_id"] == slip_id


def test_logging_does_not_move_the_bankroll(cfg):
    """The ledger records results, not stakes. Recording both double-counts a win."""
    from fv.bets import current_bankroll, log_slip

    before = current_bankroll(cfg)
    log_slip(_selections(_add_match(cfg)), cfg=cfg)
    assert current_bankroll(cfg) == pytest.approx(before)


def test_the_same_pending_bet_is_not_logged_twice(cfg):
    """Regenerating a slip mid-week must not duplicate bets already placed."""
    from fv.bets import log_slip

    mid = _add_match(cfg)
    log_slip(_selections(mid), cfg=cfg)
    _, n = log_slip(_selections(mid), cfg=cfg)
    assert n == 0


# -- settlement -------------------------------------------------------------

def test_pending_bets_without_a_result_stay_pending(cfg):
    from fv.bets import log_slip, settle_pending

    log_slip(_selections(_add_match(cfg)), cfg=cfg)
    stats = settle_pending(cfg)
    assert stats.settled == 0
    assert stats.still_pending == 1


def test_winning_bet_settles_and_credits_the_ledger(cfg):
    from fv.bets import bet_log, current_bankroll, log_slip, settle_pending

    mid = _add_match(cfg, played=True, score=(2, 1))
    before = current_bankroll(cfg)
    log_slip(_selections(mid, "H", 2.10, 10.0), cfg=cfg)

    stats = settle_pending(cfg)
    assert (stats.settled, stats.won) == (1, 1)
    assert stats.pnl == pytest.approx(11.0)  # 10 * (2.10 - 1)
    assert current_bankroll(cfg) == pytest.approx(before + 11.0)
    assert bet_log(cfg).iloc[0]["status"] == "won"


def test_losing_bet_debits_the_ledger(cfg):
    from fv.bets import current_bankroll, log_slip, settle_pending

    mid = _add_match(cfg, played=True, score=(0, 2))
    before = current_bankroll(cfg)
    log_slip(_selections(mid, "H", 2.10, 10.0), cfg=cfg)
    settle_pending(cfg)
    assert current_bankroll(cfg) == pytest.approx(before - 10.0)


def test_settling_twice_does_not_double_count(cfg):
    """Re-running settlement is routine; it must be idempotent."""
    from fv.bets import current_bankroll, log_slip, settle_pending

    mid = _add_match(cfg, played=True, score=(2, 1))
    log_slip(_selections(mid, "H", 2.10, 10.0), cfg=cfg)
    settle_pending(cfg)
    after_first = current_bankroll(cfg)
    settle_pending(cfg)
    assert current_bankroll(cfg) == pytest.approx(after_first)


def test_settlement_records_clv_when_a_closing_price_exists(cfg):
    from fv.bets import bet_log, log_slip, settle_pending

    mid = _add_match(cfg, played=True, score=(2, 1))
    with session_scope(cfg) as s:
        s.add(Odds(match_id=mid, bookmaker="B365", market="1X2", selection="H",
                   decimal_odds=2.00, odds_type="closing"))
    log_slip(_selections(mid, "H", 2.10, 10.0), cfg=cfg)
    settle_pending(cfg)
    row = bet_log(cfg).iloc[0]
    assert row["closing_odds"] == pytest.approx(2.00)
    assert row["clv"] == pytest.approx(0.05)  # took 2.10, closed 2.00


def test_manual_settlement_overrides_and_adjusts_the_ledger(cfg):
    from fv.bets import bet_log, current_bankroll, log_slip, manual_settle, settle_pending

    mid = _add_match(cfg, played=True, score=(2, 1))
    log_slip(_selections(mid, "H", 2.10, 10.0), cfg=cfg)
    settle_pending(cfg)
    won_balance = current_bankroll(cfg)

    bet_id = int(bet_log(cfg).iloc[0]["id"])
    manual_settle(bet_id, "void", cfg=cfg)

    assert bet_log(cfg).iloc[0]["status"] == "void"
    # The +11.00 win is backed out, leaving the balance where it started.
    assert current_bankroll(cfg) == pytest.approx(won_balance - 11.0)


def test_manual_settlement_rejects_an_invalid_status(cfg):
    with pytest.raises(ValueError):
        from fv.bets import manual_settle

        manual_settle(1, "cashed-out", cfg=cfg)


# -- rolling ROI ------------------------------------------------------------

def test_rolling_roi_reports_an_interval():
    from fv.bets import rolling_roi

    n = 50
    df = pd.DataFrame({
        "kickoff_utc": pd.date_range("2025-01-01", periods=n, freq="D"),
        "status": ["won", "lost"] * (n // 2),
        "pnl": [11.0, -10.0] * (n // 2),
        "stake": [10.0] * n,
    })
    out = rolling_roi(df, window=30, min_periods=10)
    assert not out.empty
    assert (out["ci_low"] <= out["roi"]).all()
    assert (out["roi"] <= out["ci_high"]).all()
    assert (out["n"] >= 10).all()


def test_rolling_roi_suppresses_the_opening_stretch():
    """ROI from the first two bets swings wildly and means nothing; it must not plot."""
    from fv.bets import rolling_roi

    n = 40
    df = pd.DataFrame({
        "kickoff_utc": pd.date_range("2025-01-01", periods=n, freq="D"),
        "status": ["won", "lost"] * (n // 2),
        "pnl": [11.0, -10.0] * (n // 2),
        "stake": [10.0] * n,
    })
    out = rolling_roi(df, window=300, min_periods=30)
    assert len(out) == n - 30 + 1
    assert out["n"].min() == 30


def test_rolling_roi_ignores_unsettled_bets():
    from fv.bets import rolling_roi

    df = pd.DataFrame({
        "kickoff_utc": pd.date_range("2025-01-01", periods=3, freq="D"),
        "status": ["pending", "pending", "pending"],
        "pnl": [None, None, None],
        "stake": [10.0, 10.0, 10.0],
    })
    assert rolling_roi(df).empty


# -- the real-money gate ----------------------------------------------------

def test_real_money_is_refused_without_a_backtest(cfg, tmp_path):
    """An absent backtest counts as a failure, not an unknown.

    A gate you can pass by not running the test is not a gate.
    """
    from fv.settings_store import real_money_readiness

    r = real_money_readiness(cfg, stage_report=tmp_path / "nonexistent.csv")
    assert r.ready is False
    assert any("No stage backtest" in reason for reason in r.reasons)


def test_real_money_is_refused_when_the_model_loses_to_the_market(cfg, tmp_path):
    from fv.settings_store import real_money_readiness

    csv = tmp_path / "stage_comparison.csv"
    pd.DataFrame([{"stage": "anchored", "log_loss": 1.0029}]).to_csv(csv, index=False)
    csv.with_suffix(".md").write_text("| **bet365 margin-free** | **1.00255** | — |")

    r = real_money_readiness(cfg, stage_report=csv)
    assert r.backtest_beats_closing is False
    assert r.ready is False


def test_real_money_is_refused_without_enough_paper_trading(cfg, tmp_path):
    """Even a model that beats the market must serve its paper-trading time."""
    from fv.settings_store import real_money_readiness

    csv = tmp_path / "stage_comparison.csv"
    pd.DataFrame([{"stage": "anchored", "log_loss": 0.95}]).to_csv(csv, index=False)
    csv.with_suffix(".md").write_text("| **bet365 margin-free** | **1.00255** | — |")

    r = real_money_readiness(cfg, stage_report=csv)
    assert r.backtest_beats_closing is True
    assert r.ready is False
    assert any("Paper trading has run" in reason for reason in r.reasons)


def test_settings_overrides_persist(cfg):
    from fv.settings_store import effective_settings, get_setting, set_setting

    set_setting("min_edge", 0.07, cfg=cfg)
    assert get_setting("min_edge", cfg=cfg) == pytest.approx(0.07)
    assert effective_settings(cfg)["min_edge"] == pytest.approx(0.07)


def test_paper_mode_defaults_to_on(cfg):
    """Real money must be a deliberate act, never a default."""
    from fv.settings_store import effective_settings

    assert effective_settings(cfg)["paper_mode"] is True


# -- filled slips -----------------------------------------------------------
# Requested feature: always return N selections so there is a slip to place each
# week. Ranked by edge rather than by win probability - probability ranking returns
# the shortest-priced favourites, which are the most likely winners and a reliable
# way to lose money at a bookmaker's margin.

def test_filled_slip_reports_its_own_negative_expected_return():
    """A filled slip must not present itself as a set of recommendations."""
    from fv.slip import Slip

    sel = pd.DataFrame({
        "stake": [20.0, 20.0, 20.0],
        "edge": [-0.05, -0.06, -0.07],
    })
    slip = Slip(sel, pd.DataFrame(), bankroll=1000.0, mode="filled")
    assert slip.total_stake == pytest.approx(60.0)
    # 20*(-0.05) + 20*(-0.06) + 20*(-0.07)
    assert slip.expected_return == pytest.approx(-3.6)
    assert slip.expected_return < 0


def test_expected_return_is_positive_on_a_genuine_value_slip():
    from fv.slip import Slip

    sel = pd.DataFrame({"stake": [10.0, 10.0], "edge": [0.05, 0.08]})
    slip = Slip(sel, pd.DataFrame(), bankroll=1000.0, mode="value")
    assert slip.expected_return == pytest.approx(1.3)


def test_slip_text_warns_that_a_filled_slip_is_expected_to_lose():
    from fv.slip import Slip, slip_to_text

    sel = pd.DataFrame({
        "kickoff_utc": [pd.Timestamp("2026-08-09 15:00")],
        "league_code": ["N1"], "home": ["A"], "away": ["B"], "selection": ["H"],
        "odds": [2.0], "stake": [20.0], "model_prob": [0.45],
        "market_prob_fair": [0.475], "edge": [-0.10],
    })
    text = slip_to_text(Slip(sel, pd.DataFrame(), bankroll=1000.0, mode="filled"))
    assert "FILLED SLIP" in text
    assert "Expected return" in text
    assert "least bad" in text


def test_a_value_slip_carries_no_filled_warning():
    from fv.slip import Slip, slip_to_text

    sel = pd.DataFrame({
        "kickoff_utc": [pd.Timestamp("2026-08-09 15:00")],
        "league_code": ["E0"], "home": ["A"], "away": ["B"], "selection": ["H"],
        "odds": [2.0], "stake": [20.0], "model_prob": [0.55],
        "market_prob_fair": [0.50], "edge": [0.10],
    })
    text = slip_to_text(Slip(sel, pd.DataFrame(), bankroll=1000.0, mode="value"))
    assert "FILLED SLIP" not in text
