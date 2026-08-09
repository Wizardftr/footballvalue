"""Accounts, and the isolation between them.

The moment this app has more than one user, "does the maths work" stops being the
only question — "can Ana see Ben's bets" becomes just as important, and it is the
kind of thing that breaks silently. Most of this file is about that.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import text

from fv import auth, chat
from fv.bets import bet_log, current_bankroll, log_slip, record_bankroll_event, settle_pending
from fv.config import Config, load_config
from fv.db.models import LeagueRow, Match, Team
from fv.db.session import get_engine, init_db, session_scope
from fv.settings_store import effective_settings, get_setting, set_setting


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Config:
    base = load_config()
    raw = dict(base.raw)
    raw["data"] = {**raw.get("data", {}), "db_path": str(tmp_path / "test.db")}
    c = Config(raw=raw, path=base.path)
    monkeypatch.delenv("FV_DB_PATH", raising=False)
    init_db(c)
    return c


def _match(cfg, home="Alpha", away="Beta", played=False, score=(2, 1)):
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
        kickoff = datetime.utcnow() + timedelta(days=2)
        m = Match(
            league_code="E0", season="2026-27", match_date=kickoff.date(), kickoff_utc=kickoff,
            home_team_id=ids[home], away_team_id=ids[away],
            status="played" if played else "scheduled",
            fthg=score[0] if played else None, ftag=score[1] if played else None,
            ftr="H" if played and score[0] > score[1] else ("D" if played else None),
        )
        s.add(m)
        s.flush()
        return m.id


def _selection(match_id, selection="H", odds=2.10, stake=10.0):
    return pd.DataFrame([{"match_id": match_id, "selection": selection,
                          "odds": odds, "stake": stake}])


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def test_password_round_trips_and_wrong_one_fails():
    h = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("correct horse battery staple", h)
    assert not auth.verify_password("correct horse battery stapl", h)


def test_long_passphrases_are_not_truncated_to_the_same_password():
    """bcrypt ignores everything past 72 bytes. Pre-hashing is what stops two
    different passphrases with a shared prefix from being interchangeable."""
    prefix = "x" * 72
    h = auth.hash_password(prefix + "AAAA")
    assert not auth.verify_password(prefix + "BBBB", h)


def test_hash_is_salted():
    assert auth.hash_password("same password") != auth.hash_password("same password")


def test_weak_passwords_are_refused(cfg):
    with pytest.raises(auth.AuthError, match="at least"):
        auth.create_user("a@b.com", "short", cfg=cfg)


def test_garbage_hash_never_verifies():
    """The CLI's placeholder account stores an unusable hash so it cannot be signed
    into from the web."""
    assert not auth.verify_password("anything", auth.UNUSABLE_HASH)


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------

def test_create_and_authenticate(cfg):
    created = auth.create_user("ana@example.com", "a-long-enough-password", "Ana", "owner", cfg)
    signed_in = auth.authenticate("  Ana@Example.com ", "a-long-enough-password", cfg)
    assert signed_in.id == created.id
    assert signed_in.is_owner


def test_unknown_account_and_wrong_password_are_indistinguishable(cfg):
    auth.create_user("ana@example.com", "a-long-enough-password", cfg=cfg)
    with pytest.raises(auth.AuthError) as wrong:
        auth.authenticate("ana@example.com", "not-the-password", cfg)
    with pytest.raises(auth.AuthError) as missing:
        auth.authenticate("nobody@example.com", "not-the-password", cfg)
    # Otherwise the error message tells a stranger which emails have accounts.
    assert str(wrong.value) == str(missing.value)


def test_duplicate_email_is_refused(cfg):
    auth.create_user("ana@example.com", "a-long-enough-password", cfg=cfg)
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.create_user("ANA@example.com", "another-long-password", cfg=cfg)


def test_repeated_failures_lock_the_account(cfg):
    auth.create_user("ana@example.com", "a-long-enough-password", cfg=cfg)
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(auth.AuthError):
            auth.authenticate("ana@example.com", "wrong", cfg)
    # Even the correct password is refused while the lockout stands.
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.authenticate("ana@example.com", "a-long-enough-password", cfg)


def test_successful_sign_in_clears_the_failure_count(cfg):
    auth.create_user("ana@example.com", "a-long-enough-password", cfg=cfg)
    for _ in range(auth.MAX_FAILED_LOGINS - 1):
        with pytest.raises(auth.AuthError):
            auth.authenticate("ana@example.com", "wrong", cfg)
    auth.authenticate("ana@example.com", "a-long-enough-password", cfg)
    with pytest.raises(auth.AuthError, match="Wrong email or password"):
        auth.authenticate("ana@example.com", "wrong", cfg)


def test_disabled_account_cannot_sign_in(cfg):
    ana = auth.create_user("ana@example.com", "a-long-enough-password", cfg=cfg)
    auth.create_user("ben@example.com", "a-long-enough-password", role="owner", cfg=cfg)
    auth.set_active(ana.id, False, cfg)
    with pytest.raises(auth.AuthError, match="disabled"):
        auth.authenticate("ana@example.com", "a-long-enough-password", cfg)


def test_the_last_owner_cannot_be_disabled(cfg):
    owner = auth.create_user("ana@example.com", "a-long-enough-password", role="owner", cfg=cfg)
    auth.create_user("ben@example.com", "a-long-enough-password", role="member", cfg=cfg)
    with pytest.raises(auth.AuthError, match="only owner"):
        auth.set_active(owner.id, False, cfg)


def test_changing_your_password_requires_the_old_one(cfg):
    ana = auth.create_user("ana@example.com", "a-long-enough-password", cfg=cfg)
    with pytest.raises(auth.AuthError, match="not correct"):
        auth.change_password(ana.id, "guessing", "a-brand-new-password", cfg)
    auth.change_password(ana.id, "a-long-enough-password", "a-brand-new-password", cfg)
    assert auth.authenticate("ana@example.com", "a-brand-new-password", cfg).id == ana.id


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def two_users(cfg):
    ana = auth.create_user("ana@example.com", "a-long-enough-password", "Ana", "owner", cfg)
    ben = auth.create_user("ben@example.com", "a-long-enough-password", "Ben", "member", cfg)
    return ana, ben


def test_bets_are_not_visible_across_accounts(cfg, two_users):
    ana, ben = two_users
    log_slip(_selection(_match(cfg)), cfg=cfg, user_id=ana.id)
    log_slip(_selection(_match(cfg, "Gamma", "Delta")), cfg=cfg, user_id=ben.id)

    assert len(bet_log(cfg, ana.id)) == 1
    assert len(bet_log(cfg, ben.id)) == 1
    assert bet_log(cfg, ana.id).iloc[0]["home"] == "Alpha"
    assert bet_log(cfg, ben.id).iloc[0]["home"] == "Gamma"
    assert len(bet_log(cfg)) == 2  # the admin view still sees everything


def test_bankrolls_are_separate(cfg, two_users):
    ana, ben = two_users
    record_bankroll_event("deposit", 500.0, cfg=cfg, user_id=ana.id)
    assert current_bankroll(cfg, ana.id) == pytest.approx(1500.0)
    assert current_bankroll(cfg, ben.id) == pytest.approx(1000.0)


def test_settings_are_per_user_but_inherit_the_house_default(cfg, two_users):
    ana, ben = two_users
    set_setting("min_edge", 0.05, cfg)                    # house default
    set_setting("min_edge", 0.09, cfg, user_id=ana.id)    # Ana's own

    assert get_setting("min_edge", cfg=cfg, user_id=ana.id) == pytest.approx(0.09)
    assert get_setting("min_edge", cfg=cfg, user_id=ben.id) == pytest.approx(0.05)
    assert effective_settings(cfg, ben.id)["min_edge"] == pytest.approx(0.05)


def test_settling_credits_each_bet_to_its_own_owner(cfg, two_users):
    ana, ben = two_users
    winner = _match(cfg, "Alpha", "Beta", played=True, score=(2, 1))
    loser = _match(cfg, "Gamma", "Delta", played=True, score=(0, 3))
    log_slip(_selection(winner, "H", 2.10, 10.0), cfg=cfg, user_id=ana.id)
    log_slip(_selection(loser, "H", 2.10, 10.0), cfg=cfg, user_id=ben.id)

    settle_pending(cfg)

    assert current_bankroll(cfg, ana.id) == pytest.approx(1011.0)
    assert current_bankroll(cfg, ben.id) == pytest.approx(990.0)


def test_one_user_cannot_settle_another_users_bet(cfg, two_users):
    from fv.bets import manual_settle

    ana, ben = two_users
    log_slip(_selection(_match(cfg)), cfg=cfg, user_id=ana.id)
    bet_id = int(bet_log(cfg, ana.id).iloc[0]["id"])

    with pytest.raises(KeyError):
        manual_settle(bet_id, "void", cfg=cfg, user_id=ben.id)
    assert bet_log(cfg, ana.id).iloc[0]["status"] == "pending"


def test_the_readiness_gate_does_not_pool_paper_trading(cfg, two_users):
    """Ben must not unlock real money on Ana's four honest weeks."""
    from fv.settings_store import real_money_readiness

    ana, ben = two_users
    log_slip(_selection(_match(cfg, "Alpha", "Beta", played=True), "H", 2.10, 10.0),
             cfg=cfg, user_id=ana.id)
    settle_pending(cfg)

    assert real_money_readiness(cfg, user_id=ana.id).paper_bets == 1
    assert real_money_readiness(cfg, user_id=ben.id).paper_bets == 0


def test_the_assistant_only_ever_sees_its_own_users_bets(cfg, two_users):
    ana, ben = two_users
    log_slip(_selection(_match(cfg)), cfg=cfg, user_id=ana.id)

    assert chat.my_bets(cfg=cfg, user_id=ana.id)["total_bets"] == 1
    assert chat.my_bets(cfg=cfg, user_id=ben.id)["rows"] == []
    assert chat.performance_summary(cfg, ben.id)["bets_logged"] == 0


def test_the_assistants_setting_change_lands_on_its_own_user(cfg, two_users):
    ana, ben = two_users
    chat.update_setting("min_edge", 0.08, cfg, user_id=ana.id)
    assert effective_settings(cfg, ana.id)["min_edge"] == pytest.approx(0.08)
    assert effective_settings(cfg, ben.id)["min_edge"] == pytest.approx(
        load_config().betting.get("min_edge", 0.04)
    )


# ---------------------------------------------------------------------------
# The command line, and migrating a database that predates accounts
# ---------------------------------------------------------------------------

def test_the_cli_account_is_created_once_and_reused(cfg):
    first = auth.local_user_id(cfg)
    assert auth.local_user_id(cfg) == first
    assert auth.user_count(cfg) == 1


def test_the_cli_account_cannot_be_signed_into(cfg):
    auth.local_user_id(cfg)
    with pytest.raises(auth.AuthError):
        auth.authenticate(auth.LOCAL_EMAIL, "", cfg)


def test_the_cli_adopts_a_real_owner_rather_than_making_its_own(cfg):
    ana = auth.create_user("ana@example.com", "a-long-enough-password", role="owner", cfg=cfg)
    assert auth.local_user_id(cfg) == ana.id
    assert auth.user_count(cfg) == 1


def test_migration_gives_pre_accounts_rows_an_owner(cfg):
    """A database from before accounts existed must keep its bets, not lose them."""
    from fv.db.migrate import ensure_schema

    # Reproduce the old shape: bets and ledger rows with no user_id column at all.
    engine = get_engine(cfg)
    match_id = _match(cfg)
    with engine.begin() as conn:
        for table in ("bets", "bankroll_events"):
            conn.execute(text(f"DROP TABLE {table}"))
        conn.execute(text(
            "CREATE TABLE bets (id INTEGER PRIMARY KEY, slip_id TEXT, match_id INTEGER, "
            "selection TEXT, odds_taken REAL, stake REAL, mode TEXT, placed_at TEXT, "
            "status TEXT, pnl REAL, closing_odds REAL, clv REAL, settled_at TEXT, "
            "settled_by TEXT, notes TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE bankroll_events (id INTEGER PRIMARY KEY, ts TEXT, type TEXT, "
            "amount REAL, balance_after REAL, bet_id INTEGER, note TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO bets (match_id, selection, odds_taken, stake, mode, status) "
            "VALUES (:m, 'H', 2.1, 10.0, 'paper', 'pending')"
        ), {"m": match_id})
        conn.execute(text(
            "INSERT INTO bankroll_events (ts, type, amount, balance_after) "
            "VALUES ('2026-01-01 00:00:00', 'deposit', 250.0, 1250.0)"
        ))

    notes = ensure_schema(cfg)

    assert any("bets" in n for n in notes)
    owner = auth.local_user_id(cfg)
    log = bet_log(cfg, owner)
    assert len(log) == 1
    assert current_bankroll(cfg, owner) == pytest.approx(1250.0)


def test_migration_is_idempotent(cfg):
    from fv.db.migrate import ensure_schema

    ensure_schema(cfg)
    assert ensure_schema(cfg) == []
