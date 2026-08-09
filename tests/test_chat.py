"""The in-app assistant's guard rails.

The assistant is the one part of the app that takes instructions from natural
language, so its limits have to hold against a determined prompt rather than
against a cooperative one. These tests exercise the limits directly — the tool
functions, not the model — because that is where the enforcement lives.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from fv import chat
from fv.config import Config, load_config
from fv.db.models import LeagueRow, Match, Team
from fv.db.session import init_db, session_scope
from fv.settings_store import effective_settings, get_setting


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Config:
    base = load_config()
    raw = dict(base.raw)
    raw["data"] = {**raw.get("data", {}), "db_path": str(tmp_path / "test.db")}
    c = Config(raw=raw, path=base.path)
    monkeypatch.delenv("FV_DB_PATH", raising=False)
    init_db(c)
    with session_scope(c) as s:
        if s.get(LeagueRow, "E0") is None:
            s.add(LeagueRow(code="E0", name="PL", country="England", tier=1))
        home = Team(canonical_name="Alpha", country="England")
        away = Team(canonical_name="Beta", country="England")
        s.add_all([home, away])
        s.flush()
        kickoff = datetime.utcnow() + timedelta(days=2)
        s.add(Match(league_code="E0", season="2026-27", match_date=kickoff.date(),
                    kickoff_utc=kickoff, home_team_id=home.id, away_team_id=away.id,
                    status="scheduled"))
    return c


# ---------------------------------------------------------------------------
# Read-only SQL
# ---------------------------------------------------------------------------

def test_select_returns_rows(cfg):
    out = chat.run_sql("SELECT canonical_name FROM teams ORDER BY canonical_name", cfg=cfg)
    assert out["columns"] == ["canonical_name"]
    assert [r[0] for r in out["rows"]] == ["Alpha", "Beta"]
    assert out["truncated"] is False


def test_with_clause_is_allowed(cfg):
    out = chat.run_sql("WITH t AS (SELECT 1 AS n) SELECT n FROM t", cfg=cfg)
    assert out["rows"] == [[1]]


@pytest.mark.parametrize("sql", [
    "INSERT INTO teams (canonical_name, country) VALUES ('X', 'Y')",
    "UPDATE bets SET stake = 0",
    "DELETE FROM matches",
    "DROP TABLE bets",
    "ALTER TABLE bets ADD COLUMN x INTEGER",
    "PRAGMA query_only = OFF",
    "ATTACH DATABASE '/tmp/other.db' AS other",
    "VACUUM",
])
def test_writes_are_refused(cfg, sql):
    with pytest.raises(chat.QueryError):
        chat.run_sql(sql, cfg=cfg)


def test_stacked_statement_is_refused(cfg):
    with pytest.raises(chat.QueryError, match="One statement"):
        chat.run_sql("SELECT 1; DELETE FROM bets", cfg=cfg)


def test_comment_cannot_hide_a_write(cfg):
    """Stripping comments before the check, not after, is what makes this fail."""
    with pytest.raises(chat.QueryError):
        chat.run_sql("SELECT 1 /* harmless */ ; DROP TABLE bets", cfg=cfg)
    with pytest.raises(chat.QueryError):
        chat.run_sql("-- please\nDELETE FROM bets", cfg=cfg)


def test_connection_itself_rejects_writes(cfg):
    """The regex is a nicer error message; this is the actual boundary.

    If the statement guard were bypassed entirely, SQLite must still refuse.
    """
    conn = chat._readonly_connection(cfg)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO teams (canonical_name, country) VALUES ('X', 'Y')")
            conn.commit()
    finally:
        conn.close()


def test_rows_are_capped_and_truncation_is_reported(cfg):
    with session_scope(cfg) as s:
        for i in range(20):
            s.add(Team(canonical_name=f"Club {i}", country="Nowhere"))
    out = chat.run_sql("SELECT canonical_name FROM teams", max_rows=5, cfg=cfg)
    assert out["row_count"] == 5
    assert out["truncated"] is True


def test_bad_sql_becomes_a_query_error(cfg):
    with pytest.raises(chat.QueryError, match="SQLite rejected"):
        chat.run_sql("SELECT no_such_column FROM teams", cfg=cfg)


def test_describe_schema_lists_the_real_tables(cfg):
    tables = chat.describe_schema(cfg)["tables"]
    assert {"matches", "odds", "bets", "settings"} <= set(tables)
    assert "CREATE TABLE" in tables["matches"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_paper_mode_cannot_be_changed(cfg):
    with pytest.raises(chat.QueryError, match="Settings page"):
        chat.update_setting("paper_mode", False, cfg)
    assert effective_settings(cfg)["paper_mode"] is True


def test_unknown_setting_is_refused(cfg):
    with pytest.raises(chat.QueryError, match="not a setting I can change"):
        chat.update_setting("db_path", "/etc/passwd", cfg)


def test_valid_change_persists(cfg):
    out = chat.update_setting("min_edge", 0.06, cfg)
    assert out["new_value"] == pytest.approx(0.06)
    assert get_setting("min_edge", None, cfg) == pytest.approx(0.06)


@pytest.mark.parametrize("key,value", [
    ("min_edge", 0.9),            # above the Settings page's own ceiling
    ("kelly_fraction", 2.0),      # over-staking compounds badly
    ("max_stake_pct", 0.5),
    ("max_bets_per_week", 0),
    ("starting_bankroll", -100),
    ("max_drawdown_pct", 0.99),
])
def test_out_of_range_values_are_refused(cfg, key, value):
    with pytest.raises(chat.QueryError, match="must be between"):
        chat.update_setting(key, value, cfg)


def test_odds_window_must_stay_a_window(cfg):
    with pytest.raises(chat.QueryError, match="must be below"):
        chat.update_setting("min_odds", 9.0, cfg)  # current max_odds is 4.00


def test_non_numeric_value_is_refused(cfg):
    with pytest.raises(chat.QueryError, match="must be a number"):
        chat.update_setting("min_edge", "lots", cfg)
    with pytest.raises(chat.QueryError, match="must be a number"):
        chat.update_setting("min_edge", True, cfg)


def test_enabled_leagues_validated_against_config(cfg):
    with pytest.raises(chat.QueryError, match="Unknown league codes"):
        chat.update_setting("enabled_leagues", ["E0", "XX9"], cfg)
    out = chat.update_setting("enabled_leagues", ["E0", "E0", "SP1"], cfg)
    assert out["new_value"] == ["E0", "SP1"]


# ---------------------------------------------------------------------------
# Tools and the loop
# ---------------------------------------------------------------------------

def test_no_tool_can_move_money(cfg):
    """The assistant may read and may tune thresholds. It may not bet."""
    names = set(chat.build_tools(cfg))
    assert names == {
        "run_sql", "describe_schema", "get_settings", "my_bets",
        "update_setting", "performance_summary", "backtest_summary",
    }


def test_every_tool_schema_is_well_formed(cfg):
    for tool in chat.build_tools(cfg).values():
        schema = tool.schema()
        assert schema["name"] and schema["description"]
        assert schema["input_schema"]["type"] == "object"


def test_performance_summary_reports_the_verdict(cfg):
    out = chat.performance_summary(cfg)
    assert out["bets_logged"] == 0
    assert out["paper_mode"] is True
    # An empty app has not earned real money, and the summary must say so.
    assert out["readiness"]["ready_for_real_money"] is False
    assert out["readiness"]["reasons_not_ready"]


class _FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResponse:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason


class _FakeClient:
    """Replays a scripted sequence of API responses and records what it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []
        self.messages = self

    def create(self, **kwargs):
        # respond() mutates the caller's list in place, so snapshot it: otherwise
        # every recorded request appears to contain the whole conversation.
        self.sent.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def test_respond_executes_a_tool_and_returns_the_final_text(cfg):
    client = _FakeClient([
        _FakeResponse(
            [_FakeBlock(type="tool_use", id="t1", name="run_sql",
                        input={"sql": "SELECT count(*) FROM teams"})],
            "tool_use",
        ),
        _FakeResponse([_FakeBlock(type="text", text="There are 2 teams.")], "end_turn"),
    ])
    messages = [{"role": "user", "content": "how many teams?"}]
    turn = chat.respond(messages, cfg=cfg, client=client)

    assert turn.text == "There are 2 teams."
    assert [c.name for c in turn.tool_calls] == ["run_sql"]
    assert turn.tool_calls[0].result["rows"] == [[2]]
    # The tool result was fed back as a user message so the model could use it.
    assert messages[-2]["content"][0]["type"] == "tool_result"


def test_respond_returns_tool_errors_to_the_model_rather_than_crashing(cfg):
    client = _FakeClient([
        _FakeResponse(
            [_FakeBlock(type="tool_use", id="t1", name="update_setting",
                        input={"key": "paper_mode", "value": False})],
            "tool_use",
        ),
        _FakeResponse([_FakeBlock(type="text", text="I can't change that.")], "end_turn"),
    ])
    turn = chat.respond([{"role": "user", "content": "go real money"}], cfg=cfg, client=client)

    assert turn.tool_calls[0].error is True
    assert effective_settings(cfg)["paper_mode"] is True
    sent_back = client.sent[1]["messages"][-1]["content"][0]
    assert sent_back["is_error"] is True
    assert "Settings page" in sent_back["content"]


def test_respond_stops_looping(cfg):
    """A model that only ever asks for tools must not spin forever."""
    forever = [
        _FakeResponse(
            [_FakeBlock(type="tool_use", id=f"t{i}", name="get_settings", input={})],
            "tool_use",
        )
        for i in range(chat.MAX_TOOL_ROUNDS + 5)
    ]
    turn = chat.respond([{"role": "user", "content": "loop"}], cfg=cfg,
                        client=_FakeClient(forever))
    assert "narrower" in turn.text
    assert len(turn.tool_calls) == chat.MAX_TOOL_ROUNDS


def test_request_uses_the_configured_model_and_no_sampling_params(cfg):
    client = _FakeClient([_FakeResponse([_FakeBlock(type="text", text="hi")], "end_turn")])
    chat.respond([{"role": "user", "content": "hi"}], cfg=cfg, client=client)
    sent = client.sent[0]
    assert sent["model"] == chat.MODEL
    assert sent["thinking"] == {"type": "adaptive"}
    # Opus 5 rejects these outright; a stray default would 400 every request.
    assert not {"temperature", "top_p", "top_k", "budget_tokens"} & set(sent)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM bets",
    "SELECT stake FROM Bets WHERE user_id = 2",
    "SELECT balance_after FROM bankroll_events ORDER BY id DESC LIMIT 1",
    "SELECT email, password_hash FROM users",
    "SELECT value_json FROM user_settings",
    "SELECT m.id FROM matches m JOIN bets b ON b.match_id = m.id",
])
def test_private_tables_are_unreachable_from_sql(cfg, sql):
    """Read-only is not enough once the app has more than one account.

    These tables hold one person's money and password hash. SQL reaches shared
    football data only; the per-user tools are the sole route to the rest.
    """
    with pytest.raises(chat.QueryError, match="private per-account data"):
        chat.run_sql(sql, cfg=cfg)
