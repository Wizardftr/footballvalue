"""The in-app assistant: ask questions of the database, adjust settings by hand.

This is a thin, deliberately boring tool-use loop over the Claude API. The
interesting decisions are all about what the assistant is *not* allowed to do:

* **It cannot write to the database.** The SQL tool opens its own connection in
  read-only mode, so a write is refused by SQLite itself, not merely by a regular
  expression over the query text. The regex is there to give a clear error, not to
  be the security boundary.
* **It cannot place, log, or settle bets.** Those are money-moving actions and they
  stay on the pages where a human is looking at them.
* **It cannot turn off paper trading.** That toggle exists precisely because the
  model has not earned real money yet, and flipping it should require a person
  reading the readiness evidence on the Settings page. An assistant that can be
  talked into it is not a gate.

Everything else — thresholds, staking, league selection — it can change, within the
ranges the Settings page enforces, because those are reversible and visible.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from fv.config import Config, load_config
from fv.settings_store import effective_settings, get_setting, real_money_readiness, set_setting

MODEL = "claude-opus-5"
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 8
DEFAULT_MAX_ROWS = 200


# ---------------------------------------------------------------------------
# Read-only SQL
# ---------------------------------------------------------------------------

class QueryError(Exception):
    """A query the assistant is not allowed to run, phrased for the assistant."""


_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|begin|commit|rollback|savepoint)\b",
    re.I,
)
# Tables holding one account's private data. SQL is shared reference data only —
# matches, odds, xG, predictions, backtests — so that no query, however it is
# phrased, can read another user's bets, balance, password hash or settings.
PRIVATE_TABLES = ("bets", "bankroll_events", "users", "user_settings")
_PRIVATE = re.compile(r"\b(" + "|".join(PRIVATE_TABLES) + r")\b", re.I)


def _readonly_connection(cfg: Config) -> sqlite3.Connection:
    """Open the app database so that writes fail at the SQLite level.

    ``mode=ro`` is the stronger guarantee, but it cannot recover a WAL journal, so
    a database left with an unclean -wal file refuses to open that way. Falling back
    to ``query_only`` keeps the same promise — SQLite rejects every write on the
    connection — without needing write access to the journal.
    """
    path = cfg.db_path
    if not path.exists():
        raise QueryError(f"No database at {path}. Run `fv init-db` and `fv download` first.")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        return conn
    except sqlite3.Error:
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA query_only = ON")
        return conn


def check_query(sql: str) -> str:
    """Reject anything that is not a single read-only statement. Returns the query."""
    stripped = _COMMENTS.sub(" ", sql).strip().rstrip(";").strip()
    if not stripped:
        raise QueryError("Empty query.")
    if ";" in stripped:
        raise QueryError("One statement per call, please — no semicolons.")
    if not re.match(r"^(select|with)\b", stripped, re.I):
        raise QueryError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    hit = _WRITE_WORDS.search(stripped)
    if hit:
        raise QueryError(
            f"'{hit.group(0)}' is not allowed. This tool is read-only: the database "
            "is changed through the app's own pages, never through SQL."
        )
    private = _PRIVATE.search(stripped)
    if private:
        raise QueryError(
            f"'{private.group(0)}' holds private per-account data and is not readable "
            "through SQL. Use my_bets or performance_summary instead — those are scoped "
            "to the signed-in user."
        )
    return stripped


def run_sql(sql: str, max_rows: int = DEFAULT_MAX_ROWS, cfg: Config | None = None) -> dict:
    """Run a read-only query and return columns, rows, and whether it was truncated."""
    cfg = cfg or load_config()
    query = check_query(sql)
    max_rows = max(1, min(int(max_rows), 1000))
    conn = _readonly_connection(cfg)
    try:
        cur = conn.execute(query)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        raise QueryError(f"SQLite rejected the query: {exc}") from exc
    finally:
        conn.close()
    truncated = len(rows) > max_rows
    return {
        "columns": columns,
        "rows": [list(r) for r in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": truncated,
    }


def describe_schema(cfg: Config | None = None) -> dict:
    """The exact CREATE statements, so the assistant never guesses a column name."""
    cfg = cfg or load_config()
    conn = _readonly_connection(cfg)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return {"tables": {name: sql for name, sql in rows}}


# ---------------------------------------------------------------------------
# Settings the assistant may change
# ---------------------------------------------------------------------------

# key -> (kind, low, high). These mirror the Settings page's own widget ranges, so
# the assistant cannot reach a value a human could not have set by hand.
NUMERIC_SETTINGS: dict[str, tuple[type, float, float]] = {
    "starting_bankroll": (float, 1.0, 10_000_000.0),
    "min_edge": (float, 0.0, 0.20),
    "min_odds": (float, 1.0, 10.0),
    "max_odds": (float, 1.0, 20.0),
    "kelly_fraction": (float, 0.05, 1.0),
    "max_stake_pct": (float, 0.005, 0.10),
    "max_bets_per_week": (int, 1, 50),
    "weekly_stop_loss_pct": (float, 0.0, 0.50),
    "max_drawdown_pct": (float, 0.05, 0.75),
    "slip_fill_n": (int, 1, 20),
}

# Settings that are never the assistant's to change, and why.
REFUSED_SETTINGS = {
    "paper_mode": (
        "Paper trading mode can only be changed on the Settings page. The project's "
        "rule is that real money is earned, not assumed: the toggle sits next to the "
        "readiness evidence so that turning it off is a decision a person makes while "
        "looking at that evidence. I will not make it for you."
    ),
}


def _validate_setting(key: str, value: Any, cfg: Config, user_id: int | None) -> Any:
    if key in REFUSED_SETTINGS:
        raise QueryError(REFUSED_SETTINGS[key])

    if key == "enabled_leagues":
        if not isinstance(value, list) or not value:
            raise QueryError("enabled_leagues must be a non-empty list of league codes.")
        known = {lg.code for lg in cfg.leagues}
        unknown = [c for c in value if c not in known]
        if unknown:
            raise QueryError(
                f"Unknown league codes: {', '.join(map(str, unknown))}. "
                f"Valid codes: {', '.join(sorted(known))}."
            )
        return list(dict.fromkeys(value))

    if key == "slip_fill_mode":
        if not isinstance(value, bool):
            raise QueryError("slip_fill_mode must be true or false.")
        return value

    if key not in NUMERIC_SETTINGS:
        raise QueryError(
            f"'{key}' is not a setting I can change. Changeable settings: "
            f"{', '.join(sorted(list(NUMERIC_SETTINGS) + ['enabled_leagues', 'slip_fill_mode']))}."
        )

    kind, low, high = NUMERIC_SETTINGS[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryError(f"{key} must be a number.")
    value = kind(value)
    if not low <= value <= high:
        raise QueryError(f"{key} must be between {low} and {high}; got {value}.")

    # The odds window has to stay a window. Checked against the value that will be in
    # force after this change, not against config.yaml's default.
    current = effective_settings(cfg, user_id)
    lo = value if key == "min_odds" else float(current["min_odds"])
    hi = value if key == "max_odds" else float(current["max_odds"])
    if key in ("min_odds", "max_odds") and lo >= hi:
        raise QueryError(f"min_odds ({lo}) must be below max_odds ({hi}).")
    return value


def update_setting(
    key: str, value: Any, cfg: Config | None = None, user_id: int | None = None
) -> dict:
    """Change one setting for one user, validated the way the Settings page would."""
    cfg = cfg or load_config()
    old = get_setting(key, effective_settings(cfg, user_id).get(key), cfg, user_id)
    new = _validate_setting(key, value, cfg, user_id)
    set_setting(key, new, cfg, user_id)
    return {"key": key, "old_value": old, "new_value": new, "saved": True}


# ---------------------------------------------------------------------------
# Summaries the database alone cannot answer
# ---------------------------------------------------------------------------

def settings_summary(cfg: Config | None = None, user_id: int | None = None) -> dict:
    cfg = cfg or load_config()
    s = dict(effective_settings(cfg, user_id))
    s["slip_fill_mode"] = bool(get_setting("slip_fill_mode", False, cfg, user_id))
    s["slip_fill_n"] = int(get_setting("slip_fill_n", 10, cfg, user_id))
    return s


def my_bets(limit: int = 50, cfg: Config | None = None, user_id: int | None = None) -> dict:
    """The signed-in user's own bets. The only route to the bets table."""
    from fv.bets import bet_log

    log = bet_log(cfg or load_config(), user_id)
    if log.empty:
        return {"rows": [], "note": "No bets logged yet."}
    keep = ["id", "kickoff_utc", "league_code", "home", "away", "selection", "odds_taken",
            "stake", "mode", "status", "pnl", "closing_odds", "clv"]
    view = log[keep].head(max(1, min(int(limit), 500)))
    return {
        "rows": json.loads(view.to_json(orient="records", date_format="iso")),
        "total_bets": int(len(log)),
        "truncated": len(log) > len(view),
    }


def performance_summary(cfg: Config | None = None, user_id: int | None = None) -> dict:
    """Bankroll, the bet log's headline numbers, and the real-money verdict."""
    cfg = cfg or load_config()
    from fv.bets import bet_log, current_bankroll, rolling_roi

    log = bet_log(cfg, user_id)
    out: dict[str, Any] = {
        "bankroll": current_bankroll(cfg, user_id),
        "bets_logged": int(len(log)),
        "paper_mode": bool(effective_settings(cfg, user_id)["paper_mode"]),
    }
    if not log.empty:
        settled = log[log["status"].isin(("won", "lost"))]
        out["bets_settled"] = int(len(settled))
        out["bets_pending"] = int((log["status"] == "pending").sum())
        if not settled.empty and settled["stake"].sum() > 0:
            out["all_time_roi"] = float(settled["pnl"].sum() / settled["stake"].sum())
            out["total_staked"] = float(settled["stake"].sum())
            out["total_pnl"] = float(settled["pnl"].sum())
        roll = rolling_roi(log, window=300)
        if not roll.empty:
            last = roll.iloc[-1]
            out["rolling_300_roi"] = {
                "roi": float(last["roi"]),
                "n": int(last["n"]),
                "ci_low": float(last["ci_low"]),
                "ci_high": float(last["ci_high"]),
            }
        else:
            out["rolling_300_roi"] = None
            out["rolling_roi_note"] = (
                "Fewer than 30 settled bets, so no rolling window is shown. Any ROI "
                "quoted from this few bets is noise."
            )

    r = real_money_readiness(cfg, user_id=user_id)
    out["readiness"] = {
        "ready_for_real_money": r.ready,
        "backtest_beats_bet365_closing": r.backtest_beats_closing,
        "backtest_log_loss_gap": r.backtest_gap,
        "best_stage": r.best_stage,
        "paper_trading_weeks": r.paper_weeks,
        "paper_bets": r.paper_bets,
        "paper_clv_mean": r.paper_clv_mean,
        "paper_clv_significant": r.paper_clv_significant,
        "reasons_not_ready": r.reasons,
    }
    return out


def backtest_summary(cfg: Config | None = None) -> dict:
    """The stage comparison report, which lives in reports/ rather than the database."""
    from fv.config import PROJECT_ROOT

    csv = PROJECT_ROOT / "reports" / "stages" / "stage_comparison.csv"
    if not csv.exists():
        return {"available": False, "note": "No stage backtest has been run yet (`fv stages`)."}
    import pandas as pd

    table = pd.read_csv(csv)
    md = csv.with_suffix(".md")
    return {
        "available": True,
        "stages": table.to_dict(orient="records"),
        "report_markdown": md.read_text()[:8000] if md.exists() else None,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def build_tools(cfg: Config, user_id: int | None = None) -> dict[str, Tool]:
    return {
        t.name: t
        for t in [
            Tool(
                name="run_sql",
                description=(
                    "Run a read-only SQL query against the app's shared football data "
                    "and get the rows back. SELECT and WITH only; writes are refused by "
                    "the database itself. Covers matches, teams, odds, xG, predictions "
                    "and backtests. Per-account tables (bets, bankroll_events, users, "
                    "user_settings) are not readable here — use my_bets or "
                    "performance_summary for those."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "A single SELECT statement."},
                        "max_rows": {
                            "type": "integer",
                            "description": f"Row cap, default {DEFAULT_MAX_ROWS}, max 1000.",
                        },
                    },
                    "required": ["sql"],
                },
                fn=lambda sql, max_rows=DEFAULT_MAX_ROWS: run_sql(sql, max_rows, cfg),
            ),
            Tool(
                name="describe_schema",
                description=(
                    "Return the CREATE TABLE statement for every table. Call this before "
                    "writing a query if you are unsure of a column name."
                ),
                input_schema={"type": "object", "properties": {}},
                fn=lambda: describe_schema(cfg),
            ),
            Tool(
                name="get_settings",
                description=(
                    "The settings currently in force: staking, thresholds, risk controls, "
                    "enabled leagues, and whether paper mode is on."
                ),
                input_schema={"type": "object", "properties": {}},
                fn=lambda: settings_summary(cfg, user_id),
            ),
            Tool(
                name="update_setting",
                description=(
                    "Change one setting. Allowed keys: starting_bankroll, min_edge, "
                    "min_odds, max_odds, kelly_fraction, max_stake_pct, max_bets_per_week, "
                    "weekly_stop_loss_pct, max_drawdown_pct, slip_fill_mode, slip_fill_n, "
                    "enabled_leagues. Values are range-checked. paper_mode cannot be "
                    "changed here. Confirm the change with the user before calling this."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {
                            "description": "Number, boolean, or list of league codes.",
                        },
                    },
                    "required": ["key", "value"],
                },
                fn=lambda key, value: update_setting(key, value, cfg, user_id),
            ),
            Tool(
                name="my_bets",
                description=(
                    "The signed-in user's own bets with status, stake, P&L and CLV. "
                    "The bets table is not reachable from run_sql, so this is the only "
                    "way to see them — and it can only ever show this user's."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Default 50, max 500."}
                    },
                },
                fn=lambda limit=50: my_bets(limit, cfg, user_id),
            ),
            Tool(
                name="performance_summary",
                description=(
                    "Bankroll, bet counts, all-time and rolling-300 ROI with its confidence "
                    "interval, and the real-money readiness verdict with its reasons."
                ),
                input_schema={"type": "object", "properties": {}},
                fn=lambda: performance_summary(cfg, user_id),
            ),
            Tool(
                name="backtest_summary",
                description=(
                    "The walk-forward stage comparison: log loss, Brier, ROI and CLV for "
                    "each model stage against bet365. This lives in reports/, not the "
                    "database, so SQL cannot reach it."
                ),
                input_schema={"type": "object", "properties": {}},
                fn=lambda: backtest_summary(cfg),
            ),
        ]
    }


SYSTEM_PROMPT = """\
You are the assistant built into footballvalue, a single-user football betting
analysis tool. The user is the tool's owner. You help them query their own data,
understand the model's results, and adjust their settings.

What the app does: it models 1X2 outcomes in eleven European leagues with a
time-decayed Dixon-Coles model, blends in xG and a LightGBM stage, anchors the
result to the market, strips bet365's margin, and reports edge = probability *
decimal odds - 1. It suggests a weekly slip. It never places bets; the user places
them by hand on bet365.

Shared tables you can query with run_sql (use describe_schema for exact columns):
  leagues, teams, team_aliases, matches, odds, match_xg, model_runs, predictions,
  backtest_runs, backtest_bets, ingest_log
The user's own bets, balance and settings are private and are NOT in run_sql's
reach. Use my_bets, performance_summary and get_settings for those. This app has
several accounts and each one sees only its own.
Notes: `odds` is long-format, one row per (match, bookmaker, market, selection,
odds_type); odds_type is 'pre', 'closing' or 'snapshot' and bookmaker 'B365' is
bet365. `matches` holds both played and scheduled fixtures — status tells you
which, and fthg/ftag are NULL until a match is played. Team names are in `teams`,
joined through matches.home_team_id / away_team_id. `bets` are the user's own
paper and real bets.

How to behave:

- Be direct and brief. This is a tool, not a chat product.
- Query before answering. Never estimate a number you could look up.
- Quote uncertainty with every performance figure. An ROI without its interval is
  misleading, and the user built this app specifically to avoid that.
- Never present projected profit as a promise, and never describe a bet as safe,
  guaranteed, or a lock.
- Singles only. Never suggest combining selections into an accumulator or parlay:
  it multiplies the bookmaker's margin. If asked for one, say why not.
- Be honest about the model. As things stand the backtest does not beat bet365's
  closing prices, and CLV is not significantly positive. If the user asks whether
  this makes money, say so plainly and show the numbers rather than softening them.
- Monthly results are variance. Judge on the rolling 300-bet window and CLV.
- Confirm before changing a setting, then call update_setting and report the old
  and new values.
- You cannot place, log, or settle bets, and you cannot turn off paper trading
  mode. If asked, say so and point at the page that can.
"""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict
    result: Any
    error: bool = False


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def api_key() -> str | None:
    load_config()  # loads .env
    return os.getenv("ANTHROPIC_API_KEY") or None


def is_available() -> tuple[bool, str]:
    """(usable, reason). Mirrors how the Odds API degrades when unconfigured."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, (
            "The `anthropic` package is not installed. Run "
            "`uv sync --extra ui` to install it."
        )
    if not api_key():
        return False, (
            "No ANTHROPIC_API_KEY found. Add a line `ANTHROPIC_API_KEY=sk-ant-...` to "
            "the `.env` file in the project root, then restart the dashboard. The key "
            "lives in .env and is never committed."
        )
    return True, ""


def _json_default(o):
    return str(o)


def respond(
    messages: list[dict],
    cfg: Config | None = None,
    client=None,
    on_tool: Callable[[str, dict], None] | None = None,
    user_id: int | None = None,
) -> Turn:
    """Run one assistant turn to completion, executing tools as they are requested.

    ``messages`` is the raw API conversation and is appended to in place, so the
    caller can keep it in session state and pass it straight back next turn.
    """
    cfg = cfg or load_config()
    tools = build_tools(cfg, user_id)
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key())

    schemas = [t.schema() for t in tools.values()]
    calls: list[ToolCall] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=schemas,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            return Turn(text=text.strip(), tool_calls=calls)

        results = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            if on_tool:
                on_tool(block.name, dict(block.input))
            tool = tools.get(block.name)
            try:
                if tool is None:
                    raise QueryError(f"No such tool: {block.name}")
                payload = tool.fn(**dict(block.input))
                body = json.dumps(payload, default=_json_default)
                calls.append(ToolCall(block.name, dict(block.input), payload))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": body})
            except Exception as exc:  # surfaced to the model so it can correct itself
                calls.append(ToolCall(block.name, dict(block.input), str(exc), error=True))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(exc),
                    "is_error": True,
                })
        messages.append({"role": "user", "content": results})

    return Turn(
        text="I kept needing more tool calls than this conversation allows. "
             "Try asking something narrower.",
        tool_calls=calls,
    )
