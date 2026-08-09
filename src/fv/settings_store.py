"""Runtime settings and the real-money gate.

Settings live in the ``settings`` table and override config.yaml, so the dashboard
can change thresholds without editing files.

The real-money gate is the important part of this module. The project's own rule is
that real money is *earned*: the model must beat bet365's closing odds out-of-sample
in the backtest and hold positive CLV through at least four weeks of paper trading.
:func:`real_money_readiness` computes that verdict from the stored results rather
than from anybody's recollection, and the UI shows it next to the toggle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fv.config import PROJECT_ROOT, Config, load_config
from fv.db.models import Setting, UserSetting
from fv.db.session import session_scope

STAGE_REPORT = PROJECT_ROOT / "reports" / "stages" / "stage_comparison.csv"
PAPER_TRADING_WEEKS_REQUIRED = 4


def _decode(row, default):
    if row is None:
        return default
    try:
        return json.loads(row.value_json)
    except ValueError:
        return default


def get_setting(key: str, default=None, cfg: Config | None = None, user_id: int | None = None):
    """Resolve a setting: the user's own value, else the house default, else ``default``.

    ``user_id=None`` reads the house default directly, which is what the CLI and the
    backtest want — they are not acting on anybody's behalf.
    """
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        if user_id is not None:
            mine = s.get(UserSetting, (user_id, key))
            if mine is not None:
                return _decode(mine, default)
        return _decode(s.get(Setting, key), default)


def set_setting(key: str, value, cfg: Config | None = None, user_id: int | None = None) -> None:
    """Store a value for one user, or as the house default when ``user_id`` is None."""
    cfg = cfg or load_config()
    payload = json.dumps(value)
    with session_scope(cfg) as s:
        if user_id is None:
            row = s.get(Setting, key)
            if row is None:
                s.add(Setting(key=key, value_json=payload, updated_at=datetime.utcnow()))
            else:
                row.value_json, row.updated_at = payload, datetime.utcnow()
            return
        row = s.get(UserSetting, (user_id, key))
        if row is None:
            s.add(UserSetting(user_id=user_id, key=key, value_json=payload,
                              updated_at=datetime.utcnow()))
        else:
            row.value_json, row.updated_at = payload, datetime.utcnow()


def effective_settings(cfg: Config | None = None, user_id: int | None = None) -> dict:
    """config.yaml defaults, then the house overrides, then this user's own."""
    cfg = cfg or load_config()
    b, r = cfg.betting, cfg.risk
    defaults = {
        "starting_bankroll": b.get("starting_bankroll", 1000.0),
        "min_edge": b.get("min_edge", 0.04),
        "min_odds": b.get("min_odds", 1.50),
        "max_odds": b.get("max_odds", 4.00),
        "kelly_fraction": b.get("kelly_fraction", 0.25),
        "max_stake_pct": b.get("max_stake_pct", 0.02),
        "max_bets_per_week": b.get("max_bets_per_week", 8),
        "weekly_stop_loss_pct": r.get("weekly_stop_loss_pct", 0.10),
        "max_drawdown_pct": r.get("max_drawdown_pct", 0.25),
        "paper_mode": r.get("paper_mode", True),
        "enabled_leagues": [lg.code for lg in cfg.enabled_leagues],
    }
    for key in list(defaults):
        stored = get_setting(key, None, cfg, user_id)
        if stored is not None:
            defaults[key] = stored
    return defaults


@dataclass
class Readiness:
    """Whether real-money mode is justified, and why."""

    backtest_beats_closing: bool
    backtest_gap: float | None  # model log loss minus market log loss; negative is good
    best_stage: str | None
    paper_weeks: float
    paper_bets: int
    paper_clv_mean: float | None
    paper_clv_significant: bool
    ready: bool
    reasons: list[str]

    @property
    def headline(self) -> str:
        return (
            "Real-money mode is justified by the evidence."
            if self.ready
            else "Real-money mode is NOT justified by the evidence."
        )


def real_money_readiness(
    cfg: Config | None = None,
    stage_report: Path | None = None,
    user_id: int | None = None,
) -> Readiness:
    """Evaluate the two gates the project set for itself.

    Deliberately strict: both conditions must hold, and an absent backtest counts as
    a failure rather than as an unknown. A gate you can pass by not running the test
    is not a gate.
    """
    cfg = cfg or load_config()
    from fv.bets import bet_log

    reasons: list[str] = []

    # Gate 1: does the model beat bet365's closing prices out-of-sample?
    path = stage_report or STAGE_REPORT
    beats = False
    gap = None
    best_stage = None
    if path.exists():
        try:
            table = pd.read_csv(path)
            # The market baseline is recomputed by the report; here we compare the
            # best stage's log loss against the market column stored alongside it.
            market_row = table.attrs.get("market_log_loss")
            best = table.loc[table["log_loss"].idxmin()]
            best_stage = str(best["stage"])
            market_ll = market_row if market_row else _market_log_loss_from_report(path)
            if market_ll is not None:
                gap = float(best["log_loss"]) - float(market_ll)
                beats = gap < 0
        except Exception:
            reasons.append("Backtest results could not be read.")
    if not path.exists():
        reasons.append("No stage backtest has been run (`fv stages`).")
    elif not beats:
        reasons.append(
            f"The model does not beat bet365's prices out-of-sample"
            + (f" (behind by {gap * 1000:.2f} millinats)." if gap is not None else ".")
        )

    # Gate 2: four weeks of paper trading with positive, significant CLV. Paper
    # trading is per-account: another user's four honest weeks are not evidence
    # about this user's discipline, and pooling them would let one account unlock
    # real money on somebody else's record.
    log = bet_log(cfg, user_id)
    paper = log[(log["mode"] == "paper") & (log["status"].isin(("won", "lost")))]
    weeks = 0.0
    clv_mean = None
    clv_sig = False
    if not paper.empty:
        span = paper["kickoff_utc"].max() - paper["kickoff_utc"].min()
        weeks = span / timedelta(weeks=1)
        clv = paper["clv"].dropna()
        if len(clv) > 1:
            clv_mean = float(clv.mean())
            se = float(clv.std(ddof=1) / np.sqrt(len(clv)))
            clv_sig = (clv_mean - 1.96 * se) > 0

    if weeks < PAPER_TRADING_WEEKS_REQUIRED:
        reasons.append(
            f"Paper trading has run {weeks:.1f} of the required "
            f"{PAPER_TRADING_WEEKS_REQUIRED} weeks."
        )
    if not clv_sig:
        if clv_mean is None:
            reasons.append("No paper-trading CLV recorded yet.")
        else:
            reasons.append(
                f"Paper-trading CLV ({clv_mean * 100:+.2f}%) is not significantly above zero."
            )

    ready = beats and weeks >= PAPER_TRADING_WEEKS_REQUIRED and clv_sig
    return Readiness(
        backtest_beats_closing=beats,
        backtest_gap=gap,
        best_stage=best_stage,
        paper_weeks=weeks,
        paper_bets=len(paper),
        paper_clv_mean=clv_mean,
        paper_clv_significant=clv_sig,
        ready=ready,
        reasons=reasons,
    )


def _market_log_loss_from_report(csv_path: Path) -> float | None:
    """Pull the market baseline out of the markdown report next to the CSV."""
    md = csv_path.with_suffix(".md")
    if not md.exists():
        return None
    for line in md.read_text().splitlines():
        if "bet365 margin-free" in line:
            parts = [p.strip().strip("*") for p in line.split("|")]
            for p in parts:
                try:
                    value = float(p)
                    if 0.1 < value < 3.0:
                        return value
                except ValueError:
                    continue
    return None
