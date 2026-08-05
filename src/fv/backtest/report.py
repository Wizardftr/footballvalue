"""Backtest reporting.

The rule this module follows: never present a number in a way that implies more
certainty than it has. ROI always carries its confidence interval, monthly figures
are labelled as variance, and if the model fails to beat the closing line the report
says so in plain words at the top rather than burying it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fv.backtest.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    longest_losing_streak,
    max_drawdown,
    roi_confidence_interval,
    summarize_clv,
)


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%"


def _market_baseline(predictions: pd.DataFrame, method: str = "proportional") -> dict:
    """Score bet365's own prices as a model, for comparison.

    This is the benchmark that matters. Beating a coin flip is meaningless; beating
    the price you are betting into is the whole question.
    """
    from fv.odds.margin import remove_margin

    rows = []
    for r in predictions.itertuples(index=False):
        trio = (r.pre_h, r.pre_d, r.pre_a)
        if any(o is None or pd.isna(o) or float(o) <= 1.0 for o in trio):
            continue
        fair = remove_margin([float(o) for o in trio], method=method)
        rows.append({"p_home": fair[0], "p_draw": fair[1], "p_away": fair[2], "ftr": r.ftr})
    if not rows:
        return {"n": 0, "brier": float("nan"), "log_loss": float("nan")}
    df = pd.DataFrame(rows)
    return {"n": len(df), "brier": brier_score(df), "log_loss": log_loss(df)}


def build_report(
    predictions: pd.DataFrame,
    bets: pd.DataFrame,
    equity: pd.DataFrame,
    params: dict,
    diagnostics: pd.DataFrame | None = None,
    halted_at=None,
    xi_table: pd.DataFrame | None = None,
    drawdown_breaches: list | None = None,
) -> str:
    """Render the full markdown report."""
    lines: list[str] = []
    add = lines.append

    add("# Phase 1 backtest: time-decayed Dixon-Coles on goals")
    add("")
    add(f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    add("")

    # ---------------------------------------------------------------- verdict
    starting = params.get("starting_bankroll", 1000.0)
    n_bets = len(bets)
    staked = float(bets["stake"].sum()) if n_bets else 0.0
    pnl = float(bets["pnl"].sum()) if n_bets else 0.0
    roi, lo, hi = (
        roi_confidence_interval(bets["pnl"].to_numpy(), bets["stake"].to_numpy())
        if n_bets > 1
        else (float("nan"), float("nan"), float("nan"))
    )
    market = _market_baseline(predictions)
    model_ll = log_loss(predictions)
    beats_market = model_ll < market["log_loss"] if market["n"] else False

    add("## Verdict")
    add("")
    if n_bets == 0:
        add("**No qualifying bets.** Nothing cleared the edge and odds thresholds.")
    else:
        add(f"- **{n_bets:,} bets** over the test window, {staked:,.0f} staked, "
            f"P&L {pnl:+,.0f}")
        add(f"- **ROI {_fmt_pct(roi)}**, 95% CI [{_fmt_pct(lo)}, {_fmt_pct(hi)}]")
        if not np.isnan(lo) and lo < 0 < hi:
            add("  - The interval spans zero: this result **does not establish an edge**.")
        elif not np.isnan(hi) and hi < 0:
            add("  - The interval is entirely below zero: this stage **loses money**.")

    add("")
    add("### Does the model beat the bookmaker?")
    add("")
    add("The only benchmark that matters. bet365's own margin-free prices are scored "
        "as if they were a model, on exactly the same matches.")
    add("")
    add("| | log loss | Brier | matches |")
    add("|---|---:|---:|---:|")
    add(f"| Dixon-Coles (this model) | {model_ll:.4f} | {brier_score(predictions):.4f} "
        f"| {len(predictions):,} |")
    if market["n"]:
        add(f"| bet365 margin-free | {market['log_loss']:.4f} | {market['brier']:.4f} "
            f"| {market['n']:,} |")
        gap = model_ll - market["log_loss"]
        add("")
        if beats_market:
            add(f"The model's log loss is **{abs(gap):.4f} lower** than the market's. "
                "That is the result this project needs, and it should be treated with "
                "suspicion until the leakage tests and out-of-sample window have been "
                "double-checked.")
        else:
            add(f"The model's log loss is **{gap:.4f} higher** than the market's — "
                "bet365 predicts these matches better than this model does. "
                "**This is the expected Phase 1 result**: a goals-only Dixon-Coles has "
                "no information the market lacks. It is the floor that the Phase 2 "
                "stages (xG, LightGBM, ensemble, market anchor) have to clear.")

    # ---------------------------------------------------------------- CLV
    add("")
    add("### Closing line value")
    add("")
    if n_bets:
        c = summarize_clv(bets["clv"])
        if c["n"]:
            add(f"- Measured on **{c['n']:,} of {n_bets:,} bets** "
                "(closing prices exist from 2019-20 onward)")
            add(f"- Mean CLV **{_fmt_pct(c['mean'])}**, median {_fmt_pct(c['median'])}")
            add(f"- Beat the closing price on **{c['pct_positive'] * 100:.1f}%** of bets")
            add("")
            if c["mean"] > 0:
                add("Positive CLV is the leading indicator of a real edge: it resolves on "
                    "every bet immediately instead of waiting for outcomes to average out.")
            else:
                add("Negative CLV means the prices taken were systematically worse than "
                    "the close. Over a long run that is incompatible with a genuine edge, "
                    "whatever the ROI happens to look like.")
        else:
            add("No closing prices available in this window.")
    add("")

    # ---------------------------------------------------------------- sample size
    add("### How much can this sample tell us?")
    add("")
    if n_bets > 1:
        per_unit = bets["pnl"].to_numpy() / bets["stake"].to_numpy()
        sd = float(np.std(per_unit, ddof=1))
        se_300 = sd / np.sqrt(300)
        add(f"Per-bet standard deviation of returns is **{sd:.2f}** units. Over a rolling "
            f"300-bet window the standard error on ROI is about **{se_300 * 100:.1f} "
            "percentage points**, so a 95% interval spans roughly "
            f"±{1.96 * se_300 * 100:.0f} points.")
        add("")
        add("**A 300-bet window cannot distinguish a 4% edge from zero.** This is why CLV "
            "is the primary metric and monthly P&L is variance, not signal.")
    add("")

    # ---------------------------------------------------------------- headline table
    add("## Performance")
    add("")
    if n_bets:
        eq = bets["bankroll_after"]
        dd = max_drawdown(pd.concat([pd.Series([starting]), eq]))
        add("| metric | value |")
        add("|---|---:|")
        add(f"| Bets | {n_bets:,} |")
        add(f"| Total staked | {staked:,.0f} |")
        add(f"| P&L | {pnl:+,.0f} |")
        add(f"| ROI | {_fmt_pct(roi)} |")
        add(f"| ROI 95% CI | [{_fmt_pct(lo)}, {_fmt_pct(hi)}] |")
        add(f"| Starting bankroll | {starting:,.0f} |")
        add(f"| Final bankroll | {float(eq.iloc[-1]):,.0f} |")
        add(f"| Max drawdown | {dd * 100:.1f}% |")
        add(f"| Longest losing streak | {longest_losing_streak(bets['result'])} |")
        add(f"| Strike rate | {(bets['result'] == 'W').mean() * 100:.1f}% |")
        add(f"| Average odds taken | {bets['odds_taken'].mean():.2f} |")
        add(f"| Average edge at bet | {_fmt_pct(bets['edge'].mean())} |")
        if halted_at is not None:
            add(f"| **Halted** | max drawdown breached at {halted_at} |")
        add("")

        if drawdown_breaches:
            add(f"### Drawdown pause would have triggered {len(drawdown_breaches)} time(s)")
            add("")
            add("In live use the app stops betting at a 25% drawdown from peak and requires "
                "a manual reset. The backtest records the breach and keeps going, so the "
                "full period stays measurable — otherwise the evaluation would end at the "
                "first bad run and hide everything after it.")
            add("")
            add("| date | bankroll | peak | drawdown |")
            add("|---|---:|---:|---:|")
            for b in drawdown_breaches[:10]:
                add(f"| {b['kickoff_utc']} | {b['bankroll']:,.0f} | {b['peak']:,.0f} "
                    f"| {b['drawdown'] * 100:.1f}% |")
            add("")

        # by league
        add("### By league")
        add("")
        add("| league | bets | staked | P&L | ROI | strike | mean CLV |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for code, g in bets.groupby("league_code"):
            r, _, _ = roi_confidence_interval(g["pnl"].to_numpy(), g["stake"].to_numpy())
            cc = summarize_clv(g["clv"])
            add(f"| {code} | {len(g):,} | {g['stake'].sum():,.0f} | {g['pnl'].sum():+,.0f} "
                f"| {_fmt_pct(r)} | {(g['result'] == 'W').mean() * 100:.1f}% "
                f"| {_fmt_pct(cc['mean'])} |")
        add("")

        # by season
        add("### By season")
        add("")
        add("| season | bets | staked | P&L | ROI |")
        add("|---|---:|---:|---:|---:|")
        for season, g in bets.groupby("season"):
            r, _, _ = roi_confidence_interval(g["pnl"].to_numpy(), g["stake"].to_numpy())
            add(f"| {season} | {len(g):,} | {g['stake'].sum():,.0f} "
                f"| {g['pnl'].sum():+,.0f} | {_fmt_pct(r)} |")
        add("")

    # ---------------------------------------------------------------- calibration
    add("## Calibration")
    add("")
    add("Predicted probability against observed frequency, pooling home/draw/away. "
        "A well-calibrated model tracks the diagonal.")
    add("")
    cal = calibration_table(predictions)
    if not cal.empty:
        add("| predicted | observed | n | error |")
        add("|---:|---:|---:|---:|")
        for r in cal.itertuples(index=False):
            add(f"| {r.predicted:.3f} | {r.observed:.3f} | {r.n:,} "
                f"| {r.observed - r.predicted:+.3f} |")
    add("")

    # ---------------------------------------------------------------- xi tuning
    if xi_table is not None and not xi_table.empty:
        add("## Time-decay tuning")
        add("")
        add("Chosen on validation seasons that end before the test window starts, so the "
            "hyperparameter never sees test data.")
        add("")
        add("| xi | half-life (days) | matches | log loss | Brier |")
        add("|---:|---:|---:|---:|---:|")
        for r in xi_table.itertuples(index=False):
            hl = "inf" if not np.isfinite(r.half_life_days) else f"{r.half_life_days:.0f}"
            add(f"| {r.xi:g} | {hl} | {r.n:,} | {r.log_loss:.4f} | {r.brier:.4f} |")
        add("")

    # ---------------------------------------------------------------- diagnostics
    if diagnostics is not None and not diagnostics.empty:
        add("## Model diagnostics")
        add("")
        conv = diagnostics["converged"].mean() * 100
        add(f"- Fits converged: **{conv:.1f}%** of {len(diagnostics):,} refits")
        add(f"- Mean home advantage: **{diagnostics['home_adv'].mean():.3f}** "
            f"(exp = {np.exp(diagnostics['home_adv'].mean()):.2f}x goal rate)")
        add(f"- Mean Dixon-Coles rho: **{diagnostics['rho'].mean():.4f}**")
        add(f"- Mean effective sample per fit: **{diagnostics['effective_sample'].mean():.0f}** "
            "matches after time decay")
        add("")

    # ---------------------------------------------------------------- monthly
    if n_bets:
        add("## Monthly P&L — this is variance, not signal")
        add("")
        add("Shown because it is asked for, labelled because it misleads. At roughly 30 "
            "bets a month, even a genuine 4% edge produces a losing month about 4 times "
            "in 10. Read the rolling window and CLV instead.")
        add("")
        m = bets.copy()
        m["month"] = pd.to_datetime(m["kickoff_utc"]).dt.to_period("M")
        agg = m.groupby("month").agg(bets=("pnl", "size"), staked=("stake", "sum"),
                                     pnl=("pnl", "sum"))
        losing = (agg["pnl"] < 0).mean()
        add(f"Losing months: **{losing * 100:.0f}%** of {len(agg)} months.")
        add("")

    return "\n".join(lines)


def write_report(
    out_dir: Path,
    report_md: str,
    predictions: pd.DataFrame,
    bets: pd.DataFrame,
    equity: pd.DataFrame,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "backtest_report.md"
    path.write_text(report_md)
    if not predictions.empty:
        predictions.to_csv(out_dir / "predictions.csv", index=False)
    if not bets.empty:
        bets.to_csv(out_dir / "bets.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / "equity.csv", index=False)
    return path
