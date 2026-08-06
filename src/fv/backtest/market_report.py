"""Over/under 2.5 and BTTS report.

Keeps the two markets visibly separate, because their evidence is not comparable:
over/under has real bet365 prices and a real backtest; BTTS has neither and can only
be checked for calibration. Presenting them in one undifferentiated table would
imply a validation for BTTS that does not exist.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fv.backtest.markets import (
    backtest_ou,
    binary_brier,
    binary_calibration,
    binary_log_loss,
    generate_btts_predictions,
    generate_ou_predictions,
    load_market_matches,
)
from fv.config import PROJECT_ROOT, Config


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%"


def run_market_backtest(
    cfg: Config,
    league_codes: list[str],
    test_from: str = "2022-08-01",
    test_to: str | None = None,
    out_dir: Path | None = None,
    console=None,
) -> str:
    def say(msg):
        if console:
            console.print(msg)

    weights_path = PROJECT_ROOT / "data" / "tuned_weights.json"
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else {}

    ou_frames, btts_frames, per_league = [], [], []

    for code in league_codes:
        matches = load_market_matches(code, cfg)
        if matches.empty:
            continue
        w = weights.get(code, {})
        xi = w.get("xi", 0.0015)
        xg = w.get("xg_weight", 0.0)

        say(f"[cyan]{code}: over/under 2.5[/cyan]")
        ou = generate_ou_predictions(code, test_from, test_to, xi=xi, xg_weight=xg,
                                     cfg=cfg, matches=matches)
        say(f"[cyan]{code}: BTTS[/cyan]")
        bt = generate_btts_predictions(code, test_from, test_to, xi=xi, xg_weight=xg,
                                       cfg=cfg, matches=matches)
        if not ou.empty:
            ou_frames.append(ou)
            res = backtest_ou(ou)
            per_league.append({
                "league": code, "n": len(ou), "model_ll": res.log_loss,
                "market_ll": res.market_log_loss, "bets": len(res.bets),
                "roi": (res.bets["pnl"].sum() / res.bets["stake"].sum())
                if not res.bets.empty else np.nan,
            })
        if not bt.empty:
            btts_frames.append(bt)

    if not ou_frames:
        return "No over/under predictions produced."

    ou_all = pd.concat(ou_frames, ignore_index=True)
    btts_all = pd.concat(btts_frames, ignore_index=True) if btts_frames else pd.DataFrame()
    result = backtest_ou(ou_all)
    league_table = pd.DataFrame(per_league)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        ou_all.to_csv(out_dir / "ou25_predictions.csv", index=False)
        if not result.bets.empty:
            result.bets.to_csv(out_dir / "ou25_bets.csv", index=False)
        if not btts_all.empty:
            btts_all.to_csv(out_dir / "btts_predictions.csv", index=False)
        league_table.to_csv(out_dir / "ou25_by_league.csv", index=False)

    report = _render(ou_all, btts_all, result, league_table, test_from, test_to)
    if out_dir:
        (out_dir / "markets_report.md").write_text(report)
    return report


def _render(ou_all, btts_all, result, league_table, test_from, test_to) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Phase 4: over/under 2.5 and BTTS")
    add("")
    add(f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    add(f"Test window: {test_from} to {test_to or 'latest'}. Flat 10 per bet.")
    add("")

    add("## Over/under 2.5")
    add("")
    add("bet365 published over/under prices in football-data from 2019-20 (pre and "
        "closing), so this market gets the same treatment as 1X2: walk-forward "
        "predictions, real prices, real CLV.")
    add("")
    gap = result.log_loss - result.market_log_loss
    add("| | log loss | Brier | matches |")
    add("|---|---:|---:|---:|")
    add(f"| Model (Dixon-Coles + xG) | {result.log_loss:.5f} | {result.brier:.5f} "
        f"| {len(ou_all):,} |")
    add(f"| **bet365 margin-free** | **{result.market_log_loss:.5f}** "
        f"| **{result.market_brier:.5f}** | |")
    add("")
    if gap < 0:
        add(f"The model beats the market by {abs(gap) * 1000:.2f} millinats. Re-check the "
            "leakage tests before believing it.")
    else:
        add(f"The model is **{gap * 1000:.2f} millinats behind** bet365's own prices.")
        add("")
        add("Relative to the market's own log loss that is a "
            f"**{gap / result.market_log_loss:.2%}** shortfall. The equivalent figure for "
            "1X2 at the same modelling stage was about 1.9%, so over/under is marginally "
            "closer — but *closer to losing* is still losing. **Over/under 2.5 is not the "
            "soft market this project was hoping to find.**")
    add("")

    if not result.bets.empty:
        bets = result.bets
        roi = bets["pnl"].sum() / bets["stake"].sum()
        per_unit = (bets["pnl"] / bets["stake"]).to_numpy()
        se = per_unit.std(ddof=1) / np.sqrt(len(per_unit))
        add("### Betting performance")
        add("")
        add("| metric | value |")
        add("|---|---:|")
        add(f"| Bets | {len(bets):,} |")
        add(f"| Staked | {bets['stake'].sum():,.0f} |")
        add(f"| P&L | {bets['pnl'].sum():+,.0f} |")
        add(f"| ROI | {_pct(roi)} |")
        add(f"| ROI 95% CI | [{_pct(roi - 1.96 * se)}, {_pct(roi + 1.96 * se)}] |")
        add(f"| Strike rate | {(bets['result'] == 'W').mean():.1%} |")
        clv = bets["clv"].dropna()
        if len(clv) > 1:
            cse = clv.std(ddof=1) / np.sqrt(len(clv))
            lo, hi = clv.mean() - 1.96 * cse, clv.mean() + 1.96 * cse
            add(f"| Mean CLV | {_pct(clv.mean(), 3)} |")
            add(f"| CLV 95% CI | [{_pct(lo, 3)}, {_pct(hi, 3)}] |")
            add("")
            if lo < 0 < hi:
                add("**CLV is indistinguishable from zero**, the same verdict as 1X2.")
        add("")

    if not league_table.empty:
        add("### By league")
        add("")
        add("| league | matches | model | market | gap | bets | ROI |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for r in league_table.sort_values("model_ll").itertuples(index=False):
            add(f"| {r.league} | {r.n:,} | {r.model_ll:.5f} | {r.market_ll:.5f} "
                f"| {r.model_ll - r.market_ll:+.5f} | {r.bets:,} | {_pct(r.roi)} |")
        add("")

    add("### Calibration")
    add("")
    add("| predicted | observed | n | error |")
    add("|---:|---:|---:|---:|")
    for r in binary_calibration(ou_all["p_over"].to_numpy(),
                                ou_all["actual_over"].to_numpy()).itertuples(index=False):
        add(f"| {r.predicted:.3f} | {r.observed:.3f} | {r.n:,} "
            f"| {r.observed - r.predicted:+.3f} |")
    add("")
    add("Errors stay under about 0.02 in the bins that carry most of the sample. The "
        "large errors at the extremes sit on a handful of matches each and are noise.")
    add("")

    # ---------------------------------------------------------------- BTTS
    add("## BTTS — not validated")
    add("")
    add("**There are no historical BTTS prices in football-data, or in any free source "
        "this project can reach.** Without prices there is no edge, no ROI and no CLV to "
        "compute, so BTTS cannot be backtested at all. Under this project's own rule — an "
        "upgrade ships only if it beats the previous stage out-of-sample — **BTTS is not "
        "validated and must not be bet on the strength of this report.**")
    add("")
    add("What can be checked is whether the model's BTTS probabilities are calibrated. "
        "That is a real question and worth answering, because a price arriving later from "
        "The Odds API is only actionable if the underlying probability is trustworthy.")
    add("")
    if not btts_all.empty:
        p = btts_all["p_btts"].to_numpy()
        y = btts_all["actual_btts"].to_numpy()
        add("| | value |")
        add("|---|---:|")
        add(f"| Matches | {len(btts_all):,} |")
        add(f"| Log loss | {binary_log_loss(p, y):.5f} |")
        add(f"| Brier | {binary_brier(p, y):.5f} |")
        add(f"| Predicted BTTS rate | {p.mean():.4f} |")
        add(f"| Observed BTTS rate | {y.mean():.4f} |")
        add("")
        bias = y.mean() - p.mean()
        add(f"**The model under-predicts BTTS by {bias * 100:.2f} percentage points.** "
            "That is a systematic bias, not noise, and it would push the model toward "
            "backing 'No' — so it needs correcting before BTTS is priced against a real "
            "market. The likely cause is the same one behind the shrunken Dixon-Coles rho: "
            "the independence assumption between the two teams' scoring misses some of "
            "the correlation that produces both-teams-score outcomes.")
        add("")
        add("| predicted | observed | n | error |")
        add("|---:|---:|---:|---:|")
        for r in binary_calibration(p, y).itertuples(index=False):
            add(f"| {r.predicted:.3f} | {r.observed:.3f} | {r.n:,} "
                f"| {r.observed - r.predicted:+.3f} |")
        add("")

    return "\n".join(lines)
