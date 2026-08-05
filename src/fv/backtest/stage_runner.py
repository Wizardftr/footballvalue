"""Run all five stages and produce the comparison.

The discipline the spec asks for: every stage is scored on the same held-out
matches, every weight is tuned on validation seasons that end before the test
window, and an upgrade only counts if it beats the previous stage out-of-sample.

Both halves of the picture are reported. Probability quality (log loss, Brier)
says whether the model got better. Betting performance (ROI, CLV) says whether it
got better *where the market is wrong*, which is a far higher bar and the one that
decides whether any of this is worth doing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fv.backtest.metrics import brier_score, log_loss, roi_confidence_interval, summarize_clv
from fv.backtest.stages import (
    anchor_to_market,
    combine,
    predict_lgbm,
    tune_market_weight,
    tune_pool_weight,
)
from fv.backtest.walkforward import (
    BettingParams,
    generate_predictions,
    load_matches,
    simulate_betting,
    tune_xg_weight,
    tune_xi,
)
from fv.config import Config
from fv.odds.edge import Thresholds
from fv.odds.kelly import StakeRules

STAGE_ORDER = ["dc", "dc_xg", "lgbm", "ensemble", "anchored"]
STAGE_LABELS = {
    "dc": "1. Dixon-Coles (goals)",
    "dc_xg": "2. Dixon-Coles (goals+xG)",
    "lgbm": "3. LightGBM (calibrated)",
    "ensemble": "4. Ensemble (log pool)",
    "anchored": "5. Market-anchored",
}


def _betting_params(cfg: Config, flat_stake: float) -> BettingParams:
    b, r = cfg.betting, cfg.risk
    return BettingParams(
        starting_bankroll=b.get("starting_bankroll", 1000.0),
        thresholds=Thresholds(
            min_edge=b.get("min_edge", 0.04),
            min_odds=b.get("min_odds", 1.50),
            max_odds=b.get("max_odds", 4.00),
        ),
        stake_rules=StakeRules(
            kelly_fraction=b.get("kelly_fraction", 0.25),
            max_stake_pct=b.get("max_stake_pct", 0.02),
            rounding=b.get("stake_rounding", 0.50),
            min_stake=b.get("min_stake", 1.00),
        ),
        max_bets_per_week=b.get("max_bets_per_week", 8),
        min_team_matches=cfg.dixon_coles.get("min_team_matches", 6),
        margin_method=cfg.odds.get("margin_method", "proportional"),
        weekly_stop_loss_pct=None,
        max_drawdown_pct=r.get("max_drawdown_pct", 0.25),
        flat_stake=flat_stake,
    )


def run_stages(
    cfg: Config,
    league_codes: list[str],
    valid_from: str,
    test_from: str,
    test_to: str | None = None,
    flat_stake: float = 10.0,
    out_dir: Path | None = None,
    console=None,
) -> str:
    """Fit and evaluate every stage; return the markdown comparison."""

    def say(msg: str) -> None:
        if console:
            console.print(msg)

    valid_to = str((pd.Timestamp(test_from) - pd.Timedelta(days=1)).date())
    xi_grid = cfg.dixon_coles.get("xi_grid", [0.0005, 0.001, 0.0015, 0.002, 0.003])
    xg_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]

    test_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in STAGE_ORDER}
    tuning_rows: list[dict] = []

    for code in league_codes:
        matches = load_matches(code, cfg)
        if matches.empty:
            say(f"[yellow]{code}: no data, skipping[/yellow]")
            continue
        has_xg = bool(matches["home_xg"].notna().any())

        # --- tune on validation ------------------------------------------
        say(f"[cyan]{code}: tuning xi[/cyan]")
        xi, _ = tune_xi(code, valid_from, valid_to, xi_grid, cfg=cfg, matches=matches)

        xg_weight = 0.0
        if has_xg:
            say(f"[cyan]{code}: tuning xG blend[/cyan]")
            xg_weight, _ = tune_xg_weight(
                code, valid_from, valid_to, xg_grid, xi=xi, cfg=cfg, matches=matches
            )

        say(f"[cyan]{code}: validation predictions[/cyan]")
        dc_v = generate_predictions(
            code, valid_from, valid_to, xi=xi, cfg=cfg, matches=matches
        ).predictions
        dcxg_v = (
            generate_predictions(
                code, valid_from, valid_to, xi=xi, cfg=cfg, matches=matches, xg_weight=xg_weight
            ).predictions
            if xg_weight > 0
            else dc_v
        )
        lgbm_v = predict_lgbm(code, valid_from, valid_to, cfg=cfg, matches=matches)

        pool_w, _ = (
            tune_pool_weight(dcxg_v, lgbm_v) if not lgbm_v.empty else (1.0, pd.DataFrame())
        )
        ens_v = combine(dcxg_v, lgbm_v, pool_w) if not lgbm_v.empty else dcxg_v
        market_w, _ = tune_market_weight(ens_v, method=cfg.odds.get("margin_method", "proportional"))

        tuning_rows.append(
            {
                "league": code,
                "has_xg": has_xg,
                "xi": xi,
                "half_life_days": np.inf if xi == 0 else np.log(2) / xi,
                "xg_weight": xg_weight,
                "dc_weight_in_ensemble": pool_w,
                "market_weight": market_w,
            }
        )

        # --- apply to the test window ------------------------------------
        say(f"[cyan]{code}: test predictions (xi={xi:g}, xg={xg_weight:g}, "
            f"pool={pool_w:g}, market={market_w:g})[/cyan]")
        dc_t = generate_predictions(
            code, test_from, test_to, xi=xi, cfg=cfg, matches=matches
        ).predictions
        dcxg_t = (
            generate_predictions(
                code, test_from, test_to, xi=xi, cfg=cfg, matches=matches, xg_weight=xg_weight
            ).predictions
            if xg_weight > 0
            else dc_t
        )
        lgbm_t = predict_lgbm(code, test_from, test_to, cfg=cfg, matches=matches)
        ens_t = combine(dcxg_t, lgbm_t, pool_w) if not lgbm_t.empty else dcxg_t
        anch_t = anchor_to_market(ens_t, market_w, method=cfg.odds.get("margin_method", "proportional"))

        for name, frame in (
            ("dc", dc_t), ("dc_xg", dcxg_t), ("lgbm", lgbm_t),
            ("ensemble", ens_t), ("anchored", anch_t),
        ):
            if not frame.empty:
                test_frames[name].append(frame)

    stages = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in test_frames.items()
        if frames
    }
    if not stages:
        return "No predictions produced."

    params = _betting_params(cfg, flat_stake)
    results = []
    for name in STAGE_ORDER:
        frame = stages.get(name)
        if frame is None or frame.empty:
            continue
        sim = simulate_betting(frame, params)
        bets = sim.bets
        roi = lo = hi = np.nan
        if len(bets) > 1:
            roi, lo, hi = roi_confidence_interval(
                bets["pnl"].to_numpy(), bets["stake"].to_numpy()
            )
        clv = summarize_clv(bets["clv"]) if len(bets) else summarize_clv(pd.Series([], dtype=float))
        results.append(
            {
                "stage": name,
                "n": len(frame),
                "log_loss": log_loss(frame),
                "brier": brier_score(frame),
                "bets": len(bets),
                "roi": roi,
                "roi_lo": lo,
                "roi_hi": hi,
                "clv_mean": clv["mean"],
                "clv_significant": clv["significant"],
                "avg_edge": float(bets["edge"].mean()) if len(bets) else np.nan,
            }
        )

    table = pd.DataFrame(results)
    tuning = pd.DataFrame(tuning_rows)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "stage_comparison.csv", index=False)
        tuning.to_csv(out_dir / "tuned_weights.csv", index=False)
        for name, frame in stages.items():
            frame.to_csv(out_dir / f"predictions_{name}.csv", index=False)

    report = _render(table, tuning, stages, test_from, test_to, flat_stake, cfg)
    if out_dir:
        (out_dir / "stage_comparison.md").write_text(report)
    return report


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%"


def _render(table, tuning, stages, test_from, test_to, flat_stake, cfg) -> str:
    from fv.backtest.report import _market_baseline

    lines: list[str] = []
    add = lines.append

    add("# Phase 2: stage-by-stage model comparison")
    add("")
    add(f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    add(f"Test window: {test_from} to {test_to or 'latest'}. "
        f"Flat stake {flat_stake:g} per bet, so the comparison measures the edge "
        "rather than a compounding path.")
    add("")

    baseline = _market_baseline(stages[list(stages)[0]])
    market_ll = baseline["log_loss"]

    add("## The comparison")
    add("")
    add("Every stage scored on the same held-out matches. Weights tuned on validation "
        "seasons ending before the test window.")
    add("")
    add("| stage | log loss | vs market | Brier | bets | ROI | 95% CI | mean CLV |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in table.itertuples(index=False):
        gap = r.log_loss - market_ll
        add(f"| {STAGE_LABELS[r.stage]} | {r.log_loss:.5f} | {gap:+.5f} | {r.brier:.5f} "
            f"| {r.bets:,} | {_pct(r.roi)} | [{_pct(r.roi_lo)}, {_pct(r.roi_hi)}] "
            f"| {_pct(r.clv_mean)} |")
    add(f"| **bet365 margin-free** | **{market_ll:.5f}** | — | "
        f"**{baseline['brier']:.5f}** | — | — | — | — |")
    add("")

    # Did each stage beat the previous one?
    add("### Did each upgrade earn its place?")
    add("")
    prev = None
    for r in table.itertuples(index=False):
        if prev is None:
            add(f"- **{STAGE_LABELS[r.stage]}** — baseline, log loss {r.log_loss:.5f}")
        else:
            delta = r.log_loss - prev.log_loss
            verdict = "**improves**" if delta < 0 else "**does not improve**"
            add(f"- **{STAGE_LABELS[r.stage]}** — {verdict} on the previous stage "
                f"({delta * 1000:+.2f} millinats, {r.log_loss:.5f})")
        prev = r
    add("")

    best = table.loc[table["log_loss"].idxmin()]
    if best.log_loss < market_ll:
        add(f"The best stage ({STAGE_LABELS[best.stage]}) beats bet365's margin-free "
            f"prices by {(market_ll - best.log_loss) * 1000:.2f} millinats. Treat this "
            "with suspicion until the leakage tests and the tuning windows have been "
            "re-checked — this is the result that would be most costly to get wrong.")
    else:
        add(f"**No stage beats bet365.** The best ({STAGE_LABELS[best.stage]}) is still "
            f"{(best.log_loss - market_ll) * 1000:.2f} millinats behind the market's own "
            "prices. Under the project's own rule — real money is earned by beating "
            "closing odds out-of-sample — **this model is not ready for real money.**")
    add("")

    add("## Tuned weights")
    add("")
    add("| league | xG? | xi | half-life | xG blend | DC weight in ensemble | market weight |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for r in tuning.itertuples(index=False):
        hl = "inf" if not np.isfinite(r.half_life_days) else f"{r.half_life_days:.0f}d"
        add(f"| {r.league} | {'yes' if r.has_xg else 'no'} | {r.xi:g} | {hl} "
            f"| {r.xg_weight:g} | {r.dc_weight_in_ensemble:g} | {r.market_weight:g} |")
    add("")
    if not tuning.empty:
        with_xg = tuning[tuning["has_xg"]]
        if not with_xg.empty:
            add(f"Mean xG blend where xG exists: **{with_xg['xg_weight'].mean():.2f}**.")
        add(f"Mean market weight: **{tuning['market_weight'].mean():.2f}**.")
    add("")

    add("## Reading the betting columns")
    add("")
    add("ROI intervals at these sample sizes are wide enough that most differences "
        "between stages are not distinguishable from noise. The log loss column is the "
        "more reliable comparison: it uses every match rather than only the ones that "
        "cleared the betting thresholds, so it has far more data behind it.")
    add("")
    add("The market-anchored stage should produce **fewer** bets than the unanchored "
        "ones. That is the anchor working as intended — it discounts model-market "
        "disagreement, and most such disagreement is model error rather than value.")
    add("")

    return "\n".join(lines)
