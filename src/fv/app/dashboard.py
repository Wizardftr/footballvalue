"""Streamlit dashboard.

Four pages: This Week, Backtest, Bankroll, Settings.

One rule runs through all of them: never present a number in a way that implies more
certainty than it has. ROI carries its interval, monthly figures are labelled as
variance, and the real-money toggle shows the evidence against itself when the
evidence is against it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from fv.bets import (
    bankroll_history,
    bet_log,
    current_bankroll,
    log_slip,
    manual_settle,
    record_bankroll_event,
    rolling_roi,
    settle_pending,
)
from fv.config import PROJECT_ROOT, load_config
from fv.settings_store import effective_settings, real_money_readiness, set_setting
from fv.slip import generate_slip, slip_to_csv, slip_to_text

st.set_page_config(page_title="footballvalue", page_icon="⚽", layout="wide")

SELECTION_WORDS = {"H": "Home", "D": "Draw", "A": "Away"}


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%"


@st.cache_data(ttl=300)
def _cached_bet_log():
    return bet_log()


def _readiness_banner():
    """The honest verdict, shown wherever real money could be a temptation."""
    r = real_money_readiness()
    if r.ready:
        st.success(f"**{r.headline}**")
        return r
    st.error(f"**{r.headline}**")
    for reason in r.reasons:
        st.caption(f"• {reason}")
    return r


# ---------------------------------------------------------------------------
# Page 1: This Week
# ---------------------------------------------------------------------------

def page_this_week():
    st.title("This Week")
    settings = effective_settings()
    bankroll = current_bankroll()

    c1, c2, c3 = st.columns(3)
    c1.metric("Bankroll", f"{bankroll:,.2f}")
    c2.metric("Mode", "Paper" if settings["paper_mode"] else "REAL MONEY")
    c3.metric("Edge threshold", f"{settings['min_edge']:.0%}")

    if not settings["paper_mode"]:
        _readiness_banner()

    with st.spinner("Fitting models and pricing fixtures…"):
        slip = generate_slip(bankroll=bankroll)

    if slip.untuned_leagues:
        st.warning(
            f"No tuned weights for {', '.join(slip.untuned_leagues)} — using config "
            "defaults. Run `fv stages` so these leagues use validated weights."
        )

    if slip.all_candidates.empty:
        st.info(
            "**No upcoming fixtures with prices.** football-data.co.uk publishes "
            "fixtures about a week ahead, and only once a season is under way. Run "
            "`fv fixtures` to refresh."
        )
        return

    st.subheader("Recommended slip")
    if slip.selections.empty:
        st.info(
            "**Nothing qualifies this week.** That is the thresholds doing their job, "
            "not a failure — a week with nothing worth backing is a normal outcome."
        )
    else:
        st.caption("**Singles only.** Never combine these into an accumulator: doing so "
                   "multiplies the bookmaker's margin.")
        display = slip.selections.copy()
        display["match"] = display["home"] + " v " + display["away"]
        display["pick"] = display["selection"].map(SELECTION_WORDS)
        st.dataframe(
            display[["kickoff_utc", "league_code", "match", "pick", "odds", "stake",
                     "model_prob", "market_prob_fair", "edge"]].rename(columns={
                "kickoff_utc": "kickoff", "league_code": "league",
                "model_prob": "model", "market_prob_fair": "market",
            }),
            hide_index=True,
            use_container_width=True,
            column_config={
                "model": st.column_config.NumberColumn(format="%.1f%%"),
                "market": st.column_config.NumberColumn(format="%.1f%%"),
                "edge": st.column_config.NumberColumn(format="%+.1f%%"),
                "odds": st.column_config.NumberColumn(format="%.2f"),
                "stake": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(f"{len(slip.selections)} selections, total stake {slip.total_stake:,.2f} "
                   f"({slip.total_stake / bankroll:.1%} of bankroll)")

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.download_button("Download .txt", slip_to_text(slip),
                           file_name=f"slip-{datetime.utcnow():%Y%m%d}.txt")
        c2.download_button("Download .csv", slip_to_csv(slip),
                           file_name=f"slip-{datetime.utcnow():%Y%m%d}.csv")
        mode = "paper" if settings["paper_mode"] else "real"
        if c3.button(f"Log these {len(slip.selections)} bets as {mode}", type="primary"):
            slip_id, n = log_slip(slip.selections, mode=mode)
            st.cache_data.clear()
            st.success(f"Logged {n} bets as {slip_id}.")

        with st.expander("Slip as text"):
            st.code(slip_to_text(slip), language=None)

    st.subheader("All fixtures and edges")
    st.caption("Every selection considered, including those that did not qualify. "
               "Useful for seeing *why* something was left out.")
    cand = slip.all_candidates.copy()
    cand["match"] = cand["home"] + " v " + cand["away"]
    cand["pick"] = cand["selection"].map(SELECTION_WORDS)
    cand["qualified"] = cand["in_odds_range"] & cand["clears_edge"] & cand["enough_history"]

    def why_not(r):
        if r.qualified:
            return "qualified"
        bits = []
        if not r.in_odds_range:
            bits.append("odds outside range")
        if not r.clears_edge:
            bits.append("edge below threshold")
        if not r.enough_history:
            bits.append("too little team history")
        return "; ".join(bits)

    cand["reason"] = [why_not(r) for r in cand.itertuples(index=False)]
    leagues = sorted(cand["league_code"].unique())
    chosen = st.multiselect("Leagues", leagues, default=leagues)
    view = cand[cand["league_code"].isin(chosen)].sort_values("edge", ascending=False)
    st.dataframe(
        view[["kickoff_utc", "league_code", "match", "pick", "odds",
              "model_prob", "market_prob_fair", "edge", "reason"]].rename(columns={
            "kickoff_utc": "kickoff", "league_code": "league",
            "model_prob": "model", "market_prob_fair": "market"}),
        hide_index=True, use_container_width=True,
        column_config={
            "model": st.column_config.NumberColumn(format="%.1f%%"),
            "market": st.column_config.NumberColumn(format="%.1f%%"),
            "edge": st.column_config.NumberColumn(format="%+.1f%%"),
            "odds": st.column_config.NumberColumn(format="%.2f"),
        },
    )


# ---------------------------------------------------------------------------
# Page 2: Backtest
# ---------------------------------------------------------------------------

def page_backtest():
    st.title("Backtest")
    stages_dir = PROJECT_ROOT / "reports" / "stages"
    flat_dir = PROJECT_ROOT / "reports" / "flat"

    stage_csv = stages_dir / "stage_comparison.csv"
    if stage_csv.exists():
        st.subheader("Stage-by-stage comparison")
        table = pd.read_csv(stage_csv)
        st.dataframe(
            table[["stage", "n", "log_loss", "brier", "bets", "roi", "roi_lo",
                   "roi_hi", "clv_mean", "clv_significant"]],
            hide_index=True, use_container_width=True,
            column_config={
                "roi": st.column_config.NumberColumn(format="%.2f%%"),
                "roi_lo": st.column_config.NumberColumn("ROI low", format="%.2f%%"),
                "roi_hi": st.column_config.NumberColumn("ROI high", format="%.2f%%"),
                "clv_mean": st.column_config.NumberColumn("CLV", format="%.3f%%"),
            },
        )
        md = stages_dir / "stage_comparison.md"
        if md.exists():
            with st.expander("Full report"):
                st.markdown(md.read_text())
    else:
        st.info("No stage comparison yet. Run `fv stages`.")

    bets_csv = flat_dir / "bets.csv"
    if not bets_csv.exists():
        st.info("No backtest bet log yet. Run `fv backtest --flat-stake 10`.")
        return

    bets = pd.read_csv(bets_csv, parse_dates=["kickoff_utc"])
    st.subheader("Cumulative P&L")
    curve = bets.sort_values("kickoff_utc").copy()
    curve["cumulative"] = curve["pnl"].cumsum()
    st.line_chart(curve.set_index("kickoff_utc")["cumulative"])

    st.subheader("Drawdown")
    peak = curve["cumulative"].cummax()
    curve["drawdown"] = curve["cumulative"] - peak
    st.area_chart(curve.set_index("kickoff_utc")["drawdown"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROI by league")
        by_league = bets.groupby("league_code").apply(
            lambda g: pd.Series({"bets": len(g), "roi": g["pnl"].sum() / g["stake"].sum()}),
            include_groups=False,
        ).reset_index()
        st.dataframe(by_league, hide_index=True, use_container_width=True,
                     column_config={"roi": st.column_config.NumberColumn(format="%.2f%%")})
    with c2:
        st.subheader("ROI by season")
        by_season = bets.groupby("season").apply(
            lambda g: pd.Series({"bets": len(g), "roi": g["pnl"].sum() / g["stake"].sum()}),
            include_groups=False,
        ).reset_index()
        st.dataframe(by_season, hide_index=True, use_container_width=True,
                     column_config={"roi": st.column_config.NumberColumn(format="%.2f%%")})

    st.subheader("CLV distribution")
    clv = bets["clv"].dropna()
    if clv.empty:
        st.caption("No closing prices in this window — CLV is unmeasurable before 2019-20.")
    else:
        counts, edges = np.histogram(clv, bins=40)
        st.bar_chart(pd.DataFrame({"count": counts},
                                  index=np.round((edges[:-1] + edges[1:]) / 2, 4)))
        se = clv.std(ddof=1) / np.sqrt(len(clv))
        lo, hi = clv.mean() - 1.96 * se, clv.mean() + 1.96 * se
        st.caption(
            f"Mean {_pct(clv.mean())} (95% CI [{_pct(lo)}, {_pct(hi)}]), "
            f"beat the close on {(clv > 0).mean():.1%} of {len(clv):,} bets."
            + ("" if lo > 0 or hi < 0 else
               "  **The interval spans zero: this is not evidence of an edge.**")
        )


# ---------------------------------------------------------------------------
# Page 3: Bankroll
# ---------------------------------------------------------------------------

def page_bankroll():
    st.title("Bankroll")
    log = _cached_bet_log()
    balance = current_bankroll()

    settled = log[log["status"].isin(("won", "lost"))] if not log.empty else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance", f"{balance:,.2f}")
    c2.metric("Bets logged", f"{len(log):,}")
    if not settled.empty:
        roi = settled["pnl"].sum() / settled["stake"].sum()
        c3.metric("All-time ROI", f"{roi:+.2%}")
        c4.metric("Longest losing streak", _longest_losing_streak(settled))

    if log.empty:
        st.info("No bets logged yet. Generate a slip on **This Week** and log it.")
        return

    st.subheader("Rolling 300-bet ROI")
    st.caption(
        "The primary performance metric, shown with its confidence interval. Over 300 "
        "bets the standard error on ROI is roughly 7 percentage points, so the band "
        "matters more than the line: **a 300-bet window cannot distinguish a 4% edge "
        "from zero.**"
    )
    roll = rolling_roi(log, window=300)
    if roll.empty:
        st.caption("Not enough settled bets to show a rolling window yet "
                   "(at least 30 settled bets are needed).")
    else:
        # Index on the date, not the timestamp: with kickoffs clustered at similar
        # times the axis otherwise renders as a row of identical "12 PM" labels.
        chart = roll.copy()
        chart["date"] = pd.to_datetime(chart["kickoff_utc"]).dt.normalize()
        st.line_chart(chart.set_index("date")[["roi", "ci_low", "ci_high"]])
        latest = roll.iloc[-1]
        st.caption(
            f"Latest window: **{latest['roi']:+.2%}** over {int(latest['n'])} bets, "
            f"95% CI [{latest['ci_low']:+.2%}, {latest['ci_high']:+.2%}]."
            + ("  The interval spans zero — this is not yet evidence of an edge."
               if latest["ci_low"] < 0 < latest["ci_high"] else "")
        )

    st.subheader("Paper vs real")
    split = log.groupby("mode").apply(
        lambda g: pd.Series({
            "bets": len(g),
            "staked": g["stake"].sum(),
            "pnl": g["pnl"].fillna(0).sum(),
            "clv": g["clv"].dropna().mean() if g["clv"].notna().any() else np.nan,
        }),
        include_groups=False,
    ).reset_index()
    st.dataframe(split, hide_index=True, use_container_width=True)

    st.subheader("Monthly P&L")
    st.warning(
        "**This is variance, not signal.** At around 30 bets a month, even a genuine "
        "4% edge produces a losing month roughly 4 times in 10. Judge the model on the "
        "rolling window and on CLV, never on a month."
    )
    if not settled.empty:
        m = settled.copy()
        m["month"] = m["kickoff_utc"].dt.to_period("M").astype(str)
        monthly = m.groupby("month").agg(bets=("pnl", "size"), staked=("stake", "sum"),
                                         pnl=("pnl", "sum")).reset_index()
        st.bar_chart(monthly.set_index("month")["pnl"])
        st.dataframe(monthly, hide_index=True, use_container_width=True)

    st.subheader("Bet log")
    c1, c2 = st.columns([1, 3])
    if c1.button("Auto-settle now"):
        stats = settle_pending()
        st.cache_data.clear()
        c2.success(stats.summary())

    view = log.copy()
    view["match"] = view["home"] + " v " + view["away"]
    view["score"] = view.apply(
        lambda r: f"{int(r.fthg)}-{int(r.ftag)}" if pd.notna(r.fthg) else "", axis=1
    )
    st.dataframe(
        view[["id", "kickoff_utc", "league_code", "match", "selection", "odds_taken",
              "stake", "mode", "status", "score", "pnl", "closing_odds", "clv"]],
        hide_index=True, use_container_width=True,
    )

    with st.expander("Manual settlement"):
        st.caption("For the cases auto-settlement cannot see: voided matches, "
                   "cashed-out bets, bookmaker corrections.")
        bet_id = st.number_input("Bet id", min_value=1, step=1)
        status = st.selectbox("Status", ["won", "lost", "void", "pending"])
        override = st.number_input("P&L override (0 to compute automatically)", value=0.0)
        if st.button("Apply manual settlement"):
            try:
                manual_settle(int(bet_id), status, override or None)
                st.cache_data.clear()
                st.success(f"Bet {int(bet_id)} set to {status}.")
            except (KeyError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Bankroll ledger"):
        st.caption("Append-only. The balance is derived from these events, never "
                   "edited in place.")
        st.dataframe(bankroll_history(), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        amount = c1.number_input("Deposit / withdrawal", value=0.0, step=10.0)
        if c2.button("Record") and amount:
            record_bankroll_event(
                "deposit" if amount > 0 else "withdrawal", amount, note="manual"
            )
            st.cache_data.clear()
            st.success("Recorded.")


def _longest_losing_streak(settled: pd.DataFrame) -> int:
    longest = current = 0
    for r in settled.sort_values("kickoff_utc")["status"]:
        if r == "lost":
            current += 1
            longest = max(longest, current)
        elif r == "won":
            current = 0
    return longest


# ---------------------------------------------------------------------------
# Page 4: Settings
# ---------------------------------------------------------------------------

def page_settings():
    st.title("Settings")
    cfg = load_config()
    s = effective_settings()

    st.subheader("Real-money mode")
    readiness = _readiness_banner()
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Backtest vs closing odds",
        "beats" if readiness.backtest_beats_closing else "does not beat",
        delta=(f"{readiness.backtest_gap * 1000:+.2f} millinats"
               if readiness.backtest_gap is not None else None),
        delta_color="inverse",
    )
    # delta_color="off" because a bet count is neither good nor bad news; the
    # default green would read as approval of "0 bets".
    c2.metric("Paper trading", f"{readiness.paper_weeks:.1f} weeks",
              delta=f"{readiness.paper_bets} bets", delta_color="off")
    c3.metric("Paper CLV",
              _pct(readiness.paper_clv_mean) if readiness.paper_clv_mean is not None else "n/a",
              delta="significant" if readiness.paper_clv_significant else "not significant",
              delta_color="normal" if readiness.paper_clv_significant else "inverse")

    paper = st.toggle("Paper trading mode", value=bool(s["paper_mode"]))
    if not paper and not readiness.ready:
        st.error(
            "Turning this off bets real money on a model that has not met the "
            "project's own criteria. The evidence above is the reason those criteria "
            "exist. You can still do it — but do it knowing that."
        )
    if st.button("Save mode"):
        set_setting("paper_mode", bool(paper))
        st.success("Saved.")

    st.divider()
    st.subheader("Staking and thresholds")
    c1, c2 = st.columns(2)
    with c1:
        bankroll = st.number_input("Starting bankroll", value=float(s["starting_bankroll"]),
                                   min_value=1.0, step=50.0)
        kelly = st.slider("Kelly fraction", 0.05, 1.0, float(s["kelly_fraction"]), 0.05,
                          help="Full Kelly assumes your probabilities are correct. They "
                               "are estimates, and overstaking compounds badly: twice the "
                               "optimal fraction has zero expected growth.")
        max_stake = st.slider("Max stake (% of bankroll)", 0.005, 0.10,
                              float(s["max_stake_pct"]), 0.005, format="%.3f")
    with c2:
        min_edge = st.slider("Minimum edge", 0.0, 0.20, float(s["min_edge"]), 0.005,
                             format="%.3f")
        odds_range = st.slider("Odds range", 1.0, 10.0,
                               (float(s["min_odds"]), float(s["max_odds"])), 0.05)
        max_bets = st.number_input("Max bets per week", value=int(s["max_bets_per_week"]),
                                   min_value=1, max_value=50)

    st.subheader("Risk controls")
    c1, c2 = st.columns(2)
    stop_loss = c1.slider("Weekly stop-loss (% of bankroll)", 0.0, 0.50,
                          float(s["weekly_stop_loss_pct"]), 0.01)
    max_dd = c2.slider("Drawdown pause (% from peak)", 0.05, 0.75,
                       float(s["max_drawdown_pct"]), 0.01)

    st.subheader("Leagues")
    all_codes = [lg.code for lg in cfg.leagues]
    labels = {lg.code: f"{lg.code} — {lg.name}" for lg in cfg.leagues}
    enabled = st.multiselect("Enabled", all_codes, default=list(s["enabled_leagues"]),
                             format_func=lambda c: labels.get(c, c))

    if st.button("Save settings", type="primary"):
        for key, value in {
            "starting_bankroll": bankroll, "kelly_fraction": kelly,
            "max_stake_pct": max_stake, "min_edge": min_edge,
            "min_odds": odds_range[0], "max_odds": odds_range[1],
            "max_bets_per_week": int(max_bets), "weekly_stop_loss_pct": stop_loss,
            "max_drawdown_pct": max_dd, "enabled_leagues": enabled,
        }.items():
            set_setting(key, value)
        st.cache_data.clear()
        st.success("Saved. These override config.yaml.")


PAGES = {
    "This Week": page_this_week,
    "Backtest": page_backtest,
    "Bankroll": page_bankroll,
    "Settings": page_settings,
}


def main():
    st.sidebar.title("⚽ footballvalue")
    st.sidebar.caption("Analysis only. It never places bets.")
    choice = st.sidebar.radio("Page", list(PAGES))
    settings = effective_settings()
    st.sidebar.divider()
    st.sidebar.metric("Bankroll", f"{current_bankroll():,.2f}")
    st.sidebar.caption("**Paper mode**" if settings["paper_mode"] else "**REAL MONEY MODE**")
    PAGES[choice]()


main()
