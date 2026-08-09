"""Streamlit dashboard.

Five pages: This Week, Backtest, Bankroll, Settings, Ask.

One rule runs through all of them: never present a number in a way that implies more
certainty than it has. ROI carries its interval, monthly figures are labelled as
variance, and the real-money toggle shows the evidence against itself when the
evidence is against it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from fv import auth, chat
from fv.app import theme
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
from fv.settings_store import (
    effective_settings,
    get_setting,
    real_money_readiness,
    set_setting,
)
from fv.db.migrate import ensure_schema
from fv.slip import generate_slip, slip_to_csv, slip_to_text

st.set_page_config(
    page_title="footballvalue",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

SELECTION_WORDS = {"H": "Home", "D": "Draw", "A": "Away"}


def _uid() -> int:
    """The signed-in account's id. Every page runs behind the sign-in gate."""
    return st.session_state.account.id


def _account():
    return st.session_state.account


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%"


@st.cache_data(ttl=300)
def _cached_bet_log(user_id: int):
    """Keyed on the user id on purpose: an unkeyed cache would hand one account
    another account's bet log for five minutes."""
    return bet_log(user_id=user_id)


def _readiness_banner():
    """The honest verdict, shown wherever real money could be a temptation."""
    r = real_money_readiness(user_id=_uid())
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
    theme.hero("This Week",
               "Fixtures the model has priced, what it disagrees with bet365 about, "
               "and the slip that follows from your thresholds.")
    settings = effective_settings(user_id=_uid())
    bankroll = current_bankroll(user_id=_uid())

    c1, c2, c3 = st.columns(3)
    c1.metric("Bankroll", f"{bankroll:,.2f}")
    c2.metric("Mode", "Paper" if settings["paper_mode"] else "REAL MONEY")
    c3.metric("Edge threshold", f"{settings['min_edge']:.0%}")

    if not settings["paper_mode"]:
        _readiness_banner()

    c1, c2 = st.columns([1, 3])
    fill_mode = c1.toggle(
        "Always give me a slip",
        value=bool(get_setting("slip_fill_mode", False, user_id=_uid())),
        help="Off: only selections that clear the edge threshold, so a week with no "
             "value gives an empty slip. On: the best N by edge regardless, so there "
             "is always something to place.",
    )
    fill_n = None
    if fill_mode:
        fill_n = int(c2.slider("Selections per week", 1, 20,
                               int(get_setting("slip_fill_n", 10, user_id=_uid()))))
        set_setting("slip_fill_mode", True, user_id=_uid())
        set_setting("slip_fill_n", fill_n, user_id=_uid())
    else:
        set_setting("slip_fill_mode", False, user_id=_uid())

    with st.spinner("Fitting models and pricing fixtures…"):
        slip = generate_slip(bankroll=bankroll, fill_to=fill_n)

    if slip.untuned_leagues:
        st.warning(
            f"No tuned weights for {', '.join(slip.untuned_leagues)} — using config "
            "defaults. Run `fv stages` so these leagues use validated weights."
        )

    if slip.no_edge_possible_leagues:
        st.error(
            f"**{', '.join(slip.no_edge_possible_leagues)}: no selection can qualify.** "
            "Validation gave the market a weight of 1.0 in these leagues, so the "
            "anchored probability *is* the market's own price and every edge equals "
            "minus the bookmaker's margin. This is structural, not a quiet week — "
            "these leagues cannot produce a bet until the model earns weight back "
            "against the market."
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
        if slip.mode == "filled":
            ev = slip.expected_return
            pct = ev / slip.total_stake if slip.total_stake else 0.0
            if ev < 0:
                st.warning(
                    f"**Filled slip — expected return {ev:+,.2f} ({pct:+.1%} of stake).** "
                    "Ranked by edge with the threshold switched off, so these are the "
                    "least-bad selections available, not good ones. The model expects "
                    "them to lose. Keep this in paper mode until four weeks of CLV say "
                    "otherwise."
                )
            else:
                st.success(f"Filled slip — expected return {ev:+,.2f} ({pct:+.1%} of stake).")
        st.caption("**Singles only.** Never combine these into an accumulator: doing so "
                   "multiplies the bookmaker's margin.")
        display = slip.selections.copy()
        display["match"] = display["home"] + " v " + display["away"]
        display["pick"] = display["selection"].map(SELECTION_WORDS)
        # Streamlit's "%.1f%%" is a printf format: it appends a percent sign but does
        # not multiply by 100. These columns hold fractions, so they must be scaled
        # here or every probability renders 100x too small.
        for col in ("model_prob", "market_prob_fair", "edge"):
            display[col] = display[col] * 100.0
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
            slip_id, n = log_slip(slip.selections, mode=mode, user_id=_uid())
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
    for col in ("model_prob", "market_prob_fair", "edge"):
        cand[col] = cand[col] * 100.0
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
    theme.hero("Backtest",
               "Walk-forward results on held-out seasons, priced against bet365's own "
               "closing odds. This is the evidence the real-money gate reads.")
    stages_dir = PROJECT_ROOT / "reports" / "stages"
    flat_dir = PROJECT_ROOT / "reports" / "flat"

    stage_csv = stages_dir / "stage_comparison.csv"
    if stage_csv.exists():
        st.subheader("Stage-by-stage comparison")
        table = pd.read_csv(stage_csv)
        for col in ("roi", "roi_lo", "roi_hi", "clv_mean"):
            if col in table.columns:
                table[col] = table[col] * 100.0
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
            lambda g: pd.Series(
                {"bets": len(g), "roi": 100.0 * g["pnl"].sum() / g["stake"].sum()}
            ),
            include_groups=False,
        ).reset_index()
        st.dataframe(by_league, hide_index=True, use_container_width=True,
                     column_config={"roi": st.column_config.NumberColumn(format="%.2f%%")})
    with c2:
        st.subheader("ROI by season")
        by_season = bets.groupby("season").apply(
            lambda g: pd.Series(
                {"bets": len(g), "roi": 100.0 * g["pnl"].sum() / g["stake"].sum()}
            ),
            include_groups=False,
        ).reset_index()
        st.dataframe(by_season, hide_index=True, use_container_width=True,
                     column_config={"roi": st.column_config.NumberColumn(format="%.2f%%")})

    markets_dir = PROJECT_ROOT / "reports" / "markets"
    if (markets_dir / "markets_report.md").exists():
        st.subheader("Other markets")
        st.caption("Over/under 2.5 has real bet365 prices and a real backtest. "
                   "**BTTS has no historical prices anywhere free, so it is not "
                   "validated** — the model produces a probability, but nothing "
                   "measures whether it beats a price.")
        with st.expander("Over/under 2.5 and BTTS report"):
            st.markdown((markets_dir / "markets_report.md").read_text())

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
    theme.hero("Bankroll",
               "Your balance, your bets, and how they are actually doing — judged on "
               "the rolling window and CLV, never on a single month.")
    log = _cached_bet_log(_uid())
    balance = current_bankroll(user_id=_uid())

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
                manual_settle(int(bet_id), status, override or None, user_id=_uid())
                st.cache_data.clear()
                st.success(f"Bet {int(bet_id)} set to {status}.")
            except (KeyError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Bankroll ledger"):
        st.caption("Append-only. The balance is derived from these events, never "
                   "edited in place.")
        st.dataframe(bankroll_history(user_id=_uid()), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        amount = c1.number_input("Deposit / withdrawal", value=0.0, step=10.0)
        if c2.button("Record") and amount:
            record_bankroll_event(
                "deposit" if amount > 0 else "withdrawal", amount, note="manual",
                user_id=_uid(),
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
    theme.hero("Settings", "Thresholds, staking and risk controls. These are yours "
                           "alone; other accounts keep their own.")
    cfg = load_config()
    s = effective_settings(user_id=_uid())

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
        set_setting("paper_mode", bool(paper), user_id=_uid())
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
            set_setting(key, value, user_id=_uid())
        st.cache_data.clear()
        st.success("Saved. These override config.yaml.")


# ---------------------------------------------------------------------------
# Page 5: Ask
# ---------------------------------------------------------------------------

def page_ask():
    theme.hero(
        "Ask",
        "An assistant with read-only access to the database. It can look things up "
        "and change your thresholds. It cannot place, log or settle bets, it cannot "
        "turn off paper trading, and it only ever sees your own bets.",
    )

    usable, reason = chat.is_available()
    if not usable:
        st.info(reason)
        return

    if "ask_messages" not in st.session_state:
        st.session_state.ask_messages = []   # raw API conversation
        st.session_state.ask_display = []    # what gets drawn, turn by turn

    if st.button("Clear conversation"):
        st.session_state.ask_messages = []
        st.session_state.ask_display = []
        st.rerun()

    for entry in st.session_state.ask_display:
        with st.chat_message(entry["role"]):
            if entry.get("tools"):
                with st.expander(f"{len(entry['tools'])} tool call(s)"):
                    for call in entry["tools"]:
                        st.markdown(f"**{call.name}** {'⚠️' if call.error else ''}")
                        if call.input:
                            st.code(
                                call.input.get("sql") or json.dumps(call.input, default=str),
                                language="sql" if "sql" in call.input else "json",
                            )
                        st.caption(str(call.result)[:2000])
            st.markdown(entry["text"])

    prompt = st.chat_input("Ask about your data, or tell me a setting to change")
    if not prompt:
        return

    st.session_state.ask_messages.append({"role": "user", "content": prompt})
    st.session_state.ask_display.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                turn = chat.respond(st.session_state.ask_messages)
            except Exception as exc:
                # A failed call leaves a dangling user message; drop it so the next
                # question starts from a valid conversation rather than a 400.
                st.session_state.ask_messages.pop()
                st.session_state.ask_display.pop()
                st.error(f"The assistant call failed: {exc}")
                return
        st.session_state.ask_display.append(
            {"role": "assistant", "text": turn.text, "tools": turn.tool_calls}
        )
        if any(c.name == "update_setting" and not c.error for c in turn.tool_calls):
            st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Page 6: Account (everyone) and People (owner only)
# ---------------------------------------------------------------------------

def page_account():
    account = _account()
    theme.hero("Account", "Your sign-in details and password.")

    c1, c2 = st.columns(2)
    theme.card("Signed in as", account.display_name, account.email, container=c1)
    theme.card("Role", account.role.title(),
               "Owners can add and disable accounts." if account.is_owner
               else "Members see only their own bets and bankroll.", container=c2)

    st.subheader("Change password")
    with st.form("change_password", clear_on_submit=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        again = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Update password", type="primary"):
            if new != again:
                st.error("The two new passwords do not match.")
            else:
                try:
                    auth.change_password(account.id, current, new)
                    st.success("Password updated.")
                except auth.AuthError as exc:
                    st.error(str(exc))

    st.caption(
        "Signing in is per browser session — refreshing the page signs you out "
        "again. Nothing about your login is stored on this device."
    )


def page_people():
    theme.hero("People", "Accounts on this instance. Each one has its own bankroll, "
                         "bet history and settings — nobody sees anybody else's.")
    if not _account().is_owner:
        st.error("Only the owner can manage accounts.")
        return

    users = auth.list_users()
    st.dataframe(
        pd.DataFrame(users)[["id", "display_name", "email", "role", "is_active",
                             "last_login_at"]].rename(columns={
            "display_name": "name", "is_active": "active", "last_login_at": "last seen"}),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Add someone")
    st.caption("There is no public signup: you create the account and send them the "
               "password yourself.")
    with st.form("add_user", clear_on_submit=True):
        c1, c2 = st.columns(2)
        email = c1.text_input("Email")
        name = c2.text_input("Display name")
        c1, c2 = st.columns(2)
        password = c1.text_input("Temporary password", type="password")
        role = c2.selectbox("Role", ["member", "owner"])
        if st.form_submit_button("Create account", type="primary"):
            try:
                created = auth.create_user(email, password, name or None, role)
                st.success(f"Created {created.email}. Send them that password and ask "
                           "them to change it on the Account page.")
            except auth.AuthError as exc:
                st.error(str(exc))

    st.subheader("Disable or re-enable")
    others = [u for u in users if u["id"] != _account().id]
    if not others:
        st.caption("No other accounts yet.")
    else:
        labels = {u["id"]: f"{u['display_name']} <{u['email']}>" for u in others}
        target = st.selectbox("Account", list(labels), format_func=labels.get)
        active = next(u["is_active"] for u in others if u["id"] == target)
        if st.button("Re-enable" if not active else "Disable"):
            try:
                auth.set_active(target, not active)
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------

def sign_in_screen():
    """The whole app sits behind this. No navigation is built until it passes."""
    # Columns, not a wrapping <div>: Streamlit renders each element in its own
    # container, so an opened div never actually wraps what follows it.
    _, middle, _ = st.columns([1, 1.15, 1])
    with middle:
        st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
        theme.wordmark()

        if auth.user_count() == 0 or _only_local_account():
            st.info(
                "**No accounts yet.** Create the first one from a terminal:\n\n"
                "```\nuv run fv user add you@example.com --owner\n```\n"
                "Then sign in here."
            )
            return

        with st.form("sign_in"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                try:
                    st.session_state.account = auth.authenticate(email, password)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
        st.caption("No public signup — the owner creates accounts.")
    theme.legal_footer()


def _only_local_account() -> bool:
    """True when the only account is the placeholder the CLI created for itself.

    That account has an unusable password hash, so showing a login form for it would
    be a dead end — better to say plainly that a real account has to be made first.
    """
    users = auth.list_users()
    return bool(users) and all(u["email"] == auth.LOCAL_EMAIL for u in users)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def sidebar():
    account = _account()
    settings = effective_settings(user_id=_uid())

    st.sidebar.markdown(
        f"<div style='color:{theme.MUTED};font-size:.8rem'>Signed in as</div>"
        f"<div style='font-weight:600;margin-bottom:.6rem'>{account.display_name}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.metric("Bankroll", f"{current_bankroll(user_id=_uid()):,.2f}")
    st.sidebar.caption(
        "Paper mode — no real money" if settings["paper_mode"]
        else "**REAL MONEY MODE**"
    )
    st.sidebar.divider()
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.pop("account", None)
        st.session_state.pop("ask_messages", None)
        st.session_state.pop("ask_display", None)
        st.cache_data.clear()
        st.rerun()


def main():
    theme.inject()
    ensure_schema()

    if "account" not in st.session_state:
        sign_in_screen()
        return

    # An account disabled while its session is open should lose access on the next
    # click, not at the next sign-in.
    live = auth.get_account(st.session_state.account.id)
    if live is None:
        st.session_state.pop("account", None)
        st.warning("That account is no longer active.")
        sign_in_screen()
        return
    st.session_state.account = live

    sidebar()

    pages = [
        st.Page(page_this_week, title="This Week", icon=":material/sports_soccer:", default=True),
        st.Page(page_backtest, title="Backtest", icon=":material/timeline:"),
        st.Page(page_bankroll, title="Bankroll", icon=":material/account_balance_wallet:"),
        st.Page(page_ask, title="Ask", icon=":material/forum:"),
        st.Page(page_settings, title="Settings", icon=":material/tune:"),
        st.Page(page_account, title="Account", icon=":material/person:"),
    ]
    if live.is_owner:
        pages.append(st.Page(page_people, title="People", icon=":material/group:"))
    st.navigation(pages).run()
    theme.legal_footer()


main()
