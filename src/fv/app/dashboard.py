"""The app.

Six pages, and only one of them is the point: **This week** tells you what to put
on. Everything else exists to answer "should I trust it?" and "how am I doing?".

Two rules run through all of it.

*Plain words.* Nothing on screen assumes the reader has heard of Kelly staking or
closing line value. Where an honest sentence needs a technical idea, the idea gets
explained instead of named. The vocabulary lives in ``plain.py``.

*Never imply more certainty than there is.* Returns carry their range, a month is
labelled as luck, and the real-money switch shows the evidence against itself when
the evidence is against it. Simplifying the language must never simplify away the
part the reader would rather not hear.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from fv import auth, chat
from fv.app import plain, theme
from fv.bets import (
    bankroll_history,
    bet_log,
    combo_legs,
    combo_log,
    current_bankroll,
    log_combo,
    log_slip,
    manual_settle,
    record_bankroll_event,
    rolling_roi,
    settle_all,
)
from fv.config import PROJECT_ROOT, load_config
from fv.db.migrate import ensure_schema
from fv.settings_store import (
    effective_settings,
    get_setting,
    real_money_readiness,
    set_setting,
)
from fv.slip import generate_slip, slip_to_csv, slip_to_text

st.set_page_config(
    page_title="footballvalue",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _uid() -> int:
    """The signed-in account. Every page runs behind the sign-in gate."""
    return st.session_state.account.id


def _account():
    return st.session_state.account


def _money(x: float) -> str:
    return f"€{x:,.2f}"


def _pct(x, digits=1, signed=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:{'+' if signed else ''}.{digits}f}%"


@st.cache_data(ttl=300)
def _cached_combo_log(user_id: int):
    return combo_log(user_id=user_id)


@st.cache_data(ttl=300)
def _cached_bet_log(user_id: int):
    """Keyed on the user id on purpose: an unkeyed cache would hand one account
    another account's history for five minutes."""
    return bet_log(user_id=user_id)


def _league_names() -> dict[str, str]:
    return {lg.code: lg.name for lg in load_config().leagues}


def glossary():
    with st.expander("What do these words mean?"):
        for term, meaning in plain.GLOSSARY:
            st.markdown(f"**{term}** — {meaning}")


# ---------------------------------------------------------------------------
# This week
# ---------------------------------------------------------------------------

def page_this_week():
    settings = effective_settings(user_id=_uid())
    balance = current_bankroll(user_id=_uid())
    names = _league_names()
    practice = bool(settings["paper_mode"])

    theme.hero(
        "This week",
        "Your picks, ready to place by hand at the bookmaker. One bet per match.",
    )

    c1, c2, c3 = st.columns(3)
    theme.card("Your balance", _money(balance), container=c1)
    theme.card("Mode", "Practice" if practice else "REAL MONEY",
               "No real money involved" if practice else "You are staking real money",
               tone="" if practice else "bad", container=c2)

    # How many picks, in one control instead of three.
    always = bool(get_setting("slip_fill_mode", True, user_id=_uid()))
    how_many = int(get_setting("slip_fill_n", 10, user_id=_uid()))
    theme.card("Picks this week", str(how_many) if always else "Only when worth it",
               container=c3)

    rank_by = str(get_setting("rank_by", "value", user_id=_uid()))
    with st.expander("Change what you get"):
        choice = st.radio(
            "Every week…",
            ["Give me my best picks", "Only give me picks that look genuinely worth it"],
            index=0 if always else 1,
            help="The second option is stricter and will often give you nothing at "
                 "all. A week with no bets is a normal outcome, not a fault.",
        )
        always = choice.startswith("Give me")
        if always:
            how_many = int(st.slider("How many picks", 1, 20, how_many))
            set_setting("slip_fill_n", how_many, user_id=_uid())
        set_setting("slip_fill_mode", always, user_id=_uid())

        st.markdown("**Mix of markets**")
        st.caption("Leave both at 0 to just take the best picks whatever the market.")
        m1, m2 = st.columns(2)
        n_goals = int(m1.number_input("Goals picks (over/under 2.5)", 0, 10,
                                      int(get_setting("mix_goals", 0, user_id=_uid()))))
        n_winner = int(m2.number_input("Winner picks (home/draw/away)", 0, 10,
                                       int(get_setting("mix_winner", 0, user_id=_uid()))))
        set_setting("mix_goals", n_goals, user_id=_uid())
        set_setting("mix_winner", n_winner, user_id=_uid())

        order = st.radio(
            "Put in order by",
            ["Best value", "Most likely to win"],
            index=0 if rank_by == "value" else 1,
            help="Most likely to win gives you shorter odds, so more of them come in "
                 "— but each one pays less, and it is not a better bet. The expected "
                 "profit shown below tells you the truth either way.",
        )
        rank_by = "value" if order == "Best value" else "likely"
        set_setting("rank_by", rank_by, user_id=_uid())

    n_goals = int(get_setting("mix_goals", 0, user_id=_uid()))
    n_winner = int(get_setting("mix_winner", 0, user_id=_uid()))
    mix = {"OU25": n_goals, "1X2": n_winner} if (n_goals or n_winner) else None

    with st.spinner("Working out this week's prices…"):
        slip = generate_slip(bankroll=balance, fill_to=how_many if always else None,
                             rank_by=rank_by, market_mix=mix)

    if slip.all_candidates.empty:
        st.info(
            "**No matches to price yet.** Fixtures appear about a week ahead, and only "
            "once the season is under way. Check back in a day or two."
        )
        glossary()
        return

    if slip.selections.empty:
        st.info(
            "**Nothing worth backing this week.** That is the filter doing its job. "
            "A quiet week is normal — switch on *Give me my best picks* above if you "
            "would rather always have something."
        )
        _why_not_table(slip, names)
        glossary()
        return

    # The honest health warning, in plain words, before the table rather than after.
    if slip.mode == "filled" and slip.expected_return < 0:
        st.warning(
            f"**These are the best available, not good ones.** The model expects this "
            f"list to lose about {_money(abs(slip.expected_return))} of the "
            f"{_money(slip.total_stake)} staked. You asked for a set number of picks, "
            f"so it gave you the least-bad {len(slip.selections)}. Keep this in "
            "practice mode."
        )

    st.subheader(f"Your {len(slip.selections)} picks")
    st.caption(f"Total to stake: **{_money(slip.total_stake)}** — "
               f"{slip.total_stake / balance:.0%} of your balance. "
               "**Best placed as separate bets** — see below for exactly what "
               "combining them costs.")

    table = slip.selections.copy()
    table["Match"] = table["home"] + "  v  " + table["away"]
    table["Bet on"] = table["selection"].map(plain.PICK_WORDS)
    table["Market"] = table["market"].map(plain.MARKET_WORDS).fillna(table["market"])
    table["League"] = table["league_code"].map(names).fillna(table["league_code"])
    table["Kick-off"] = pd.to_datetime(table["kickoff_utc"]).dt.strftime("%a %d %b, %H:%M")
    table["Stake"] = table["stake"]
    table["Odds"] = table["odds"]
    table["Returns"] = table["stake"] * table["odds"]

    st.dataframe(
        table[["Kick-off", "League", "Match", "Market", "Bet on", "Odds", "Stake",
               "Returns"]],
        hide_index=True, use_container_width=True,
        column_config={
            "Odds": st.column_config.NumberColumn(format="%.2f"),
            "Stake": st.column_config.NumberColumn(format="€%.2f"),
            "Returns": st.column_config.NumberColumn(
                "Returns if it wins", format="€%.2f",
                help="Your stake back plus the profit."),
        },
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.download_button("Save as text", slip_to_text(slip),
                       file_name=f"picks-{datetime.utcnow():%Y-%m-%d}.txt",
                       use_container_width=True)
    c2.download_button("Save as spreadsheet", slip_to_csv(slip),
                       file_name=f"picks-{datetime.utcnow():%Y-%m-%d}.csv",
                       use_container_width=True)
    label = ("Save these to my history" if practice
             else "Save these as REAL bets")
    if c3.button(label, type="primary", use_container_width=True):
        slip_id, n = log_slip(slip.selections, mode="paper" if practice else "real",
                              user_id=_uid())
        st.cache_data.clear()
        st.success(f"Saved {n} picks. Results fill in automatically as matches finish.")

    if len(slip.selections) > 1:
        _combination_panel(slip)

    with st.expander("Why these matches?"):
        st.caption(
            "A match only appears when the model thinks a result is more likely than "
            "the bookmaker's price suggests. The two columns below are those two "
            "opinions side by side."
        )
        detail = slip.selections.copy()
        detail["Match"] = detail["home"] + " v " + detail["away"]
        detail["Bet on"] = detail["selection"].map(plain.PICK_WORDS)
        if (detail["model_prob"] - detail["market_prob_fair"]).abs().max() < 1e-9:
            st.warning(
                "**In these leagues the two columns are identical, and that is the "
                "honest answer.** Testing showed the model added nothing to the "
                "bookmaker's price here, so it was told to use the price as-is. "
                "These picks are the bookmaker's own opinion, sorted — not an "
                "insight of ours."
            )
        for col in ("model_prob", "market_prob_fair", "edge"):
            detail[col] = detail[col] * 100.0
        st.dataframe(
            detail[["Match", "Bet on", "odds", "model_prob", "market_prob_fair", "edge"]]
            .rename(columns={"odds": "Odds", "model_prob": "Our chance",
                             "market_prob_fair": "Their chance", "edge": "Value"}),
            hide_index=True, use_container_width=True,
            column_config={
                "Odds": st.column_config.NumberColumn(format="%.2f"),
                "Our chance": st.column_config.NumberColumn(format="%.0f%%"),
                "Their chance": st.column_config.NumberColumn(
                    format="%.0f%%", help="The bookmaker's price, with their cut removed."),
                "Value": st.column_config.NumberColumn(
                    format="%+.1f%%", help="How much better than the price the model "
                                           "thinks this is. Negative means worse."),
            },
        )

    _why_not_table(slip, names)
    glossary()


def _combination_panel(slip):
    """What the slip pays as one combined bet, and what that costs.

    This exists because the alternative is worse. Somebody who wants a big return
    from a small stake will build the combination on the bookmaker's site, where
    nothing shows the chance of it landing or how much the combining itself takes.
    Here the same numbers are on screen next to it.
    """
    with st.expander("Combine these into one bet?"):
        stake = st.number_input("Stake on the combined bet (€)", min_value=0.50,
                                value=5.00, step=0.50)
        v = slip.combination_verdict(stake)
        c1, c2, c3 = st.columns(3)
        theme.card("Combined odds", f"{v['odds']:.2f}",
                   f"{_money(stake)} returns {_money(v['returns'])}", container=c1)
        theme.card("Chance all of them land", f"{v['chance']:.1%}",
                   f"about 1 in {v['one_in']:.0f}", container=c2)
        theme.card("Expected profit", _money(v["expected_profit"]),
                   tone="bad" if v["expected_profit"] < 0 else "accent", container=c3)
        st.warning(
            f"**The same {_money(stake)} spread over {len(slip.selections)} separate "
            f"bets has an expected profit of {_money(v['expected_profit_as_singles'])}, "
            f"against {_money(v['expected_profit'])} combined.** Combining does not "
            "change any single prediction — it multiplies the bookmaker's cut by the "
            f"number of legs. As singles you would expect about "
            f"{v['expected_winners_as_singles']:.1f} of {len(slip.selections)} to come "
            "in and get paid on each; combined, one loser pays nothing at all."
        )


def _why_not_table(slip, names):
    """Everything considered and left out, with a reason a person can act on."""
    if slip.no_edge_possible_leagues:
        left_out = ", ".join(names.get(c, c) for c in slip.no_edge_possible_leagues)
        st.warning(
            f"**{left_out} will never produce a pick.** In these leagues the model has "
            "not shown it can price matches better than the bookmaker, so testing told "
            "it to simply copy the bookmaker's price — and copying the price always "
            "loses by the size of the bookmaker's cut. This is honest, not broken."
        )

    with st.expander("Every match we looked at"):
        cand = slip.all_candidates.copy()
        cand["Match"] = cand["home"] + " v " + cand["away"]
        cand["Bet on"] = cand["selection"].map(plain.PICK_WORDS)
        cand["League"] = cand["league_code"].map(names).fillna(cand["league_code"])
        cand["Market"] = cand["market"].map(plain.MARKET_WORDS).fillna(cand["market"])
        qualified = cand["in_odds_range"] & cand["clears_edge"] & cand["enough_history"]

        def why(r, ok):
            if ok:
                return "On your list"
            bits = []
            if not r.in_odds_range:
                bits.append("odds outside your range")
            if not r.clears_edge:
                bits.append("not enough value")
            if not r.enough_history:
                bits.append("too little history on these teams")
            return "; ".join(bits)

        cand["Why not"] = [why(r, ok) for r, ok in
                           zip(cand.itertuples(index=False), qualified, strict=True)]
        cand["Value"] = cand["edge"] * 100.0
        chosen = st.multiselect("Leagues", sorted(cand["League"].unique()),
                                default=sorted(cand["League"].unique()))
        view = cand[cand["League"].isin(chosen)].sort_values("edge", ascending=False)
        st.dataframe(
            view[["League", "Match", "Market", "Bet on", "odds", "Value", "Why not"]]
            .rename(columns={"odds": "Odds"}),
            hide_index=True, use_container_width=True,
            column_config={
                "Odds": st.column_config.NumberColumn(format="%.2f"),
                "Value": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------

def page_track_record():
    theme.hero(
        "Track record",
        "How the model would have done on seasons it had never seen. This is the "
        "evidence for or against trusting it with real money.",
    )

    stages_csv = PROJECT_ROOT / "reports" / "stages" / "stage_comparison.csv"
    if not stages_csv.exists():
        st.info("No test results yet. Ask whoever set this up to run the season test.")
        return

    table = pd.read_csv(stages_csv)
    r = real_money_readiness(user_id=_uid())
    best = table.sort_values("log_loss").iloc[0]

    if r.backtest_beats_closing:
        st.success("**In testing, the model priced matches better than the bookmaker.**")
    else:
        st.error(
            "**In testing, the model did not price matches better than the "
            "bookmaker.** Everything else on this page follows from that. It is why "
            "the app stays in practice mode, and it is not something a good few weeks "
            "can overturn."
        )

    c1, c2, c3 = st.columns(3)
    roi = float(best["roi"])
    lo, hi = float(best["roi_lo"]), float(best["roi_hi"])
    theme.card(
        "Return in testing", _pct(roi),
        f"From {int(best['bets']):,} bets — and anywhere between {_pct(lo)} and "
        f"{_pct(hi)} would have looked the same. The range is the honest answer.",
        tone="bad" if hi < 0 else "warn", container=c1,
    )
    theme.card("Matches priced", f"{int(best['n']):,}",
               "Every one on a season the model had never seen.", container=c2)
    theme.card("Beat the closing price?",
               "Yes" if bool(best["clv_significant"]) else "No",
               "Regularly taking a better price than the bookmaker's final one is the "
               "earliest sign of a real advantage. There isn't one here.",
               tone="bad" if not bool(best["clv_significant"]) else "accent",
               container=c3)

    st.subheader("What each version of the model achieved")
    st.caption("Each row adds something to the one above it. More complicated did not "
               "turn out to mean better.")
    # The bookmaker's own score, backed out of the gap the readiness gate computed for
    # the best stage. Comparing each row against it is what turns a log-loss column
    # nobody can read into a yes/no anybody can.
    market_ll = (float(best["log_loss"]) - r.backtest_gap
                 if r.backtest_gap is not None else None)
    show = table.copy()
    show["Version"] = show["stage"].map(plain.STAGE_NAMES).fillna(show["stage"])
    show["Return"] = show["roi"] * 100.0
    show["Better than the bookmaker"] = (
        show["log_loss"] < market_ll if market_ll is not None else False
    )
    st.dataframe(
        show[["Version", "bets", "Return", "Better than the bookmaker"]]
        .rename(columns={"bets": "Bets"}),
        hide_index=True, use_container_width=True,
        column_config={"Return": st.column_config.NumberColumn(format="%+.2f%%")},
    )

    flat = PROJECT_ROOT / "reports" / "flat" / "bets.csv"
    if flat.exists():
        bets = pd.read_csv(flat, parse_dates=["kickoff_utc"])
        st.subheader("How a €10 bet on every pick would have gone")
        curve = bets.sort_values("kickoff_utc").copy()
        curve["Running profit (€)"] = curve["pnl"].cumsum()
        st.line_chart(curve.set_index("kickoff_utc")["Running profit (€)"])
        worst = (curve["Running profit (€)"] - curve["Running profit (€)"].cummax()).min()
        st.caption(
            f"Worst losing run: **{_money(abs(worst))}** below the best point it had "
            "reached. Any real betting plan has to survive a stretch like that."
        )

        with st.expander("More detail"):
            st.caption("By league and by season, same €10 bets.")
            c1, c2 = st.columns(2)
            names = _league_names()
            by_league = bets.groupby("league_code").apply(
                lambda g: pd.Series({"Bets": len(g),
                                     "Return": 100.0 * g["pnl"].sum() / g["stake"].sum()}),
                include_groups=False).reset_index()
            by_league["League"] = by_league["league_code"].map(names)
            c1.dataframe(by_league[["League", "Bets", "Return"]], hide_index=True,
                         use_container_width=True,
                         column_config={"Return": st.column_config.NumberColumn(
                             format="%+.2f%%")})
            by_season = bets.groupby("season").apply(
                lambda g: pd.Series({"Bets": len(g),
                                     "Return": 100.0 * g["pnl"].sum() / g["stake"].sum()}),
                include_groups=False).reset_index()
            c2.dataframe(by_season.rename(columns={"season": "Season"}), hide_index=True,
                         use_container_width=True,
                         column_config={"Return": st.column_config.NumberColumn(
                             format="%+.2f%%")})

    md = stages_csv.with_suffix(".md")
    if md.exists():
        with st.expander("The full technical report"):
            st.markdown(md.read_text())
    glossary()


# ---------------------------------------------------------------------------
# My results
# ---------------------------------------------------------------------------

def page_results():
    theme.hero("My results", "Every pick you have saved, and how they turned out.")
    log = _cached_bet_log(_uid())
    combos = _cached_combo_log(_uid())
    balance = current_bankroll(user_id=_uid())

    if log.empty and combos.empty:
        theme.card("Your balance", _money(balance), container=st)
        st.info("Nothing saved yet. Go to **This week**, then press *Save these to my "
                "history*. Results fill in on their own once the matches are played.")
        glossary()
        return

    done = log[log["status"].isin(("won", "lost"))] if not log.empty else log
    done_combos = (combos[combos["status"].isin(("won", "lost"))]
                   if not combos.empty else combos)

    # Both kinds of bet count once each. A combined bet is one bet, however many
    # matches it rides on.
    n_bets = len(log) + len(combos)
    staked = (done["stake"].sum() if not done.empty else 0.0) + (
        done_combos["stake"].sum() if not done_combos.empty else 0.0)
    profit = (done["pnl"].sum() if not done.empty else 0.0) + (
        done_combos["pnl"].sum() if not done_combos.empty else 0.0)
    pending = (int((log["status"] == "pending").sum()) if not log.empty else 0) + (
        int((combos["status"] == "pending").sum()) if not combos.empty else 0)
    n_done = len(done) + len(done_combos)
    n_won = (int((done["status"] == "won").sum()) if not done.empty else 0) + (
        int((done_combos["status"] == "won").sum()) if not done_combos.empty else 0)

    c1, c2, c3, c4 = st.columns(4)
    theme.card("Your balance", _money(balance), container=c1)
    theme.card("Bets saved", f"{n_bets:,}", f"{pending} still to play", container=c2)
    if n_done:
        theme.card("Profit so far", _money(profit), f"from {_money(staked)} staked",
                   tone="bad" if profit < 0 else "accent", container=c3)
        theme.card("Won", f"{n_won} of {n_done}", container=c4)

    c1, c2 = st.columns([1, 3])
    if c1.button("Check for results", use_container_width=True):
        stats = settle_all()
        st.cache_data.clear()
        c2.success(f"Updated {stats.settled} bet(s). {stats.still_pending} still to play.")

    if not combos.empty:
        st.subheader("Combined bets")
        st.caption("One stake riding on several matches. It pays only if every leg "
                   "lands — a voided match drops out and shortens the price.")
        view = combos.copy()
        view["Placed"] = view["placed_at"].dt.strftime("%a %d %b")
        view["Legs"] = (view["legs_won"].fillna(0).astype(int).astype(str) + " of "
                        + view["legs"].astype(int).astype(str) + " landed")
        view["Result"] = view["status"].map(
            {"won": "Won", "lost": "Lost", "void": "Void", "pending": "Waiting"})
        st.dataframe(
            view[["id", "Placed", "combined_odds", "stake", "Legs", "Result", "pnl",
                  "mode"]].rename(columns={"id": "#", "combined_odds": "Odds",
                                           "stake": "Stake", "pnl": "Profit",
                                           "mode": "Real or practice"}),
            hide_index=True, use_container_width=True,
            column_config={
                "Odds": st.column_config.NumberColumn(format="%.2f"),
                "Stake": st.column_config.NumberColumn(format="€%.2f"),
                "Profit": st.column_config.NumberColumn(format="€%.2f"),
            },
        )
        legs = combo_legs([int(i) for i in combos["id"]])
        if not legs.empty:
            with st.expander("What was in each one"):
                legs["Match"] = legs["home"] + " v " + legs["away"]
                legs["Bet on"] = legs["selection"].map(plain.PICK_WORDS)
                legs["Score"] = [f"{int(h)}-{int(a)}" if pd.notna(h) else "-"
                                 for h, a in zip(legs["fthg"], legs["ftag"], strict=True)]
                legs["Landed"] = legs["result"].map(
                    {"won": "Yes", "lost": "No", "void": "Void"}).fillna("Waiting")
                st.dataframe(
                    legs[["combo_id", "Match", "Bet on", "odds_taken", "Score", "Landed"]]
                    .rename(columns={"combo_id": "Bet #", "odds_taken": "Odds"}),
                    hide_index=True, use_container_width=True,
                    column_config={"Odds": st.column_config.NumberColumn(format="%.2f")},
                )

    _record_bet_form()

    if log.empty:
        glossary()
        return

    st.subheader("Your single bets")
    view = log.copy()
    view["Match"] = view["home"] + " v " + view["away"]
    view["Bet on"] = view["selection"].map(plain.PICK_WORDS)
    view["Market"] = view["market"].map(plain.MARKET_WORDS).fillna(view["market"])
    view["Score"] = [f"{int(h)}-{int(a)}" if pd.notna(h) else "not played yet"
                     for h, a in zip(view["fthg"], view["ftag"], strict=True)]
    view["Result"] = view["status"].map(
        {"won": "Won", "lost": "Lost", "void": "Void", "pending": "Waiting"})
    view["Kick-off"] = view["kickoff_utc"].dt.strftime("%a %d %b")
    st.dataframe(
        view[["Kick-off", "Match", "Market", "Bet on", "odds_taken", "stake", "Score",
              "Result", "pnl"]].rename(columns={"odds_taken": "Odds", "stake": "Stake",
                                      "pnl": "Profit"}),
        hide_index=True, use_container_width=True,
        column_config={
            "Odds": st.column_config.NumberColumn(format="%.2f"),
            "Stake": st.column_config.NumberColumn(format="€%.2f"),
            "Profit": st.column_config.NumberColumn(format="€%.2f"),
        },
    )

    if len(done) >= 30:
        st.subheader("Are you actually winning?")
        st.warning(
            "**Not enough bets can tell you.** Over 300 bets, luck alone moves your "
            "return by around 7 percentage points either way — so a run of good or bad "
            "results proves very little. The shaded range below is the honest answer; "
            "the line in the middle is not."
        )
        roll = rolling_roi(log, window=300)
        if not roll.empty:
            chart = roll.copy()
            chart["date"] = pd.to_datetime(chart["kickoff_utc"]).dt.normalize()
            chart = chart.rename(columns={"roi": "Return", "ci_low": "Could be as low as",
                                          "ci_high": "Could be as high as"})
            for col in ("Return", "Could be as low as", "Could be as high as"):
                chart[col] = chart[col] * 100
            st.line_chart(chart.set_index("date")[
                ["Return", "Could be as low as", "Could be as high as"]])
            last = roll.iloc[-1]
            st.caption(
                f"Latest: **{_pct(last['roi'])}** over {int(last['n'])} picks, but "
                f"anywhere from {_pct(last['ci_low'])} to {_pct(last['ci_high'])} would "
                "look the same."
                + ("  That range includes zero, so this is not yet evidence of anything."
                   if last["ci_low"] < 0 < last["ci_high"] else "")
            )

    with st.expander("Month by month"):
        st.caption("**This is luck, not skill.** At around 30 picks a month, even a "
                   "genuinely good model has a losing month about four times in ten. "
                   "Do not change anything because of one bad month.")
        if not done.empty:
            m = done.copy()
            m["Month"] = m["kickoff_utc"].dt.to_period("M").astype(str)
            monthly = m.groupby("Month").agg(Picks=("pnl", "size"),
                                             Staked=("stake", "sum"),
                                             Profit=("pnl", "sum")).reset_index()
            st.bar_chart(monthly.set_index("Month")["Profit"])
            st.dataframe(monthly, hide_index=True, use_container_width=True)

    with st.expander("Money in and out"):
        st.caption("Add money you have deposited, or take out what you have withdrawn.")
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount (negative to take out)", value=0.0, step=10.0)
        if c2.button("Record it") and amount:
            record_bankroll_event("deposit" if amount > 0 else "withdrawal", amount,
                                  note="manual", user_id=_uid())
            st.cache_data.clear()
            st.rerun()
        st.dataframe(bankroll_history(user_id=_uid()), hide_index=True,
                     use_container_width=True)

    with st.expander("Fix a result by hand"):
        st.caption("For the cases the app cannot see: a match voided, a bet cashed out "
                   "early, or the bookmaker correcting something.")
        bet_id = st.number_input("Pick number (from the table above)", min_value=1, step=1)
        status = st.selectbox("What happened", ["won", "lost", "void", "pending"],
                              format_func=lambda s: {"won": "It won", "lost": "It lost",
                                                     "void": "Voided — stake returned",
                                                     "pending": "Not settled yet"}[s])
        override = st.number_input("Profit, if it was not the usual amount", value=0.0)
        if st.button("Save this correction"):
            try:
                manual_settle(int(bet_id), status, override or None, user_id=_uid())
                st.cache_data.clear()
                st.success("Saved.")
            except (KeyError, ValueError) as exc:
                st.error(str(exc))
    glossary()


RECENT_MATCH_QUERY = """
SELECT m.id, m.league_code, m.kickoff_utc, m.status,
       th.canonical_name AS home, ta.canonical_name AS away
FROM matches m
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
WHERE m.kickoff_utc >= :since
ORDER BY m.kickoff_utc DESC
LIMIT 300
"""


@st.cache_data(ttl=300)
def _selectable_matches():
    """Recent and upcoming fixtures, for entering a bet you placed yourself."""
    from sqlalchemy import text as sql_text

    from fv.db.session import get_engine

    since = (datetime.utcnow() - pd.Timedelta(days=14)).isoformat(sep=" ")
    with get_engine(load_config()).connect() as conn:
        df = pd.read_sql(sql_text(RECENT_MATCH_QUERY), conn, params={"since": since})
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
        df["label"] = (df["kickoff_utc"].dt.strftime("%a %d %b") + " \u00b7 "
                       + df["home"] + " v " + df["away"])
    return df


def _record_bet_form():
    """Enter a bet placed at the bookmaker by hand.

    The app cannot see your bookmaker account, so anything placed outside the
    *This week* button is invisible to it - and a history with holes in it is worse
    than no history, because the return on screen looks real. This is how to fill
    them in.
    """
    with st.expander("Record a bet you placed yourself"):
        matches = _selectable_matches()
        if matches.empty:
            st.caption("No fixtures loaded yet. Run `uv run fv refresh` first.")
            return

        if "draft_legs" not in st.session_state:
            st.session_state.draft_legs = []

        labels = dict(zip(matches["id"], matches["label"], strict=True))
        c1, c2, c3 = st.columns([3, 2, 1])
        match_id = c1.selectbox("Match", list(labels), format_func=labels.get)
        pick = c2.selectbox("What you backed", ["H", "D", "A", "O", "U"],
                            format_func=lambda s: plain.PICK_WORDS[s])
        odds = c3.number_input("Odds", min_value=1.01, value=2.00, step=0.01, format="%.2f")
        if st.button("Add to this bet"):
            st.session_state.draft_legs.append({
                "match_id": int(match_id),
                "market": "OU25" if pick in ("O", "U") else "1X2",
                "selection": pick,
                "odds": float(odds),
                "label": labels[match_id],
            })
            st.rerun()

        if not st.session_state.draft_legs:
            st.caption("Add one selection for a single bet, or several for a combined one.")
            return

        draft = pd.DataFrame(st.session_state.draft_legs)
        st.dataframe(
            draft[["label", "selection", "odds"]]
            .assign(selection=draft["selection"].map(plain.PICK_WORDS))
            .rename(columns={"label": "Match", "selection": "Bet on", "odds": "Odds"}),
            hide_index=True, use_container_width=True,
        )
        product = float(draft["odds"].prod())
        if st.button("Clear these"):
            st.session_state.draft_legs = []
            st.rerun()

        c1, c2, c3 = st.columns(3)
        stake = c1.number_input("Total stake (\u20ac)", min_value=0.01, value=5.00, step=0.50)
        placed = c2.date_input("Date placed", value=datetime.utcnow().date())
        real = c3.selectbox("Real money?", [True, False],
                            format_func=lambda b: "Real money" if b else "Practice")

        if len(draft) > 1:
            combined = st.number_input(
                "Combined odds", min_value=1.01, value=round(product, 2), step=0.01,
                format="%.2f",
                help="Defaults to the legs multiplied together. Change it to whatever "
                     "your betting slip actually says - bookmakers round.",
            )
            st.caption(f"{_money(stake)} at {combined:.2f} returns "
                       f"{_money(stake * combined)} if every leg lands. One leg missing "
                       "and it returns nothing.")
            if st.button("Save this combined bet", type="primary"):
                log_combo(draft, stake=stake, combined_odds=float(combined),
                          mode="real" if real else "paper",
                          placed_at=datetime.combine(placed, datetime.min.time()),
                          notes="entered by hand", user_id=_uid())
                st.session_state.draft_legs = []
                st.cache_data.clear()
                st.rerun()
        elif st.button("Save this single bet", type="primary"):
            log_slip(draft.assign(stake=stake), mode="real" if real else "paper",
                     user_id=_uid())
            st.session_state.draft_legs = []
            st.cache_data.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def page_settings():
    cfg = load_config()
    s = effective_settings(user_id=_uid())
    theme.hero("Settings", "Yours alone — other people using this app keep their own.")

    st.subheader("Real money")
    r = real_money_readiness(user_id=_uid())
    if r.ready:
        st.success("**The evidence supports betting real money.**")
    else:
        st.error("**The evidence does not support betting real money yet.**")
    for line in plain.readiness_lines(r):
        st.markdown(line)

    practice = st.toggle("Practice mode — no real money", value=bool(s["paper_mode"]))
    if not practice and not r.ready:
        st.error(
            "Switching this off means staking real money on a model that has not "
            "passed its own tests. You can do it. Do it knowing that."
        )
    if st.button("Save"):
        set_setting("paper_mode", bool(practice), user_id=_uid())
        st.success("Saved.")

    st.divider()
    st.subheader("How much to stake")
    current_style = plain.style_of(s)
    options = list(plain.STAKING_STYLES)
    style = st.radio(
        "Style",
        options,
        index=options.index(current_style) if current_style in options else 1,
        format_func=lambda name: f"{name} — {plain.STAKING_STYLES[name]['blurb']}",
        label_visibility="collapsed",
    )
    if current_style == "Custom":
        st.caption("Your current numbers do not match any of these three. Picking one "
                   "will replace them.")

    balance = st.number_input("Starting balance (€)", value=float(s["starting_bankroll"]),
                              min_value=1.0, step=50.0)

    st.subheader("Leagues")
    labels = {lg.code: lg.name for lg in cfg.leagues}
    leagues = st.multiselect("Which leagues to look at", list(labels),
                             default=list(s["enabled_leagues"]),
                             format_func=lambda c: labels.get(c, c))

    if st.button("Save settings", type="primary"):
        preset = {k: v for k, v in plain.STAKING_STYLES[style].items() if k != "blurb"}
        for key, value in {**preset, "starting_bankroll": balance,
                           "enabled_leagues": leagues}.items():
            set_setting(key, value, user_id=_uid())
        st.cache_data.clear()
        st.success("Saved.")

    with st.expander("Advanced — the individual numbers"):
        st.caption("The three styles above are shortcuts for these. Change them here if "
                   "you know what you want; it will show as *Custom*.")
        c1, c2 = st.columns(2)
        kelly = c1.slider(
            "Share of the mathematically optimal stake", 0.05, 1.0,
            float(s["kelly_fraction"]), 0.05,
            help="Staking the full optimal amount assumes the model's probabilities "
                 "are exactly right. They are estimates, and overstaking compounds "
                 "badly — twice the optimal amount grows your money not at all.")
        max_stake = c1.slider("Most to risk on one match (share of balance)",
                              0.005, 0.10, float(s["max_stake_pct"]), 0.005,
                              format="%.3f")
        min_value = c2.slider("Minimum value before a match qualifies", 0.0, 0.20,
                              float(s["min_edge"]), 0.005, format="%.3f")
        odds = c2.slider("Only suggest odds between", 1.0, 10.0,
                         (float(s["min_odds"]), float(s["max_odds"])), 0.05)
        max_bets = c1.number_input("Most picks in a week", value=int(s["max_bets_per_week"]),
                                   min_value=1, max_value=50)
        stop_loss = c2.slider("Stop for the week after losing this share of the balance",
                              0.0, 0.50, float(s["weekly_stop_loss_pct"]), 0.01)
        max_dd = c2.slider("Pause everything after falling this far from your peak",
                           0.05, 0.75, float(s["max_drawdown_pct"]), 0.01)
        if st.button("Save these numbers"):
            for key, value in {
                "kelly_fraction": kelly, "max_stake_pct": max_stake,
                "min_edge": min_value, "min_odds": odds[0], "max_odds": odds[1],
                "max_bets_per_week": int(max_bets),
                "weekly_stop_loss_pct": stop_loss, "max_drawdown_pct": max_dd,
            }.items():
                set_setting(key, value, user_id=_uid())
            st.cache_data.clear()
            st.success("Saved.")
    glossary()


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

def page_ask():
    theme.hero(
        "Ask",
        "Ask anything about your picks, your results, or the data behind them. It can "
        "also change your settings for you. It cannot place bets, and it cannot turn "
        "off practice mode.",
    )

    usable, reason = chat.is_available()
    if not usable:
        st.info(reason)
        return

    if "ask_messages" not in st.session_state:
        st.session_state.ask_messages = []
        st.session_state.ask_display = []

    if st.session_state.ask_display:
        if st.button("Start over"):
            st.session_state.ask_messages = []
            st.session_state.ask_display = []
            st.rerun()
    else:
        st.caption("Try: *how did I do last month?* · *which league gives me the most "
                   "picks?* · *make me more careful*")

    for entry in st.session_state.ask_display:
        with st.chat_message(entry["role"]):
            if entry.get("tools"):
                with st.expander(f"Looked up {len(entry['tools'])} thing(s)"):
                    for call in entry["tools"]:
                        st.markdown(f"**{call.name}** {'⚠️' if call.error else ''}")
                        if call.input:
                            st.code(call.input.get("sql")
                                    or json.dumps(call.input, default=str),
                                    language="sql" if "sql" in call.input else "json")
                        st.caption(str(call.result)[:2000])
            st.markdown(entry["text"])

    prompt = st.chat_input("Ask a question, or tell it what to change")
    if not prompt:
        return

    st.session_state.ask_messages.append({"role": "user", "content": prompt})
    st.session_state.ask_display.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            turn = chat.respond(st.session_state.ask_messages, user_id=_uid())
        except Exception as exc:
            # A failed call leaves a dangling question; drop it so the next one
            # starts from a valid conversation rather than an error.
            st.session_state.ask_messages.pop()
            st.session_state.ask_display.pop()
            st.error(f"That didn't work: {exc}")
            return
    st.session_state.ask_display.append(
        {"role": "assistant", "text": turn.text, "tools": turn.tool_calls})
    if any(c.name == "update_setting" and not c.error for c in turn.tool_calls):
        st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Account and People
# ---------------------------------------------------------------------------

def page_account():
    account = _account()
    theme.hero("Account", "Your sign-in details.")

    c1, c2 = st.columns(2)
    theme.card("Signed in as", account.display_name, account.email, container=c1)
    theme.card("Role", "Owner" if account.is_owner else "Member",
               "You can add and remove people." if account.is_owner
               else "You see only your own picks and balance.", container=c2)

    st.subheader("Change your password")
    with st.form("change_password", clear_on_submit=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        again = st.text_input("New password again", type="password")
        if st.form_submit_button("Update", type="primary"):
            if new != again:
                st.error("The two new passwords are not the same.")
            else:
                try:
                    auth.change_password(account.id, current, new)
                    st.success("Done.")
                except auth.AuthError as exc:
                    st.error(str(exc))

    st.caption("Refreshing the page signs you out. Nothing about your login is kept "
               "on this device.")


def page_people():
    theme.hero("People", "Everyone who can sign in. Each person has their own balance, "
                         "picks and settings — nobody sees anybody else's.")
    if not _account().is_owner:
        st.error("Only the owner can manage people.")
        return

    users = auth.list_users()
    show = pd.DataFrame(users)
    show["Role"] = show["role"].str.title()
    show["Can sign in"] = show["is_active"]
    st.dataframe(
        show[["display_name", "email", "Role", "Can sign in", "last_login_at"]]
        .rename(columns={"display_name": "Name", "email": "Email",
                         "last_login_at": "Last seen"}),
        hide_index=True, use_container_width=True,
    )

    st.subheader("Invite someone")
    st.caption("You create the account and send them the password yourself — there is "
               "no public sign-up.")
    with st.form("add_user", clear_on_submit=True):
        c1, c2 = st.columns(2)
        email = c1.text_input("Their email")
        name = c2.text_input("Their name")
        c1, c2 = st.columns(2)
        password = c1.text_input("A password to give them", type="password")
        role = c2.selectbox("Role", ["member", "owner"],
                            format_func=lambda r: "Member" if r == "member" else "Owner")
        if st.form_submit_button("Create", type="primary"):
            try:
                created = auth.create_user(email, password, name or None, role)
                st.success(f"Created {created.email}. Send them that password and ask "
                           "them to change it on their Account page.")
            except auth.AuthError as exc:
                st.error(str(exc))

    st.subheader("Remove access")
    others = [u for u in users if u["id"] != _account().id]
    if not others:
        st.caption("Nobody else yet.")
        return
    labels = {u["id"]: f"{u['display_name']} ({u['email']})" for u in others}
    target = st.selectbox("Who", list(labels), format_func=labels.get)
    active = next(u["is_active"] for u in others if u["id"] == target)
    if st.button("Let them back in" if not active else "Stop them signing in"):
        try:
            auth.set_active(target, not active)
            st.rerun()
        except auth.AuthError as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------------
# Getting in
# ---------------------------------------------------------------------------

def _needs_first_account() -> bool:
    """No real account yet — only the placeholder the command line owns, if any.

    That placeholder has an unusable password, so offering a sign-in form for it
    would be a dead end. Offer to make a real account instead.
    """
    users = auth.list_users()
    return not users or all(u["email"] == auth.LOCAL_EMAIL for u in users)


def first_run_screen():
    """Set-up, in the browser. Nobody should need a terminal to get an account.

    Only reachable while no real account exists, so it cannot become a public
    sign-up route: the moment the first owner is created, this screen is gone for
    good and further accounts come from that owner.
    """
    _, middle, _ = st.columns([1, 1.15, 1])
    with middle:
        st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
        theme.wordmark()
        st.subheader("Set up your account")
        st.caption("This is the first account, so it will be the owner. Nobody else "
                   "can sign up on their own — you invite them.")
        with st.form("first_account"):
            name = st.text_input("Your name")
            email = st.text_input("Email")
            password = st.text_input("Choose a password", type="password",
                                     help="At least 10 characters. Make it up — there "
                                          "is nothing to look up.")
            again = st.text_input("Type it again", type="password")
            if st.form_submit_button("Create my account", type="primary",
                                     use_container_width=True):
                if password != again:
                    st.error("The two passwords are not the same.")
                else:
                    try:
                        st.session_state.account = auth.create_user(
                            email, password, name or None, "owner")
                        st.rerun()
                    except auth.AuthError as exc:
                        st.error(str(exc))
    theme.legal_footer()


def sign_in_screen():
    # Columns, not a wrapping <div>: Streamlit renders each element in its own
    # container, so an opened div never actually wraps what follows it.
    _, middle, _ = st.columns([1, 1.15, 1])
    with middle:
        st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
        theme.wordmark()
        with st.form("sign_in"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                try:
                    st.session_state.account = auth.authenticate(email, password)
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))
        st.caption("Forgotten it? Ask whoever set this up to give you a new one.")
    theme.legal_footer()


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
    st.sidebar.metric("Balance", _money(current_bankroll(user_id=_uid())))
    st.sidebar.caption("Practice mode — no real money" if settings["paper_mode"]
                       else "**REAL MONEY**")
    st.sidebar.divider()
    if st.sidebar.button("Sign out", use_container_width=True):
        for key in ("account", "ask_messages", "ask_display"):
            st.session_state.pop(key, None)
        st.cache_data.clear()
        st.rerun()


def main():
    theme.inject()
    ensure_schema()

    if "account" not in st.session_state:
        first_run_screen() if _needs_first_account() else sign_in_screen()
        return

    # Someone whose access is removed mid-session should lose it on the next click,
    # not at the next sign-in.
    live = auth.get_account(st.session_state.account.id)
    if live is None:
        st.session_state.pop("account", None)
        st.warning("That account can no longer sign in.")
        sign_in_screen()
        return
    st.session_state.account = live

    sidebar()
    pages = [
        st.Page(page_this_week, title="This week", icon=":material/sports_soccer:",
                default=True),
        st.Page(page_results, title="My results", icon=":material/receipt_long:"),
        st.Page(page_track_record, title="Track record", icon=":material/timeline:"),
        st.Page(page_ask, title="Ask", icon=":material/forum:"),
        st.Page(page_settings, title="Settings", icon=":material/tune:"),
        st.Page(page_account, title="Account", icon=":material/person:"),
    ]
    if live.is_owner:
        pages.append(st.Page(page_people, title="People", icon=":material/group:"))
    st.navigation(pages).run()
    theme.legal_footer()


main()
