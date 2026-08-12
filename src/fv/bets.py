"""Bet logging, settlement, and the bankroll ledger.

The bankroll is derived from an append-only ledger rather than stored as a mutable
number. That makes every balance explainable — you can always point at the events
that produced it — and it means a settlement bug can be corrected by appending a
adjustment rather than by editing history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import bindparam, func, select, text

from fv.auth import local_user_id
from fv.config import Config, load_config
from fv.db.models import BankrollEvent, Bet, ComboBet, ComboLeg, Match
from fv.db.session import get_engine, session_scope
from fv.odds.settlement import clv as clv_of
from fv.odds.settlement import settle_bet


def new_slip_id() -> str:
    return f"slip-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"


def current_bankroll(cfg: Config | None = None, user_id: int | None = None) -> float:
    """Balance from one user's ledger; falls back to the configured starting bankroll.

    Every bankroll function takes ``user_id=None`` to mean "the account the command
    line acts as", so scripts and tests keep working unchanged while the web app
    passes the signed-in user explicitly.
    """
    cfg = cfg or load_config()
    uid = user_id if user_id is not None else local_user_id(cfg)
    with session_scope(cfg) as s:
        last = s.scalar(
            select(BankrollEvent)
            .where(BankrollEvent.user_id == uid)
            .order_by(BankrollEvent.ts.desc(), BankrollEvent.id.desc())
            .limit(1)
        )
        if last is not None:
            return float(last.balance_after)
    # Before any ledger entry the balance is whatever the user said they started
    # with. Reading only config.yaml here ignored the figure they set on the
    # Settings page, so the number on screen was not the number they chose.
    from fv.settings_store import get_setting

    default = float(cfg.betting.get("starting_bankroll", 1000.0))
    return float(get_setting("starting_bankroll", default, cfg, uid))


def record_bankroll_event(
    event_type: str,
    amount: float,
    note: str | None = None,
    bet_id: int | None = None,
    cfg: Config | None = None,
    user_id: int | None = None,
) -> float:
    """Append a ledger entry and return the new balance."""
    cfg = cfg or load_config()
    uid = user_id if user_id is not None else local_user_id(cfg)
    balance = current_bankroll(cfg, uid) + amount
    with session_scope(cfg) as s:
        s.add(
            BankrollEvent(
                user_id=uid,
                ts=datetime.utcnow(),
                type=event_type,
                amount=float(amount),
                balance_after=float(balance),
                bet_id=bet_id,
                note=note,
            )
        )
    return balance


def log_slip(
    selections: pd.DataFrame,
    mode: str = "paper",
    slip_id: str | None = None,
    cfg: Config | None = None,
    user_id: int | None = None,
) -> tuple[str, int]:
    """Record a slip's selections as pending bets. Returns (slip_id, count).

    Staking money does not move the bankroll here: the ledger records the *result*
    when the bet settles. Tracking stake-out and return separately would double-count
    every winning bet's stake.
    """
    cfg = cfg or load_config()
    uid = user_id if user_id is not None else local_user_id(cfg)
    slip_id = slip_id or new_slip_id()
    if selections.empty:
        return slip_id, 0

    n = 0
    with session_scope(cfg) as s:
        for r in selections.itertuples(index=False):
            market = getattr(r, "market", "1X2")
            already = s.scalar(
                select(Bet).where(
                    Bet.user_id == uid,
                    Bet.match_id == int(r.match_id),
                    Bet.market == market,
                    Bet.selection == r.selection,
                    Bet.status == "pending",
                )
            )
            if already is not None:
                continue  # never log the same pending bet twice
            s.add(
                Bet(
                    user_id=uid,
                    slip_id=slip_id,
                    match_id=int(r.match_id),
                    market=market,
                    selection=r.selection,
                    odds_taken=float(r.odds),
                    stake=float(r.stake),
                    mode=mode,
                    placed_at=datetime.utcnow(),
                    status="pending",
                )
            )
            n += 1
    return slip_id, n


@dataclass
class SettlementStats:
    settled: int = 0
    won: int = 0
    lost: int = 0
    void: int = 0
    pnl: float = 0.0
    still_pending: int = 0

    def summary(self) -> str:
        return (
            f"settled {self.settled} ({self.won}W/{self.lost}L/{self.void}V), "
            f"P&L {self.pnl:+,.2f}, {self.still_pending} still pending"
        )


def settle_all(cfg: Config | None = None) -> SettlementStats:
    """Settle singles and combined bets in one pass, and add up both."""
    singles = settle_pending(cfg)
    combos = settle_combos(cfg)
    return SettlementStats(
        settled=singles.settled + combos.settled,
        won=singles.won + combos.won,
        lost=singles.lost + combos.lost,
        void=singles.void + combos.void,
        pnl=singles.pnl + combos.pnl,
        still_pending=singles.still_pending + combos.still_pending,
    )


def settle_pending(cfg: Config | None = None) -> SettlementStats:
    """Settle every pending single bet whose match now has a result.

    Also records CLV where a closing price has arrived, which is what makes the
    Bankroll page's CLV column fill in as results load.
    """
    cfg = cfg or load_config()
    stats = SettlementStats()

    with session_scope(cfg) as s:
        pending = s.scalars(select(Bet).where(Bet.status == "pending")).all()
        for bet in pending:
            match = s.get(Match, bet.match_id)
            # Still to be played is pending. Played-with-no-score, or explicitly
            # void, is an abandoned match: that settles as a void and returns the
            # stake, rather than sitting pending forever.
            if match is None or match.status == "scheduled":
                stats.still_pending += 1
                continue

            result = settle_bet(
                bet.market or "1X2", bet.selection, bet.stake, bet.odds_taken,
                match.fthg, match.ftag,
            )
            bet.status = result.status
            bet.pnl = result.pnl
            bet.settled_at = datetime.utcnow()
            bet.settled_by = "auto"

            closing = s.scalar(
                text(
                    "SELECT decimal_odds FROM odds WHERE match_id=:m AND bookmaker='B365' "
                    "AND market=:k AND selection=:s AND odds_type='closing' LIMIT 1"
                ),
                {"m": bet.match_id, "k": bet.market or "1X2", "s": bet.selection},
            )
            if closing:
                bet.closing_odds = float(closing)
                bet.clv = clv_of(bet.odds_taken, float(closing))

            stats.settled += 1
            stats.pnl += result.pnl
            if result.status == "won":
                stats.won += 1
            elif result.status == "lost":
                stats.lost += 1
            else:
                stats.void += 1

    # Ledger entries are appended outside the loop so the balance is computed once
    # per settled bet in a consistent order.
    if stats.settled:
        with session_scope(cfg) as s:
            just_settled = s.scalars(
                select(Bet)
                .where(Bet.status.in_(("won", "lost", "void")), Bet.settled_by == "auto")
                .order_by(Bet.settled_at.desc())
                .limit(stats.settled)
            ).all()
        for bet in sorted(just_settled, key=lambda b: b.id):
            already = _has_ledger_entry(bet.id, cfg)
            if not already and bet.pnl is not None:
                record_bankroll_event(
                    "settlement", float(bet.pnl), note=f"bet {bet.id}", bet_id=bet.id,
                    cfg=cfg, user_id=bet.user_id,
                )
    return stats


def _has_ledger_entry(bet_id: int, cfg: Config | None = None) -> bool:
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        return s.scalar(select(func.count(BankrollEvent.id)).where(
            BankrollEvent.bet_id == bet_id)) > 0


def manual_settle(
    bet_id: int,
    status: str,
    pnl: float | None = None,
    cfg: Config | None = None,
    user_id: int | None = None,
) -> None:
    """Override a settlement by hand, for the cases auto-settlement can't see.

    ``user_id`` is an authorisation check, not a filter: passing one means "this
    person is editing their own bet", and editing somebody else's is refused rather
    than silently ignored.
    """
    if status not in ("won", "lost", "void", "pending"):
        raise ValueError(f"invalid status: {status}")
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        bet = s.get(Bet, bet_id)
        if bet is None:
            raise KeyError(f"no bet with id {bet_id}")
        if user_id is not None and bet.user_id != user_id:
            raise KeyError(f"no bet with id {bet_id}")
        owner_id = bet.user_id
        old_pnl = bet.pnl or 0.0
        if pnl is None:
            pnl = (
                bet.stake * (bet.odds_taken - 1.0) if status == "won"
                else (-bet.stake if status == "lost" else 0.0)
            )
        bet.status = status
        bet.pnl = None if status == "pending" else float(pnl)
        bet.settled_at = None if status == "pending" else datetime.utcnow()
        bet.settled_by = "manual"
        delta = (0.0 if status == "pending" else float(pnl)) - old_pnl

    if delta:
        record_bankroll_event(
            "adjustment", delta, note=f"manual settle of bet {bet_id}", bet_id=bet_id,
            cfg=cfg, user_id=owner_id,
        )


BET_LOG_QUERY = """
SELECT b.id, b.slip_id, b.mode, b.status, b.market, b.selection, b.odds_taken, b.stake,
       b.pnl, b.closing_odds, b.clv, b.placed_at, b.settled_at, b.settled_by,
       m.league_code, m.season, m.kickoff_utc,
       th.canonical_name AS home, ta.canonical_name AS away,
       m.fthg, m.ftag, b.user_id
FROM bets b
JOIN matches m ON m.id = b.match_id
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
WHERE (:user_id IS NULL OR b.user_id = :user_id)
ORDER BY m.kickoff_utc DESC, b.id DESC
"""


def bet_log(cfg: Config | None = None, user_id: int | None = None) -> pd.DataFrame:
    """One user's bets, or every user's when ``user_id`` is None."""
    cfg = cfg or load_config()
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(text(BET_LOG_QUERY), conn, params={"user_id": user_id})
    if not df.empty:
        for col in ("placed_at", "settled_at", "kickoff_utc"):
            df[col] = pd.to_datetime(df[col])
    return df


def bankroll_history(cfg: Config | None = None, user_id: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    uid = user_id if user_id is not None else local_user_id(cfg)
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(
            text("SELECT ts, type, amount, balance_after, bet_id, note "
                 "FROM bankroll_events WHERE user_id = :user_id ORDER BY ts, id"),
            conn,
            params={"user_id": uid},
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def rolling_roi(bets: pd.DataFrame, window: int = 300, min_periods: int = 30) -> pd.DataFrame:
    """Rolling ROI over the last ``window`` settled bets, with its interval.

    The interval is the point of this function. Over 300 bets the standard error on
    ROI is around 7 percentage points, so the number alone invites exactly the
    misreading the project is trying to avoid.

    ``min_periods`` suppresses the opening stretch. An ROI computed from the first
    two bets swings between +110% and -100% and would dominate the chart's scale
    while meaning nothing at all — precisely the kind of display this project exists
    to avoid.
    """
    settled = bets[bets["status"].isin(("won", "lost"))].sort_values("kickoff_utc")
    if settled.empty:
        return pd.DataFrame()
    pnl = settled["pnl"].to_numpy(dtype=float)
    stake = settled["stake"].to_numpy(dtype=float)
    rows = []
    for i in range(len(settled)):
        lo = max(0, i - window + 1)
        p, s_ = pnl[lo : i + 1], stake[lo : i + 1]
        if s_.sum() <= 0 or len(p) < min_periods:
            continue
        roi = p.sum() / s_.sum()
        per_unit = p / s_
        se = per_unit.std(ddof=1) / (len(per_unit) ** 0.5) if len(per_unit) > 1 else float("nan")
        rows.append({
            "kickoff_utc": settled["kickoff_utc"].iloc[i],
            "n": i - lo + 1,
            "roi": roi,
            "ci_low": roi - 1.96 * se,
            "ci_high": roi + 1.96 * se,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined bets
# ---------------------------------------------------------------------------

def log_combo(
    legs: pd.DataFrame,
    stake: float,
    combined_odds: float | None = None,
    mode: str = "paper",
    ref: str | None = None,
    placed_at: datetime | None = None,
    notes: str | None = None,
    cfg: Config | None = None,
    user_id: int | None = None,
) -> int:
    """Record one stake riding on several matches at once. Returns the combo id.

    ``combined_odds`` defaults to the product of the legs. Pass the bookmaker's own
    figure when you have it: they round, and the slip in your hand is the authority
    on what you were actually offered.
    """
    if legs.empty:
        raise ValueError("a combined bet needs at least one leg")
    if stake <= 0:
        raise ValueError(f"stake must be positive, got {stake}")
    cfg = cfg or load_config()
    uid = user_id if user_id is not None else local_user_id(cfg)
    odds = float(combined_odds) if combined_odds else float(legs["odds"].prod())
    if odds <= 1.0:
        raise ValueError(f"combined odds must be > 1.0, got {odds}")

    with session_scope(cfg) as s:
        combo = ComboBet(
            user_id=uid,
            ref=ref or f"combo-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}",
            stake=float(stake),
            combined_odds=odds,
            mode=mode,
            placed_at=placed_at or datetime.utcnow(),
            status="pending",
            notes=notes,
        )
        s.add(combo)
        s.flush()
        for r in legs.itertuples(index=False):
            s.add(
                ComboLeg(
                    combo_id=combo.id,
                    match_id=int(r.match_id),
                    market=getattr(r, "market", "1X2"),
                    selection=r.selection,
                    odds_taken=float(r.odds),
                )
            )
        return combo.id


def settle_combos(cfg: Config | None = None) -> SettlementStats:
    """Settle combined bets whose matches have all finished.

    A combination is all-or-nothing, with one exception that is not a detail: a
    voided leg drops out and the price is recomputed from the legs that remain,
    which is what bookmakers actually do. Treating a void as a loss would take money
    the bet never risked.
    """
    cfg = cfg or load_config()
    stats = SettlementStats()
    settled: list[tuple[int, float]] = []

    with session_scope(cfg) as s:
        pending = s.scalars(select(ComboBet).where(ComboBet.status == "pending")).all()
        for combo in pending:
            legs = s.scalars(select(ComboLeg).where(ComboLeg.combo_id == combo.id)).all()
            outcomes = []
            for leg in legs:
                match = s.get(Match, leg.match_id)
                if match is None or match.status == "scheduled":
                    outcomes = None
                    break
                result = settle_bet(leg.market or "1X2", leg.selection, 1.0,
                                    leg.odds_taken, match.fthg, match.ftag)
                leg.result = result.status
                outcomes.append((leg, result.status))
            if outcomes is None:
                stats.still_pending += 1
                continue

            live = [(leg, r) for leg, r in outcomes if r != "void"]
            if not live:
                combo.status, combo.pnl = "void", 0.0
            elif all(r == "won" for _, r in live):
                # Recompute from the surviving legs so a void reduces the price
                # rather than the stake.
                price = 1.0
                for leg, _ in live:
                    price *= leg.odds_taken
                if len(live) == len(outcomes):
                    price = combo.combined_odds
                combo.status = "won"
                combo.pnl = round(combo.stake * (price - 1.0), 2)
            else:
                combo.status, combo.pnl = "lost", -round(combo.stake, 2)

            combo.settled_at = datetime.utcnow()
            combo.settled_by = "auto"
            stats.settled += 1
            stats.pnl += combo.pnl
            if combo.status == "won":
                stats.won += 1
            elif combo.status == "lost":
                stats.lost += 1
            else:
                stats.void += 1
            settled.append((combo.id, combo.pnl, combo.user_id))

    for combo_id, pnl, uid in settled:
        if not _has_combo_ledger_entry(combo_id, cfg):
            _record_combo_settlement(combo_id, pnl, uid, cfg)
    return stats


def _has_combo_ledger_entry(combo_id: int, cfg: Config | None = None) -> bool:
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        return (s.scalar(select(func.count(BankrollEvent.id)).where(
            BankrollEvent.combo_id == combo_id)) or 0) > 0


def _record_combo_settlement(combo_id: int, pnl: float, uid: int, cfg: Config) -> None:
    balance = current_bankroll(cfg, uid) + pnl
    with session_scope(cfg) as s:
        s.add(
            BankrollEvent(
                user_id=uid,
                ts=datetime.utcnow(),
                type="settlement",
                amount=float(pnl),
                balance_after=float(balance),
                combo_id=combo_id,
                note=f"combined bet {combo_id}",
            )
        )


COMBO_LOG_QUERY = """
SELECT c.id, c.ref, c.stake, c.combined_odds, c.mode, c.status, c.pnl,
       c.placed_at, c.settled_at, c.notes, c.user_id,
       COUNT(l.id) AS legs,
       SUM(CASE WHEN l.result = 'won' THEN 1 ELSE 0 END) AS legs_won
FROM combo_bets c
LEFT JOIN combo_legs l ON l.combo_id = c.id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
GROUP BY c.id
ORDER BY c.placed_at DESC, c.id DESC
"""

COMBO_LEG_QUERY = """
SELECT l.combo_id, l.market, l.selection, l.odds_taken, l.result,
       m.league_code, m.kickoff_utc, m.fthg, m.ftag,
       th.canonical_name AS home, ta.canonical_name AS away
FROM combo_legs l
JOIN matches m ON m.id = l.match_id
JOIN teams th ON th.id = m.home_team_id
JOIN teams ta ON ta.id = m.away_team_id
WHERE l.combo_id IN :ids
ORDER BY m.kickoff_utc
"""


def combo_log(cfg: Config | None = None, user_id: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(text(COMBO_LOG_QUERY), conn, params={"user_id": user_id})
    if not df.empty:
        for col in ("placed_at", "settled_at"):
            df[col] = pd.to_datetime(df[col])
    return df


def combo_legs(combo_ids: list[int], cfg: Config | None = None) -> pd.DataFrame:
    if not combo_ids:
        return pd.DataFrame()
    cfg = cfg or load_config()
    with get_engine(cfg).connect() as conn:
        df = pd.read_sql(
            text(COMBO_LEG_QUERY).bindparams(bindparam("ids", expanding=True)),
            conn, params={"ids": [int(i) for i in combo_ids]},
        )
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    return df
