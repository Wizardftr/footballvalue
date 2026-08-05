"""SQLAlchemy schema.

Design notes worth keeping in mind:

* `odds` is long-format (one row per price) so that football-data closing prices,
  The Odds API snapshots (Phase 4), and CLV comparisons all live in one table and
  CLV is a self-join rather than a schema change.
* `predictions.trained_through` records the exact training cutoff for every
  prediction, so lookahead leakage is auditable after the fact and not just
  asserted by a test.
* `bankroll_events` is append-only. The balance is derived from the ledger, never
  stored as mutable state.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class LeagueRow(Base):
    """Reference data + per-league config mirrored from config.yaml."""

    __tablename__ = "leagues"

    code: Mapped[str] = mapped_column(String(8), primary_key=True)  # E0, SP1, ...
    name: Mapped[str] = mapped_column(String(64))
    country: Mapped[str] = mapped_column(String(32))
    tier: Mapped[int] = mapped_column(Integer)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    has_xg: Mapped[bool] = mapped_column(Boolean, default=False)


class Team(Base):
    """Canonical team registry. One row per real-world club.

    Names are unique per country, not globally: clubs move between tiers within a
    country (so E0 and E1 must share one team id), but two different clubs in two
    countries can legitimately share a short name.
    """

    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("country", "canonical_name", name="uq_team_country_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(96), index=True)
    country: Mapped[str] = mapped_column(String(32), index=True)
    first_seen_season: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_seen_season: Mapped[str | None] = mapped_column(String(8), nullable=True)

    aliases: Mapped[list["TeamAlias"]] = relationship(back_populates="team")


class TeamAlias(Base):
    """Every raw team string ever seen, mapped to a canonical team.

    A wrongly-split alias ("Man Utd" vs "Man United") silently corrupts 25 years of
    ratings, so unresolved names are surfaced as errors rather than auto-created.
    """

    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("alias", "source", name="uq_alias_source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alias: Mapped[str] = mapped_column(String(96), index=True)
    source: Mapped[str] = mapped_column(String(24))  # football_data | odds_api | understat | fbref
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)

    team: Mapped[Team] = relationship(back_populates="aliases")


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "league_code", "season", "home_team_id", "away_team_id", name="uq_match_fixture"
        ),
        Index("ix_matches_kickoff", "kickoff_utc"),
        Index("ix_matches_league_season", "league_code", "season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_code: Mapped[str] = mapped_column(ForeignKey("leagues.code"), index=True)
    season: Mapped[str] = mapped_column(String(8), index=True)  # "2024-25"
    match_date: Mapped[date] = mapped_column(Date, index=True)
    # Kickoff time is absent in older football-data files; falls back to match_date
    # at 12:00 UTC so the walk-forward cutoff always has something to order by.
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    fthg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ftag: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ftr: Mapped[str | None] = mapped_column(String(1), nullable=True)  # H | D | A
    hthg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    htag: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(12), default="played")  # scheduled|played|void
    source: Mapped[str] = mapped_column(String(24), default="football_data")
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])


class Odds(Base):
    """Long-format prices. One row per (match, book, market, selection, type, capture)."""

    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "bookmaker",
            "market",
            "selection",
            "odds_type",
            "captured_at",
            name="uq_odds_point",
        ),
        Index("ix_odds_match_type", "match_id", "odds_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    bookmaker: Mapped[str] = mapped_column(String(24), default="B365")
    market: Mapped[str] = mapped_column(String(16), default="1X2")
    selection: Mapped[str] = mapped_column(String(8))  # H | D | A
    decimal_odds: Mapped[float] = mapped_column(Float)
    # pre      = price published ahead of the round (what you could realistically take)
    # closing  = final price before kickoff (the CLV benchmark)
    # snapshot = a timestamped pull from The Odds API (Phase 4)
    odds_type: Mapped[str] = mapped_column(String(12), default="pre")
    captured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(24), default="football_data")


class MatchXG(Base):
    """Phase 2. Defined now so the schema doesn't churn later."""

    __tablename__ = "match_xg"

    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), primary_key=True)
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(24))
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    """One row per fitted model configuration, so stages stay comparable."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stage: Mapped[str] = mapped_column(String(32))  # dixon_coles | dc_xg | lgbm | ensemble | anchored
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_run_id", "match_id", name="uq_pred_run_match"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)
    # The training cutoff used to produce this prediction. Nothing at or after this
    # timestamp was visible to the model.
    trained_through: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), index=True)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    season_from: Mapped[str | None] = mapped_column(String(8), nullable=True)
    season_to: Mapped[str | None] = mapped_column(String(8), nullable=True)

    n_predictions: Mapped[int] = mapped_column(Integer, default=0)
    n_bets: Mapped[int] = mapped_column(Integer, default=0)
    total_staked: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BacktestBet(Base):
    """Every hypothetical bet, so any headline number can be drilled into."""

    __tablename__ = "backtest_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    league_code: Mapped[str] = mapped_column(String(8), index=True)
    season: Mapped[str] = mapped_column(String(8), index=True)
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime)
    selection: Mapped[str] = mapped_column(String(8))

    model_prob: Mapped[float] = mapped_column(Float)
    market_prob_raw: Mapped[float] = mapped_column(Float)
    market_prob_fair: Mapped[float] = mapped_column(Float)
    odds_taken: Mapped[float] = mapped_column(Float)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge: Mapped[float] = mapped_column(Float)
    stake: Mapped[float] = mapped_column(Float)
    result: Mapped[str] = mapped_column(String(8))  # W | L | void
    pnl: Mapped[float] = mapped_column(Float)
    bankroll_after: Mapped[float] = mapped_column(Float)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)


class Bet(Base):
    """Phase 3: real and paper bets placed manually on bet365."""

    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slip_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    selection: Mapped[str] = mapped_column(String(8))
    odds_taken: Mapped[float] = mapped_column(Float)
    stake: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(8), default="paper")  # paper | real
    placed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending|won|lost|void
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    settled_by: Mapped[str | None] = mapped_column(String(8), nullable=True)  # auto | manual
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class BankrollEvent(Base):
    """Append-only ledger. Balance is derived, never mutated in place."""

    __tablename__ = "bankroll_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    type: Mapped[str] = mapped_column(String(16))  # deposit|withdrawal|settlement|adjustment
    amount: Mapped[float] = mapped_column(Float)
    balance_after: Mapped[float] = mapped_column(Float)
    bet_id: Mapped[int | None] = mapped_column(ForeignKey("bets.id"), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class Setting(Base):
    """UI overrides of config.yaml defaults (Phase 3)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IngestLog(Base):
    """One row per downloaded file, so the weekly re-run is cheap and auditable."""

    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(24), default="football_data")
    url: Mapped[str] = mapped_column(String(255))
    league_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    season: Mapped[str | None] = mapped_column(String(8), nullable=True)
    rows_parsed: Mapped[int] = mapped_column(Integer, default=0)
    rows_upserted: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok|missing|error|unchanged
