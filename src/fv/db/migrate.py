"""Schema migrations for databases created before accounts existed.

There is no Alembic here on purpose: this is one product with one deployment and a
handful of forward-only changes. What matters is that an existing 93MB database
full of real matches keeps working, so every step is idempotent and additive —
nothing is dropped, nothing is rewritten in place.
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from fv.config import Config, load_config
from fv.db.models import Base
from fv.db.session import get_engine


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}


def _add_user_column(conn, table: str, owner_id: int) -> str | None:
    """Give an existing personal table an owner. Returns a note if it did anything.

    The column is added nullable because SQLite cannot add a NOT NULL column to a
    populated table without a default, and a default user id is exactly the kind of
    silent wrong answer this project avoids. The ORM declares it non-nullable, so
    every row written from here on must name its owner.
    """
    if "user_id" in _columns(conn, table):
        return None
    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER"))
    result = conn.execute(
        text(f"UPDATE {table} SET user_id = :uid WHERE user_id IS NULL"), {"uid": owner_id}
    )
    conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_user_id ON {table} (user_id)"))
    return f"{table}: added user_id, assigned {result.rowcount} existing row(s) to the owner"


def _add_market_column(conn) -> str | None:
    """Bets written before the goals markets existed were all 1X2."""
    if "market" in _columns(conn, "bets"):
        return None
    conn.execute(text("ALTER TABLE bets ADD COLUMN market VARCHAR(16)"))
    result = conn.execute(text("UPDATE bets SET market = '1X2' WHERE market IS NULL"))
    return f"bets: added market, marked {result.rowcount} existing bet(s) as 1X2"


def ensure_schema(cfg: Config | None = None) -> list[str]:
    """Create anything missing and migrate anything old. Safe to run every startup."""
    cfg = cfg or load_config()
    engine = get_engine(cfg)
    existing = set(inspect(engine).get_table_names())
    Base.metadata.create_all(engine)

    notes: list[str] = []
    if "users" not in existing:
        notes.append("created the users and user_settings tables")

    with engine.connect() as conn:
        needs_owner = any(
            table in existing and "user_id" not in _columns(conn, table)
            for table in ("bets", "bankroll_events")
        )
        needs_market = "bets" in existing and "market" not in _columns(conn, "bets")

    if needs_market:
        with engine.begin() as conn:
            note = _add_market_column(conn)
            if note:
                notes.append(note)

    if not needs_owner:
        return notes

    # Importing here keeps fv.auth out of the import cycle: auth needs a database,
    # and the database module is what creates it.
    from fv.auth import local_user_id

    owner_id = local_user_id(cfg)
    with engine.begin() as conn:
        for table in ("bets", "bankroll_events"):
            if table in existing:
                note = _add_user_column(conn, table, owner_id)
                if note:
                    notes.append(note)
    return notes
