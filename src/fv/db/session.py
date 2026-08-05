"""Engine/session helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from fv.config import Config, load_config
from fv.db.models import Base, LeagueRow

_engines: dict[str, object] = {}


def get_engine(cfg: Config | None = None):
    cfg = cfg or load_config()
    path = cfg.db_path
    key = str(path)
    if key not in _engines:
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", future=True)

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):  # pragma: no cover - driver level
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        _engines[key] = engine
    return _engines[key]


def get_sessionmaker(cfg: Config | None = None):
    return sessionmaker(bind=get_engine(cfg), future=True, expire_on_commit=False)


@contextmanager
def session_scope(cfg: Config | None = None):
    sm = get_sessionmaker(cfg)
    session: Session = sm()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(cfg: Config | None = None, echo_path: bool = False) -> Path:
    """Create tables and sync the league reference rows from config.yaml."""
    cfg = cfg or load_config()
    engine = get_engine(cfg)
    Base.metadata.create_all(engine)

    with session_scope(cfg) as s:
        for lg in cfg.leagues:
            row = s.get(LeagueRow, lg.code)
            if row is None:
                s.add(
                    LeagueRow(
                        code=lg.code,
                        name=lg.name,
                        country=lg.country,
                        tier=lg.tier,
                        enabled=lg.enabled,
                        has_xg=lg.has_xg,
                    )
                )
            else:
                row.name, row.country, row.tier = lg.name, lg.country, lg.tier
                row.enabled, row.has_xg = lg.enabled, lg.has_xg

    if echo_path:
        print(cfg.db_path)
    return cfg.db_path
