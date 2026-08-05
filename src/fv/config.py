"""Configuration loading: config.yaml + .env, with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class League:
    code: str
    name: str
    country: str
    tier: int
    enabled: bool = True
    has_xg: bool = False


@dataclass
class Config:
    raw: dict
    path: Path

    # -- sections ---------------------------------------------------------
    @property
    def data(self) -> dict:
        return self.raw.get("data", {})

    @property
    def model(self) -> dict:
        return self.raw.get("model", {})

    @property
    def dixon_coles(self) -> dict:
        return self.model.get("dixon_coles", {})

    @property
    def odds(self) -> dict:
        return self.raw.get("odds", {})

    @property
    def betting(self) -> dict:
        return self.raw.get("betting", {})

    @property
    def risk(self) -> dict:
        return self.raw.get("risk", {})

    @property
    def backtest(self) -> dict:
        return self.raw.get("backtest", {})

    # -- derived ----------------------------------------------------------
    @property
    def leagues(self) -> list[League]:
        return [League(**entry) for entry in self.raw.get("leagues", [])]

    @property
    def enabled_leagues(self) -> list[League]:
        return [lg for lg in self.leagues if lg.enabled]

    def league(self, code: str) -> League:
        for lg in self.leagues:
            if lg.code == code:
                return lg
        raise KeyError(f"unknown league code: {code}")

    @property
    def db_path(self) -> Path:
        override = os.getenv("FV_DB_PATH")
        raw_path = override or self.data.get("db_path", "data/footballvalue.db")
        p = Path(raw_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw_cache(self) -> Path:
        p = Path(self.data.get("raw_cache", "data/raw"))
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def odds_api_key(self) -> str | None:
        return os.getenv("ODDS_API_KEY") or None


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw, path=cfg_path)
