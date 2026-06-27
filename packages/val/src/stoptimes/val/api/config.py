"""API configuration via environment variables.

Environment variables
---------------------
ETAVAL_API_HOST
    Bind host (default ``"0.0.0.0"``).
ETAVAL_API_PORT
    Bind port (default ``8010`` — not 8000, which the databus dev stack occupies).
ETAVAL_DATA_DIR
    Root directory for run data (default ``"./.etaval_data"``).
    Per-run SQLite databases are stored under ``<DATA_DIR>/runs/``.
ETAVAL_CORS_ORIGINS
    Comma-separated allowed CORS origins (default ``"http://localhost:3000"``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Typed API settings read from environment variables."""

    host: str
    port: int
    data_dir: str
    cors_origins: list[str]


def get_settings() -> Settings:
    """Read and return API settings from the current environment.

    All values are read at call time so tests can override via monkeypatch or
    ``os.environ`` manipulation before calling ``get_settings()``.
    """
    raw_origins = os.environ.get("ETAVAL_CORS_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    return Settings(
        host=os.environ.get("ETAVAL_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("ETAVAL_API_PORT", "8010")),
        data_dir=os.environ.get("ETAVAL_DATA_DIR", "./.etaval_data"),
        cors_origins=origins,
    )
