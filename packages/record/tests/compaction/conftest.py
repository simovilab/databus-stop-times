"""Shared fixtures: a local-directory Storage + DuckDB, so nothing under
`rt_pipeline/compaction/tests/` touches S3, `mc`, or the `duckdb` CLI binary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stoptimes.record.compaction.duckdb_ops import DuckDB
from stoptimes.record.compaction.storage import LocalStorage


@pytest.fixture
def store(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def duck() -> DuckDB:
    return DuckDB.connect_local()
