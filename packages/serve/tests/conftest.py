"""Shared test setup for `packages/serve`.

`stoptimes.models.common.data.load_dataset` no longer falls back to another
package's test fixtures (Phase 2 T3 — production code must not know test
paths). The serve wiring tests load the same sample dataset, which lives in
this package's own `tests/fixtures/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _default_dataset_dir(monkeypatch):
    monkeypatch.setenv("STOPTIMES_DATASET_DIR", str(_FIXTURES_DIR))
