"""Shared test setup for `packages/models`.

`stoptimes.models.common.data.load_dataset` no longer falls back to another
package's test fixtures when called with no explicit `dataset_dir` (Phase 2
T3 fix — production code must not know test paths, see
`docs/notes/p2-t3.md`). Tests that call `load_dataset("sample")` with no
directory rely on `STOPTIMES_DATASET_DIR`, set here to this package's own
`tests/fixtures/` (a copy of the same tracked sample dataset).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _default_dataset_dir(monkeypatch):
    monkeypatch.setenv("STOPTIMES_DATASET_DIR", str(_FIXTURES_DIR))
