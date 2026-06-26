"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory holding captured GTFS-RT JSON snapshots and mini_gtfs.zip."""
    return FIXTURES
