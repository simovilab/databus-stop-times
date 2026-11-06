"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory holding captured GTFS-RT JSON snapshots and mini_gtfs.zip."""
    return FIXTURES


@pytest.fixture(scope="session")
def kit():
    """The synthetic live-serving kit (`livekit.py`), loaded by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("val_livekit", Path(__file__).parent / "livekit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
