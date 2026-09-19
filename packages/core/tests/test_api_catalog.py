"""Tests for stoptimes.core.api's catalog verbs (add/list/get/remove/load)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stoptimes.core.api import (
    DatasetExistsError,
    add_dataset,
    get_dataset,
    guess_format,
    list_datasets,
    load_observations,
    remove_dataset,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from stoptimes.core.config import reset_config

    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    reset_config()
    yield
    reset_config()


def _archive(root: Path) -> Path:
    rows = [
        {
            "source_id": "demo",
            "kind": "vehicle_position",
            "trip_id": "t1",
            "route_id": "r1",
            "vehicle_id": "v1",
            "start_date": "20260917",
            "ts": pd.Timestamp("2026-09-17T08:00:00Z") + pd.Timedelta(seconds=15 * i),
            "lat": 0.0,
            "lon": round(i * 0.001, 6),
            "bearing": 90.0,
        }
        for i in range(10)
    ]
    partition = root / "year=2026" / "month=09" / "day=17" / "route_id=r1"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part-0.parquet")
    return root


def test_guess_format():
    assert guess_format("a.csv") == "avl_csv"
    assert guess_format("a.pb") == "gtfsrt_pb"
    assert guess_format("a.json") == "gtfsrt_json"
    assert guess_format("archive/") == "archive_parquet"
    assert guess_format("a.json", mapping="mapping.yaml") == "avl_csv"


def test_add_list_get_remove_roundtrip(tmp_path):
    archive_dir = _archive(tmp_path / "archive")

    entry = add_dataset("positions", str(archive_dir), source="demo", kind="observations")
    assert entry.name == "positions"
    assert entry.row_count == 10
    assert entry.kind.value == "observations"

    assert [e.name for e in list_datasets()] == ["positions"]
    assert get_dataset("positions") is not None
    assert get_dataset("missing") is None

    with pytest.raises(DatasetExistsError):
        add_dataset("positions", str(archive_dir), source="demo")
    add_dataset("positions", str(archive_dir), source="demo", overwrite=True)

    assert remove_dataset("positions") is True
    assert remove_dataset("positions") is False
    assert list_datasets() == []


def test_load_observations_by_catalog_name_or_path(tmp_path):
    archive_dir = _archive(tmp_path / "archive")
    add_dataset("positions", str(archive_dir), source="demo", kind="observations")

    by_name = load_observations("positions", source_id="demo")
    by_path = load_observations(str(archive_dir), source_id="demo")
    assert len(by_name) == len(by_path) == 10
