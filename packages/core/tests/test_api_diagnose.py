"""Tests for stoptimes.core.api.diagnose and inspect_dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stoptimes.core.api import add_dataset, diagnose, inspect_dataset


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from stoptimes.core.config import reset_config

    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    reset_config()
    yield
    reset_config()


def test_diagnose_with_nothing_reports_static_missing():
    report = diagnose()
    assert report.static_present is False
    assert report.vp_present is False
    assert report.tu_present is False
    assert any(f.id == "static_missing" for f in report.findings)


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


def test_diagnose_reads_a_catalog_dataset_by_name(tmp_path):
    archive_dir = _archive(tmp_path / "archive")
    add_dataset("positions", str(archive_dir), source="demo", kind="observations")

    report = diagnose(vehicle_positions="positions")
    assert report.vp_present is True
    assert report.vp_rows == 10


def test_inspect_dataset_reports_coverage(tmp_path):
    archive_dir = _archive(tmp_path / "archive")
    add_dataset("positions", str(archive_dir), source="demo", kind="observations")

    report = inspect_dataset("positions")
    assert report["kind"] == "vehicle_position"
    assert report["rows"] == 10


def test_inspect_dataset_raises_for_unknown_name():
    with pytest.raises(KeyError):
        inspect_dataset("nope")
