"""Tests for readers.read_archive_parquet: Hive-partitioned VP/TU archives."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from stoptimes.core.contracts.data import (
    OBSERVATION_TU_SCHEMA,
    OBSERVATION_VP_SCHEMA,
    ObservationKind,
    validate_frame,
)
from stoptimes.core.readers import read_archive_parquet


def _write_vp_partition(root: Path, year, month, day, route_id, rows: list[dict]) -> None:
    directory = root / f"year={year}" / f"month={month}" / f"day={day}" / f"route_id={route_id}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "data.parquet")


def _vp_row(**overrides) -> dict:
    row = {
        "feed_name": "mbta_vp",
        "vehicle_id": "v1",
        "ts": pd.Timestamp("2026-01-01T08:00:00", tz="UTC"),
        "lat": 42.35,
        "lon": -71.06,
        "bearing": 10.0,
        "speed": 5.0,
        "trip_id": "t1",
        "current_stop_sequence": 2,
        "current_status": "IN_TRANSIT_TO",
        "stop_id": "s1",
        "ingested_at": pd.Timestamp("2026-01-01T08:00:05", tz="UTC"),
    }
    row.update(overrides)
    return row


@pytest.fixture
def vp_archive(tmp_path) -> Path:
    root = tmp_path / "vp_archive"
    _write_vp_partition(root, 2026, 1, 1, "10", [_vp_row(trip_id="t1")])
    _write_vp_partition(root, 2026, 1, 1, "20", [_vp_row(trip_id="t2")])
    late_ts = pd.Timestamp("2026-01-05T08:00:00", tz="UTC")
    _write_vp_partition(root, 2026, 1, 5, "10", [_vp_row(trip_id="t3", ts=late_ts)])
    return root


def test_read_archive_vp_reads_all_partitions(vp_archive):
    df = read_archive_parquet(str(vp_archive), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []
    assert len(df) == 3
    assert (df["source_id"] == "mbta").all()


def test_read_archive_vp_date_filter_pushed_down_to_file_list(vp_archive):
    df = read_archive_parquet(
        str(vp_archive),
        ObservationKind.VEHICLE_POSITION,
        source_id="mbta",
        start_date="2026-01-01",
        end_date="2026-01-01",
    )
    assert len(df) == 2
    assert set(df["trip_id"]) == {"t1", "t2"}


def test_read_archive_vp_route_filter(vp_archive):
    df = read_archive_parquet(
        str(vp_archive), ObservationKind.VEHICLE_POSITION, source_id="mbta", routes=["10"]
    )
    assert len(df) == 2
    assert set(df["trip_id"]) == {"t1", "t3"}


def test_read_archive_vp_empty_result_has_correct_schema(tmp_path):
    root = tmp_path / "empty_archive"
    root.mkdir()
    df = read_archive_parquet(str(root), ObservationKind.VEHICLE_POSITION, source_id="mbta")
    assert len(df) == 0
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []


def test_read_archive_vp_no_matching_dates_is_empty(vp_archive):
    df = read_archive_parquet(
        str(vp_archive), ObservationKind.VEHICLE_POSITION, source_id="mbta", start_date="2030-01-01"
    )
    assert len(df) == 0
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []


def _tu_row(**overrides) -> dict:
    row = {
        "feed_name": "mbta_tu",
        "route_id": "r1",
        "trip_id": "t1",
        "start_date": "20260101",
        "start_time": "08:00:00",
        "direction_id": 0,
        "vehicle_id": "v1",
        "trip_schedule_relationship": "SCHEDULED",
        "revenue": True,
        "stop_id": "s1",
        "stop_sequence": 1,
        "arrival_time": pd.Timestamp("2026-01-01T08:05:00", tz="UTC"),
        "arrival_delay": 30,
        "arrival_uncertainty": 15,
        "departure_time": pd.Timestamp("2026-01-01T08:05:30", tz="UTC"),
        "departure_delay": 30,
        "departure_uncertainty": 15,
        "stu_schedule_relationship": "SCHEDULED",
        "ts": pd.Timestamp("2026-01-01T08:00:00", tz="UTC"),
        "header_ts": pd.Timestamp("2026-01-01T08:00:00", tz="UTC"),
        "ingested_at": pd.Timestamp("2026-01-01T08:00:05", tz="UTC"),
    }
    row.update(overrides)
    return row


@pytest.fixture
def tu_archive(tmp_path) -> Path:
    root = tmp_path / "tu_archive"
    directory = root / "year=2026" / "month=1" / "day=1"
    directory.mkdir(parents=True)
    pd.DataFrame([_tu_row(), _tu_row(trip_id="t2", route_id="r2", stop_sequence=2)]).to_parquet(
        directory / "data.parquet"
    )
    return root


def test_read_archive_tu_reads_rows_via_adapter(tu_archive):
    df = read_archive_parquet(str(tu_archive), ObservationKind.TRIP_UPDATE, source_id="mbta")
    assert validate_frame(df, OBSERVATION_TU_SCHEMA) == []
    assert len(df) == 2
    assert set(df["trip_id"]) == {"t1", "t2"}


def test_read_archive_tu_route_filter_applied_post_read(tu_archive):
    # TU archives have no route_id partition to push the filter down to —
    # the reader still filters the loaded rows.
    df = read_archive_parquet(
        str(tu_archive), ObservationKind.TRIP_UPDATE, source_id="mbta", routes=["r1"]
    )
    assert len(df) == 1
    assert df.iloc[0]["trip_id"] == "t1"


def test_read_archive_parquet_missing_extra_gives_clear_error(monkeypatch, tmp_path):
    import sys

    import stoptimes.core.readers as readers_module

    root = tmp_path / "archive"
    directory = root / "year=2026" / "month=1" / "day=1"
    directory.mkdir(parents=True)
    pd.DataFrame([_vp_row()]).to_parquet(directory / "data.parquet")

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"\[parquet\]"):
        readers_module.read_archive_parquet(
            str(root), ObservationKind.VEHICLE_POSITION, source_id="mbta"
        )
