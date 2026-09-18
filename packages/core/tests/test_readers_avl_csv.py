"""Tests for readers.read_avl_csv + ColumnMapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from stoptimes.core.contracts.data import OBSERVATION_VP_SCHEMA, validate_frame
from stoptimes.core.readers import ColumnMapping, read_avl_csv

FIXTURES = Path(__file__).parent / "fixtures"
AVL_CSV = FIXTURES / "avl_sample.csv"

BUCR_MAPPING = ColumnMapping(
    timestamp="cr_datetime",
    lat="latitude",
    lon="longitude",
    vehicle="plate_number",
    source_timezone="America/Costa_Rica",
)


def test_read_avl_csv_happy_path_drops_missing_position_rows():
    df = read_avl_csv(str(AVL_CSV), BUCR_MAPPING, source_id="bucr")
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []
    # 4 raw rows; the one with blank lat/lon is dropped.
    assert len(df) == 3
    assert set(df["vehicle_id"]) == {"SJB-001", "SJB-002"}


def test_read_avl_csv_naive_timestamp_interpreted_in_source_timezone():
    df = read_avl_csv(str(AVL_CSV), BUCR_MAPPING, source_id="bucr")
    row = df[(df["vehicle_id"] == "SJB-001")].sort_values("ts").iloc[0]
    # 2026-01-01 08:00:00 America/Costa_Rica (UTC-6, no DST) == 14:00:00 UTC.
    assert row["ts"] == pd.Timestamp("2026-01-01T14:00:00", tz="UTC")


def test_read_avl_csv_bad_timestamp_kept_as_nat_not_dropped():
    df = read_avl_csv(str(AVL_CSV), BUCR_MAPPING, source_id="bucr")
    bad_row = df[df["ts"].isna()]
    assert len(bad_row) == 1
    assert bad_row.iloc[0]["lat"] == pytest.approx(9.94)


def test_read_avl_csv_no_trip_column_defaults_to_empty_string():
    df = read_avl_csv(str(AVL_CSV), BUCR_MAPPING, source_id="bucr")
    assert (df["trip_id"] == "").all()
    assert df["route_id"].isna().all()


def test_read_avl_csv_missing_mapped_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing mapped column"):
        read_avl_csv(str(bad_csv), BUCR_MAPPING, source_id="bucr")


def test_read_avl_csv_empty_file(tmp_path):
    header_only = tmp_path / "empty.csv"
    header_only.write_text("cr_datetime,latitude,longitude,plate_number\n")
    df = read_avl_csv(str(header_only), BUCR_MAPPING, source_id="bucr")
    assert len(df) == 0
    assert validate_frame(df, OBSERVATION_VP_SCHEMA) == []


def test_read_avl_csv_explicit_timestamp_format(tmp_path):
    csv_path = tmp_path / "explicit_fmt.csv"
    csv_path.write_text("ts,lat,lon\n01/02/2026 03:04:05,9.9,-84.0\n")
    mapping = ColumnMapping(
        timestamp="ts", lat="lat", lon="lon", timestamp_format="%m/%d/%Y %H:%M:%S"
    )
    df = read_avl_csv(str(csv_path), mapping, source_id="bucr")
    assert len(df) == 1
    assert df.iloc[0]["ts"] == pd.Timestamp("2026-01-02T03:04:05", tz="UTC")


def test_column_mapping_from_file(tmp_path):
    mapping_file = tmp_path / "mapping.yaml"
    mapping_file.write_text(
        "timestamp: cr_datetime\nlat: latitude\nlon: longitude\nvehicle: plate_number\n"
        "source_timezone: America/Costa_Rica\n"
    )
    mapping = ColumnMapping.from_file(mapping_file)
    assert mapping == BUCR_MAPPING


def test_read_avl_csv_already_tz_aware_timestamp_is_respected(tmp_path):
    csv_path = tmp_path / "aware.csv"
    csv_path.write_text("ts,lat,lon\n2026-01-01T00:00:00-06:00,9.9,-84.0\n")
    mapping = ColumnMapping(timestamp="ts", lat="lat", lon="lon", source_timezone="UTC")
    df = read_avl_csv(str(csv_path), mapping, source_id="bucr")
    assert df.iloc[0]["ts"] == pd.Timestamp("2026-01-01T06:00:00", tz="UTC")
