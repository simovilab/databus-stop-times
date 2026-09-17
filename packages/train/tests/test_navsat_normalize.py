"""Tests for the pure real-schema -> canonical-schema navsat normalizer."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from feature_engineering.navsat_normalize import (
    COSTA_RICA_TZ,
    NavsatSchemaError,
    normalize_raw_navsat,
)


def _real_schema_raw(n: int = 3, **overrides) -> pd.DataFrame:
    """A frame shaped like the actual S3 navsat export -- latitude/longitude,
    cr_datetime (local, naive), cr_datetime_utc absent."""
    df = pd.DataFrame(
        {
            "plate_number": [f"P{i}" for i in range(n)],
            "cr_datetime": [f"2026-08-18 {6 + i:02d}:00:00" for i in range(n)],
            "latitude": [9.9 + 0.001 * i for i in range(n)],
            "longitude": [-84.0 - 0.001 * i for i in range(n)],
            "speed_kmh": [36] * n,
            "odometer_km": [100 + i for i in range(n)],
            "estado": ["movimiento" if i % 2 == 0 else "detenido" for i in range(n)],
            "lugar": ["San Jose"] * n,
            "provided_by": ["navsat"] * n,
            "ingested_at_utc": [f"2026-08-18T{12 + i:02d}:00:04.084809+00:00" for i in range(n)],
        }
    )
    for col, val in overrides.items():
        df[col] = val
    return df


def test_renames_latitude_longitude_to_lat_lon():
    out = normalize_raw_navsat(_real_schema_raw(3))
    assert "lat" in out.columns and "lon" in out.columns
    assert "latitude" not in out.columns and "longitude" not in out.columns


def test_idempotent_when_lat_lon_already_canonical():
    already = _real_schema_raw(2).rename(columns={"latitude": "lat", "longitude": "lon"})
    out = normalize_raw_navsat(already)
    assert list(out["lat"]) == list(already["lat"])
    assert list(out["lon"]) == list(already["lon"])


def test_idempotent_running_twice_gives_same_result():
    raw = _real_schema_raw(3)
    once = normalize_raw_navsat(raw)
    twice = normalize_raw_navsat(once)
    pd.testing.assert_frame_equal(once, twice)


def test_derives_cr_datetime_utc_from_local_costa_rica_time_exact_value():
    """Costa Rica noon on a known date -> exactly 18:00 UTC (fixed UTC-6, no DST)."""
    raw = _real_schema_raw(1, **{"cr_datetime": ["2026-08-18 12:00:00"]})
    out = normalize_raw_navsat(raw)
    expected = pd.Timestamp("2026-08-18 18:00:00", tz="UTC")
    assert out["cr_datetime_utc"].iloc[0] == expected


def test_costa_rica_tz_is_fixed_utc_minus_6():
    sample = dt.datetime(2026, 8, 18, 12, 0, tzinfo=COSTA_RICA_TZ)
    assert sample.utcoffset() == dt.timedelta(hours=-6)


def test_prefers_existing_non_null_cr_datetime_utc_over_derived():
    raw = _real_schema_raw(
        1,
        **{
            "cr_datetime": ["2026-08-18 12:00:00"],  # would derive to 18:00Z
            "cr_datetime_utc": ["2026-08-18T20:00:00+00:00"],  # explicit, different
        },
    )
    out = normalize_raw_navsat(raw)
    assert out["cr_datetime_utc"].iloc[0] == pd.Timestamp("2026-08-18 20:00:00", tz="UTC")


def test_falls_back_to_derived_only_for_null_existing_utc_rows():
    raw = _real_schema_raw(
        2,
        **{
            "cr_datetime": ["2026-08-18 06:00:00", "2026-08-18 07:00:00"],
            "cr_datetime_utc": [None, "2026-08-18T20:00:00+00:00"],
        },
    )
    out = normalize_raw_navsat(raw)
    # row 0: null existing -> derived from cr_datetime (06:00 CR -> 12:00Z)
    assert out["cr_datetime_utc"].iloc[0] == pd.Timestamp("2026-08-18 12:00:00", tz="UTC")
    # row 1: existing present -> kept as-is, NOT overwritten by derivation (would be 13:00Z)
    assert out["cr_datetime_utc"].iloc[1] == pd.Timestamp("2026-08-18 20:00:00", tz="UTC")


def test_null_cr_datetime_yields_null_cr_datetime_utc_not_raise():
    raw = _real_schema_raw(2, **{"cr_datetime": [None, "2026-08-18 06:00:00"]})
    out = normalize_raw_navsat(raw)
    assert pd.isna(out["cr_datetime_utc"].iloc[0])
    assert pd.notna(out["cr_datetime_utc"].iloc[1])


def test_unparseable_cr_datetime_yields_null_not_raise():
    raw = _real_schema_raw(1, **{"cr_datetime": ["not-a-datetime"]})
    out = normalize_raw_navsat(raw)
    assert pd.isna(out["cr_datetime_utc"].iloc[0])


def test_other_columns_pass_through_untouched():
    raw = _real_schema_raw(3)
    out = normalize_raw_navsat(raw)
    for col in ["plate_number", "ingested_at_utc", "provided_by", "odometer_km", "estado", "speed_kmh"]:
        assert list(out[col]) == list(raw[col])


def test_missing_coordinate_columns_raises_navsat_schema_error():
    raw = _real_schema_raw(2).drop(columns=["latitude", "longitude"])
    with pytest.raises(NavsatSchemaError):
        normalize_raw_navsat(raw)


def test_missing_both_timestamp_columns_raises_navsat_schema_error():
    raw = _real_schema_raw(2).drop(columns=["cr_datetime"])
    with pytest.raises(NavsatSchemaError):
        normalize_raw_navsat(raw)


def test_not_a_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        normalize_raw_navsat([{"a": 1}])


def test_empty_frame_with_valid_schema_returns_empty_with_canonical_columns():
    raw = _real_schema_raw(0)
    out = normalize_raw_navsat(raw)
    assert out.empty
    assert "lat" in out.columns and "lon" in out.columns and "cr_datetime_utc" in out.columns


def test_empty_frame_missing_columns_still_raises():
    raw = _real_schema_raw(0).drop(columns=["latitude", "longitude"])
    with pytest.raises(NavsatSchemaError):
        normalize_raw_navsat(raw)


def test_input_frame_not_mutated():
    raw = _real_schema_raw(3)
    raw_copy = raw.copy(deep=True)
    normalize_raw_navsat(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)
