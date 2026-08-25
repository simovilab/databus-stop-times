"""Tests for the pure BUCR navsat trace-cleaning row filter."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from feature_engineering.navsat_adapter import navsat_to_vehicle_positions
from feature_engineering.navsat_cleaning import (
    DEFAULT_STALENESS_THRESHOLD_SECONDS,
    REQUIRED_RAW_COLUMNS,
    TraceCleaningReport,
    clean_navsat_trace,
)


def _raw(n: int = 3, **overrides) -> pd.DataFrame:
    t0 = dt.datetime(2026, 8, 24, 12, 0, tzinfo=dt.timezone.utc)
    df = pd.DataFrame(
        {
            "plate_number": [f"P{i}" for i in range(n)],
            "cr_datetime": [(t0 - dt.timedelta(hours=6)) + dt.timedelta(seconds=30 * i) for i in range(n)],
            "cr_datetime_utc": [t0 + dt.timedelta(seconds=30 * i) for i in range(n)],
            "ingested_at_utc": [t0 + dt.timedelta(seconds=30 * i, milliseconds=200) for i in range(n)],
            "lat": [9.9 + 0.001 * i for i in range(n)],
            "lon": [-84.0 - 0.001 * i for i in range(n)],
            "speed_kmh": [36.0] * n,
            "odometer_km": [100.0 + i for i in range(n)],
            "estado": ["movimiento" if i % 2 == 0 else "detenido" for i in range(n)],
            "lugar": ["San Jose"] * n,
        }
    )
    for col, val in overrides.items():
        df[col] = val
    return df


# --- basic contract ---------------------------------------------------


def test_clean_frame_passes_through_unchanged_with_zero_drop_report():
    raw = _raw(5)
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 5
    assert report.dropped_total == 0
    assert report.rows_kept == 5
    assert report.rows_in == 5
    assert report.drop_rate == 0.0
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), raw.reset_index(drop=True))


def test_empty_frame_returns_empty_with_zero_report():
    raw = _raw(0)
    cleaned, report = clean_navsat_trace(raw)
    assert cleaned.empty
    assert report.rows_in == 0
    assert report.rows_kept == 0
    assert report.dropped_total == 0
    assert report.drop_rate == 0.0


def test_missing_required_column_raises_value_error():
    raw = _raw(2).drop(columns=["ingested_at_utc"])
    with pytest.raises(ValueError, match="ingested_at_utc"):
        clean_navsat_trace(raw)


def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError):
        clean_navsat_trace([{"lat": 1.0}])


def test_negative_threshold_raises_value_error():
    raw = _raw(2)
    with pytest.raises(ValueError, match="non-negative"):
        clean_navsat_trace(raw, staleness_threshold_seconds=-1.0)


def test_does_not_mutate_input_frame():
    raw = _raw(4)
    raw.loc[1, "lat"] = 0.0
    raw.loc[1, "lon"] = 0.0
    raw_copy = raw.copy(deep=True)
    clean_navsat_trace(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)


def test_order_preservation_among_survivors():
    raw = _raw(6)
    raw.loc[2, "lat"] = np.nan  # drop row 2
    cleaned, _ = clean_navsat_trace(raw)
    assert list(cleaned["plate_number"]) == ["P0", "P1", "P3", "P4", "P5"]


# --- staleness ----------------------------------------------------------


def test_stale_fix_dropped_over_threshold():
    raw = _raw(1)
    raw.loc[0, "ingested_at_utc"] = raw.loc[0, "cr_datetime_utc"] + dt.timedelta(
        seconds=DEFAULT_STALENESS_THRESHOLD_SECONDS + 1
    )
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_stale == 1
    assert report.dropped_total == 1


def test_fix_kept_at_threshold_boundary():
    raw = _raw(1)
    raw.loc[0, "ingested_at_utc"] = raw.loc[0, "cr_datetime_utc"] + dt.timedelta(
        seconds=DEFAULT_STALENESS_THRESHOLD_SECONDS
    )
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 1
    assert report.dropped_stale == 0
    assert report.rows_kept == 1


def test_fix_kept_just_under_threshold():
    raw = _raw(1)
    raw.loc[0, "ingested_at_utc"] = raw.loc[0, "cr_datetime_utc"] + dt.timedelta(
        seconds=DEFAULT_STALENESS_THRESHOLD_SECONDS - 1
    )
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 1
    assert report.dropped_stale == 0


def test_custom_staleness_threshold_used_and_recorded():
    raw = _raw(1)
    raw.loc[0, "ingested_at_utc"] = raw.loc[0, "cr_datetime_utc"] + dt.timedelta(seconds=61)
    cleaned, report = clean_navsat_trace(raw, staleness_threshold_seconds=60.0)
    assert len(cleaned) == 0
    assert report.dropped_stale == 1
    assert report.staleness_threshold_seconds == 60.0


# --- bad timestamps -------------------------------------------------------


def test_bad_cr_datetime_utc_dropped_and_not_double_counted_as_stale():
    raw = _raw(1)
    raw["cr_datetime_utc"] = raw["cr_datetime_utc"].astype(object)
    raw.loc[0, "cr_datetime_utc"] = "not-a-timestamp"
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_bad_timestamp == 1
    assert report.dropped_stale == 0
    assert report.dropped_total == 1


def test_null_ingested_at_utc_dropped_as_bad_timestamp():
    raw = _raw(1)
    raw.loc[0, "ingested_at_utc"] = None
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_bad_timestamp == 1


# --- coordinates ------------------------------------------------------


def test_null_lat_dropped():
    raw = _raw(1)
    raw.loc[0, "lat"] = np.nan
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_null_coord == 1


def test_null_lon_dropped():
    raw = _raw(1)
    raw.loc[0, "lon"] = None
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_null_coord == 1


def test_zero_zero_coord_dropped():
    raw = _raw(1)
    raw.loc[0, "lat"] = 0.0
    raw.loc[0, "lon"] = 0.0
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_zero_coord == 1


def test_zero_lat_nonzero_lon_is_not_zero_coord():
    raw = _raw(1)
    raw.loc[0, "lat"] = 0.0
    raw.loc[0, "lon"] = -84.0
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 1
    assert report.dropped_zero_coord == 0


@pytest.mark.parametrize(
    "lat, lon",
    [(91.0, -84.0), (-91.0, -84.0), (9.9, 181.0), (9.9, -181.0)],
)
def test_out_of_range_coord_dropped(lat, lon):
    raw = _raw(1)
    raw.loc[0, "lat"] = lat
    raw.loc[0, "lon"] = lon
    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 0
    assert report.dropped_out_of_range == 1


# --- precedence / report correctness -------------------------------------


def test_report_counts_sum_to_dropped_total_and_kept():
    raw = _raw(5)
    raw.loc[0, "cr_datetime_utc"] = None  # bad timestamp
    raw.loc[1, "ingested_at_utc"] = raw.loc[1, "cr_datetime_utc"] + dt.timedelta(
        seconds=DEFAULT_STALENESS_THRESHOLD_SECONDS + 5
    )  # stale
    raw.loc[2, "lat"] = np.nan  # null coord
    raw.loc[3, "lat"] = 0.0
    raw.loc[3, "lon"] = 0.0  # zero coord
    # row 4 stays clean
    cleaned, report = clean_navsat_trace(raw)
    assert report.rows_in == 5
    assert report.dropped_bad_timestamp == 1
    assert report.dropped_stale == 1
    assert report.dropped_null_coord == 1
    assert report.dropped_zero_coord == 1
    assert report.dropped_out_of_range == 0
    assert report.dropped_total == 4
    assert report.rows_kept == 1
    assert len(cleaned) == 1
    assert report.rows_in == report.rows_kept + report.dropped_total


def test_bad_timestamp_takes_precedence_over_stale_when_row_qualifies_for_both():
    # A row with a bad cr_datetime_utc cannot have staleness computed at all,
    # so it must be counted only as bad-timestamp, never as stale too.
    raw = _raw(1)
    raw.loc[0, "cr_datetime_utc"] = None
    raw.loc[0, "ingested_at_utc"] = raw.loc[0, "ingested_at_utc"] + dt.timedelta(hours=5)
    cleaned, report = clean_navsat_trace(raw)
    assert report.dropped_bad_timestamp == 1
    assert report.dropped_stale == 0


def test_stale_takes_precedence_over_bad_coord_when_row_qualifies_for_both():
    raw = _raw(1)
    raw.loc[0, "ingested_at_utc"] = raw.loc[0, "cr_datetime_utc"] + dt.timedelta(
        seconds=DEFAULT_STALENESS_THRESHOLD_SECONDS + 1
    )
    raw.loc[0, "lat"] = np.nan
    cleaned, report = clean_navsat_trace(raw)
    assert report.dropped_stale == 1
    assert report.dropped_null_coord == 0


def test_report_is_frozen():
    report = TraceCleaningReport(
        rows_in=1,
        dropped_bad_timestamp=0,
        dropped_stale=0,
        dropped_null_coord=0,
        dropped_zero_coord=0,
        dropped_out_of_range=0,
        rows_kept=1,
        staleness_threshold_seconds=900.0,
    )
    with pytest.raises(Exception):
        report.rows_in = 2  # type: ignore[misc]


def test_deterministic_same_input_same_output():
    raw = _raw(10)
    raw.loc[3, "lat"] = np.nan
    cleaned_a, report_a = clean_navsat_trace(raw)
    cleaned_b, report_b = clean_navsat_trace(raw)
    pd.testing.assert_frame_equal(cleaned_a, cleaned_b)
    assert report_a == report_b


def test_required_raw_columns_includes_ingested_at_utc():
    assert "ingested_at_utc" in REQUIRED_RAW_COLUMNS


# --- integration smoke: cleaned output feeds the Step-1 adapter -----------


def test_cleaned_output_is_valid_input_to_navsat_to_vehicle_positions():
    raw = _raw(6)
    raw.loc[1, "lat"] = 0.0
    raw.loc[1, "lon"] = 0.0  # dropped: zero coord
    raw.loc[2, "ingested_at_utc"] = raw.loc[2, "cr_datetime_utc"] + dt.timedelta(
        seconds=DEFAULT_STALENESS_THRESHOLD_SECONDS + 100
    )  # dropped: stale

    cleaned, report = clean_navsat_trace(raw)
    assert len(cleaned) == 4
    assert report.dropped_total == 2

    vp = navsat_to_vehicle_positions(cleaned)
    assert len(vp) == len(cleaned)
    assert list(vp["vehicle_id"]) == list(cleaned["plate_number"])
