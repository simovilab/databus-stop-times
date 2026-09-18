"""Tests for core.catalog.inspect.build_coverage_report."""

from __future__ import annotations

import pandas as pd
import pytest
from stoptimes.core.catalog.inspect import build_coverage_report


def _row(trip_id, route_id, lat, lon, ts) -> dict:
    return {"trip_id": trip_id, "route_id": route_id, "lat": lat, "lon": lon, "ts": ts}


def _vp_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_empty_frame_reports_zero_rows():
    report = build_coverage_report(pd.DataFrame(), "vehicle_position")
    assert report["rows"] == 0
    assert report["trips_seen"] == 0
    assert report["date_range"] is None


def test_basic_counts():
    df = _vp_frame(
        [
            _row("t1", "r1", 1.0, 2.0, "2026-01-01T08:00:00"),
            _row("t1", "r1", 1.1, 2.1, "2026-01-01T08:00:15"),
            _row("t2", "r2", None, None, "2026-01-01T08:00:30"),
        ]
    )
    report = build_coverage_report(df, "vehicle_position")
    assert report["rows"] == 3
    assert report["trips_seen"] == 2
    assert report["routes_seen"] == 2
    assert report["usable_position_share"] == pytest.approx(2 / 3)
    assert report["usable_timestamp_share"] == 1.0


def test_gap_detection():
    df = _vp_frame(
        [
            _row("t1", "r1", 1.0, 2.0, "2026-01-01T08:00:00"),
            _row("t1", "r1", 1.0, 2.0, "2026-01-01T08:00:10"),
            # 20-minute gap
            _row("t1", "r1", 1.0, 2.0, "2026-01-01T08:20:10"),
        ]
    )
    report = build_coverage_report(df, "vehicle_position", gap_threshold_s=300.0)
    assert report["gap_count"] == 1
    assert report["max_gap_s"] == pytest.approx(1200.0)


def test_trips_matched_static():
    df = _vp_frame(
        [
            _row("t1", "r1", 1.0, 2.0, "2026-01-01T08:00:00"),
            _row("t2", "r1", 1.0, 2.0, "2026-01-01T08:00:10"),
        ]
    )
    report = build_coverage_report(df, "vehicle_position", static_trip_ids={"t1", "t9"})
    assert report["trips_matched_static"] == {"matched": 1, "total": 2, "share": 0.5}


def test_trip_update_kind_has_no_position_share():
    df = pd.DataFrame(
        {
            "trip_id": ["t1"],
            "route_id": ["r1"],
            "ts": pd.to_datetime(["2026-01-01T08:00:00"], utc=True),
        }
    )
    report = build_coverage_report(df, "trip_update")
    assert report["usable_position_share"] is None


def test_unusable_timestamps_reduce_share_and_leave_date_range_from_usable_only():
    df = _vp_frame([_row("t1", "r1", 1.0, 2.0, "2026-01-01T08:00:00")])
    df.loc[len(df)] = _row("t2", "r1", 1.0, 2.0, pd.NaT)
    report = build_coverage_report(df, "vehicle_position")
    assert report["usable_timestamp_share"] == 0.5
    assert report["date_range"] is not None
