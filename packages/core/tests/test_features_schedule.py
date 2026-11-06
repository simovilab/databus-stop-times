"""Tests for core.features.schedule: GTFS anchoring, interpolation, delay features."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest

from stoptimes.core.features.schedule import (
    TripSchedule,
    build_trip_schedule,
    parse_gtfs_time,
    resolve_timezone,
    schedule_features,
    scheduled_at_progress,
    scheduled_epochs,
    service_day_origin,
)


def _utc(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> float:
    return datetime(y, m, d, h, mi, tzinfo=UTC).timestamp()


# --- service_day_origin -------------------------------------------------------


def test_origin_regular_day_is_local_midnight():
    # EST (UTC-5): midnight local = 05:00Z
    assert service_day_origin("20260105", "America/New_York") == _utc(2026, 1, 5, 5)


def test_origin_spring_forward_is_noon_minus_12h_not_midnight():
    # noon EDT = 16:00Z -> origin 04:00Z (23:00 EST the night before); midnight would be 05:00Z
    assert service_day_origin("20260308", "America/New_York") == _utc(2026, 3, 8, 4)


def test_origin_fall_back_is_noon_minus_12h_not_midnight():
    # noon EST = 17:00Z -> origin 05:00Z (01:00 EDT); midnight would be 04:00Z
    assert service_day_origin("20261101", "America/New_York") == _utc(2026, 11, 1, 5)


def test_origin_without_dst():
    # Costa Rica is UTC-6 all year
    assert service_day_origin("20260308", "America/Costa_Rica") == _utc(2026, 3, 8, 6)
    assert service_day_origin("20261101", "America/Costa_Rica") == _utc(2026, 11, 1, 6)


def test_origin_rejects_bad_date_and_tz():
    with pytest.raises(ValueError):
        service_day_origin("2026-03-08", "America/New_York")
    with pytest.raises(ValueError):
        service_day_origin("20260308", "Not/AZone")


# --- parse_gtfs_time ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("08:00:00", 28800.0), ("25:10:00", 90600.0), (" 7:05:09 ", 25509.0), ("24:00:00", 86400.0)],
)
def test_parse_gtfs_time(text, expected):
    assert parse_gtfs_time(text) == expected


@pytest.mark.parametrize("text", [None, "", "  ", "nan", float("nan")])
def test_parse_gtfs_time_blank_is_nan(text):
    assert np.isnan(parse_gtfs_time(text))


def test_parse_gtfs_time_garbage_raises():
    with pytest.raises(ValueError):
        parse_gtfs_time("8am")


# --- build_trip_schedule ------------------------------------------------------


def test_build_sorts_by_sequence_and_prefers_arrival():
    sched = build_trip_schedule(
        "t",
        stop_ids=["c", "a", "b"],
        stop_sequences=[30, 10, 20],
        arrival_times=["08:20:00", "08:00:00", ""],
        departure_times=["08:21:00", "08:01:00", "08:11:00"],
    )
    assert sched.stop_ids == ("a", "b", "c")
    assert sched.stop_sequences == (10, 20, 30)
    assert sched.offsets_s == (28800.0, 29460.0, 30000.0)
    assert sched.interpolated == (False, False, False)


def test_build_past_midnight_times():
    sched = build_trip_schedule(
        "t", ["a", "b"], [1, 2], ["23:50:00", "25:10:00"], ["23:50:00", "25:10:00"]
    )
    assert sched.offsets_s == (85800.0, 90600.0)


def test_build_blank_times_interpolated_by_index():
    sched = build_trip_schedule(
        "t",
        ["a", "b", "c", "d"],
        [1, 2, 3, 4],
        ["08:00:00", "", "", "08:09:00"],
        ["08:00:00", "", "", "08:09:00"],
    )
    assert sched.offsets_s[1:3] == pytest.approx((28800 + 180, 28800 + 360))
    assert sched.interpolated == (False, True, True, False)


def test_build_blank_times_interpolated_by_shape_dist():
    sched = build_trip_schedule(
        "t",
        ["a", "b", "c"],
        [1, 2, 3],
        ["08:00:00", "", "08:10:00"],
        ["08:00:00", "", "08:10:00"],
        shape_dist_traveled=[0.0, 750.0, 1000.0],
    )
    assert sched.offsets_s[1] == pytest.approx(28800 + 450)  # 75% of 600 s
    assert sched.interpolated == (False, True, False)


def test_build_partial_shape_dist_falls_back_to_index():
    sched = build_trip_schedule(
        "t",
        ["a", "b", "c"],
        [1, 2, 3],
        ["08:00:00", "", "08:10:00"],
        ["08:00:00", "", "08:10:00"],
        shape_dist_traveled=[0.0, float("nan"), 1000.0],
    )
    assert sched.offsets_s[1] == pytest.approx(28800 + 300)


def test_build_leading_blank_stays_nan_not_flagged():
    sched = build_trip_schedule(
        "t", ["a", "b"], [1, 2], ["", "08:00:00"], ["", "08:00:00"]
    )
    assert np.isnan(sched.offsets_s[0])
    assert sched.interpolated == (False, False)


def test_build_departure_fallback_and_length_mismatch():
    sched = build_trip_schedule("t", ["a"], [1], [""], ["08:00:00"])
    assert sched.offsets_s == (28800.0,)
    with pytest.raises(ValueError):
        build_trip_schedule("t", ["a", "b"], [1], [""], [""])


def test_trip_schedule_is_frozen():
    sched = build_trip_schedule("t", ["a"], [1], ["08:00:00"], ["08:00:00"])
    assert isinstance(sched, TripSchedule)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sched.trip_id = "x"  # type: ignore[misc]


# --- scheduled_epochs ---------------------------------------------------------


def test_scheduled_epochs_on_dst_days():
    sched = build_trip_schedule(
        "t", ["a", "b"], [1, 2], ["01:30:00", "25:10:00"], ["01:30:00", "25:10:00"]
    )
    spring = scheduled_epochs(sched, "20260308", "America/New_York")
    origin = _utc(2026, 3, 8, 4)
    assert spring.tolist() == [origin + 5400, origin + 90600]
    fall = scheduled_epochs(sched, "20261101", "America/New_York")
    assert fall[0] == _utc(2026, 11, 1, 5) + 5400
    cr = scheduled_epochs(sched, "20261101", "America/Costa_Rica")
    assert cr[1] == _utc(2026, 11, 1, 6) + 90600


# --- scheduled_at_progress ----------------------------------------------------


def test_scheduled_at_progress_interpolates_and_clamps():
    prog = np.array([0.0, 100.0, 300.0])
    se = np.array([1000.0, 1100.0, 1300.0])
    assert scheduled_at_progress(prog, se, 50.0) == pytest.approx(1050.0)
    assert scheduled_at_progress(prog, se, -10.0) == 1000.0
    assert scheduled_at_progress(prog, se, 999.0) == 1300.0
    out = scheduled_at_progress(prog, se, np.array([0.0, 200.0]))
    assert out.tolist() == [1000.0, 1200.0]


def test_scheduled_at_progress_applies_cummax():
    prog = np.array([0.0, 200.0, 150.0, 300.0])  # non-monotone dip
    se = np.array([0.0, 20.0, 25.0, 40.0])
    # cummax -> [0,200,200,300]; at 250: between (200,25) and (300,40)
    assert scheduled_at_progress(prog, se, 250.0) == pytest.approx(32.5)


# --- schedule_features vs reference formulas ----------------------------------


def test_schedule_features_match_reference_formulas():
    prog = np.array([0.0, 500.0, 450.0, 1500.0])  # non-monotone on purpose
    se = np.array([1000.0, 1100.0, 1150.0, 1400.0])
    progress_m, ping_ts, t = 700.0, 1130.0, 1200.0
    tgt = np.array([1, 3])

    sched_now = np.interp(progress_m, np.maximum.accumulate(prog), se)  # reference
    ref_delay = ping_ts - sched_now
    ref_rem = se[tgt] - sched_now
    ref_rem_t = se[tgt] - t

    out = schedule_features(prog, se, progress_m, ping_ts, t, tgt)
    assert set(out) == {"delay_now", "sched_rem", "sched_rem_t"}
    np.testing.assert_allclose(out["delay_now"], np.full(2, ref_delay))
    np.testing.assert_allclose(out["sched_rem"], ref_rem)
    np.testing.assert_allclose(out["sched_rem_t"], ref_rem_t)


def test_schedule_features_nan_safe():
    prog = np.array([0.0, 500.0, 1000.0])
    se = np.array([1000.0, np.nan, 1400.0])
    out = schedule_features(prog, se, 250.0, 1210.0, 1220.0, np.array([1, 2]))
    assert out["delay_now"][0] == pytest.approx(1210.0 - 1100.0)  # NaN stop skipped
    assert np.isnan(out["sched_rem"][0]) and np.isnan(out["sched_rem_t"][0])
    assert out["sched_rem"][1] == pytest.approx(1400.0 - 1100.0)


def test_schedule_features_all_missing_is_nan():
    out = schedule_features(
        np.array([0.0, 1.0]), np.array([np.nan, np.nan]), 0.5, 10.0, 11.0, np.array([0, 1])
    )
    assert all(np.isnan(v).all() for v in out.values())


# --- resolve_timezone ---------------------------------------------------------


def test_resolve_timezone_precedence():
    assert resolve_timezone("America/New_York", "America/Costa_Rica") == "America/New_York"
    assert resolve_timezone(None, "America/Costa_Rica") == "America/Costa_Rica"
    assert resolve_timezone("", "America/Costa_Rica") == "America/Costa_Rica"


def test_resolve_timezone_errors():
    with pytest.raises(ValueError):
        resolve_timezone(None, None)
    with pytest.raises(ValueError):
        resolve_timezone("Not/AZone", None)
