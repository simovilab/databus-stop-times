"""Tests for stoptimes.core.arrivals: derive_arrivals and ArrivalParams.

Geometry is a straight north-south line (constant longitude) so the exact
math is easy to reason about: stops sit on the line at fixed latitude steps,
and a synthetic vehicle trace moves along it at constant speed. Distance to
any stop-on-the-line is then monotonic while approaching, so "first sample
within radius" and "crossing-time interpolation" have an unambiguous
geometric meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from stoptimes.core.arrivals import ArrivalParams, derive_arrivals
from stoptimes.core.contracts.data import ARRIVAL_SCHEMA, validate_frame
from stoptimes.core.contracts.source import GroundTruthDetector, Source
from stoptimes.core.domain.models import Stop
from stoptimes.core.spatial.polyline import haversine_m

LON = -84.0
BASE_LAT = 9.0
LAT_STEP = 0.003  # ~334 m between consecutive stops at this latitude


def _stop(seq: int, lat: float, stop_id: str | None = None) -> Stop:
    return Stop(stop_id=stop_id or f"s{seq}", stop_sequence=seq, lat=lat, lon=LON, progress_m=None)


def _stops(n: int) -> list[Stop]:
    return [_stop(i + 1, BASE_LAT + i * LAT_STEP) for i in range(n)]


def _shape_points(n: int, margin: float = 0.5) -> list[tuple[float, float, None]]:
    lo = BASE_LAT - margin * LAT_STEP
    hi = BASE_LAT + (n - 1 + margin) * LAT_STEP
    # A handful of intermediate points keeps build_polyline's cumulative
    # distance close to true haversine distance without needing exactness.
    n_pts = 20
    return [(lo + (hi - lo) * i / (n_pts - 1), LON, None) for i in range(n_pts)]


class FakeStatic:
    """Minimal `StaticGeometry`: a fixed trip_id -> (shape_points, stops) map."""

    def __init__(self, trips: dict[str, tuple[list, list[Stop]]]):
        self._trips = trips

    def trip_geometry(self, trip_id: str):
        if trip_id not in self._trips:
            raise ValueError(f"unknown trip {trip_id!r}")
        return self._trips[trip_id]


def make_source(**detector_overrides) -> Source:
    return Source(
        id="test",
        name="Test Agency",
        timezone="UTC",
        gtfs_static_url="local:///fake.zip",
        detector=GroundTruthDetector(**detector_overrides),
    )


def make_trace(
    trip_id: str,
    start_date: str,
    lat_start: float,
    lat_end: float,
    n_samples: int,
    t0: pd.Timestamp,
    dt_s: float,
    route_id: str = "r1",
    vehicle_id: str = "v1",
) -> list[dict]:
    rows = []
    for i in range(n_samples):
        frac = i / max(n_samples - 1, 1)
        lat = lat_start + (lat_end - lat_start) * frac
        rows.append(
            {
                "kind": "vehicle_position",
                "source_id": "test",
                "trip_id": trip_id,
                "start_date": start_date,
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "ts": t0 + pd.Timedelta(seconds=i * dt_s),
                "lat": lat,
                "lon": LON,
            }
        )
    return rows


def obs_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


T0 = pd.Timestamp("2026-09-17T12:00:00Z")


# ---------------------------------------------------------------------------
# Schema / basic shape
# ---------------------------------------------------------------------------


def test_empty_observations_yields_empty_valid_frame():
    source = make_source()
    static = FakeStatic({})
    result = derive_arrivals(pd.DataFrame(), static, source=source)
    assert list(result.columns) == ARRIVAL_SCHEMA.column_names
    assert result.empty
    assert validate_frame(result, ARRIVAL_SCHEMA) == []


def test_output_conforms_to_arrival_schema():
    stops = _stops(3)
    static = FakeStatic({"t1": (_shape_points(3), stops)})
    source = make_source()
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 0.008, 40, T0, 15.0)
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert not result.empty
    assert validate_frame(result, ARRIVAL_SCHEMA) == []


# ---------------------------------------------------------------------------
# reported
# ---------------------------------------------------------------------------


def test_reported_arrivals_taken_directly_from_trip_update():
    source = make_source()
    static = FakeStatic({})
    arrival_time = pd.Timestamp("2026-09-17T12:05:00Z")
    rows = [
        {
            "kind": "trip_update",
            "source_id": "test",
            "trip_id": "t1",
            "start_date": "20260917",
            "route_id": "r1",
            "vehicle_id": "v1",
            "stop_id": "s1",
            "stop_sequence": 1,
            "ts": arrival_time,
            "arrival_time": arrival_time,
            "schedule_relationship": "SCHEDULED",
        }
    ]
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["method"] == "reported"
    assert row["arrival_time"] == arrival_time
    assert pd.isna(row["confidence"])


def test_reported_skips_skipped_stop_time_updates():
    source = make_source()
    static = FakeStatic({})
    rows = [
        {
            "kind": "trip_update",
            "source_id": "test",
            "trip_id": "t1",
            "start_date": "20260917",
            "stop_id": "s1",
            "stop_sequence": 1,
            "ts": T0,
            "arrival_time": T0,
            "schedule_relationship": "SKIPPED",
        }
    ]
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert result.empty


def test_reported_keeps_latest_report_for_same_stop():
    source = make_source()
    static = FakeStatic({})
    early = pd.Timestamp("2026-09-17T12:05:00Z")
    late = pd.Timestamp("2026-09-17T12:05:30Z")
    rows = [
        {
            "kind": "trip_update", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
            "stop_id": "s1", "stop_sequence": 1, "ts": early, "ingested_at": early,
            "arrival_time": early,
        },
        {
            "kind": "trip_update", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
            "stop_id": "s1", "stop_sequence": 1, "ts": late, "ingested_at": late,
            "arrival_time": late,
        },
    ]
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert len(result) == 1
    assert result.iloc[0]["arrival_time"] == late


# ---------------------------------------------------------------------------
# map_matched: crossing interpolation
# ---------------------------------------------------------------------------


def test_map_matched_crossing_is_interpolated_between_bracketing_samples():
    stops = _stops(2)
    static = FakeStatic({"t1": (_shape_points(2), stops)})
    source = make_source(arrival_radius_m=40.0)
    # Coarse sampling: consecutive samples ~55 m apart so the confirmed-entry
    # sample is not exactly at the stop, and the previous sample is outside
    # the 40 m radius -- a genuine bracket to interpolate across.
    rows = make_trace("t1", "20260917", BASE_LAT - 0.0015, BASE_LAT + 0.0015, 61, T0, 10.0)
    result = derive_arrivals(obs_frame(rows), static, source=source)
    stop1 = result[result["stop_sequence"] == 1].iloc[0]

    # Coarser sampling for the same geometry should give a "close but not
    # identical" crossing time -- proof the value came from interpolation,
    # not from snapping to the confirmed sample's own timestamp.
    coarse_rows = make_trace("t1", "20260917", BASE_LAT - 0.0015, BASE_LAT + 0.0015, 7, T0, 90.0)
    coarse_result = derive_arrivals(obs_frame(coarse_rows), static, source=source)
    coarse_stop1 = coarse_result[coarse_result["stop_sequence"] == 1].iloc[0]

    assert abs((stop1["arrival_time"] - coarse_stop1["arrival_time"]).total_seconds()) < 90.0


def test_crossing_time_lies_strictly_between_bracketing_samples_not_snapped():
    stops = [_stop(1, BASE_LAT)]
    static = FakeStatic({"t1": (_shape_points(1, margin=1.0), stops)})
    source = make_source(arrival_radius_m=40.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 0.001, 11, T0, 20.0)
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert len(result) == 1
    crossing = result.iloc[0]["arrival_time"]

    # Find the confirmed-entry sample (first within radius) and its predecessor.
    df = obs_frame(rows)
    dists = [haversine_m(r["lat"], r["lon"], BASE_LAT, LON) for _, r in df.iterrows()]
    entry_idx = next(i for i, d in enumerate(dists) if d <= 40.0)
    entry_ts = df.iloc[entry_idx]["ts"]
    prev_ts = df.iloc[entry_idx - 1]["ts"]

    assert prev_ts < crossing < entry_ts


# ---------------------------------------------------------------------------
# map_matched: radius / dwell edge cases
# ---------------------------------------------------------------------------


def test_bus_stopping_short_of_radius_yields_no_arrival():
    stops = _stops(2)
    static = FakeStatic({"t1": (_shape_points(2), stops)})
    source = make_source(arrival_radius_m=40.0)
    # Trace approaches stop 2 but turns back (ends) 150 m short of it.
    short_of_stop2 = BASE_LAT + LAT_STEP - (150.0 / 111_195.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, short_of_stop2, 30, T0, 10.0)
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert 1 in set(result["stop_sequence"])
    assert 2 not in set(result["stop_sequence"])


def test_dwell_requires_continuous_time_in_radius():
    stops = [_stop(1, BASE_LAT)]
    static = FakeStatic({"t1": (_shape_points(1, margin=1.0), stops)})
    source = make_source(arrival_radius_m=40.0, dwell_s=60.0)

    # Vehicle sits exactly at the stop for only 20s (5 samples * 5s) -- not
    # enough to satisfy a 60s dwell requirement.
    brief = [
        {
            "kind": "vehicle_position", "source_id": "test", "trip_id": "t1",
            "start_date": "20260917", "route_id": "r1", "vehicle_id": "v1",
            "ts": T0 + pd.Timedelta(seconds=i * 5), "lat": BASE_LAT, "lon": LON,
        }
        for i in range(5)
    ]
    result_brief = derive_arrivals(obs_frame(brief), static, source=source)
    assert result_brief.empty

    # Vehicle sits for 75s -- dwell satisfied; arrival is confirmed at first entry.
    long_dwell = [
        {
            "kind": "vehicle_position", "source_id": "test", "trip_id": "t1",
            "start_date": "20260917", "route_id": "r1", "vehicle_id": "v1",
            "ts": T0 + pd.Timedelta(seconds=i * 15), "lat": BASE_LAT, "lon": LON,
        }
        for i in range(6)
    ]
    result_long = derive_arrivals(obs_frame(long_dwell), static, source=source)
    assert len(result_long) == 1
    assert result_long.iloc[0]["arrival_time"] == T0


def test_shape_threshold_rejects_off_route_proximity():
    # The stop sits ON the shape, but the vehicle's whole trace runs ~110 m
    # off it (in longitude) -- close enough to the stop for a generous
    # radius, but the confirming sample's cross-track error exceeds
    # shape_threshold_m, so it must not confirm.
    off_route_lon = LON + 0.001  # ~110 m east of the shape line
    stops = [_stop(1, BASE_LAT, stop_id="s1")]
    static = FakeStatic({"t1": (_shape_points(1, margin=1.0), stops)})
    source = make_source(arrival_radius_m=150.0, shape_threshold_m=30.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 0.001, 11, T0, 20.0)
    for r in rows:
        r["lon"] = off_route_lon
    result = derive_arrivals(obs_frame(rows), static, source=source)
    assert result.empty


# ---------------------------------------------------------------------------
# interpolated
# ---------------------------------------------------------------------------


def test_interpolated_fills_unobserved_intermediate_stop():
    stops = _stops(3)  # s1, s2 (unobserved), s3
    static = FakeStatic({"t1": (_shape_points(3), stops)})
    source = make_source(arrival_radius_m=15.0)  # tight radius: sampling will skip s2
    # s1..s3 span ~670 m along-shape -- above the default 600 m interpolation
    # gap, so widen it explicitly; this test is about gap-filling itself, not
    # the gap-size cutoff (see test_interpolation_skipped_when_gap_too_large).
    params = ArrivalParams.from_detector(source.detector, max_interpolation_gap_m=1000.0)
    # Offsets chosen (see module comment) so no sample lands within 15 m of
    # s2's latitude while samples do land within 15 m of s1 and s3.
    rows = make_trace(
        "t1", "20260917", BASE_LAT - 0.0015, BASE_LAT + 2 * LAT_STEP + 0.0019, 25, T0, 15.0
    )
    result = derive_arrivals(obs_frame(rows), static, source=source, params=params)
    methods = dict(zip(result["stop_sequence"], result["method"]))
    assert methods.get(1) == "map_matched"
    assert methods.get(3) == "map_matched"
    assert methods.get(2) == "interpolated"

    row2 = result[result["stop_sequence"] == 2].iloc[0]
    row1 = result[result["stop_sequence"] == 1].iloc[0]
    row3 = result[result["stop_sequence"] == 3].iloc[0]
    assert row1["arrival_time"] <= row2["arrival_time"] <= row3["arrival_time"]
    assert row2["confidence"] < row1["confidence"]


def test_interpolation_skipped_when_gap_too_large():
    stops = _stops(3)
    static = FakeStatic({"t1": (_shape_points(3), stops)})
    source = make_source(arrival_radius_m=15.0)
    params = ArrivalParams.from_detector(source.detector, max_interpolation_gap_s=1.0)
    rows = make_trace(
        "t1", "20260917", BASE_LAT - 0.0015, BASE_LAT + 2 * LAT_STEP + 0.0019, 25, T0, 15.0
    )
    result = derive_arrivals(obs_frame(rows), static, source=source, params=params)
    assert 2 not in set(result["stop_sequence"])


# ---------------------------------------------------------------------------
# Physical plausibility: monotonicity, teleports
# ---------------------------------------------------------------------------


def test_non_monotonic_out_of_order_arrival_is_dropped():
    """Simulates a doubled-back / mismatched trace: stop 2's confirmed entry
    happens chronologically *before* stop 1's, which cannot really happen for
    two stops in increasing stop_sequence order on the same trip.
    `_drop_impossible` walks stops in stop_sequence order and keeps the
    longest monotonically-increasing-in-time prefix, so stop 1 (processed
    first) is kept and stop 2's earlier-in-time reading is dropped as the
    non-monotonic one.
    """
    stops = _stops(2)
    static = FakeStatic({"t1": (_shape_points(2), stops)})
    source = make_source(arrival_radius_m=40.0)

    rows = [
        # First visits near stop 2 early...
        {"kind": "vehicle_position", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
         "route_id": "r1", "vehicle_id": "v1", "ts": T0, "lat": BASE_LAT + LAT_STEP, "lon": LON},
        # ...then drifts back and visits stop 1 much later.
        {"kind": "vehicle_position", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
         "route_id": "r1", "vehicle_id": "v1", "ts": T0 + pd.Timedelta(seconds=120), "lat": BASE_LAT, "lon": LON},
    ]
    result = derive_arrivals(obs_frame(rows), static, source=source)
    # stop 2's out-of-order (earlier-time-but-later-sequence) reading is
    # dropped; only stop 1 survives.
    assert list(result["stop_sequence"]) == [1]


def test_teleport_between_consecutive_stops_is_dropped():
    stops = _stops(2)
    static = FakeStatic({"t1": (_shape_points(2), stops)})
    source = make_source(arrival_radius_m=40.0)
    params = ArrivalParams.from_detector(source.detector, max_teleport_speed_mps=45.0)

    rows = [
        {"kind": "vehicle_position", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
         "route_id": "r1", "vehicle_id": "v1", "ts": T0, "lat": BASE_LAT, "lon": LON},
        # Reaches stop 2 (~334 m away) one second later: ~334 m/s, way above
        # any plausible bus speed -- a teleport.
        {"kind": "vehicle_position", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
         "route_id": "r1", "vehicle_id": "v1", "ts": T0 + pd.Timedelta(seconds=1), "lat": BASE_LAT + LAT_STEP, "lon": LON},
    ]
    result = derive_arrivals(obs_frame(rows), static, source=source, params=params)
    assert list(result["stop_sequence"]) == [1]


# ---------------------------------------------------------------------------
# Loop route: same stop id visited twice
# ---------------------------------------------------------------------------


def test_loop_route_same_stop_twice_keeps_both_visits():
    loop_stop_lat0 = BASE_LAT
    loop_stop_lat1 = BASE_LAT + 2 * LAT_STEP
    stops = [
        _stop(1, loop_stop_lat0, stop_id="loop"),
        _stop(2, BASE_LAT + LAT_STEP, stop_id="mid"),
        _stop(3, loop_stop_lat1, stop_id="loop"),
    ]
    static = FakeStatic({"t1": (_shape_points(3), stops)})
    source = make_source(arrival_radius_m=40.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 2 * LAT_STEP + 0.001, 60, T0, 10.0)
    result = derive_arrivals(obs_frame(rows), static, source=source)

    loop_rows = result[result["stop_id"] == "loop"]
    assert len(loop_rows) == 2
    assert set(loop_rows["stop_sequence"]) == {1, 3}
    times = loop_rows.sort_values("stop_sequence")["arrival_time"].tolist()
    assert times[0] < times[1]


# ---------------------------------------------------------------------------
# Input hygiene: out-of-order timestamps, duplicate feeds
# ---------------------------------------------------------------------------


def test_out_of_order_input_rows_are_sorted_internally():
    stops = _stops(2)
    static = FakeStatic({"t1": (_shape_points(2), stops)})
    source = make_source(arrival_radius_m=40.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + LAT_STEP + 0.001, 20, T0, 10.0)
    shuffled = [rows[i] for i in [5, 0, 12, 3, 19, 7, 1, 15, 2, 4, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18]]

    result_sorted = derive_arrivals(obs_frame(rows), static, source=source)
    result_shuffled = derive_arrivals(obs_frame(shuffled), static, source=source)

    pd.testing.assert_frame_equal(
        result_sorted.reset_index(drop=True), result_shuffled.reset_index(drop=True)
    )


def test_duplicate_feed_rows_do_not_change_result():
    stops = _stops(2)
    static = FakeStatic({"t1": (_shape_points(2), stops)})
    source = make_source(arrival_radius_m=40.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + LAT_STEP + 0.001, 20, T0, 10.0)
    duplicated = rows + rows  # same observations reported by two feeds

    result = derive_arrivals(obs_frame(rows), static, source=source)
    result_dup = derive_arrivals(obs_frame(duplicated), static, source=source)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), result_dup.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Dedup preference across methods
# ---------------------------------------------------------------------------


def test_reported_wins_over_map_matched_for_same_stop():
    stops = _stops(1)
    static = FakeStatic({"t1": (_shape_points(1, margin=1.0), stops)})
    source = make_source(arrival_radius_m=40.0)
    reported_time = pd.Timestamp("2026-09-17T12:00:05Z")

    vp_rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 0.001, 11, T0, 20.0)
    tu_row = {
        "kind": "trip_update", "source_id": "test", "trip_id": "t1", "start_date": "20260917",
        "stop_id": "s1", "stop_sequence": 1, "ts": reported_time, "arrival_time": reported_time,
    }
    result = derive_arrivals(obs_frame(vp_rows + [tu_row]), static, source=source)
    assert len(result) == 1
    assert result.iloc[0]["method"] == "reported"
    assert result.iloc[0]["arrival_time"] == reported_time


# ---------------------------------------------------------------------------
# Property-style checks
# ---------------------------------------------------------------------------


def test_arrivals_are_monotonic_in_stop_sequence_per_trip():
    stops = _stops(5)
    static = FakeStatic({"t1": (_shape_points(5), stops)})
    source = make_source(arrival_radius_m=40.0)
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 4 * LAT_STEP + 0.001, 120, T0, 5.0)
    result = derive_arrivals(obs_frame(rows), static, source=source)
    for _, group in result.groupby(["trip_id", "start_date"]):
        ordered = group.sort_values("stop_sequence")
        times = ordered["arrival_time"].tolist()
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


def test_map_matched_arrivals_stay_within_the_observation_window():
    stops = _stops(3)
    static = FakeStatic({"t1": (_shape_points(3), stops)})
    source = make_source(arrival_radius_m=40.0)
    t_start = T0
    n = 90
    dt = 5.0
    rows = make_trace("t1", "20260917", BASE_LAT - 0.001, BASE_LAT + 2 * LAT_STEP + 0.001, n, t_start, dt)
    t_end = t_start + pd.Timedelta(seconds=(n - 1) * dt)
    result = derive_arrivals(obs_frame(rows), static, source=source)
    matched = result[result["method"].isin(["map_matched", "interpolated"])]
    assert (matched["arrival_time"] >= t_start).all()
    assert (matched["arrival_time"] <= t_end).all()
