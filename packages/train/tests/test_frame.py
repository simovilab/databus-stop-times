"""Tests for `stoptimes.train.frame.build_training_frame`.

Fixture geometry mirrors `packages/cli/tests/test_cli_data_arrivals.py`: a
one-trip, three-stop feed on a straight west-to-east line at the equator,
where haversine distances are hand-computable (1 degree of longitude at the
equator is ~111,320 m, so 0.01 deg ~= 1113.2 m).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stoptimes.core.contracts.adapters import source_from_val_yaml
from stoptimes.core.contracts.data import ObservationKind
from stoptimes.train.frame import FrameParams, build_training_frame

pytest.importorskip("pandas")

METERS_PER_DEG = 111_320.0


def _source(**overrides):
    data = {
        "name": "demo",
        "gtfs_static_url": "unused.zip",
        "vehicle_positions_url": "http://example.invalid/vp.json",
        "trip_updates_url": "http://example.invalid/tu.json",
        "poll_interval_s": 15,
        "timezone": "UTC",
        "arrival_radius_m": 40.0,
        "dwell_s": 0.0,
        "shape_threshold_m": 30.0,
        **overrides,
    }
    return source_from_val_yaml(data)


class FakeStatic:
    """A minimal `StaticGeometry`: a fixed table of (shape_points, stops) per trip_id."""

    def __init__(self, geometries: dict):
        self._geometries = geometries

    def trip_geometry(self, trip_id: str):
        if trip_id not in self._geometries:
            raise ValueError(f"unknown trip_id {trip_id!r}")
        return self._geometries[trip_id]


def _straight_line_geometry(with_shape: bool = True):
    """t1: s1 (0,0) -> s2 (0, 0.01) -> s3 (0, 0.02), stop_sequence 1..3."""
    from stoptimes.core.domain.models import Stop

    stops = [
        Stop(stop_id="s1", stop_sequence=1, lat=0.0, lon=0.0),
        Stop(stop_id="s2", stop_sequence=2, lat=0.0, lon=0.01),
        Stop(stop_id="s3", stop_sequence=3, lat=0.0, lon=0.02),
    ]
    shape_points = (
        [(0.0, 0.0, 0.0), (0.0, 0.01, 1113.2), (0.0, 0.02, 2226.4)] if with_shape else []
    )
    return shape_points, stops


def _vp_rows(lons: list[float], trip_id: str = "t1", start_date: str = "20260917", **extra):
    base_ts = pd.Timestamp("2026-09-17T08:00:00Z")
    rows = []
    for i, lon in enumerate(lons):
        row = {
            "kind": ObservationKind.VEHICLE_POSITION.value,
            "trip_id": trip_id,
            "route_id": "r1",
            "vehicle_id": "v1",
            "start_date": start_date,
            "ts": base_ts + pd.Timedelta(seconds=30 * i),
            "lat": 0.0,
            "lon": lon,
            "bearing": 90.0,
            "speed": 5.0,
            "current_status": "IN_TRANSIT_TO",
        }
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def _arrivals_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)
    return df


def test_basic_frame_matches_hand_computed_distances_and_target():
    source = _source()
    static = FakeStatic({"t1": _straight_line_geometry()})
    # One VP sitting exactly at s1.
    observations = _vp_rows([0.0])
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s1", "stop_sequence": 1,
                "arrival_time": "2026-09-17T08:00:00Z", "method": "reported", "confidence": None,
            },
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s2", "stop_sequence": 2,
                "arrival_time": "2026-09-17T08:02:00Z", "method": "reported", "confidence": None,
            },
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s3", "stop_sequence": 3,
                "arrival_time": "2026-09-17T08:04:00Z", "method": "reported", "confidence": None,
            },
        ]
    )

    frame = build_training_frame(observations, arrivals, static, source=source)

    assert list(frame["stop_id"]) == ["s1", "s2", "s3"]
    assert list(frame["stops_ahead"]) == [0, 1, 2]

    # distance_to_stop for s2/s3 from (0,0): great-circle ~ 0.01/0.02 deg longitude.
    row_s2 = frame[frame["stop_id"] == "s2"].iloc[0]
    row_s3 = frame[frame["stop_id"] == "s3"].iloc[0]
    assert row_s2["distance_to_stop"] == pytest.approx(1113.2, rel=0.01)
    assert row_s3["distance_to_stop"] == pytest.approx(2226.4, rel=0.01)

    # Target: time from vp_ts (08:00:00) to each stop's arrival.
    assert list(frame["time_to_arrival_seconds"]) == [0.0, 120.0, 240.0]
    assert frame["actual_arrival"].dt.tz is not None

    # Kinematic: current_speed_kmh = speed(m/s) * 3.6.
    assert frame["current_speed_kmh"].tolist() == pytest.approx([18.0, 18.0, 18.0])

    # Superset check: every non-temporal feature xgb-baseline.yaml needs is present.
    xgb_baseline_nontemporal = {
        "distance_to_stop", "distance_to_next_stop", "progress_ratio",
        "stops_ahead", "current_speed_kmh",
    }
    assert xgb_baseline_nontemporal <= set(frame.columns)


def test_trip_with_no_shape_leaves_shape_columns_nan_not_fabricated():
    source = _source()
    static = FakeStatic({"t1": _straight_line_geometry(with_shape=False)})
    observations = _vp_rows([0.0])
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s1", "stop_sequence": 1,
                "arrival_time": "2026-09-17T08:00:00Z", "method": "reported", "confidence": None,
            },
        ]
    )

    frame = build_training_frame(observations, arrivals, static, source=source)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert pd.isna(row["shape_distance_to_stop"])
    assert pd.isna(row["shape_progress"])
    assert pd.isna(row["cross_track_error"])
    # distance_to_stop is still computed via haversine (doesn't need a shape).
    assert row["distance_to_stop"] == pytest.approx(0.0, abs=1.0)


def test_position_after_last_stop_emits_one_row_for_it():
    source = _source()
    static = FakeStatic({"t1": _straight_line_geometry()})
    # VP well past s3.
    observations = _vp_rows([0.03])
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s3", "stop_sequence": 3,
                "arrival_time": "2026-09-17T08:04:00Z", "method": "reported", "confidence": None,
            },
        ]
    )

    frame = build_training_frame(observations, arrivals, static, source=source)

    assert list(frame["stop_id"]) == ["s3"]
    assert frame.iloc[0]["stops_ahead"] == 0


def test_arrival_missing_for_one_stop_is_dropped_by_default():
    source = _source()
    static = FakeStatic({"t1": _straight_line_geometry()})
    observations = _vp_rows([0.0])
    # No arrival row at all for s2.
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s1", "stop_sequence": 1,
                "arrival_time": "2026-09-17T08:00:00Z", "method": "reported", "confidence": None,
            },
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s3", "stop_sequence": 3,
                "arrival_time": "2026-09-17T08:04:00Z", "method": "reported", "confidence": None,
            },
        ]
    )

    frame = build_training_frame(observations, arrivals, static, source=source)
    assert set(frame["stop_id"]) == {"s1", "s3"}

    kept = build_training_frame(
        observations, arrivals, static, source=source, params=FrameParams(keep_unlabeled=True)
    )
    assert set(kept["stop_id"]) == {"s1", "s2", "s3"}
    row_s2 = kept[kept["stop_id"] == "s2"].iloc[0]
    assert pd.isna(row_s2["actual_arrival"])
    assert math.isnan(row_s2["time_to_arrival_seconds"])


def test_max_stops_ahead_truncates():
    source = _source()
    static = FakeStatic({"t1": _straight_line_geometry()})
    observations = _vp_rows([0.0])
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": sid, "stop_sequence": seq,
                "arrival_time": ts, "method": "reported", "confidence": None,
            }
            for sid, seq, ts in [
                ("s1", 1, "2026-09-17T08:00:00Z"),
                ("s2", 2, "2026-09-17T08:02:00Z"),
                ("s3", 3, "2026-09-17T08:04:00Z"),
            ]
        ]
    )

    frame = build_training_frame(
        observations, arrivals, static, source=source, params=FrameParams(max_stops_ahead=1)
    )
    assert list(frame["stop_id"]) == ["s1"]


def test_ordering_and_determinism():
    source = _source()
    static = FakeStatic({"t1": _straight_line_geometry()})
    observations = _vp_rows([0.0, 0.005])
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": sid, "stop_sequence": seq,
                "arrival_time": ts, "method": "reported", "confidence": None,
            }
            for sid, seq, ts in [
                ("s1", 1, "2026-09-17T08:00:00Z"),
                ("s2", 2, "2026-09-17T08:02:00Z"),
                ("s3", 3, "2026-09-17T08:04:00Z"),
            ]
        ]
    )

    frame1 = build_training_frame(observations, arrivals, static, source=source)
    # Shuffle input row order -- output must be identical (sorted internally).
    shuffled = observations.iloc[::-1].reset_index(drop=True)
    frame2 = build_training_frame(shuffled, arrivals, static, source=source)

    pd.testing.assert_frame_equal(frame1, frame2)
    # vp_ts non-decreasing within a trip; ties broken by stop_sequence.
    assert frame1["vp_ts"].is_monotonic_increasing


def test_unknown_trip_id_contributes_no_rows():
    source = _source()
    static = FakeStatic({})  # no geometry registered for any trip
    observations = _vp_rows([0.0])
    arrivals = _arrivals_frame(
        [
            {
                "source_id": "demo", "trip_id": "t1", "start_date": "20260917",
                "route_id": "r1", "vehicle_id": "v1", "stop_id": "s1", "stop_sequence": 1,
                "arrival_time": "2026-09-17T08:00:00Z", "method": "reported", "confidence": None,
            },
        ]
    )

    frame = build_training_frame(observations, arrivals, static, source=source)
    assert frame.empty
    assert list(frame.columns) == [
        "trip_id", "route_id", "vehicle_id", "stop_id", "stop_sequence",
        "vp_ts", "vp_lat", "vp_lon", "vp_bearing", "stop_lat", "stop_lon",
        "distance_to_stop", "distance_to_next_stop", "shape_distance_to_stop",
        "shape_progress", "cross_track_error", "progress_ratio", "stops_ahead",
        "current_speed_kmh", "bearing_to_stop", "bearing_diff",
        "is_at_stop", "actual_arrival", "time_to_arrival_seconds",
    ]


def test_no_observations_returns_empty_frame_with_columns():
    source = _source()
    static = FakeStatic({})
    observations = pd.DataFrame(columns=["kind", "trip_id", "lat", "lon", "ts"])
    arrivals = pd.DataFrame(columns=["source_id", "trip_id", "start_date", "stop_id", "stop_sequence", "arrival_time"])

    frame = build_training_frame(observations, arrivals, static, source=source)
    assert frame.empty
