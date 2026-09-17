"""Tests for the shared segment core
(``stoptimes.core.segments``).

The two agency builders (bUCR / MBTA) are covered end-to-end by
``test_segment_dataset_builder.py`` and ``test_mbta_segments.py``; those exercise
this core through their resolvers. These tests lock the core's *own* public seam
directly -- ``TripInstance`` emission, ``resolve_agency_temporal``, and the
empty/degenerate cases -- so a future change to a resolver can't quietly rely on
core behavior no test pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytest

from stoptimes.core.segments import (
    OUTPUT_COLUMNS,
    TripInstance,
    build_segments_from_instances,
    resolve_agency_temporal,
)


@dataclass(frozen=True)
class _Stop:
    """Minimal stop matching what the core reads (lat/lon/id/seq/progress_m)."""
    stop_id: str
    stop_sequence: int
    lat: float
    lon: float
    progress_m: Optional[float]


# Three stops ~ every 0.0027 deg lon (~222 m at this latitude), clear of the
# 200 m closest-approach fallback so a neighbour's VP isn't misdetected.
_LAT = 42.36
_STOPS = [
    _Stop("A", 1, _LAT, -71.0600, 0.0),
    _Stop("B", 2, _LAT, -71.0573, 100.0),
    _Stop("C", 3, _LAT, -71.0546, 200.0),
]

_KW = dict(
    arrival_radius_m=50.0,
    closest_fallback_m=200.0,
    max_segment_seconds=1800.0,
    temporal_tz="America/New_York",
    temporal_region="US_MA",
)


def _vps(minutes=(0, 3, 8)) -> pd.DataFrame:
    base = pd.Timestamp("2026-08-17T08:00:00Z")
    return pd.DataFrame(
        [
            {
                "vehicle_id": "V1",
                "ts": base + pd.Timedelta(minutes=m),
                "lat": s.lat,
                "lon": s.lon,
                "bearing": 90.0,
                "speed": 0.0,
                "current_status": "STOPPED_AT",
            }
            for s, m in zip(_STOPS, minutes)
        ]
    )


def _instance(stops=_STOPS, vps=None) -> TripInstance:
    return TripInstance(
        trip_id="T1",
        route_id="R1",
        vehicle_id="V1",
        stops=list(stops),
        polyline=None,  # exercises the no-shape path (cross_track / progress NaN)
        trip_vps=vps if vps is not None else _vps(),
    )


def test_emits_one_row_per_traversed_segment():
    out = build_segments_from_instances([_instance()], **_KW)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 2
    assert list(out["segment_index"]) == [0, 1]
    assert list(out["from_stop_id"]) == ["A", "B"]
    assert list(out["stop_id"]) == ["B", "C"]
    assert list(out["time_to_arrival_seconds"]) == [180.0, 300.0]
    assert (out["trip_id"] == "T1").all()
    assert (out["vehicle_id"] == "V1").all()


def test_no_polyline_leaves_shape_features_null():
    out = build_segments_from_instances([_instance()], **_KW)
    assert out["cross_track_error"].isna().all()
    assert out["shape_progress"].isna().all()
    # along-shape segment length still comes from stop progress_m, not the shape
    assert list(out["distance_to_stop"]) == [100.0, 100.0]


def test_out_of_range_traversal_dropped():
    out = build_segments_from_instances([_instance(vps=_vps((0, 3, 43)))], **_KW)
    assert list(out["segment_index"]) == [0]  # 40-min last leg > 30-min cap


def test_fewer_than_two_stops_yields_nothing():
    out = build_segments_from_instances([_instance(stops=_STOPS[:1])], **_KW)
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_null_progress_stops_filtered():
    stops = [_STOPS[0], _Stop("X", 2, _LAT, -71.0573, None), _STOPS[2]]
    out = build_segments_from_instances([_instance(stops=stops)], **_KW)
    # X is dropped -> A->C becomes the single remaining segment
    assert len(out) == 1
    assert list(out["from_stop_id"]) == ["A"]
    assert list(out["stop_id"]) == ["C"]


def test_empty_instances_returns_schema():
    out = build_segments_from_instances([], **_KW)
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_multiple_instances_concatenated_in_order():
    out = build_segments_from_instances([_instance(), _instance()], **_KW)
    assert len(out) == 4
    assert list(out["segment_index"]) == [0, 1, 0, 1]


def test_resolve_agency_temporal_known_and_override():
    tz, region = resolve_agency_temporal("mbta", None)
    assert tz == "America/New_York"
    assert region == "US_MA"
    tz2, _ = resolve_agency_temporal("bucr", "UTC")
    assert tz2 == "UTC"  # explicit override wins


def test_resolve_agency_temporal_unknown_raises():
    with pytest.raises(ValueError):
        resolve_agency_temporal("nope", None)
