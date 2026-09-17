"""Tests for the BUCR segment dataset builder
(``feature_engineering.segment_dataset_builder``).

Deterministic synthetic candidates + inferred VP frames: a straight
west->east shape with three evenly spaced stops, a vehicle that arrives at
each in turn, and assertions on segment count, targets, feature semantics,
schema parity with ``ETADataset.FEATURE_GROUPS``, and the drop rules
(un-reached endpoints, out-of-range traversals, non-BUCR/unknown trip ids).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etaval.domain.models import Stop as EtavalStop
from etaval.spatial.polyline import assign_stops_monotonic, build_polyline

from feature_engineering.bucr_gtfs import RouteDirectionCandidate
from feature_engineering.segment_dataset_builder import (
    OUTPUT_COLUMNS,
    build_bucr_segment_dataset,
)
from models.common.data import ETADataset

_LAT = 9.9300
# Three stops ~ every 0.0020 deg lon (~220 m at this latitude).
_STOP_LONS = [-84.0500, -84.0480, -84.0460]
_ROUTE = "R1"
_DIR = 0
_SHAPE = "shapeOut"


def _candidate() -> RouteDirectionCandidate:
    # Dense polyline so cross-track projection is well defined.
    lons = np.linspace(-84.0500, -84.0460, 21)
    polyline = build_polyline([(_LAT, float(x)) for x in lons])
    stops = [
        EtavalStop(stop_id=f"S{i}", stop_sequence=i + 1, lat=_LAT, lon=lon)
        for i, lon in enumerate(_STOP_LONS)
    ]
    stops = assign_stops_monotonic(stops, polyline)
    return RouteDirectionCandidate(
        route_id=_ROUTE,
        direction_id=_DIR,
        shape_id=_SHAPE,
        representative_trip_id="t0",
        polyline=polyline,
        stops=stops,
    )


def _trip_id(start: str = "20260101T080000Z", vehicle: str = "BUS1") -> str:
    return f"bucr:{vehicle}:{_ROUTE}:{_DIR}:{_SHAPE}:{start}"


def _inferred_frame(arrival_minutes=(0, 3, 8), vehicle="BUS1", trip_id=None) -> pd.DataFrame:
    """One VP exactly at each stop, at the given minute offsets from 08:00 UTC."""
    trip_id = trip_id or _trip_id(vehicle=vehicle)
    base = pd.Timestamp("2026-01-01T08:00:00Z")
    rows = []
    for lon, minute in zip(_STOP_LONS, arrival_minutes):
        rows.append(
            {
                "trip_id": trip_id,
                "vehicle_id": vehicle,
                "ts": base + pd.Timedelta(minutes=minute),
                "lat": _LAT,
                "lon": lon,
                "bearing": 90.0,
                "speed": 0.0,
                "current_stop_sequence": None,
                "current_status": "STOPPED_AT",
                "stop_id": None,
                "route_id": _ROUTE,
            }
        )
    return pd.DataFrame(rows)


def test_two_segments_from_three_stops():
    df = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    assert len(df) == 2
    assert list(df["segment_index"]) == [0, 1]
    # Targets = 3 min and 5 min between arrivals.
    assert df.loc[0, "time_to_arrival_seconds"] == 180.0
    assert df.loc[1, "time_to_arrival_seconds"] == 300.0


def test_segment_identifiers_and_stops_remaining():
    df = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    # segment 0: from S0 -> S1 ; segment 1: from S1 -> S2
    assert list(df["from_stop_id"]) == ["S0", "S1"]
    assert list(df["stop_id"]) == ["S1", "S2"]
    assert list(df["stop_sequence"]) == [2, 3]
    # stops-remaining after destination: 1 then 0.
    assert list(df["stops_ahead"]) == [1, 0]


def test_segment_length_is_along_shape_positive():
    df = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    assert (df["distance_to_stop"] > 0).all()
    # distance_to_next_stop is set for the first segment, NaN for the last.
    assert df.loc[0, "distance_to_next_stop"] > 0
    assert pd.isna(df.loc[1, "distance_to_next_stop"])
    # shape_distance_to_stop mirrors the along-shape segment length.
    assert (df["shape_distance_to_stop"] == df["distance_to_stop"]).all()


def test_schema_satisfies_model_feature_contract():
    df = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    needed = set()
    for cols in ETADataset.FEATURE_GROUPS.values():
        needed.update(cols)
    missing = needed - set(df.columns)
    assert not missing, f"segment schema missing model-contract columns: {missing}"


def test_entry_kinematics_and_temporal():
    df = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    assert df["is_at_stop"].all()  # entry VPs reported STOPPED_AT
    # 08:00 UTC -> 02:00 America/Costa_Rica (UTC-6).
    assert (df["hour"] == 2).all()
    assert set(df.columns) >= {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def test_unreached_endpoint_drops_segment():
    # Move the middle stop's arrival VP far away so S1 is never "reached".
    df_in = _inferred_frame()
    df_in.loc[1, "lat"] = _LAT + 0.05  # ~5.5 km off -> beyond 200 m fallback
    df_in.loc[1, "lon"] = -84.0480
    out = build_bucr_segment_dataset(df_in, [_candidate()])
    # Both segments touch S1, so both drop.
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_out_of_range_traversal_dropped():
    # Second arrival 40 min later -> exceeds MAX_SEGMENT_SECONDS default (30 min).
    out = build_bucr_segment_dataset(_inferred_frame(arrival_minutes=(0, 3, 43)), [_candidate()])
    assert list(out["segment_index"]) == [0]  # only S0->S1 survives


def test_non_bucr_trip_id_skipped():
    df_in = _inferred_frame(trip_id="mbta:BUS1:R1:0:shapeOut:x")
    out = build_bucr_segment_dataset(df_in, [_candidate()])
    assert out.empty


def test_unknown_candidate_key_skipped():
    df_in = _inferred_frame(trip_id=_trip_id().replace(_SHAPE, "ghostShape"))
    out = build_bucr_segment_dataset(df_in, [_candidate()])
    assert out.empty


def test_deterministic():
    a = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    b = build_bucr_segment_dataset(_inferred_frame(), [_candidate()])
    pd.testing.assert_frame_equal(a, b)


def test_empty_input_returns_schema():
    out = build_bucr_segment_dataset(pd.DataFrame(columns=["trip_id", "vehicle_id", "ts", "lat", "lon"]), [_candidate()])
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_two_trips_kept_separate():
    a = _inferred_frame(vehicle="BUS1")
    b = _inferred_frame(arrival_minutes=(0, 2, 6), vehicle="BUS2")
    out = build_bucr_segment_dataset(pd.concat([a, b], ignore_index=True), [_candidate()])
    assert out["trip_id"].nunique() == 2
    assert len(out) == 4
