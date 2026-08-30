"""Tests for the MBTA segment dataset builder
(``feature_engineering.mbta_segments``).

Deterministic synthetic GTFS (routes/trips/stops/stop_times/shapes frames,
mirroring ``feature_engineering.bucr_gtfs.load_gtfs``'s output shape) + a
synthetic raw VP frame carrying REAL trip_id/route_id -- no trip inference.
Mirrors ``feature_engineering/tests/test_segment_dataset_builder.py``'s style
and asserts schema/behavior parity with the bUCR builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.bucr_gtfs import BucrGtfs
from feature_engineering.mbta_segments import build_mbta_segment_dataset
from feature_engineering.segment_dataset_builder import OUTPUT_COLUMNS
from models.common.data import ETADataset

_LAT = 42.3600
# Three stops ~ every 0.0027 deg lon (~222 m at this latitude -- wider than
# bUCR's fixture spacing since MBTA's higher latitude shrinks a degree of
# longitude; must clear CLOSEST_APPROACH_FALLBACK_M=200m between adjacent
# stops or a neighbor's VP would be misdetected as "arrived").
_STOP_LONS = [-71.0600, -71.0573, -71.0546]
_ROUTE = "222"
_SHAPE = "shape_222_0"
_TRIP = "mbta_trip_1"


def _gtfs() -> BucrGtfs:
    routes = pd.DataFrame([{"route_id": _ROUTE, "route_short_name": "222", "route_type": 3}])
    trips = pd.DataFrame(
        [{"route_id": _ROUTE, "trip_id": _TRIP, "direction_id": 0, "shape_id": _SHAPE}]
    )
    stops = pd.DataFrame(
        [
            {"stop_id": f"S{i}", "stop_lat": _LAT, "stop_lon": lon}
            for i, lon in enumerate(_STOP_LONS)
        ]
    )
    stop_times = pd.DataFrame(
        [
            {
                "trip_id": _TRIP,
                "stop_id": f"S{i}",
                "stop_sequence": i + 1,
                "arrival_time": f"08:0{i}:00",
                "departure_time": f"08:0{i}:00",
            }
            for i in range(len(_STOP_LONS))
        ]
    )
    lons = np.linspace(-71.0600, -71.0546, 21)
    shapes = pd.DataFrame(
        [
            {
                "shape_id": _SHAPE,
                "shape_pt_lat": _LAT,
                "shape_pt_lon": float(lon),
                "shape_pt_sequence": i,
            }
            for i, lon in enumerate(lons)
        ]
    )
    return BucrGtfs(routes=routes, trips=trips, stops=stops, stop_times=stop_times, shapes=shapes)


def _vps_frame(arrival_minutes=(0, 3, 8), vehicle="V1", trip_id=None) -> pd.DataFrame:
    """One VP exactly at each stop, at the given minute offsets from 08:00 UTC."""
    trip_id = trip_id if trip_id is not None else _TRIP
    base = pd.Timestamp("2026-08-17T08:00:00Z")
    rows = []
    for lon, minute in zip(_STOP_LONS, arrival_minutes):
        rows.append(
            {
                "vehicle_id": vehicle,
                "ts": base + pd.Timedelta(minutes=minute),
                "lat": _LAT,
                "lon": lon,
                "bearing": 90.0,
                "speed": 0.0,
                "trip_id": trip_id,
                "current_stop_sequence": None,
                "current_status": "STOPPED_AT",
                "route_id": _ROUTE,
            }
        )
    return pd.DataFrame(rows)


def test_two_segments_from_three_stops():
    df = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    assert len(df) == 2
    assert list(df["segment_index"]) == [0, 1]
    assert df.loc[0, "time_to_arrival_seconds"] == 180.0
    assert df.loc[1, "time_to_arrival_seconds"] == 300.0


def test_segment_identifiers_and_stops_remaining():
    df = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    assert list(df["from_stop_id"]) == ["S0", "S1"]
    assert list(df["stop_id"]) == ["S1", "S2"]
    assert list(df["stop_sequence"]) == [2, 3]
    assert list(df["stops_ahead"]) == [1, 0]
    assert (df["route_id"] == _ROUTE).all()
    assert (df["trip_id"] == _TRIP).all()


def test_segment_length_is_along_shape_positive():
    df = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    assert (df["distance_to_stop"] > 0).all()
    assert df.loc[0, "distance_to_next_stop"] > 0
    assert pd.isna(df.loc[1, "distance_to_next_stop"])
    assert (df["shape_distance_to_stop"] == df["distance_to_stop"]).all()


def test_schema_matches_output_columns():
    df = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    assert list(df.columns) == OUTPUT_COLUMNS


def test_schema_satisfies_model_feature_contract():
    df = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    needed = set()
    for cols in ETADataset.FEATURE_GROUPS.values():
        needed.update(cols)
    missing = needed - set(df.columns)
    assert not missing, f"segment schema missing model-contract columns: {missing}"


def test_entry_kinematics_and_temporal():
    df = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    assert df["is_at_stop"].all()
    # 08:00 UTC -> 04:00 America/New_York (UTC-4 in August, EDT).
    assert (df["hour"] == 4).all()
    assert set(df.columns) >= {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def test_unreached_endpoint_drops_segment():
    df_in = _vps_frame()
    df_in.loc[1, "lat"] = _LAT + 0.05  # ~5.5 km off -> beyond 200 m fallback
    df_in.loc[1, "lon"] = -71.0573
    out = build_mbta_segment_dataset(df_in, _gtfs())
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_out_of_range_traversal_dropped():
    out = build_mbta_segment_dataset(
        _vps_frame(arrival_minutes=(0, 3, 43)), _gtfs()
    )  # 40 min last leg > 30 min MAX_SEGMENT_SECONDS
    assert list(out["segment_index"]) == [0]


def test_unknown_trip_id_skipped():
    df_in = _vps_frame(trip_id="not_in_gtfs")
    out = build_mbta_segment_dataset(df_in, _gtfs())
    assert out.empty


def test_null_trip_id_dropped():
    df_in = _vps_frame()
    df_in.loc[0, "trip_id"] = None
    out = build_mbta_segment_dataset(df_in, _gtfs())
    # Only S0's row (entry to segment 0) is dropped -- but the arrival index
    # for S0 now has no VP, so segment 0 loses its whole trip group (all rows
    # share the same trip_id here) once null trip_id rows are excluded from
    # matching -- assert no crash and a schema-correct result.
    assert list(out.columns) == OUTPUT_COLUMNS


def test_deterministic():
    a = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    b = build_mbta_segment_dataset(_vps_frame(), _gtfs())
    pd.testing.assert_frame_equal(a, b)


def test_empty_input_returns_schema():
    out = build_mbta_segment_dataset(
        pd.DataFrame(columns=["trip_id", "vehicle_id", "ts", "lat", "lon"]), _gtfs()
    )
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_two_trips_kept_separate():
    trip2 = "mbta_trip_2"
    gtfs = _gtfs()
    gtfs.trips.loc[len(gtfs.trips)] = {
        "route_id": _ROUTE,
        "trip_id": trip2,
        "direction_id": 0,
        "shape_id": _SHAPE,
    }
    extra_st = pd.DataFrame(
        [
            {
                "trip_id": trip2,
                "stop_id": f"S{i}",
                "stop_sequence": i + 1,
                "arrival_time": f"08:0{i}:00",
                "departure_time": f"08:0{i}:00",
            }
            for i in range(len(_STOP_LONS))
        ]
    )
    gtfs = BucrGtfs(
        routes=gtfs.routes,
        trips=gtfs.trips,
        stops=gtfs.stops,
        stop_times=pd.concat([gtfs.stop_times, extra_st], ignore_index=True),
        shapes=gtfs.shapes,
    )

    a = _vps_frame(vehicle="V1")
    b = _vps_frame(arrival_minutes=(0, 2, 6), vehicle="V2", trip_id=trip2)
    out = build_mbta_segment_dataset(pd.concat([a, b], ignore_index=True), gtfs)
    assert out["trip_id"].nunique() == 2
    assert len(out) == 4


def test_type_errors():
    with pytest.raises(TypeError):
        build_mbta_segment_dataset([1, 2, 3], _gtfs())
    with pytest.raises(TypeError):
        build_mbta_segment_dataset(_vps_frame(), object())


def test_missing_required_column_raises():
    df = _vps_frame().drop(columns=["lat"])
    with pytest.raises(ValueError):
        build_mbta_segment_dataset(df, _gtfs())
