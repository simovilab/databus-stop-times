"""Tests for BUCR trip/route/stop inference (feature_engineering.bucr_trip_inference).

Two kinds of coverage, mirroring ``test_bucr_gtfs.py``:
  * Deterministic SYNTHETIC candidates/traces (hand-built, geographically
    well-separated shapes) for the core behaviors -- direction+variant
    assignment, trip-instance segmentation, stop assignment, anomaly
    rejection, ambiguity parking, determinism, output shape.
  * A couple of smoke checks against the REAL BUCR feed (skipped if the
    local scratchpad snapshot isn't present), including the real feed's
    inherent near-ambiguity between shape variants that share most of their
    path (see the module-level docstring note in this file's own report).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from etaval.domain.models import Stop as EtavalStop
from etaval.spatial.polyline import assign_stops_monotonic, build_polyline

from feature_engineering.bucr_gtfs import (
    RouteDirectionCandidate,
    load_gtfs,
    route_direction_candidates,
    route_directions,
)
from feature_engineering.bucr_trip_inference import (
    ANOMALY_CROSS_TRACK_M,
    OUTPUT_COLUMNS,
    REQUIRED_VP_COLUMNS,
    InferenceStats,
    infer_bucr_trips,
)

_REAL_FEED_ZIP = Path(
    "/private/tmp/claude-501/-Users-dotj-Desktop-SIMOVI-git-no-sync-gtfs-eta"
    "/011eb36e-aa69-42fd-a73f-95fb1c198afb/scratchpad/bucr_gtfs.zip"
)

requires_real_feed = pytest.mark.skipif(
    not _REAL_FEED_ZIP.exists(), reason="real BUCR GTFS snapshot not present locally"
)

# ---------------------------------------------------------------------------
# Synthetic candidates: two geographically well-separated straight shapes, so
# cross-track scoring cleanly distinguishes them (unlike the real feed's
# milla-detour variants, which mostly overlap -- see requires_real_feed
# ambiguity test below for that case).
# ---------------------------------------------------------------------------


def _line_points(lat0: float, lon0: float, lat1: float, lon1: float, n: int = 40) -> List[tuple]:
    lats = np.linspace(lat0, lat1, n)
    lons = np.linspace(lon0, lon1, n)
    return list(zip(lats.tolist(), lons.tolist()))


def _make_candidate(
    route_id: str, direction_id: int, shape_id: str, pts: List[tuple], n_stops: int = 5
) -> RouteDirectionCandidate:
    polyline = build_polyline(pts)
    idxs = np.linspace(0, len(polyline) - 1, n_stops).astype(int)
    stops = [
        EtavalStop(
            stop_id=f"{shape_id}-S{i + 1}",
            stop_sequence=i + 1,
            lat=polyline[idx].lat,
            lon=polyline[idx].lon,
        )
        for i, idx in enumerate(idxs)
    ]
    assigned_stops = assign_stops_monotonic(stops, polyline)
    return RouteDirectionCandidate(
        route_id=route_id,
        direction_id=direction_id,
        shape_id=shape_id,
        representative_trip_id=f"trip-{shape_id}",
        polyline=polyline,
        stops=assigned_stops,
    )


# Shape A: a ~300 m diagonal line near UCR. Shape B: an unrelated shape
# ~1.7 km north -- far enough that cross-track scoring never confuses them.
_SHAPE_A_PTS = _line_points(9.9300, -84.0500, 9.9320, -84.0480)
_SHAPE_B_PTS = _line_points(9.9450, -84.0500, 9.9470, -84.0480)

_CANDIDATE_A = _make_candidate("R1", 0, "shapeA", _SHAPE_A_PTS)
_CANDIDATE_B = _make_candidate("R1", 1, "shapeB", _SHAPE_B_PTS)
_CANDIDATES = [_CANDIDATE_A, _CANDIDATE_B]

# Shape C: an exact geographic duplicate of shape A (different route/shape
# id) -- deliberately used to force a near-zero scoring gap so a trace
# walking shape A is genuinely ambiguous between A and C.
_CANDIDATE_C = _make_candidate("R2", 0, "shapeC", _SHAPE_A_PTS)
_AMBIGUOUS_CANDIDATES = [_CANDIDATE_A, _CANDIDATE_C]


def _walk_shape_trace(
    candidate: RouteDirectionCandidate,
    vehicle_id: str = "V1",
    start_ts: pd.Timestamp = pd.Timestamp("2026-08-24T12:00:00Z"),
    seed: int = 0,
    dt_seconds: float = 8.0,
    noise_sigma_deg: float = 0.00003,  # ~3 m
    status: str = "IN_TRANSIT_TO",
) -> pd.DataFrame:
    """Build a canonical-VP-shaped synthetic trace walking one candidate's polyline in order."""
    rng = np.random.default_rng(seed)
    poly = candidate.polyline
    n = len(poly)
    rows = {
        "vehicle_id": [vehicle_id] * n,
        "ts": [start_ts + pd.Timedelta(seconds=dt_seconds * i) for i in range(n)],
        "lat": [p.lat + rng.normal(0, noise_sigma_deg) for p in poly],
        "lon": [p.lon + rng.normal(0, noise_sigma_deg) for p in poly],
        "bearing": [np.nan] * n,
        "speed": [3.0] * n,
        "current_status": [status] * n,
        "odometer_km": [np.nan] * n,
    }
    return pd.DataFrame(rows)


def _concat_traces(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _dwell_trace(
    lat: float,
    lon: float,
    start_ts: pd.Timestamp,
    duration_seconds: float,
    dt_seconds: float = 8.0,
    vehicle_id: str = "V1",
    seed: int = 100,
    noise_sigma_deg: float = 0.00003,  # ~3 m, inside PRESEGMENT_DWELL_RADIUS_M
) -> pd.DataFrame:
    """A stationary dwell: fixes hovering (within GPS noise) around one point.

    Models the terminal turnaround between an out trip and the back trip --
    the direction-agnostic boundary signal ``presegment_boundaries`` splits on.
    """
    rng = np.random.default_rng(seed)
    n = max(2, int(duration_seconds // dt_seconds) + 1)
    rows = {
        "vehicle_id": [vehicle_id] * n,
        "ts": [start_ts + pd.Timedelta(seconds=dt_seconds * i) for i in range(n)],
        "lat": [lat + rng.normal(0, noise_sigma_deg) for _ in range(n)],
        "lon": [lon + rng.normal(0, noise_sigma_deg) for _ in range(n)],
        "bearing": [np.nan] * n,
        "speed": [0.0] * n,
        "current_status": ["STOPPED_AT"] * n,
        "odometer_km": [np.nan] * n,
    }
    return pd.DataFrame(rows)


# An out-and-back loop for one route -- the realistic BUCR round trip. OUT and
# BACK SHARE their terminal endpoints (west terminal P0, east terminal P1 --
# where the turnaround dwell sits, on-track for both) but take GEOMETRICALLY
# DISTINCT middles: OUT runs straight along the south street (direction 0),
# BACK returns bowing ~110 m north along a parallel street (direction 1). Like
# real BUCR one-way campus streets, the divergent middle lets cross-track
# scoring tell the two directions apart per-segment; the point of the
# pre-segmentation architecture is that a SINGLE vehicle-day trace contains
# BOTH, so assigning one direction to the whole day is wrong -- each leg must be
# assigned on its own coarse segment.
_WEST_TERMINAL = (9.9300, -84.0500)
_EAST_TERMINAL = (9.9300, -84.0460)
_OUT_PTS = _line_points(*_WEST_TERMINAL, *_EAST_TERMINAL)
# BACK: east terminal -> north-bowed midpoint -> west terminal (shares both ends).
_BACK_PTS = (
    _line_points(*_EAST_TERMINAL, 9.9310, -84.0480, n=20)
    + _line_points(9.9310, -84.0480, *_WEST_TERMINAL, n=20)
)
_CANDIDATE_OUT = _make_candidate("R1", 0, "shapeOut", _OUT_PTS)
_CANDIDATE_BACK = _make_candidate("R1", 1, "shapeBack", _BACK_PTS)
_ROUNDTRIP_CANDIDATES = [_CANDIDATE_OUT, _CANDIDATE_BACK]


# ---------------------------------------------------------------------------
# Direction+variant assignment, stop monotonicity
# ---------------------------------------------------------------------------


def test_single_loop_assigns_correct_direction_and_variant() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1)
    out, stats = infer_bucr_trips(trace, _CANDIDATES)

    assert not out.empty
    assert (out["route_id"] == "R1").all()
    assert stats.points_parked_ambiguous == 0
    assert stats.points_anomaly_rejected == 0
    assert stats.trips_per_direction_variant == (("R1", 0, "shapeA", 1),)
    assert stats.trips_inferred == 1


def test_single_loop_stop_sequence_monotonic() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1)
    out, _ = infer_bucr_trips(trace, _CANDIDATES)

    seqs = out["current_stop_sequence"].to_numpy()
    assert (np.diff(seqs) >= 0).all()
    assert seqs.min() == 1
    assert seqs.max() == 5  # 5 stops synthesized for shapeA
    assert out["stop_id"].notna().all()


def test_single_loop_single_trip_instance() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1)
    out, stats = infer_bucr_trips(trace, _CANDIDATES)

    assert out["trip_id"].nunique() == 1
    assert stats.trips_inferred == 1


# ---------------------------------------------------------------------------
# Trip-instance segmentation: two consecutive loops (progress reset)
# ---------------------------------------------------------------------------


def test_two_consecutive_loops_yield_two_trip_instances() -> None:
    loop1 = _walk_shape_trace(_CANDIDATE_A, seed=1, start_ts=pd.Timestamp("2026-08-24T12:00:00Z"))
    loop2_start = loop1["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    loop2 = _walk_shape_trace(_CANDIDATE_A, seed=2, start_ts=loop2_start)
    trace = _concat_traces(loop1, loop2)

    out, stats = infer_bucr_trips(trace, _CANDIDATES)

    trip_ids = out["trip_id"].unique().tolist()
    assert len(trip_ids) == 2
    assert stats.trips_inferred == 2
    assert stats.trips_per_direction_variant == (("R1", 0, "shapeA", 2),)

    # Each instance's stop sequence is independently monotonic, within
    # 1..5 (GPS noise can occasionally push the very first fix past stop 1's
    # progress before it's observed, so the instance minimum is not always
    # exactly 1 -- see this file's report for the full explanation).
    for tid in trip_ids:
        seqs = out.loc[out["trip_id"] == tid, "current_stop_sequence"].to_numpy()
        assert (np.diff(seqs) >= 0).all()
        assert 1 <= seqs.min() <= 2
        assert seqs.max() == 5


def test_large_time_gap_also_splits_trip_instance() -> None:
    loop1 = _walk_shape_trace(_CANDIDATE_A, seed=1, start_ts=pd.Timestamp("2026-08-24T12:00:00Z"))
    # A big silent gap (> MAX_GAP_SECONDS) even without a full progress reset.
    loop2_start = loop1["ts"].iloc[-1] + pd.Timedelta(hours=2)
    loop2 = _walk_shape_trace(_CANDIDATE_A, seed=3, start_ts=loop2_start)
    trace = _concat_traces(loop1, loop2)

    out, stats = infer_bucr_trips(trace, _CANDIDATES)
    assert out["trip_id"].nunique() == 2
    assert stats.trips_inferred == 2


# ---------------------------------------------------------------------------
# Interleaved directions within one vehicle-day (the Step-4 architectural fix):
# a round trip (out in direction 0, back in direction 1) separated by a
# terminal dwell must assign BOTH directions -- NOT park the whole day as
# ambiguous, which is what assigning a single direction to the mixed-direction
# vehicle-day did before pre-segmentation was introduced.
# ---------------------------------------------------------------------------


def test_roundtrip_separated_by_dwell_assigns_both_directions() -> None:
    t0 = pd.Timestamp("2026-08-24T12:00:00Z")
    out_leg = _walk_shape_trace(_CANDIDATE_OUT, seed=1, start_ts=t0)
    # Terminal turnaround at P1 (the OUT shape's end == the BACK shape's start),
    # longer than PRESEGMENT_DWELL_SECONDS so it becomes a coarse boundary.
    dwell_start = out_leg["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    dwell = _dwell_trace(
        lat=_BACK_PTS[0][0], lon=_BACK_PTS[0][1], start_ts=dwell_start, duration_seconds=180.0
    )
    back_start = dwell["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    back_leg = _walk_shape_trace(_CANDIDATE_BACK, seed=2, start_ts=back_start)
    trace = _concat_traces(out_leg, dwell, back_leg)

    out, stats = infer_bucr_trips(trace, _ROUNDTRIP_CANDIDATES)

    # Both directions recovered, nothing parked.
    assert stats.points_parked_ambiguous == 0
    assert stats.trips_inferred == 2
    assert set(stats.trips_per_direction_variant) == {
        ("R1", 0, "shapeOut", 1),
        ("R1", 1, "shapeBack", 1),
    }


def test_roundtrip_without_a_dwell_collapses_into_a_single_direction() -> None:
    """Guards the premise of the fix: the SAME out+back traversal with NO dwell
    between the legs stays in one coarse segment, so a single direction is
    forced onto the whole trace -- the opposite leg is silently mislabeled
    rather than recovered as its own direction. This is exactly the loss the
    dwell-separated case above avoids, so the win there comes from
    pre-segmentation, not from the two shapes being trivially separable."""
    t0 = pd.Timestamp("2026-08-24T12:00:00Z")
    out_leg = _walk_shape_trace(_CANDIDATE_OUT, seed=1, start_ts=t0)
    back_start = out_leg["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    back_leg = _walk_shape_trace(_CANDIDATE_BACK, seed=2, start_ts=back_start)
    trace = _concat_traces(out_leg, back_leg)  # NO dwell -> one coarse segment

    _out, stats = infer_bucr_trips(trace, _ROUNDTRIP_CANDIDATES)

    # Only ONE direction is recovered -- both legs collapse together (contrast
    # the two directions recovered once a dwell splits them, above).
    recovered_directions = {d for _r, d, _s, _c in stats.trips_per_direction_variant}
    assert len(recovered_directions) == 1


# ---------------------------------------------------------------------------
# Anomaly rejection
# ---------------------------------------------------------------------------


def test_off_route_excursion_is_anomaly_rejected_and_trip_still_infers() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1).reset_index(drop=True)
    # Inject two off-route points (~1.1 km away -- far beyond ANOMALY_CROSS_TRACK_M)
    # in the middle of the trace, keeping their timestamps in-sequence so no
    # large gap is introduced once they're dropped.
    mid = len(trace) // 2
    anomaly_rows = trace.iloc[[mid, mid]].copy().reset_index(drop=True)
    anomaly_rows["lat"] = anomaly_rows["lat"] + 0.01
    anomaly_rows["lon"] = anomaly_rows["lon"] + 0.01
    anomaly_rows["ts"] = [
        trace["ts"].iloc[mid] + pd.Timedelta(seconds=2),
        trace["ts"].iloc[mid] + pd.Timedelta(seconds=4),
    ]

    trace_with_anomaly = pd.concat(
        [trace.iloc[: mid + 1], anomaly_rows, trace.iloc[mid + 1 :]], ignore_index=True
    ).sort_values("ts").reset_index(drop=True)

    out, stats = infer_bucr_trips(trace_with_anomaly, _CANDIDATES)

    assert stats.points_anomaly_rejected == 2
    assert stats.points_in == len(trace_with_anomaly)
    # The surrounding trip still infers as (at least) one instance.
    assert stats.trips_inferred >= 1
    assert not out.empty
    assert (out["route_id"] == "R1").all()


# ---------------------------------------------------------------------------
# Ambiguity parking
# ---------------------------------------------------------------------------


def test_ambiguous_trace_is_parked_not_force_assigned() -> None:
    # Shape C is a geographic duplicate of shape A -- a trace walking A's
    # path scores nearly identically against both, so it must be parked.
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1)
    out, stats = infer_bucr_trips(trace, _AMBIGUOUS_CANDIDATES)

    assert out.empty
    assert stats.points_parked_ambiguous == len(trace)
    assert stats.points_assigned == 0
    assert stats.trips_inferred == 0
    assert stats.trips_per_direction_variant == ()


# ---------------------------------------------------------------------------
# Output shape / dtypes
# ---------------------------------------------------------------------------


def test_output_columns_match_expected_schema() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1)
    out, _ = infer_bucr_trips(trace, _CANDIDATES)

    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["trip_id"].notna().all()
    assert out["route_id"].notna().all()
    assert out["stop_id"].notna().all()
    assert out["current_stop_sequence"].notna().all()
    assert str(out["current_stop_sequence"].dtype) == "Int64"
    assert isinstance(out["ts"].dtype, pd.DatetimeTZDtype)


def test_empty_input_returns_empty_frame_with_correct_columns() -> None:
    empty = pd.DataFrame(columns=REQUIRED_VP_COLUMNS)
    out, stats = infer_bucr_trips(empty, _CANDIDATES)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.empty
    assert stats == InferenceStats(
        points_in=0,
        points_assigned=0,
        points_parked_ambiguous=0,
        points_anomaly_rejected=0,
        points_dropped_as_noise=0,
        trips_inferred=0,
        instances_dropped_as_noise=0,
        trips_per_direction_variant=(),
    )


# ---------------------------------------------------------------------------
# presegment_boundaries (direction-agnostic coarse splitting) -- unit tests
# ---------------------------------------------------------------------------


def test_presegment_splits_on_long_stationary_dwell() -> None:
    from feature_engineering.bucr_trip_scoring import presegment_boundaries

    t0 = pd.Timestamp("2026-08-24T12:00:00Z")
    out_leg = _walk_shape_trace(_CANDIDATE_OUT, seed=1, start_ts=t0)
    dwell_start = out_leg["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    dwell = _dwell_trace(
        lat=_BACK_PTS[0][0], lon=_BACK_PTS[0][1], start_ts=dwell_start, duration_seconds=180.0
    )
    back_start = dwell["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    back_leg = _walk_shape_trace(_CANDIDATE_BACK, seed=2, start_ts=back_start)
    trace = _concat_traces(out_leg, dwell, back_leg).sort_values("ts").reset_index(drop=True)

    boundary = presegment_boundaries(
        trace["ts"], trace["lat"].to_numpy(float), trace["lon"].to_numpy(float)
    )
    assert boundary[0]
    # Exactly one interior boundary: where movement resumes after the dwell.
    assert boundary.sum() == 2
    resume_idx = int(np.flatnonzero(boundary)[1])
    # It lands at the first BACK-leg fix (movement out of the dwell radius).
    assert resume_idx >= len(out_leg) + len(dwell) - 1


def test_presegment_does_not_split_a_single_continuous_trip() -> None:
    from feature_engineering.bucr_trip_scoring import presegment_boundaries

    trace = _walk_shape_trace(_CANDIDATE_OUT, seed=1)
    boundary = presegment_boundaries(
        trace["ts"], trace["lat"].to_numpy(float), trace["lon"].to_numpy(float)
    )
    assert boundary[0]
    assert boundary.sum() == 1  # only the start; no interior split


def test_presegment_splits_on_long_time_gap() -> None:
    from feature_engineering.bucr_trip_scoring import presegment_boundaries

    leg1 = _walk_shape_trace(_CANDIDATE_OUT, seed=1, start_ts=pd.Timestamp("2026-08-24T12:00:00Z"))
    leg2_start = leg1["ts"].iloc[-1] + pd.Timedelta(hours=1)  # > PRESEGMENT_MAX_GAP_SECONDS
    leg2 = _walk_shape_trace(_CANDIDATE_OUT, seed=2, start_ts=leg2_start)
    trace = _concat_traces(leg1, leg2)

    boundary = presegment_boundaries(
        trace["ts"], trace["lat"].to_numpy(float), trace["lon"].to_numpy(float)
    )
    assert boundary.sum() == 2
    assert bool(boundary[len(leg1)])  # boundary exactly at the post-gap fix


def test_presegment_ignores_a_brief_stop() -> None:
    """A short in-service stop (below PRESEGMENT_DWELL_SECONDS) must NOT split a
    trip -- only real terminal turnarounds do."""
    from feature_engineering.bucr_trip_scoring import presegment_boundaries

    t0 = pd.Timestamp("2026-08-24T12:00:00Z")
    leg1 = _walk_shape_trace(_CANDIDATE_OUT, seed=1, start_ts=t0)
    stop_start = leg1["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    brief_stop = _dwell_trace(
        lat=_OUT_PTS[-1][0], lon=_OUT_PTS[-1][1], start_ts=stop_start, duration_seconds=24.0
    )
    resume_start = brief_stop["ts"].iloc[-1] + pd.Timedelta(seconds=8)
    leg2 = _walk_shape_trace(_CANDIDATE_OUT, seed=2, start_ts=resume_start)
    trace = _concat_traces(leg1, brief_stop, leg2).sort_values("ts").reset_index(drop=True)

    boundary = presegment_boundaries(
        trace["ts"], trace["lat"].to_numpy(float), trace["lon"].to_numpy(float)
    )
    assert boundary.sum() == 1  # brief stop ignored


def test_presegment_empty_and_single_point() -> None:
    from feature_engineering.bucr_trip_scoring import presegment_boundaries

    empty = presegment_boundaries(pd.Series([], dtype="datetime64[ns, UTC]"),
                                  np.array([]), np.array([]))
    assert empty.shape == (0,)

    one = presegment_boundaries(
        pd.Series([pd.Timestamp("2026-08-24T12:00:00Z")]), np.array([9.93]), np.array([-84.05])
    )
    assert one.tolist() == [True]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_input_twice() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=7)
    out1, stats1 = infer_bucr_trips(trace, _CANDIDATES)
    out2, stats2 = infer_bucr_trips(trace, _CANDIDATES)

    pd.testing.assert_frame_equal(out1, out2)
    assert stats1 == stats2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_non_dataframe_raises_type_error() -> None:
    with pytest.raises(TypeError):
        infer_bucr_trips([1, 2, 3], _CANDIDATES)  # type: ignore[arg-type]


def test_missing_required_columns_raises_value_error() -> None:
    bad = pd.DataFrame({"vehicle_id": ["V1"], "ts": [pd.Timestamp.utcnow()]})
    with pytest.raises(ValueError, match="missing required columns"):
        infer_bucr_trips(bad, _CANDIDATES)


def test_empty_candidates_raises_value_error() -> None:
    trace = _walk_shape_trace(_CANDIDATE_A, seed=1)
    with pytest.raises(ValueError, match="candidates must not be empty"):
        infer_bucr_trips(trace, [])


# ---------------------------------------------------------------------------
# Real BUCR feed (S3 snapshot) -- skipped if not present locally.
# ---------------------------------------------------------------------------


@requires_real_feed
def test_real_feed_smoke_no_crash_and_stats_consistent() -> None:
    data = load_gtfs(_REAL_FEED_ZIP)
    all_candidates: List[RouteDirectionCandidate] = []
    for route_id, direction_id in route_directions(data):
        all_candidates.extend(route_direction_candidates(data, route_id, direction_id))

    cand = next(c for c in all_candidates if c.shape_id == "desde_educacion_a_odontologia_sin_milla")
    trace = _walk_shape_trace(cand, seed=1)

    out, stats = infer_bucr_trips(trace, all_candidates)

    assert stats.points_in == len(trace)
    assert stats.points_accounted_for == stats.points_in
    assert len(out) == stats.points_assigned


@requires_real_feed
def test_real_feed_no_detour_trace_resolves_to_its_own_sin_milla_shape() -> None:
    """A no-detour ("sin_milla") trace is NOT parked as ambiguous against its
    same-origin "con_milla" sibling, even though the two variants share
    almost their entire path.

    This replaces the old
    ``test_real_feed_milla_variants_are_ambiguous_for_a_shared_leg_trace``,
    which encoded the OLD (broken) median-cross-track behavior: it asserted
    a sin_milla trace got parked because it scored within a few metres of
    its con_milla sibling. That was wrong -- the sin_milla trace never
    visits the con_milla variant's detour-only stops, which is itself
    decisive evidence for sin_milla, not grounds for ambiguity. The fixed
    ``resolve_variant_by_stop_coverage`` (see bucr_trip_scoring.py) detects
    exactly this: the con_milla candidate is missing required (detour) stop
    visitation, so it loses to the sin_milla candidate outright. Parking is
    reserved for genuine cross-direction ambiguity or a real all-stops tie,
    neither of which applies here.
    """
    data = load_gtfs(_REAL_FEED_ZIP)
    all_candidates: List[RouteDirectionCandidate] = []
    for route_id, direction_id in route_directions(data):
        all_candidates.extend(route_direction_candidates(data, route_id, direction_id))

    cand = next(c for c in all_candidates if c.shape_id == "desde_artes_a_odontologia_sin_milla")
    trace = _walk_shape_trace(cand, seed=1)

    out, stats = infer_bucr_trips(trace, all_candidates)

    assert not out.empty
    assert stats.points_parked_ambiguous == 0
    assert len(stats.trips_per_direction_variant) == 1
    got_route, got_direction, got_shape, _n_trips = stats.trips_per_direction_variant[0]
    assert (got_route, got_direction, got_shape) == (
        "bUCR",
        0,
        "desde_artes_a_odontologia_sin_milla",
    )


@requires_real_feed
@pytest.mark.parametrize(
    "shape_id",
    [
        "desde_educacion_a_odontologia_sin_milla",
        "desde_educacion_a_odontologia_con_milla",
        "desde_artes_a_odontologia_sin_milla",
        "desde_artes_a_odontologia_con_milla",
        "desde_odontologia_a_educacion",
        "desde_edufi_a_educacion",
        "desde_odontologia_a_artes",
    ],
)
def test_real_feed_each_shape_recovers_exactly_itself(shape_id: str) -> None:
    """For each of the 7 real BUCR shapes, a clean walk of THAT shape's
    polyline, scored against the FULL 7-shape candidate set, recovers
    exactly that shape_id -- proving both discriminating dimensions (origin
    AND milla-detour) are resolved correctly, not just direction.

    This asserts exact (route_id, direction_id, shape_id) recovery, full
    point coverage (no parking, no anomaly rejection), and per-instance
    monotonic stop sequences. It deliberately does NOT assert a single trip
    instance overall: two of the seven real shapes (the "sin_milla"
    variants) contain a short self-crossing loop in their raw GTFS shape
    points (two points ~80-215 m apart in cumulative distance sit only a
    few metres apart geographically -- verified by inspecting the raw
    polyline; see this file's report). Walking through that loop makes
    nearest-point projection briefly jump backward past
    MONOTONIC_JITTER_TOLERANCE_M, which segment_boundaries (correctly, and
    unrelated to this candidate-selection fix -- segmentation is out of
    scope here and left as-is) treats as a new trip instance. It is a
    pre-existing property of those two shapes' geometry, not a
    scoring/selection defect: every point still lands on the correct
    shape_id, just split across more than one instance.
    """
    data = load_gtfs(_REAL_FEED_ZIP)
    all_candidates: List[RouteDirectionCandidate] = []
    for route_id, direction_id in route_directions(data):
        all_candidates.extend(route_direction_candidates(data, route_id, direction_id))
    assert len(all_candidates) == 7

    cand = next(c for c in all_candidates if c.shape_id == shape_id)
    trace = _walk_shape_trace(cand, seed=1)

    out, stats = infer_bucr_trips(trace, all_candidates)

    assert not out.empty
    assert stats.points_parked_ambiguous == 0
    assert stats.points_anomaly_rejected == 0
    assert len(out) == len(trace)
    assert (out["route_id"] == cand.route_id).all()
    # Exactly one (route, direction, shape) key -- no leakage to any rival
    # candidate, whatever the instance count.
    assert len(stats.trips_per_direction_variant) == 1
    got_route, got_direction, got_shape, _n_trips = stats.trips_per_direction_variant[0]
    assert (got_route, got_direction, got_shape) == (cand.route_id, cand.direction_id, shape_id)

    for tid in out["trip_id"].unique():
        seqs = out.loc[out["trip_id"] == tid, "current_stop_sequence"].to_numpy()
        assert (np.diff(seqs) >= 0).all()
