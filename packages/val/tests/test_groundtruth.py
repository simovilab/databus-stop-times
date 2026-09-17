"""Tests for P4 — ground-truth detectors.

Scenario
--------
A bus travels along a straight route with 4 scheduled stops (A, B, C, FAR).
The position history is a chronological list of VehicleState samples with known
progress_m, lat/lon, and optional current_status fields.

Stop layout (progress_m along shape):
  A → 100 m    B → 200 m    C → 300 m    FAR → 5 000 m (vehicle never reaches it)

Position history (progress advances monotonically):
  t=1000  pm=50   (before A)
  t=1010  pm=100  (at A)
  t=1020  pm=200  (at B)
  t=1030  pm=300  (at C)
  t=1040  pm=400  (past C, before FAR)
"""

from __future__ import annotations

import pytest

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.groundtruth.position_radius import PositionRadiusDetector
from stoptimes.val.groundtruth.shape_distance import ShapeDistanceDetector
from stoptimes.val.groundtruth.stop_status import StopStatusDetector

# ---------------------------------------------------------------------------
# Constants for stop lat/lon placement
# At the equator, 1° latitude ≈ 111 194 m, so:
#   0.0001° ≈ 11 m  → well inside 40 m radius
#   0.001°  ≈ 111 m → clearly outside 40 m radius
# ---------------------------------------------------------------------------

_STOP_LAT_A = 10.0000
_STOP_LON_A = 20.0000
_STOP_LAT_B = 10.0100
_STOP_LON_B = 20.0100
_STOP_LAT_C = 10.0200
_STOP_LON_C = 20.0200
_STOP_LAT_FAR = 11.0000
_STOP_LON_FAR = 21.0000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vs(
    epoch: float,
    lat: float,
    lon: float,
    *,
    pm: float | None = None,
    status: str | None = None,
    seq: int | None = None,
) -> VehicleState:
    """Compact VehicleState factory for test readability."""
    return VehicleState(
        epoch=epoch,
        lat=lat,
        lon=lon,
        progress_m=pm,
        current_status=status,
        current_stop_sequence=seq,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trip() -> TripInstance:
    return TripInstance(
        source="test",
        trip_id="T1",
        vehicle_id="V1",
        route_id="R1",
        direction_id=0,
        shape_id="S1",
        start_date="20240101",
        started_at=1000.0,
    )


@pytest.fixture()
def stops_progress() -> list[Stop]:
    """Stops with progress_m assigned (shape-distance tests)."""
    return [
        Stop(stop_id="A", stop_sequence=1, lat=_STOP_LAT_A, lon=_STOP_LON_A, progress_m=100.0),
        Stop(stop_id="B", stop_sequence=2, lat=_STOP_LAT_B, lon=_STOP_LON_B, progress_m=200.0),
        Stop(stop_id="C", stop_sequence=3, lat=_STOP_LAT_C, lon=_STOP_LON_C, progress_m=300.0),
        Stop(
            stop_id="FAR", stop_sequence=4,
            lat=_STOP_LAT_FAR, lon=_STOP_LON_FAR, progress_m=5000.0,
        ),
    ]


@pytest.fixture()
def history_progress() -> list[VehicleState]:
    """Position history with monotonic progress_m passing through stops A, B, C."""
    return [
        _vs(1000.0, _STOP_LAT_A - 0.001, _STOP_LON_A, pm=50.0),
        _vs(1010.0, _STOP_LAT_A, _STOP_LON_A, pm=100.0),
        _vs(1020.0, _STOP_LAT_B, _STOP_LON_B, pm=200.0),
        _vs(1030.0, _STOP_LAT_C, _STOP_LON_C, pm=300.0),
        _vs(1040.0, _STOP_LAT_C + 0.001, _STOP_LON_C, pm=400.0),
    ]


# ---------------------------------------------------------------------------
# ShapeDistanceDetector
# ---------------------------------------------------------------------------


class TestShapeDistanceDetector:
    def test_fires_for_all_stops_vehicle_reaches(
        self,
        trip: TripInstance,
        stops_progress: list[Stop],
        history_progress: list[VehicleState],
    ) -> None:
        det = ShapeDistanceDetector(shape_threshold_m=30.0)
        arrivals = det.detect(trip, history_progress, stops_progress[:3])  # A, B, C only

        assert len(arrivals) == 3
        by_stop = {a.stop_id: a for a in arrivals}
        assert by_stop["A"].actual_arrival_epoch == 1010.0
        assert by_stop["B"].actual_arrival_epoch == 1020.0
        assert by_stop["C"].actual_arrival_epoch == 1030.0

    def test_does_not_fire_for_unreached_stop(
        self,
        trip: TripInstance,
        stops_progress: list[Stop],
        history_progress: list[VehicleState],
    ) -> None:
        det = ShapeDistanceDetector(shape_threshold_m=30.0)
        arrivals = det.detect(trip, history_progress, stops_progress)  # includes FAR

        stop_ids = {a.stop_id for a in arrivals}
        assert "FAR" not in stop_ids
        assert {"A", "B", "C"}.issubset(stop_ids)

    def test_threshold_not_breached_no_fire(self, trip: TripInstance) -> None:
        # Closest approach to stop at pm=100 is pm=60 (delta=40 m > 30 m threshold).
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0, progress_m=100.0)]
        history = [
            _vs(1000.0, 0.0, 0.0, pm=60.0),   # delta=40 m
            _vs(1010.0, 0.0, 0.0, pm=200.0),  # delta=100 m
        ]
        det = ShapeDistanceDetector(shape_threshold_m=30.0)
        assert det.detect(trip, history, stops) == []

    def test_threshold_breached_fires(self, trip: TripInstance) -> None:
        # Closest approach delta is 20 m < 30 m threshold → should fire.
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0, progress_m=100.0)]
        history = [
            _vs(1000.0, 0.0, 0.0, pm=80.0),   # delta=20 m ← closest
            _vs(1010.0, 0.0, 0.0, pm=200.0),
        ]
        det = ShapeDistanceDetector(shape_threshold_m=30.0)
        arrivals = det.detect(trip, history, stops)
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1000.0

    def test_closest_approach_epoch_selected(self, trip: TripInstance) -> None:
        # Middle sample is closest in progress space; its epoch must be used.
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0, progress_m=100.0)]
        history = [
            _vs(1000.0, 0.0, 0.0, pm=80.0),   # delta=20
            _vs(1010.0, 0.0, 0.0, pm=95.0),   # delta=5 ← closest
            _vs(1020.0, 0.0, 0.0, pm=130.0),  # delta=30
        ]
        det = ShapeDistanceDetector(shape_threshold_m=30.0)
        arrivals = det.detect(trip, history, stops)
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1010.0

    def test_method_string(self) -> None:
        assert ShapeDistanceDetector().method == "shape_distance"

    def test_arrivals_carry_method(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0, progress_m=100.0)]
        history = [_vs(1000.0, 0.0, 0.0, pm=100.0)]
        arrivals = ShapeDistanceDetector().detect(trip, history, stops)
        assert len(arrivals) == 1
        assert arrivals[0].method == "shape_distance"

    def test_skips_states_without_progress_m(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0, progress_m=100.0)]
        history = [_vs(1000.0, 0.0, 0.0), _vs(1010.0, 0.0, 0.0)]
        assert ShapeDistanceDetector().detect(trip, history, stops) == []

    def test_skips_stops_without_progress_m(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]  # progress_m=None
        history = [_vs(1000.0, 0.0, 0.0, pm=100.0)]
        assert ShapeDistanceDetector().detect(trip, history, stops) == []

    def test_at_most_one_arrival_per_stop(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0, progress_m=100.0)]
        history = [
            _vs(1000.0, 0.0, 0.0, pm=100.0),
            _vs(1010.0, 0.0, 0.0, pm=100.0),
        ]
        arrivals = ShapeDistanceDetector().detect(trip, history, stops)
        assert len(arrivals) == 1


# ---------------------------------------------------------------------------
# PositionRadiusDetector — first-entry (no dwell)
# ---------------------------------------------------------------------------


class TestPositionRadiusDetectorFirstEntry:
    @pytest.fixture()
    def stops_latlon(self) -> list[Stop]:
        return [
            Stop(stop_id="A", stop_sequence=1, lat=_STOP_LAT_A, lon=_STOP_LON_A),
            Stop(stop_id="B", stop_sequence=2, lat=_STOP_LAT_B, lon=_STOP_LON_B),
            Stop(stop_id="C", stop_sequence=3, lat=_STOP_LAT_C, lon=_STOP_LON_C),
        ]

    def test_fires_for_each_stop_vehicle_approaches(
        self,
        trip: TripInstance,
        stops_latlon: list[Stop],
        history_progress: list[VehicleState],
    ) -> None:
        det = PositionRadiusDetector(arrival_radius_m=40.0, dwell_s=0.0)
        arrivals = det.detect(trip, history_progress, stops_latlon)
        stop_ids = {a.stop_id for a in arrivals}
        assert {"A", "B", "C"}.issubset(stop_ids)

    def test_does_not_fire_for_stop_vehicle_never_approaches(
        self, trip: TripInstance, history_progress: list[VehicleState]
    ) -> None:
        stops = [Stop(stop_id="FAR", stop_sequence=1, lat=_STOP_LAT_FAR, lon=_STOP_LON_FAR)]
        det = PositionRadiusDetector(arrival_radius_m=40.0)
        assert det.detect(trip, history_progress, stops) == []

    def test_arrival_epoch_is_first_entry(self, trip: TripInstance) -> None:
        # outside → inside → inside: epoch must be the first entry, not second.
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [
            _vs(1000.0, 0.001, 0.0),   # ~111 m → outside 40 m
            _vs(1010.0, 0.0001, 0.0),  # ~11 m  → inside  (first entry)
            _vs(1020.0, 0.0, 0.0),     # 0 m    → still inside
        ]
        arrivals = PositionRadiusDetector(arrival_radius_m=40.0).detect(trip, history, [stop])
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1010.0  # first entry epoch

    def test_method_string(self) -> None:
        assert PositionRadiusDetector().method == "position_radius"

    def test_arrivals_carry_method(self, trip: TripInstance) -> None:
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [_vs(1000.0, 0.0, 0.0)]
        arrivals = PositionRadiusDetector(arrival_radius_m=40.0).detect(trip, history, [stop])
        assert len(arrivals) == 1
        assert arrivals[0].method == "position_radius"

    def test_at_most_one_arrival_per_stop(self, trip: TripInstance) -> None:
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [_vs(float(t), 0.0, 0.0) for t in range(1000, 1030, 10)]
        arrivals = PositionRadiusDetector(arrival_radius_m=40.0).detect(trip, history, [stop])
        assert len(arrivals) == 1


# ---------------------------------------------------------------------------
# PositionRadiusDetector — dwell_s requirement
# ---------------------------------------------------------------------------


class TestPositionRadiusDetectorDwell:
    def test_brief_single_sample_does_not_confirm(self, trip: TripInstance) -> None:
        """One sample inside radius then exits: span=0 s < dwell_s=5 → no arrival."""
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [
            _vs(1000.0, 0.001, 0.0),   # outside (~111 m)
            _vs(1010.0, 0.0001, 0.0),  # inside  (~11 m) — span=0
            _vs(1011.0, 0.001, 0.0),   # outside again → reset
        ]
        det = PositionRadiusDetector(arrival_radius_m=40.0, dwell_s=5.0)
        assert det.detect(trip, history, [stop]) == []

    def test_sustained_presence_confirms_at_first_entry_epoch(
        self, trip: TripInstance
    ) -> None:
        """Vehicle stays inside ≥ dwell_s → confirmed at first_entry_epoch, not current."""
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [
            _vs(1000.0, 0.001, 0.0),   # outside
            _vs(1005.0, 0.0001, 0.0),  # inside, first_entry=1005, span=0
            _vs(1009.0, 0.0001, 0.0),  # inside, span=4 s < 5 → not yet
            _vs(1010.0, 0.0, 0.0),     # inside, span=5 s ≥ 5 → confirm
        ]
        det = PositionRadiusDetector(arrival_radius_m=40.0, dwell_s=5.0)
        arrivals = det.detect(trip, history, [stop])
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1005.0  # first entry, not 1010

    def test_exit_and_reentry_resets_dwell(self, trip: TripInstance) -> None:
        """Exiting the radius resets the window; re-entry starts a fresh window."""
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [
            _vs(1000.0, 0.0001, 0.0),  # inside, first_entry=1000
            _vs(1003.0, 0.001, 0.0),   # outside → reset (span was 3 s < 5)
            _vs(1010.0, 0.0001, 0.0),  # inside, new first_entry=1010, span=0
            _vs(1015.0, 0.0, 0.0),     # inside, span=5 s → confirm
        ]
        det = PositionRadiusDetector(arrival_radius_m=40.0, dwell_s=5.0)
        arrivals = det.detect(trip, history, [stop])
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1010.0  # second entry's first epoch

    def test_dwell_zero_behaves_like_first_entry(self, trip: TripInstance) -> None:
        stop = Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)
        history = [_vs(1000.0, 0.0, 0.0)]
        arrivals = PositionRadiusDetector(arrival_radius_m=40.0, dwell_s=0.0).detect(
            trip, history, [stop]
        )
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1000.0


# ---------------------------------------------------------------------------
# StopStatusDetector
# ---------------------------------------------------------------------------


class TestStopStatusDetector:
    def test_fires_on_stopped_at_with_matching_sequence(self, trip: TripInstance) -> None:
        stops = [
            Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0),
            Stop(stop_id="B", stop_sequence=2, lat=0.0, lon=0.0),
        ]
        history = [
            _vs(1000.0, 0.0, 0.0, status="IN_TRANSIT_TO", seq=1),
            _vs(1010.0, 0.0, 0.0, status="STOPPED_AT", seq=1),
            _vs(1015.0, 0.0, 0.0, status="STOPPED_AT", seq=1),  # duplicate — ignored
            _vs(1020.0, 0.0, 0.0, status="STOPPED_AT", seq=2),
        ]
        det = StopStatusDetector()
        arrivals = det.detect(trip, history, stops)

        assert len(arrivals) == 2
        by_stop = {a.stop_id: a for a in arrivals}
        assert by_stop["A"].actual_arrival_epoch == 1010.0
        assert by_stop["B"].actual_arrival_epoch == 1020.0

    def test_ignores_non_matching_sequence(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]
        history = [_vs(1010.0, 0.0, 0.0, status="STOPPED_AT", seq=99)]
        assert StopStatusDetector().detect(trip, history, stops) == []

    def test_ignores_in_transit_and_incoming_status(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]
        history = [
            _vs(1000.0, 0.0, 0.0, status="IN_TRANSIT_TO", seq=1),
            _vs(1010.0, 0.0, 0.0, status="INCOMING_AT", seq=1),
        ]
        assert StopStatusDetector().detect(trip, history, stops) == []

    def test_first_sample_epoch_used_not_subsequent(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]
        history = [
            _vs(1010.0, 0.0, 0.0, status="STOPPED_AT", seq=1),
            _vs(1020.0, 0.0, 0.0, status="STOPPED_AT", seq=1),
        ]
        arrivals = StopStatusDetector().detect(trip, history, stops)
        assert len(arrivals) == 1
        assert arrivals[0].actual_arrival_epoch == 1010.0

    def test_method_string(self) -> None:
        assert StopStatusDetector().method == "stop_status"

    def test_arrivals_carry_method(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]
        history = [_vs(1010.0, 0.0, 0.0, status="STOPPED_AT", seq=1)]
        arrivals = StopStatusDetector().detect(trip, history, stops)
        assert len(arrivals) == 1
        assert arrivals[0].method == "stop_status"

    def test_no_arrivals_for_none_current_stop_sequence(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]
        history = [_vs(1010.0, 0.0, 0.0, status="STOPPED_AT", seq=None)]
        assert StopStatusDetector().detect(trip, history, stops) == []

    def test_at_most_one_arrival_per_stop(self, trip: TripInstance) -> None:
        stops = [Stop(stop_id="A", stop_sequence=1, lat=0.0, lon=0.0)]
        history = [
            _vs(1010.0, 0.0, 0.0, status="STOPPED_AT", seq=1),
            _vs(1020.0, 0.0, 0.0, status="STOPPED_AT", seq=1),
            _vs(1030.0, 0.0, 0.0, status="STOPPED_AT", seq=1),
        ]
        arrivals = StopStatusDetector().detect(trip, history, stops)
        assert len(arrivals) == 1
