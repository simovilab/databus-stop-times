"""Tests for Phase A4 — etaval/predictors/schedule_baseline.py.

Coverage
--------
- _parse_gtfs_time: HH:MM:SS parsing, including past-midnight (>= 24:00:00).
- Starvation: no loader attached / trip not in schedule / no usable progress -> [].
- Constant-offset cancellation: adding a constant to every scheduled time in
  the feed leaves every predicted_arrival_epoch bit-for-bit unchanged.
- Basic ETA correctness on a small synthetic 3-stop schedule.
- PREDICTOR_REGISTRY contains "baseline:schedule".
"""

from __future__ import annotations

import pandas as pd
import pytest

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.engine.assembly import PREDICTOR_REGISTRY
from stoptimes.val.predictors.schedule_baseline import ScheduleBaselinePredictor, _parse_gtfs_time

TRIP = TripInstance(
    source="test",
    trip_id="trip1",
    vehicle_id="v1",
    route_id="r1",
    direction_id=0,
    shape_id="shape1",
    start_date="20240101",
    started_at=1_000_000.0,
)


class _FakeLoader:
    """Minimal GtfsStaticLoader-shaped double: only .stop_times is needed."""

    def __init__(self, stop_times: pd.DataFrame) -> None:
        self.stop_times = stop_times


def _stop_times_df(
    trip_id: str,
    rows: list[tuple[str, int, str, str]],
) -> pd.DataFrame:
    """rows: (stop_id, stop_sequence, arrival_time, departure_time)."""
    return pd.DataFrame(
        {
            "trip_id": [trip_id] * len(rows),
            "stop_id": [r[0] for r in rows],
            "stop_sequence": [r[1] for r in rows],
            "arrival_time": [r[2] for r in rows],
            "departure_time": [r[3] for r in rows],
        }
    )


def _stop(stop_id: str, stop_sequence: int, progress_m: float | None) -> Stop:
    return Stop(
        stop_id=stop_id, stop_sequence=stop_sequence, lat=10.0, lon=10.0, progress_m=progress_m
    )


def _state(epoch: float, progress_m: float | None) -> VehicleState:
    return VehicleState(epoch=epoch, lat=10.0, lon=10.0, progress_m=progress_m)


# ---------------------------------------------------------------------------
# _parse_gtfs_time
# ---------------------------------------------------------------------------


class TestParseGtfsTime:
    def test_ordinary_time(self) -> None:
        assert _parse_gtfs_time("08:00:00") == 8 * 3600

    def test_past_midnight(self) -> None:
        assert _parse_gtfs_time("25:15:00") == 25 * 3600 + 15 * 60

    def test_far_past_midnight(self) -> None:
        assert _parse_gtfs_time("30:00:00") == 30 * 3600

    def test_none_returns_none(self) -> None:
        assert _parse_gtfs_time(None) is None

    def test_blank_returns_none(self) -> None:
        assert _parse_gtfs_time("") is None
        assert _parse_gtfs_time("   ") is None

    def test_malformed_returns_none(self) -> None:
        assert _parse_gtfs_time("not-a-time") is None
        assert _parse_gtfs_time("08:00") is None

    def test_nan_returns_none(self) -> None:
        assert _parse_gtfs_time(float("nan")) is None


# ---------------------------------------------------------------------------
# Starvation
# ---------------------------------------------------------------------------


class TestStarvation:
    def test_no_loader_attached_returns_empty(self) -> None:
        predictor = ScheduleBaselinePredictor()
        upcoming = [_stop("s1", 1, 0.0), _stop("s2", 2, 500.0)]
        state = _state(1_000_000.0, 250.0)
        assert predictor.predict(TRIP, upcoming, state) == []

    def test_trip_not_in_schedule_returns_empty(self) -> None:
        predictor = ScheduleBaselinePredictor()
        loader = _FakeLoader(
            _stop_times_df("other_trip", [("s1", 1, "08:00:00", "08:00:30")])
        )
        predictor.attach_static_loader(loader)
        upcoming = [_stop("s1", 1, 0.0), _stop("s2", 2, 500.0)]
        state = _state(1_000_000.0, 250.0)
        assert predictor.predict(TRIP, upcoming, state) == []

    def test_fewer_than_two_upcoming_stops_returns_empty(self) -> None:
        predictor = ScheduleBaselinePredictor()
        loader = _FakeLoader(
            _stop_times_df(
                "trip1",
                [("s1", 1, "08:00:00", "08:00:30"), ("s2", 2, "08:05:00", "08:05:30")],
            )
        )
        predictor.attach_static_loader(loader)
        upcoming = [_stop("s1", 1, 0.0)]
        state = _state(1_000_000.0, 0.0)
        assert predictor.predict(TRIP, upcoming, state) == []

    def test_no_vehicle_progress_returns_empty(self) -> None:
        predictor = ScheduleBaselinePredictor()
        loader = _FakeLoader(
            _stop_times_df(
                "trip1",
                [("s1", 1, "08:00:00", "08:00:30"), ("s2", 2, "08:05:00", "08:05:30")],
            )
        )
        predictor.attach_static_loader(loader)
        upcoming = [_stop("s1", 1, 0.0), _stop("s2", 2, 500.0)]
        state = _state(1_000_000.0, None)
        assert predictor.predict(TRIP, upcoming, state) == []

    def test_no_schedule_entry_for_first_two_stops_returns_empty(self) -> None:
        predictor = ScheduleBaselinePredictor()
        loader = _FakeLoader(
            _stop_times_df("trip1", [("s9", 9, "08:00:00", "08:00:30")])
        )
        predictor.attach_static_loader(loader)
        upcoming = [_stop("s1", 1, 0.0), _stop("s2", 2, 500.0)]
        state = _state(1_000_000.0, 250.0)
        assert predictor.predict(TRIP, upcoming, state) == []

    def test_degenerate_equal_progress_returns_empty(self) -> None:
        predictor = ScheduleBaselinePredictor()
        loader = _FakeLoader(
            _stop_times_df(
                "trip1",
                [("s1", 1, "08:00:00", "08:00:30"), ("s2", 2, "08:05:00", "08:05:30")],
            )
        )
        predictor.attach_static_loader(loader)
        upcoming = [_stop("s1", 1, 100.0), _stop("s2", 2, 100.0)]  # equal progress_m
        state = _state(1_000_000.0, 100.0)
        assert predictor.predict(TRIP, upcoming, state) == []


# ---------------------------------------------------------------------------
# Basic ETA correctness
# ---------------------------------------------------------------------------


class TestBasicCorrectness:
    def _build(self) -> tuple[ScheduleBaselinePredictor, list[Stop]]:
        predictor = ScheduleBaselinePredictor()
        loader = _FakeLoader(
            _stop_times_df(
                "trip1",
                [
                    ("s1", 1, "08:00:00", "08:00:30"),  # 28800
                    ("s2", 2, "08:05:00", "08:05:30"),  # 29100
                    ("s3", 3, "08:10:00", "08:10:30"),  # 29400
                ],
            )
        )
        predictor.attach_static_loader(loader)
        upcoming = [_stop("s1", 1, 0.0), _stop("s2", 2, 500.0), _stop("s3", 3, 1000.0)]
        return predictor, upcoming

    def test_predictions_at_halfway_progress(self) -> None:
        predictor, upcoming = self._build()
        now = 1_000_000.0
        state = _state(now, 250.0)  # halfway between s1 (0) and s2 (500)

        preds = predictor.predict(TRIP, upcoming, state)
        by_stop = {p.stop_id: p for p in preds}

        assert set(by_stop) == {"s1", "s2", "s3"}
        # progress_sched_sec = 28800 + 0.5 * (29100 - 28800) = 28950
        assert by_stop["s1"].predicted_arrival_epoch == pytest.approx(now + (28800 - 28950))
        assert by_stop["s2"].predicted_arrival_epoch == pytest.approx(now + (29100 - 28950))
        assert by_stop["s3"].predicted_arrival_epoch == pytest.approx(now + (29400 - 28950))

    def test_stops_remaining_and_dist_remaining(self) -> None:
        predictor, upcoming = self._build()
        state = _state(1_000_000.0, 250.0)
        preds = predictor.predict(TRIP, upcoming, state)
        by_stop = {p.stop_id: p for p in preds}

        assert by_stop["s1"].stops_remaining == 3
        assert by_stop["s2"].stops_remaining == 2
        assert by_stop["s3"].stops_remaining == 1
        assert by_stop["s1"].dist_remaining_m == pytest.approx(0.0 - 250.0)
        assert by_stop["s2"].dist_remaining_m == pytest.approx(500.0 - 250.0)

    def test_made_at_epoch_is_now(self) -> None:
        predictor, upcoming = self._build()
        now = 1_234_567.0
        state = _state(now, 250.0)
        preds = predictor.predict(TRIP, upcoming, state)
        assert all(p.made_at_epoch == pytest.approx(now) for p in preds)


# ---------------------------------------------------------------------------
# Constant-offset cancellation (the key property)
# ---------------------------------------------------------------------------


def _shift_time(hhmmss: str, delta_seconds: int) -> str:
    total = _parse_gtfs_time(hhmmss) + delta_seconds
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TestConstantOffsetCancellation:
    def _predict_with_offset(self, offset_seconds: int) -> list[float]:
        rows = [
            ("s1", 1, "08:00:00", "08:00:30"),
            ("s2", 2, "08:05:00", "08:05:30"),
            ("s3", 3, "08:10:00", "08:10:30"),
        ]
        shifted_rows = [
            (sid, seq, _shift_time(arr, offset_seconds), _shift_time(dep, offset_seconds))
            for sid, seq, arr, dep in rows
        ]
        predictor = ScheduleBaselinePredictor()
        predictor.attach_static_loader(_FakeLoader(_stop_times_df("trip1", shifted_rows)))
        upcoming = [_stop("s1", 1, 0.0), _stop("s2", 2, 500.0), _stop("s3", 3, 1000.0)]
        state = _state(1_000_000.0, 317.0)  # arbitrary, non-round progress

        preds = predictor.predict(TRIP, upcoming, state)
        return [p.predicted_arrival_epoch for p in sorted(preds, key=lambda p: p.stop_id)]

    def test_constant_offset_cancellation(self) -> None:
        """A constant added to every scheduled time changes nothing predicted."""
        baseline = self._predict_with_offset(0)
        shifted_by_1h = self._predict_with_offset(3600)
        shifted_by_7h_and_change = self._predict_with_offset(7 * 3600 + 123)

        assert baseline == pytest.approx(shifted_by_1h)
        assert baseline == pytest.approx(shifted_by_7h_and_change)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_schedule_baseline() -> None:
    assert "baseline:schedule" in PREDICTOR_REGISTRY
    instance = PREDICTOR_REGISTRY["baseline:schedule"]()
    assert isinstance(instance, ScheduleBaselinePredictor)
    assert instance.predictor_id == "baseline:schedule"
    assert hasattr(instance, "attach_static_loader")
