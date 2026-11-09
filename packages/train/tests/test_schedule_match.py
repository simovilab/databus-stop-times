"""Inferred trip instances -> scheduled GTFS trips (stoptimes.train.sources.schedule_match)."""

from __future__ import annotations

import pandas as pd
import pytest
from stoptimes.core.features.trip_state import StopArrival, TripState
from stoptimes.train.live_frame import with_radius_arrivals
from stoptimes.train.sources.schedule_match import (
    MATCH_COLUMNS,
    MatchParams,
    gtfs_time_to_seconds,
    match_instances,
    scheduled_stop_times,
    stop_passages,
)

T0 = pd.Timestamp("2026-10-20T12:00:00Z")


def _pings(progress: list[float], dt: float = 10.0) -> pd.DataFrame:
    """Pings whose latitude encodes their progress (decoded by ``_project``)."""
    return pd.DataFrame(
        {
            "ts": [T0 + pd.Timedelta(seconds=dt * i) for i in range(len(progress))],
            "lat": [p / 1000 for p in progress],
            "lon": [0.0] * len(progress),
        }
    )


def _project(lat: float, lon: float) -> float:
    return lat * 1000


def test_gtfs_times_past_midnight():
    assert gtfs_time_to_seconds("06:20:00") == 6 * 3600 + 20 * 60
    assert gtfs_time_to_seconds("25:01:02") == 25 * 3600 + 62


def test_passages_interpolate_between_pings_and_skip_unspanned_stops():
    pings = _pings([50, 100, 200, 300])
    got = stop_passages(
        pings, [("before", 10.0), ("s1", 150.0), ("s2", 300.0), ("after", 900.0)], _project
    )
    assert set(got) == {"s1", "s2"}
    assert got["s1"] == pytest.approx(T0.timestamp() + 15.0)
    assert got["s2"] == pytest.approx(T0.timestamp() + 30.0)


def test_passages_ignore_backward_jitter():
    pings = _pings([0, 120, 90, 200])  # the third ping jumps back 30 m
    got = stop_passages(pings, [("s", 100.0)], _project)
    assert got["s"] == pytest.approx(T0.timestamp() + 10 * (100 / 120))


def _schedule():
    rows = []
    for trip, _shape, dep in [
        ("t0620", "via_A", 6 * 3600 + 1200),
        ("t0630", "via_B", 6 * 3600 + 1800),
        ("t0640", "via_A", 6 * 3600 + 2400),
        ("t0650", "via_A", 6 * 3600 + 3000),
    ]:
        for k, stop in enumerate(["s1", "s2", "s3", "s4"]):
            rows.append({"trip_id": trip, "stop_id": stop, "arrival_time": _hms(dep + 120 * k)})
    stop_times = pd.DataFrame(rows)
    trips = pd.DataFrame(
        {
            "trip_id": ["t0620", "t0630", "t0640", "t0650"],
            "direction_id": [0, 0, 0, 0],
            "shape_id": ["via_A", "via_B", "via_A", "via_A"],
        }
    )
    return scheduled_stop_times(stop_times, trips)


def _hms(s: int) -> str:
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _instance(inst: str, dep_s: float, shape: str = "via_A", stops=("s1", "s2", "s3", "s4")):
    passages = [
        {"instance_id": inst, "stop_id": s, "passage_s": dep_s + 120 * k}
        for k, s in enumerate(stops)
    ]
    meta = {"instance_id": inst, "service_date": "20261020", "direction_id": 0, "shape_id": shape}
    return passages, meta


def _match(*instances, active=lambda d: {"t0620", "t0630", "t0640", "t0650"}, **params):
    passages = pd.DataFrame([p for ps, _ in instances for p in ps])
    meta = pd.DataFrame([m for _, m in instances])
    out = match_instances(passages, meta, _schedule(), active, MatchParams(**params))
    assert list(out.columns) == MATCH_COLUMNS
    return out.set_index("instance_id")


def test_each_instance_gets_the_scheduled_trip_it_ran_late_or_early():
    out = _match(_instance("late", 6 * 3600 + 1200 + 90), _instance("early", 6 * 3600 + 2400 - 60))
    assert out.loc["late", "trip_id"] == "t0620" and out.loc["late", "offset_s"] == pytest.approx(
        90
    )
    assert out.loc["early", "trip_id"] == "t0640" and out.loc["early", "offset_s"] == pytest.approx(
        -60
    )
    assert out["matched"].all() and out["pattern_agrees"].all()
    assert out["spread_s"].tolist() == pytest.approx([0.0, 0.0])


def test_assignment_is_one_to_one():
    out = _match(_instance("a", 6 * 3600 + 1200 + 30), _instance("b", 6 * 3600 + 1200 + 60))
    assert out["trip_id"].nunique() == 2


def test_the_geometry_is_a_prior_the_timetable_decides():
    # inferred via_A but departing exactly at the via_B trip's time, 10 min from any via_A trip
    out = _match(_instance("x", 6 * 3600 + 1800, shape="via_A"))
    assert out.loc["x", "trip_id"] == "t0630" and not out.loc["x", "pattern_agrees"]
    # ... unless the matching-pattern trip is close enough: 200 s before t0630 costs
    # |400| for t0620 against |200| + 300 (pattern penalty) for t0630
    out = _match(_instance("y", 6 * 3600 + 1800 - 200, shape="via_A"))
    assert out.loc["y", "trip_id"] == "t0620" and out.loc["y", "pattern_agrees"]


def test_far_from_every_trip_is_unmatched():
    out = _match(_instance("x", 9 * 3600))
    assert not out.loc["x", "matched"] and out.loc["x", "trip_id"] is None


def test_trips_not_running_that_day_are_never_matched():
    out = _match(_instance("x", 6 * 3600 + 1200 - 100), active=lambda d: {"t0640"})
    assert not out.loc["x", "matched"]  # t0620 is not running; t0640 is 21m40s away


def test_too_few_shared_stops_is_unmatched():
    out = _match(_instance("x", 6 * 3600 + 1200, stops=("s1", "s2")))
    assert not out.loc["x", "matched"]


def test_margin_is_small_when_two_trips_fit_equally():
    out = _match(_instance("x", 6 * 3600 + 2700))  # halfway between t0640 and t0650
    assert out.loc["x", "margin_s"] == pytest.approx(0, abs=1)


# ---------------------------------------------------------------------------
# with_radius_arrivals (live_frame): a bus that lays over just short of the stop
# ---------------------------------------------------------------------------


def _state(pings, arrival):
    return TripState("k", (100.0,), tuple(pings), (arrival,), False, pings[-1][0])


def test_radius_arrival_replaces_a_late_crossing():
    late = StopArrival(epoch=1795.0, detect_epoch=1800.0, gap_s=10.0)
    state = _state(
        [(0.0, 0.0, True), (60.0, 94.0, True), (1790.0, 95.0, True), (1800.0, 130.0, True)], late
    )
    out = with_radius_arrivals(state, 10.0)
    assert out.arrivals[0] == StopArrival(epoch=60.0, detect_epoch=60.0, gap_s=0.0)
    assert with_radius_arrivals(state, 2.0).arrivals[0] == late  # 6 m short is outside 2 m


def test_radius_arrival_fills_a_missing_crossing_and_ignores_invalid_pings():
    state = _state([(0.0, 0.0, True), (30.0, 99.0, False), (60.0, 96.0, True)], None)
    assert with_radius_arrivals(state, 5.0).arrivals[0].epoch == 60.0


def test_radius_never_moves_an_arrival_later():
    early = StopArrival(epoch=50.0, detect_epoch=55.0, gap_s=10.0)
    state = _state([(0.0, 0.0, True), (55.0, 120.0, True)], early)
    assert with_radius_arrivals(state, 30.0).arrivals[0] == early
