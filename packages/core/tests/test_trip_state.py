"""Tests for core/features/trip_state.py -- online trip memory (F3).

Covers: forward-only matching (straight, loop, jitter, off-route), arrival
detection with the 60 s gap rule, cold start, duplicate/out-of-order pings,
store eviction, causality of delay_last, and online-vs-batch parity.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from stoptimes.core.features.trip_state import (
    ForwardMatcher,
    InMemoryTripStateStore,
    Ping,
    TripStateStore,
    arrivals_from_pings,
    delay_last,
    update_trip_state,
)
from stoptimes.core.spatial.polyline import build_polyline

LAT0, LON0 = 42.0, -71.0
M_DEG = 6371000.0 * math.pi / 180.0
COS = math.cos(math.radians(LAT0))


def east(progress_m: float, north_m: float = 0.0) -> tuple[float, float]:
    return LAT0 + north_m / M_DEG, LON0 + progress_m / (COS * M_DEG)


def straight_polyline(length_m: float = 5000.0, step_m: float = 250.0):
    n = int(length_m / step_m) + 1
    return build_polyline([east(i * step_m) for i in range(n)])


def pings_at(progresses, t0=1000.0, dt=10.0, north_m=0.0):
    return [Ping(t0 + i * dt, *east(p, north_m)) for i, p in enumerate(progresses)]


def replay(pings, polyline, stops, key="trip"):
    state = None
    for p in pings:
        state = update_trip_state(state, p, polyline=polyline, stop_progress_m=stops, trip_key=key)
    return state


STOPS = (0.0, 1000.0, 2000.0, 3000.0, 4000.0)


def arr_epochs(state):
    return [None if a is None else a.epoch for a in state.arrivals]


def test_straight_route_arrivals_interpolated():
    poly = straight_polyline()
    # 10 m/s, ping every 10 s => 100 m per ping, starting exactly at the first stop
    state = replay(pings_at([i * 100.0 for i in range(0, 43)]), poly, STOPS)
    ep = arr_epochs(state)
    assert ep[0] is None  # first ping already at stop 0: before_first_ping
    for j in range(1, 5):
        assert ep[j] == pytest.approx(1000.0 + STOPS[j] / 10.0, abs=0.5)
    assert state.arrivals[2].gap_s == pytest.approx(10.0)
    # detection time = epoch of the first ping at/after the stop
    assert state.arrivals[2].detect_epoch >= state.arrivals[2].epoch
    assert state.started_mid_trip is False
    assert state.last_update_epoch == 1000.0 + 420.0


def test_state_updates_are_immutable():
    poly = straight_polyline()
    s1 = replay(pings_at([800.0, 900.0]), poly, STOPS)
    s2 = update_trip_state(
        s1, Ping(1020.0, *east(1050.0)), polyline=poly, stop_progress_m=STOPS, trip_key="trip"
    )
    assert s2 is not s1
    assert len(s1.pings) == 2 and len(s2.pings) == 3
    assert s1.arrivals[1] is None and s2.arrivals[1] is not None
    with pytest.raises(FrozenInstanceError):
        s1.last_update_epoch = 0.0  # type: ignore[misc]


def test_loop_route_does_not_snap_back():
    # out east 3000 m, 50 m north, back west: return leg ends 50 m from the start
    pts = [east(x) for x in range(0, 3001, 250)] + [east(x, 50.0) for x in range(3000, -1, -250)]
    poly = build_polyline(pts)
    total = poly[-1].cum_m
    assert total > 6000
    progresses = list(np.arange(0.0, total - 20.0, 100.0))
    pings = []
    for i, p in enumerate(progresses):
        # position of the vehicle at progress p
        if p <= 3000:
            lat, lon = east(p)
        elif p <= 3050:
            lat, lon = east(3000, p - 3000)
        else:
            lat, lon = east(3000 - (p - 3050), 50.0)
        pings.append(Ping(1000.0 + 10 * i, lat, lon))
    state = replay(pings, poly, (0.0, 3000.0, total - 150.0))
    prog = [m[1] for m in state.pings if m[2]]
    assert all(b >= a for a, b in zip(prog, prog[1:], strict=False))
    assert prog[-1] > 5900  # never snapped back to the outbound leg near the start
    assert state.arrivals[2] is not None


def test_backward_gps_jitter_is_clamped():
    poly = straight_polyline()
    state = replay(pings_at([500.0, 600.0, 560.0, 700.0]), poly, STOPS)
    prog = [m[1] for m in state.pings]
    assert all(m[2] for m in state.pings)
    assert prog[2] == pytest.approx(prog[1])  # clamped, not moved back
    assert prog[3] == pytest.approx(700.0, abs=1.0)


def test_big_backward_jump_is_invalid_then_recovers():
    poly = straight_polyline()
    pings = pings_at([1000.0, 1100.0, 700.0, 1300.0])
    state = replay(pings, poly, STOPS)
    valid = [m[2] for m in state.pings]
    assert valid[:2] == [True, True]
    assert valid[2] is False  # 400 m behind: outside the allowed window
    assert valid[3] is True


def test_off_route_ping_ignored():
    poly = straight_polyline()
    pings = (
        pings_at([500.0, 600.0], dt=10.0)
        + [Ping(1020.0, *east(700.0, 150.0))]
        + [Ping(1030.0, *east(800.0))]
    )
    state = replay(pings, poly, STOPS)
    assert [m[2] for m in state.pings] == [True, True, False, True]
    assert math.isnan(state.pings[2][1])


def test_gap_over_60s_rejects_arrival():
    poly = straight_polyline()
    pings = pings_at([500.0, 900.0], t0=1000.0, dt=10.0)
    pings = pings + [Ping(1000.0 + 10 + 90.0, *east(1100.0))]  # 90 s gap across stop at 1000
    pings = pings + [Ping(1000.0 + 10 + 100.0, *east(1200.0))]
    state = replay(pings, poly, STOPS)
    assert state.arrivals[1] is None
    # a later stop crossed with a short gap is still detected
    for i, prog in enumerate((1500.0, 1800.0, 2100.0), start=1):
        state = update_trip_state(
            state,
            Ping(1110.0 + 10.0 * i, *east(prog)),
            polyline=poly,
            stop_progress_m=STOPS,
            trip_key="trip",
        )
    assert state.arrivals[2] is not None


def test_cold_start_mid_trip():
    poly = straight_polyline()
    state = replay(pings_at([2500.0, 2600.0, 3100.0]), poly, STOPS)
    assert state.started_mid_trip is True
    assert arr_epochs(state)[:3] == [None, None, None]  # stops passed before first ping: unknown
    assert state.arrivals[3] is not None
    sched = [0.0, 0.0, 0.0, 1015.0, 0.0]
    d = delay_last(state, sched, 3100.0, state.last_update_epoch)
    assert d == pytest.approx(state.arrivals[3].epoch - 1015.0)


def test_started_at_route_start_is_not_mid_trip():
    poly = straight_polyline()
    state = replay(pings_at([20.0, 120.0]), poly, STOPS)
    assert state.started_mid_trip is False


def test_duplicate_and_out_of_order_pings_ignored():
    poly = straight_polyline()
    base = pings_at([100.0, 200.0, 300.0, 400.0])
    clean = replay(base, poly, STOPS)
    noisy = replay([base[0], base[1], base[1], base[2], base[0], base[3], base[2]], poly, STOPS)
    assert noisy == clean
    s = replay(base, poly, STOPS)
    again = update_trip_state(s, base[1], polyline=poly, stop_progress_m=STOPS, trip_key="trip")
    assert again is s


def test_eviction_ttl_and_finished():
    store = InMemoryTripStateStore(ttl_s=300.0, finished_grace_s=60.0)
    assert isinstance(store, TripStateStore)
    poly = straight_polyline()
    live = replay(pings_at([100.0, 200.0], t0=1000.0), poly, STOPS, key="live")
    stale = replay(pings_at([100.0, 200.0], t0=0.0), poly, STOPS, key="stale")
    done = replay(pings_at([3900.0, 4100.0], t0=1000.0), poly, STOPS, key="done")
    for s in (live, stale, done):
        store.put(s)
    assert len(store) == 3 and store.get("live") is live and store.get("nope") is None
    assert store.evict(1015.0) == 1  # stale: no update for > ttl
    assert store.get("stale") is None
    assert store.evict(1010.0 + 59.0) == 0
    assert store.evict(1010.0 + 61.0) == 1  # done: past last stop for > grace
    assert store.get("done") is None and store.get("live") is live
    assert store.evict(1010.0 + 400.0) == 1 and len(store) == 0


def test_delay_last_basic_and_none():
    poly = straight_polyline()
    state = replay(pings_at([100.0, 200.0]), poly, STOPS)
    sched = [900.0, 1050.0, 1150.0, 1250.0, 1350.0]
    assert math.isnan(delay_last(state, sched, 200.0, 1010.0))
    state = replay(pings_at([i * 100.0 for i in range(1, 25)]), poly, STOPS)
    a2 = state.arrivals[2]
    last = state.pings[-1]
    d = delay_last(state, sched, last[1], last[0])
    # last passed stop is index 4? progress 2400 => stops 0..2 passed; last with arrival is 2
    assert d == pytest.approx(a2.epoch - sched[2])


def test_delay_last_is_causal():
    poly = straight_polyline()
    state = replay(pings_at([i * 100.0 for i in range(1, 31)]), poly, STOPS)
    sched = [0.0, 1050.0, 1150.0, 1250.0, 1350.0]
    a1, a2 = state.arrivals[1], state.arrivals[2]
    # progress is past stop 2, but at ping_epoch just before a2 was detected only stop 1 counts
    t = a2.detect_epoch - 1.0
    assert t >= a1.detect_epoch
    assert delay_last(state, sched, 2500.0, t) == pytest.approx(a1.epoch - sched[1])
    assert math.isnan(delay_last(state, sched, 2500.0, a1.detect_epoch - 1.0))
    assert delay_last(state, sched, 2500.0, a2.detect_epoch) == pytest.approx(a2.epoch - sched[2])


def test_delay_last_ignores_stops_not_yet_passed():
    poly = straight_polyline()
    state = replay(pings_at([i * 100.0 for i in range(1, 31)]), poly, STOPS)
    sched = [0.0, 1050.0, 1150.0, 1250.0, 1350.0]
    # asking about an earlier progress must not use a later stop's arrival
    assert delay_last(state, sched, 1500.0, 1e9) == pytest.approx(
        state.arrivals[1].epoch - sched[1]
    )


def _synthetic_trips():
    rng = np.random.default_rng(7)
    out = []
    for _ in range(12):
        n = 60
        gaps = rng.choice(
            [6.0, 10.0, 30.0, 45.0, 70.0, 95.0], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
        )
        t = 1000.0 + np.cumsum(gaps)
        prog = np.cumsum(rng.uniform(0, 160, size=n)) + rng.uniform(0, 1500)
        prog = np.minimum(prog, 4900.0)
        noise_n = rng.normal(0, 8.0, size=n)
        noise_p = rng.normal(0, 12.0, size=n)
        pings = []
        for ti, p, nn, pn in zip(t, prog, noise_n, noise_p, strict=True):
            off = 160.0 if rng.random() < 0.05 else nn
            pings.append(Ping(float(ti), *east(max(p + pn, 0.0), off)))
        # duplicates and a shuffled-late ping
        pings.insert(5, pings[4])
        pings.insert(9, pings[2])
        out.append(pings)
    return out


def test_online_replay_matches_batch():
    poly = straight_polyline()
    for pings in _synthetic_trips():
        online = replay(pings, poly, STOPS)
        batch = arrivals_from_pings(pings, poly, STOPS, trip_key="trip")
        assert online.arrivals == batch.arrivals
        assert online.started_mid_trip == batch.started_mid_trip
        assert online.pings == batch.pings
        a_on, d_on, g_on = online.arrival_arrays()
        a_b, d_b, g_b = batch.arrival_arrays()
        np.testing.assert_array_equal(a_on, a_b)
        np.testing.assert_array_equal(d_on, d_b)
        np.testing.assert_array_equal(g_on, g_b)


def test_batch_empty_and_single():
    poly = straight_polyline()
    assert arrivals_from_pings([], poly, STOPS).arrivals == (None,) * 5
    one = arrivals_from_pings(pings_at([700.0]), poly, STOPS)
    assert one.arrivals == (None,) * 5


def test_matcher_first_ping_is_global_and_windowed_after():
    m = ForwardMatcher(straight_polyline())
    lat, lon = east(2000.0)
    first = m.match(None, lat, lon)
    assert first == pytest.approx(2000.0, abs=1.0)
    assert m.match(None, *east(100.0)) == pytest.approx(100.0, abs=1.0)
