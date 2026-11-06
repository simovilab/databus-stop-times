"""Tests for core/features/live.py: the 19-feature assembly and its train/serve parity (F4a)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.features.identity import RouteVocabulary
from stoptimes.core.features.live import (
    IDENTITY_COLUMNS,
    LIVE_FEATURES,
    LiveAssets,
    MissingAssetError,
    TripContext,
    assemble_live_batch,
    assemble_live_features,
    build_trip_context,
    minute_of_day,
    temporal_columns,
)
from stoptimes.core.features.prior import fit, segments_from_arrivals
from stoptimes.core.features.registry import build as build_registry
from stoptimes.core.features.schedule import TripSchedule, build_trip_schedule
from stoptimes.core.features.trip_state import (
    ForwardMatcher,
    Ping,
    arrivals_from_pings,
    update_trip_state,
)
from stoptimes.core.spatial.polyline import build_polyline

LAT0, LON0 = 42.0, -71.0
M_DEG = 6371000.0 * math.pi / 180.0
COS = math.cos(math.radians(LAT0))
TZ = "America/New_York"
DAY = "20261014"
STOP_PROGRESS = tuple(float(x) for x in range(0, 6001, 500))  # 13 stops
STOP_IDS = tuple(f"s{i}" for i in range(len(STOP_PROGRESS)))
NOON_ORIGIN = pd.Timestamp("2026-10-14 00:00", tz=TZ).timestamp()


def east(progress_m: float, north_m: float = 0.0) -> tuple[float, float]:
    return LAT0 + north_m / M_DEG, LON0 + progress_m / (COS * M_DEG)


def make_ctx(route: str = "R1", day: str = DAY) -> TripContext:
    sched = tuple(NOON_ORIGIN + 8 * 3600 + 60.0 * i for i in range(len(STOP_IDS)))
    return TripContext(
        trip_id="trip1",
        route_id=route,
        service_date=day,
        tz=TZ,
        stop_ids=STOP_IDS,
        stop_sequences=tuple(range(1, len(STOP_IDS) + 1)),
        stop_progress_m=STOP_PROGRESS,
        sched_epochs=sched,
        holiday_region="US_MA",
    )


def make_matcher() -> ForwardMatcher:
    return ForwardMatcher(build_polyline([east(i * 250.0) for i in range(31)]))


def make_pings(seed: int = 1, start_m: float = 0.0, t0: float | None = None) -> list[Ping]:
    """A noisy drive with dwell, an off-route excursion, a >60 s gap and a duplicate timestamp."""
    rng = np.random.default_rng(seed)
    t = (NOON_ORIGIN + 8 * 3600 + 20.0) if t0 is None else t0
    p = start_m
    pings: list[Ping] = []
    while p < 6100:
        pings.append(Ping(t, *east(p, north_m=float(rng.normal(0, 3)))))
        dt = float(rng.uniform(5, 12))
        if len(pings) == 40:
            dt = 90.0  # gap > 60 s: arrivals across it are rejected
        t += dt
        p += float(rng.uniform(0, 140)) if len(pings) % 17 else 0.0  # dwell now and then
    pings.insert(25, Ping(pings[24].epoch + 2.0, *east(3000.0, north_m=500.0)))  # off route
    pings.insert(30, pings[29])  # duplicate timestamp -> ignored
    return pings


def make_assets(route: str = "R1") -> LiveAssets:
    rng = np.random.default_rng(7)
    rows = []
    for d in range(20):
        date = f"2026-09-{(d % 10) + 1:02d}"
        for trip in range(6):
            t0 = pd.Timestamp(f"{date} 07:00", tz=TZ).timestamp() + trip * 1800.0
            tt = np.cumsum(rng.uniform(40, 90, len(STOP_IDS)))
            for i, sid in enumerate(STOP_IDS):
                rows.append(
                    {
                        "trip_key": f"{d}-{trip}",
                        "service_date": date,
                        "route_id": route,
                        "stop_index": i,
                        "stop_id": sid,
                        "stop_progress_m": STOP_PROGRESS[i],
                        "arrival_epoch": t0 + tt[i],
                    }
                )
    stats = fit(segments_from_arrivals(pd.DataFrame(rows), TZ))
    return LiveAssets(stats, RouteVocabulary.fit(["R0", route, "R9"]))


def replay_features(ctx, assets, pings, matcher, features=LIVE_FEATURES):
    state = None
    parts = []
    for p in pings:
        state = update_trip_state(
            state, p, polyline=matcher, stop_progress_m=ctx.stop_progress_m, trip_key=ctx.trip_key
        )
        parts.append(assemble_live_features(ctx, assets, state, p.epoch, features=features))
    return pd.concat(parts, ignore_index=True)


def assert_frames_identical(a: pd.DataFrame, b: pd.DataFrame, cols) -> None:
    assert len(a) == len(b)
    for c in (*IDENTITY_COLUMNS, *cols):
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            np.testing.assert_array_equal(x, y, err_msg=c)
        else:
            assert (x == y).all(), c


@pytest.mark.parametrize("seed,start_m", [(1, 0.0), (2, 0.0), (3, 2300.0)])
def test_batch_equals_one_ping_at_a_time_replay(seed, start_m):
    ctx, assets, matcher = make_ctx(), make_assets(), make_matcher()
    pings = make_pings(seed, start_m)  # seed 3 starts mid-trip: cold start
    replayed = replay_features(ctx, assets, pings, matcher)
    final = arrivals_from_pings(pings, matcher, ctx.stop_progress_m, trip_key=ctx.trip_key)
    batch = assemble_live_batch(ctx, assets, final, [p.epoch for p in pings])
    assert len(batch) > 100
    assert_frames_identical(replayed, batch, LIVE_FEATURES)
    # every feature is exercised, not just present
    assert batch["delay_last"].notna().any() and batch["delay_last"].isna().any()
    assert batch["prior_rem"].notna().all()
    assert batch["delay_now"].notna().all()


def test_batch_equals_replay_between_pings_and_with_unseen_route():
    ctx, assets, matcher = make_ctx(route="R-new"), make_assets(), make_matcher()
    pings = make_pings(5)
    # prediction times that fall between pings (the vehicle state is the latest ping <= t)
    times = [p.epoch + 3.5 for p in pings[10:60]]
    parts = []
    for t in times:  # the state at t holds exactly the pings with epoch <= t
        state = None
        for p in (q for q in pings if q.epoch <= t):
            state = update_trip_state(state, p, polyline=matcher, stop_progress_m=ctx.stop_progress_m, trip_key="k")
        parts.append(assemble_live_features(ctx, assets, state, t))
    replayed = pd.concat(parts, ignore_index=True)
    final = arrivals_from_pings(pings, matcher, ctx.stop_progress_m, trip_key="k")
    batch = assemble_live_batch(ctx, assets, final, times)
    assert_frames_identical(replayed, batch, LIVE_FEATURES)
    assert batch["route_code"].isna().all()  # unseen route -> missing, not an error
    assert (batch["t"] > batch["ping_ts"]).all()


def test_feature_semantics_on_a_clean_drive():
    ctx, assets, matcher = make_ctx(), make_assets(), make_matcher()
    t0 = ctx.sched_epochs[0] + 30.0  # 30 s behind schedule at stop 0
    pings = [Ping(t0 + 10.0 * i, *east(100.0 * i)) for i in range(0, 61)]  # 10 m/s
    final = arrivals_from_pings(pings, matcher, ctx.stop_progress_m)
    tq = t0 + 260.0  # vehicle at 2600 m: stop 5 (2500 m) passed, stop 6 (3000 m) next
    f = assemble_live_batch(ctx, assets, final, [tq])
    assert list(f["stop_idx"]) == list(range(6, 13))
    assert list(f["stops_ahead"]) == list(range(7))
    np.testing.assert_allclose(f["distance_to_stop"], [400.0 + 500.0 * k for k in range(7)], atol=1e-6)
    # schedule: stop i at 08:00 + 60 s * i, so 2600 m is 312 s into the trip
    sched_now = ctx.sched_epochs[0] + 312.0
    np.testing.assert_allclose(f["delay_now"], tq - sched_now, atol=1e-6)
    np.testing.assert_allclose(f["sched_rem"], [48.0 + 60.0 * k for k in range(7)], atol=1e-6)
    np.testing.assert_allclose(f["sched_rem_t"], [ctx.sched_epochs[6 + k] - tq for k in range(7)], atol=1e-6)
    # stop 5 was crossed at t0 + 250 (interpolated); scheduled at origin + 300 s
    assert f["delay_last"].iloc[0] == pytest.approx((t0 + 250.0) - ctx.sched_epochs[5], abs=1e-6)
    assert (f["prior_rem"].diff().dropna() > 0).all()
    assert f["stop_sequence"].tolist() == [7, 8, 9, 10, 11, 12, 13]
    assert f["route_code"].iloc[0] == 1.0  # vocabulary ["R0","R1","R9"]


def test_nothing_comes_back_before_the_first_valid_ping_or_after_the_last_stop():
    ctx, assets, matcher = make_ctx(), make_assets(), make_matcher()
    pings = [Ping(1000.0 + 10 * i, *east(7000.0 + 5 * i)) for i in range(5)]  # beyond the shape: off route
    state = None
    for p in pings:
        state = update_trip_state(state, p, polyline=matcher, stop_progress_m=ctx.stop_progress_m, trip_key="k")
    assert assemble_live_features(ctx, assets, state, 1100.0).empty
    final = arrivals_from_pings(pings, matcher, ctx.stop_progress_m)
    assert assemble_live_batch(ctx, assets, final, [1100.0]).empty
    on_route = [Ping(1000.0, *east(6000.0)), Ping(1010.0, *east(6100.0))]
    s2 = arrivals_from_pings(on_route, matcher, ctx.stop_progress_m)
    assert assemble_live_batch(ctx, assets, s2, [1015.0]).empty  # all stops passed
    assert assemble_live_batch(ctx, assets, s2, [900.0]).empty  # t before any ping


def test_state_is_not_read_beyond_t():
    ctx, assets, matcher = make_ctx(), make_assets(), make_matcher()
    pings = make_pings(4)
    full = replay_features(ctx, assets, pings[:50], matcher)
    state = None
    for p in pings[:80]:
        state = update_trip_state(state, p, polyline=matcher, stop_progress_m=ctx.stop_progress_m, trip_key="k")
    at = pings[49].epoch
    early = replay_features(ctx, assets, pings[:50], matcher)
    late = assemble_live_features(ctx, assets, state, at)  # state knows later pings; t hides them
    last_rows = early[early["t"] == at].reset_index(drop=True)
    assert_frames_identical(last_rows, late.reset_index(drop=True), LIVE_FEATURES)
    assert len(full) == len(early)


def test_missing_assets_are_named():
    ctx, matcher = make_ctx(), make_matcher()
    pings = make_pings(1)
    state = arrivals_from_pings(pings, matcher, ctx.stop_progress_m)
    with pytest.raises(MissingAssetError, match="segment_stats"):
        assemble_live_batch(ctx, LiveAssets(None, RouteVocabulary.fit(["R1"])), state, [pings[5].epoch])
    with pytest.raises(MissingAssetError, match="route_vocabulary"):
        assemble_live_features(ctx, LiveAssets(make_assets().segment_stats, None), state, pings[5].epoch)
    # a model that does not use them can be served without them
    out = assemble_live_batch(ctx, LiveAssets(), state, [pings[5].epoch], features=["distance_to_stop", "delay_now"])
    assert list(out.columns) == [*IDENTITY_COLUMNS, "distance_to_stop", "delay_now"]
    with pytest.raises(ValueError, match="not live features"):
        assemble_live_batch(ctx, LiveAssets(), state, [1.0], features=["vp_lat"])


def test_context_validates_alignment_and_is_frozen():
    with pytest.raises(ValueError, match="equal length"):
        TripContext("t", "r", DAY, TZ, ("a", "b"), (1,), (0.0, 1.0), (0.0, 1.0))
    ctx = make_ctx()
    with pytest.raises(Exception):
        ctx.trip_id = "x"  # type: ignore[misc]


def test_temporal_columns_equal_the_registry_per_timestamp():
    base = pd.Timestamp("2026-10-14 08:00", tz=TZ).timestamp()
    ts = np.array([base, base + 1800, base + 3 * 3600, base + 86400 * 4 + 7 * 3600, base - 3600 * 9])
    cols = temporal_columns(ts, TZ, "US_MA")
    frame = pd.DataFrame({"_ts": pd.to_datetime(ts, unit="s", utc=True)})
    ref = build_registry(list(cols.columns), frame, {"timestamp_column": "_ts", "tz": TZ, "holiday_region": "US_MA"})
    for c in cols.columns:
        np.testing.assert_array_equal(cols[c].to_numpy(), ref[c].astype(float).to_numpy(), err_msg=c)
    assert temporal_columns([], TZ, None).empty


def test_minute_of_day_uses_the_service_date_offset():
    t = pd.Timestamp("2026-10-14 08:30", tz=TZ).timestamp()
    assert minute_of_day(t, DAY, TZ) == pytest.approx(8 * 60 + 30)
    after_midnight = pd.Timestamp("2026-10-15 00:30", tz=TZ).timestamp()
    assert minute_of_day(after_midnight, DAY, TZ) == pytest.approx(30.0)


class _FakeStatic:
    def __init__(self, ids=STOP_IDS):
        from stoptimes.core.domain.models import Stop

        self._stops = [
            Stop(stop_id=s, stop_sequence=i + 1, lat=east(STOP_PROGRESS[i])[0], lon=east(STOP_PROGRESS[i])[1], progress_m=None)
            for i, s in enumerate(ids)
        ]
        self._ids = ids

    def trip_geometry(self, trip_id):
        return [(*east(i * 250.0), None) for i in range(25)], list(self._stops)

    def trip_schedule(self, trip_id) -> TripSchedule:
        n = len(self._ids)
        times = [f"08:{i:02d}:00" for i in range(n)]
        return build_trip_schedule(trip_id, list(self._ids), list(range(1, n + 1)), times, times)


def test_build_trip_context_anchors_schedule_and_progress():
    ctx, matcher = build_trip_context(_FakeStatic(), "t1", "R1", DAY, TZ, holiday_region="US_MA")
    assert ctx.stop_progress_m == pytest.approx(STOP_PROGRESS, abs=1.0)
    assert ctx.sched_epochs[0] == pytest.approx(NOON_ORIGIN + 8 * 3600, abs=3700)  # origin = noon - 12 h
    assert ctx.sched_epochs[3] - ctx.sched_epochs[0] == 180.0
    assert isinstance(matcher, ForwardMatcher) and ctx.holiday_region == "US_MA"


def test_build_trip_context_rejects_misaligned_stops():
    class Bad(_FakeStatic):
        def trip_schedule(self, trip_id):
            return build_trip_schedule(trip_id, ["a", "b"], [1, 2], ["08:00:00"] * 2, ["08:00:00"] * 2)

    with pytest.raises(ValueError, match="do not align"):
        build_trip_context(Bad(), "t1", "R1", DAY, TZ)
