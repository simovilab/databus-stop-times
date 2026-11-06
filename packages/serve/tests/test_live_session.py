"""LiveSession: per-run state, staleness, causality, cold start, isolation, service date."""

from __future__ import annotations

import numpy as np
import pytest

from stoptimes.core.features.live import LIVE_FEATURES
from stoptimes.serve.live_session import (
    LiveSession,
    LiveSessionError,
    derive_service_date,
    needs_live_session,
)


def new_session(kit, **kw):
    return LiveSession(
        kit.StubStatic(), kit.make_assets(), features=LIVE_FEATURES, tz=kit.TZ, holiday_region="US_MA", **kw
    )


def feed(session, trip, pings, *, start_date="20261014", vehicle="v1"):
    key = None
    for p in pings:
        key = session.observe(trip, "R1", start_date, vehicle, p)
    return key


def test_needs_live_session_only_for_features_with_inputs():
    assert needs_live_session(["distance_to_stop", "delay_now"])
    assert not needs_live_session(["distance_to_stop", "hour", "not_in_registry"])


def test_predicts_all_downstream_stops_in_one_model_call(kit):
    s = new_session(kit)
    pings = kit.drive("trip1", seed=1)
    k1 = feed(s, "trip1", pings[:30])
    p2 = kit.drive("trip2", seed=2, delay_s=-1200)
    k2 = feed(s, "trip2", [p for p in p2 if p.epoch <= pings[29].epoch], vehicle="v2")
    t = pings[29].epoch + 1.0
    calls = []

    class Counting(kit.SumModel):
        def predict(self, x):
            calls.append(len(x))
            return super().predict(x)

    frame = s.predict(Counting(), [k1, k2], t)
    assert len(calls) == 1 and calls[0] == len(frame) > 10
    assert set(frame.trip_id) == {"trip1", "trip2"} and frame.horizon_s.notna().all()


def test_stale_vehicle_is_skipped_but_fresh_one_is_not(kit):
    s = new_session(kit)
    pings = kit.drive("trip1")
    key = feed(s, "trip1", pings[:20])
    last = pings[19].epoch
    assert s.fresh_slot(key, last + 29.0) is not None
    assert s.fresh_slot(key, last + 31.0) is None
    assert s.assemble([key], last + 31.0).empty
    assert key in s.fresh_keys(last + 10.0) and key not in s.fresh_keys(last + 31.0)


def test_prediction_at_t_ignores_pings_after_t(kit):
    pings = kit.drive("trip1", seed=4)
    t = pings[24].epoch + 3.0
    a = new_session(kit)
    ka = feed(a, "trip1", pings[:25])
    b = new_session(kit)
    kb = feed(b, "trip1", pings)  # the whole future is already in the state
    fa, fb = a.assemble([ka], t), b.assemble([kb], t)
    assert len(fa) == len(fb) > 0
    for col in LIVE_FEATURES:
        np.testing.assert_array_equal(fa[col].to_numpy(), fb[col].to_numpy(), err_msg=col)


def test_changing_a_later_ping_does_not_change_an_earlier_prediction(kit):
    pings = kit.drive("trip1", seed=5)
    t = pings[20].epoch + 2.0
    base = new_session(kit)
    k = feed(base, "trip1", pings[:21])
    ref = base.predict(kit.SumModel(), [k], t).horizon_s.to_numpy()
    other = new_session(kit)
    mutated = pings[:21] + [type(p)(p.epoch, p.lat + 0.01, p.lon + 0.01) for p in pings[21:]]
    k2 = feed(other, "trip1", mutated)
    np.testing.assert_array_equal(other.predict(kit.SumModel(), [k2], t).horizon_s.to_numpy(), ref)


def test_cold_start_mid_trip_gives_predictions_without_delay_last(kit):
    s = new_session(kit)
    pings = kit.drive("trip1", start_m=3000.0, seed=6)  # first ping seen is mid-trip
    key = feed(s, "trip1", pings[:6])
    frame = s.assemble([key], pings[5].epoch + 1.0)
    assert len(frame) > 0 and frame.stops_ahead.min() == 0
    assert frame.delay_last.isna().all()
    assert frame.distance_to_stop.min() > 0
    assert (frame.stop_idx.min() >= 6)  # stops behind 3000 m are not predicted


def test_state_is_isolated_between_trips(kit):
    p1, p2 = kit.drive("trip1", seed=1), kit.drive("trip2", seed=2, delay_s=-1200)
    solo = new_session(kit)
    k_solo = feed(solo, "trip1", p1[:30])
    t = p1[29].epoch + 1.0
    ref = solo.assemble([k_solo], t)
    both = new_session(kit)
    ka = kb = None
    for i in range(30):  # interleaved arrival order
        ka = both.observe("trip1", "R1", "20261014", "v1", p1[i])
        if i < len(p2):
            kb = both.observe("trip2", "R1", "20261014", "v2", p2[i])
    got = both.assemble([ka], t)
    assert ka != kb
    for col in LIVE_FEATURES:
        np.testing.assert_array_equal(got[col].to_numpy(), ref[col].to_numpy(), err_msg=col)


def test_same_trip_two_vehicles_do_not_share_state(kit):
    s = new_session(kit)
    pings = kit.drive("trip1", seed=1)
    ka = feed(s, "trip1", pings[:10], vehicle="vA")
    kb = feed(s, "trip1", pings[:3], vehicle="vB")
    assert ka != kb
    assert len(s.store.get(ka).pings) == 10 and len(s.store.get(kb).pings) == 3


def test_duplicate_and_out_of_order_pings_are_ignored(kit):
    s = new_session(kit)
    pings = kit.drive("trip1")
    key = feed(s, "trip1", pings[:10])
    before = s.store.get(key)
    s.observe("trip1", "R1", "20261014", "v1", pings[9])
    s.observe("trip1", "R1", "20261014", "v1", pings[3])
    assert s.store.get(key) is before


def test_unknown_trip_raises_once_and_is_negative_cached(kit):
    static = kit.StubStatic()
    s = LiveSession(static, kit.make_assets(), features=LIVE_FEATURES, tz=kit.TZ)
    p = kit.drive("trip1")[0]
    for _ in range(3):
        with pytest.raises(LiveSessionError, match="not found"):
            s.observe("ghost", "R1", "20261014", "v1", p)
    assert static.calls["trip_schedule"] == 0


def test_trip_context_is_built_once_per_trip_and_date(kit):
    static = kit.StubStatic()
    s = LiveSession(static, kit.make_assets(), features=LIVE_FEATURES, tz=kit.TZ)
    feed(s, "trip1", kit.drive("trip1")[:25])
    assert static.calls == {"trip_schedule": 1, "trip_geometry": 1}


def test_unknown_feature_is_rejected(kit):
    with pytest.raises(LiveSessionError):
        LiveSession(kit.StubStatic(), kit.make_assets(), features=["cross_track_error"], tz=kit.TZ)


# -- service date --------------------------------------------------------------------


def test_start_date_from_feed_wins(kit):
    s = new_session(kit)
    assert s.service_date("trip1", "v1", "20261013", kit.MIDNIGHT + 8 * 3600) == "20261013"


def test_derived_service_date_plain_and_after_midnight(kit):
    static = kit.StubStatic()
    morning = kit.MIDNIGHT + 8 * 3600 + 120  # trip1 08:00, observed 08:02 on 10-14
    assert derive_service_date(static.trip_schedule("trip1"), morning, kit.TZ) == "20261014"
    # trip_late departs 25:30 of the 13th = 01:30 on the 14th; seen at 01:40 local on the 14th
    late = kit.MIDNIGHT + 3600 + 40 * 60
    assert derive_service_date(static.trip_schedule("trip_late"), late, kit.TZ) == "20261013"
    # the pre-departure layover (01:10 local) is still the 13th, not the 14th
    assert derive_service_date(static.trip_schedule("trip_late"), kit.MIDNIGHT + 3600 + 600, kit.TZ) == "20261013"


def test_session_derives_and_binds_the_date_without_start_date(kit):
    s = new_session(kit)
    pings = kit.drive("trip_late")
    key = feed(s, "trip_late", pings[:15], start_date=None)
    assert key[1] == "20261013"
    assert s.fresh_slot(key, pings[14].epoch + 1) is not None


def test_timezone_and_region_come_from_the_artifact(kit, tmp_path):
    meta = kit.write_live_registry(tmp_path)
    s = LiveSession.from_artifact(meta, tmp_path, kit.StubStatic(), fallback_timezone="UTC", fallback_holiday_region="XX")
    assert (s.tz, s.holiday_region) == ("America/New_York", "US_MA")
    assert s.assets.segment_stats is not None and s.assets.route_vocabulary is not None
