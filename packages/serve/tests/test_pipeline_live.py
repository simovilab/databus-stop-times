"""The default pipeline serving a live-feature model across ticks (synthetic feed, no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import PipelineContext
from stoptimes.core.contracts.source import FeedFormat, FeedKind, RealtimeFeed, Source
from stoptimes.core.features.live import LIVE_FEATURES
from stoptimes.serve.live_session import LiveSession
from stoptimes.serve.pipeline import build_default_pipeline


def _source(**kw) -> Source:
    return Source(id="synth", name="Synth", gtfs_static_url="/dev/null", **kw)


def _vp(trip, ping, vehicle="v1", start_date="20261014"):
    return Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id="synth",
        trip_id=trip,
        route_id="R1",
        vehicle_id=vehicle,
        start_date=start_date,
        ts=pd.Timestamp(ping.epoch, unit="s", tz="UTC"),
        lat=ping.lat,
        lon=ping.lon,
    )


def _tick(kit, session, pipeline, model, observations, now):
    ctx = PipelineContext(
        source=_source(),
        static_feed=None,
        model=model,
        now=datetime.fromtimestamp(now, tz=UTC),
        config={"observations": observations, "live_session": session},
    )
    pipeline.run(ctx)
    return ctx


def _session(kit):
    return LiveSession(kit.StubStatic(), kit.make_assets(), features=LIVE_FEATURES, tz=kit.TZ, holiday_region="US_MA")


def test_state_persists_across_ticks_and_predictions_follow_the_model(kit):
    session, pipeline, model = _session(kit), build_default_pipeline(), kit.SumModel()
    pings = kit.drive("trip1", seed=1)
    last = None
    for i in range(0, 40, 4):  # one poll every ~4 pings, as a serve loop would
        batch = pings[i : i + 4]
        now = batch[-1].epoch + 1.0
        last = _tick(kit, session, pipeline, model, [_vp("trip1", p) for p in batch], now)
        assert last.state.get("errors", []) == []
    (tu,) = last.state["trip_updates"]
    assert tu.trip_id == "trip1" and tu.start_date == "20261014"
    assert len(tu.stop_time_updates) > 3
    # state survived: delay_last needs arrivals detected on earlier ticks
    state = session.store.get(("trip1", "20261014", "v1"))
    assert len(state.pings) == 40 and any(a is not None for a in state.arrivals)
    assert all(stu.predicted_arrival_time >= last.now for stu in tu.stop_time_updates)


def test_a_tick_without_new_pings_still_predicts_fresh_vehicles_then_goes_quiet(kit):
    session, pipeline, model = _session(kit), build_default_pipeline(), kit.SumModel()
    pings = kit.drive("trip1")[:30]
    _tick(kit, session, pipeline, model, [_vp("trip1", p) for p in pings], pings[-1].epoch + 1.0)
    fresh = _tick(kit, session, pipeline, model, [], pings[-1].epoch + 20.0)
    assert len(fresh.state["trip_updates"]) == 1
    stale = _tick(kit, session, pipeline, model, [], pings[-1].epoch + 31.0)
    assert stale.state["trip_updates"] == [] and stale.state["stale_vehicles_skipped"] == 1


def test_one_batched_model_call_per_tick_for_all_vehicles(kit):
    calls = []

    class Counting(kit.SumModel):
        def predict(self, x):
            calls.append(len(x))
            return super().predict(x)

    session, pipeline = _session(kit), build_default_pipeline()
    p1 = kit.drive("trip1", seed=1)[:25]
    p2 = [p for p in kit.drive("trip2", seed=2, delay_s=-1200) if p.epoch <= p1[-1].epoch]
    obs = [_vp("trip1", p) for p in p1] + [_vp("trip2", p, vehicle="v2") for p in p2]
    ctx = _tick(kit, session, pipeline, Counting(), obs, p1[-1].epoch + 1.0)
    assert len(calls) == 1 and calls[0] > 13
    assert {tu.trip_id for tu in ctx.state["trip_updates"]} == {"trip1", "trip2"}


def test_unknown_trip_is_an_error_record_not_a_crash(kit):
    session, pipeline = _session(kit), build_default_pipeline()
    ping = kit.drive("trip1")[0]
    ctx = _tick(kit, session, pipeline, kit.SumModel(), [_vp("ghost", ping)], ping.epoch + 1)
    assert ctx.state["errors"][0]["stage"] == "match" and "ghost" in ctx.state["errors"][0]["trip_id"]
    assert ctx.state["trip_updates"] == []


def test_live_model_without_a_session_reports_what_is_missing(kit):
    ping = kit.drive("trip1")[0]
    ctx = PipelineContext(
        source=_source(),
        static_feed=None,
        model=kit.SumModel(),
        now=datetime.fromtimestamp(ping.epoch + 1, tz=UTC),
        config={"observations": [_vp("trip1", ping)]},
    )
    build_default_pipeline().run(ctx)
    assert "live_session" in ctx.state["errors"][0]["error"]


def test_non_live_models_ignore_the_session_machinery(kit):
    class Plain:
        family = "plain"
        required_features = ["stop_sequence"]

        def predict(self, x):
            return np.array([42.0])

    tu = Observation(
        kind=ObservationKind.TRIP_UPDATE, source_id="synth", trip_id="t", route_id="r", start_date="20261014",
        stop_id="s1", stop_sequence=1, ts=pd.Timestamp("2026-10-14T12:00:00Z"),
    )
    ctx = PipelineContext(
        source=_source(), static_feed=None, model=Plain(),
        now=datetime(2026, 10, 14, 12, tzinfo=UTC), config={"observations": [tu]},
    )
    build_default_pipeline().run(ctx)
    assert ctx.state["trip_updates"][0].stop_time_updates[0].horizon_s == 42.0


def test_ingest_polls_vehicle_positions_when_a_session_is_configured(kit, monkeypatch):
    ping = kit.drive("trip1")[5]
    feed = {
        "header": {"timestamp": int(ping.epoch)},
        "entity": [
            {
                "id": "e1",
                "vehicle": {
                    "trip": {"trip_id": "trip1", "route_id": "R1", "start_date": "20261014", "direction_id": 0},
                    "vehicle": {"id": "v1"},
                    "position": {"latitude": ping.lat, "longitude": ping.lon},
                    "timestamp": int(ping.epoch),
                },
            }
        ],
    }
    urls = []

    class Resp:
        content = json.dumps(feed).encode()

        def raise_for_status(self):
            pass

        def json(self):
            return feed

    import httpx

    monkeypatch.setattr(httpx, "get", lambda url, **kw: (urls.append(url), Resp())[1])
    source = _source(
        realtime_feeds=[
            RealtimeFeed(kind=FeedKind.TRIP_UPDATES, url="http://x/tu", format=FeedFormat.JSON),
            RealtimeFeed(kind=FeedKind.VEHICLE_POSITIONS, url="http://x/vp", format=FeedFormat.JSON),
        ]
    )
    from stoptimes.serve.pipeline import default_ingest

    ctx = PipelineContext(
        source=source, static_feed=None, model=kit.SumModel(),
        now=datetime.fromtimestamp(ping.epoch + 1, tz=UTC), config={"live_session": _session(kit)},
    )
    (obs,) = default_ingest(None, ctx)
    assert urls == ["http://x/vp"] and obs.kind == ObservationKind.VEHICLE_POSITION
    assert (obs.trip_id, obs.vehicle_id, obs.start_date) == ("trip1", "v1", "20261014")
    assert obs.ts.timestamp() == int(ping.epoch)
