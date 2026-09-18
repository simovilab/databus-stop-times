"""test_any_tripupdates_source — validating a plain external TripUpdates feed
(T6 deliverable 4: "validate anything," not just a shipped agency profile).

Nothing in `val`'s feed/predictor code branches on the source ("mbta"/"bucr"
vs. anything else) -- any published GTFS-RT TripUpdates URL, with or without
a `SourceConfig` for it, is a first-class case. This test proves it against
a generic, unbranded domain and an ad hoc `SourceConfig` built without
touching `examples/agencies/`, then checks the same is true when the feed is
protobuf rather than JSON (someone else's published predictions come in
whatever format they publish).
"""

from __future__ import annotations

import httpx
import pytest
import respx
from google.transit import gtfs_realtime_pb2

from stoptimes.core.domain.models import SourceConfig, Stop, TripInstance, VehicleState
from stoptimes.core.realtime.gtfsrt_json import parse_tu_feed
from stoptimes.val.feeds.pb import AutoTripUpdatesPoller
from stoptimes.val.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor, TripUpdateInput

# A source that is not "mbta" or "bucr" and was never added to examples/agencies/.
_EXTERNAL_SOURCE = SourceConfig(
    name="generic-transit-authority",
    gtfs_static_url="https://transit.example-city.gov/gtfs/static.zip",
    vehicle_positions_url="https://transit.example-city.gov/gtfsrt/VehiclePositions.json",
    trip_updates_url="https://transit.example-city.gov/gtfsrt/TripUpdates.json",
    poll_interval_s=20,
)

_TRIP = TripInstance(
    source="generic-transit-authority",
    trip_id="ext-trip-1",
    vehicle_id="ext-veh-1",
    route_id="ext-route-1",
    direction_id=0,
    shape_id="ext-shape-1",
    start_date="20260917",
    started_at=1_700_000_000.0,
)

_STATE = VehicleState(epoch=1_700_000_050.0, lat=10.0, lon=20.0, progress_m=None)


def _stops() -> list[Stop]:
    return [
        Stop(stop_id="ext-s1", stop_sequence=1, lat=10.001, lon=20.001),
        Stop(stop_id="ext-s2", stop_sequence=2, lat=10.002, lon=20.002),
    ]


def test_source_config_accepts_an_arbitrary_third_party_url() -> None:
    """No code path special-cases a known agency -- any URL round-trips."""
    assert _EXTERNAL_SOURCE.name == "generic-transit-authority"
    assert "example-city.gov" in _EXTERNAL_SOURCE.trip_updates_url


async def test_gtfsrt_predictor_scores_an_arbitrary_json_tu_feed() -> None:
    feed = {
        "header": {"timestamp": 1700000000},
        "entity": [
            {
                "id": "1",
                "trip_update": {
                    "trip": {"trip_id": "ext-trip-1", "route_id": "ext-route-1"},
                    "stop_time_update": [
                        {"stop_id": "ext-s1", "stop_sequence": 1, "arrival": {"time": 1700000300}},
                        {"stop_id": "ext-s2", "stop_sequence": 2, "arrival": {"time": 1700000600}},
                    ],
                },
            }
        ],
    }

    with respx.mock:
        respx.get(_EXTERNAL_SOURCE.trip_updates_url).mock(return_value=httpx.Response(200, json=feed))
        poller = AutoTripUpdatesPoller(_EXTERNAL_SOURCE.trip_updates_url)
        result = await poller.poll()

    matching = next(r for r in result.records if r.trip_id == "ext-trip-1")
    tu_input = TripUpdateInput(
        header_timestamp=result.header_timestamp, stop_time_updates=matching.stop_time_updates
    )

    predictor = GtfsRtTripUpdatesPredictor()
    preds = predictor.predict(_TRIP, _stops(), _STATE, tu_input)

    assert len(preds) == 2
    assert preds[0].stop_id == "ext-s1"
    assert preds[0].predicted_arrival_epoch == pytest.approx(1700000300.0)
    assert preds[1].predicted_arrival_epoch == pytest.approx(1700000600.0)


async def test_gtfsrt_predictor_scores_an_arbitrary_protobuf_tu_feed() -> None:
    """Same external agency, but it only publishes protobuf -- still first-class."""
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = 1700000000
    entity = msg.entity.add()
    entity.id = "1"
    entity.trip_update.trip.trip_id = "ext-trip-1"
    entity.trip_update.trip.route_id = "ext-route-1"
    stu = entity.trip_update.stop_time_update.add()
    stu.stop_id = "ext-s1"
    stu.stop_sequence = 1
    stu.arrival.time = 1700000300
    body = msg.SerializeToString()

    with respx.mock:
        respx.get(_EXTERNAL_SOURCE.trip_updates_url).mock(
            return_value=httpx.Response(200, content=body, headers={"content-type": "application/x-protobuf"})
        )
        poller = AutoTripUpdatesPoller(_EXTERNAL_SOURCE.trip_updates_url)
        result = await poller.poll()

    matching = next(r for r in result.records if r.trip_id == "ext-trip-1")
    tu_input = TripUpdateInput(
        header_timestamp=result.header_timestamp, stop_time_updates=matching.stop_time_updates
    )

    predictor = GtfsRtTripUpdatesPredictor()
    preds = predictor.predict(_TRIP, _stops(), _STATE, tu_input)

    assert len(preds) == 1
    assert preds[0].predicted_arrival_epoch == pytest.approx(1700000300.0)


def test_parse_tu_feed_has_no_agency_specific_branching() -> None:
    """`parse_tu_feed` reads any spec-valid feed dict -- no allowlist of known agencies."""
    header_ts, records = parse_tu_feed(
        {
            "header": {"timestamp": 1},
            "entity": [{"id": "1", "trip_update": {"trip": {"trip_id": "whatever-anyone-publishes"}}}],
        }
    )
    assert records[0].trip_id == "whatever-anyone-publishes"
