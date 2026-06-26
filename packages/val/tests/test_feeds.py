"""test_feeds — feed adapter tests for P0 (SourceConfig) and P1 (GTFS-RT + static + tracker).

P0 tests: YAML → SourceConfig round-trips.
P1 tests: GTFS-RT JSON parsing, async pollers (respx mock), GtfsStaticLoader,
          and ActiveTripTracker scripted sequences.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from etaval.domain.models import VehicleState
from etaval.feeds.gtfsrt_json import (
    TripUpdatesPoller,
    VehiclePositionRecord,
    VehiclePositionsPoller,
    parse_tu_feed,
    parse_vp_feed,
)
from etaval.feeds.source_config import list_sources, load_named_source, load_source_config
from etaval.feeds.static import GtfsStaticLoader
from etaval.feeds.tracker import ActiveTripTracker

# ── P0: shipped YAML round-trips ──────────────────────────────────────────────


def test_load_mbta_source_config() -> None:
    cfg = load_named_source("mbta")

    assert cfg.name == "mbta"
    assert cfg.gtfs_static_url == "https://cdn.mbta.com/MBTA_GTFS.zip"
    assert cfg.vehicle_positions_url == "https://cdn.mbta.com/realtime/VehiclePositions.json"
    assert cfg.trip_updates_url == "https://cdn.mbta.com/realtime/TripUpdates.json"
    assert cfg.poll_interval_s == 15
    assert isinstance(cfg.headers, dict)
    assert cfg.arrival_radius_m == pytest.approx(40.0)
    assert cfg.dwell_s == pytest.approx(0.0)
    assert cfg.shape_threshold_m == pytest.approx(30.0)


def test_load_bucr_source_config() -> None:
    cfg = load_named_source("bucr")

    assert cfg.name == "bucr"
    assert "localhost:8000" in cfg.gtfs_static_url
    assert "vehicle_positions" in cfg.vehicle_positions_url
    assert "trip_updates" in cfg.trip_updates_url
    assert cfg.poll_interval_s == 5
    assert isinstance(cfg.headers, dict)
    assert cfg.arrival_radius_m == pytest.approx(30.0)
    assert cfg.dwell_s == pytest.approx(0.0)
    assert cfg.shape_threshold_m == pytest.approx(25.0)


# ── P0: list_sources ──────────────────────────────────────────────────────────


def test_list_sources_includes_shipped_sources() -> None:
    sources = list_sources()
    assert "mbta" in sources
    assert "bucr" in sources


def test_list_sources_is_sorted() -> None:
    sources = list_sources()
    assert sources == sorted(sources)


def test_list_sources_custom_dir(tmp_path: Path) -> None:
    (tmp_path / "alpha.yaml").write_text(
        "name: alpha\n"
        "gtfs_static_url: http://a.example.com/gtfs.zip\n"
        "vehicle_positions_url: http://a.example.com/vp.json\n"
        "trip_updates_url: http://a.example.com/tu.json\n"
        "poll_interval_s: 10\n"
    )
    (tmp_path / "beta.yaml").write_text(
        "name: beta\n"
        "gtfs_static_url: http://b.example.com/gtfs.zip\n"
        "vehicle_positions_url: http://b.example.com/vp.json\n"
        "trip_updates_url: http://b.example.com/tu.json\n"
        "poll_interval_s: 20\n"
    )
    result = list_sources(config_dir=tmp_path)
    assert result == ["alpha", "beta"]


# ── P0: load_source_config from arbitrary path ────────────────────────────────


def test_load_source_config_from_path(tmp_path: Path) -> None:
    config_file = tmp_path / "test_source.yaml"
    config_file.write_text(
        "name: test_source\n"
        "gtfs_static_url: http://example.com/gtfs.zip\n"
        "vehicle_positions_url: http://example.com/vp.json\n"
        "trip_updates_url: http://example.com/tu.json\n"
        "poll_interval_s: 10\n"
        "headers: {}\n"
        "arrival_radius_m: 50.0\n"
        "dwell_s: 2.0\n"
        "shape_threshold_m: 35.0\n"
    )
    cfg = load_source_config(config_file)

    assert cfg.name == "test_source"
    assert cfg.poll_interval_s == 10
    assert cfg.arrival_radius_m == pytest.approx(50.0)
    assert cfg.dwell_s == pytest.approx(2.0)
    assert cfg.shape_threshold_m == pytest.approx(35.0)


def test_load_source_config_default_optional_fields(tmp_path: Path) -> None:
    """Omitting optional fields should use model defaults."""
    config_file = tmp_path / "minimal.yaml"
    config_file.write_text(
        "name: minimal\n"
        "gtfs_static_url: http://example.com/gtfs.zip\n"
        "vehicle_positions_url: http://example.com/vp.json\n"
        "trip_updates_url: http://example.com/tu.json\n"
        "poll_interval_s: 30\n"
    )
    cfg = load_source_config(config_file)

    assert cfg.headers == {}
    assert cfg.auth is None
    assert cfg.arrival_radius_m == pytest.approx(40.0)  # model default
    assert cfg.dwell_s == pytest.approx(0.0)
    assert cfg.shape_threshold_m == pytest.approx(30.0)


# ── P0: error cases ───────────────────────────────────────────────────────────


def test_load_named_source_unknown_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_named_source("__nonexistent_source__")


def test_load_source_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_source_config("/tmp/__etaval_no_such_file__.yaml")


# ── P1: VP fixture parsing ────────────────────────────────────────────────────


def test_parse_mbta_vp_entity_count(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "mbta_vp.json").read_text())
    header_ts, records = parse_vp_feed(feed)
    assert header_ts == pytest.approx(1700000000.0)
    assert len(records) == 2


def test_parse_mbta_vp_first_record_fields(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "mbta_vp.json").read_text())
    _, records = parse_vp_feed(feed)
    r = records[0]
    assert r.trip_id == "trip_1"
    assert r.route_id == "route_1"
    assert r.direction_id == 0
    assert r.start_date == "20240101"
    assert r.vehicle_id == "vehicle_1"
    assert r.state.lat == pytest.approx(42.3601)
    assert r.state.lon == pytest.approx(-71.0589)
    assert r.state.speed == pytest.approx(10.5)
    assert r.state.current_status == "IN_TRANSIT_TO"
    assert r.state.current_stop_sequence == 2
    assert r.state.epoch == pytest.approx(1700000000.0)


def test_parse_mbta_vp_second_record_status(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "mbta_vp.json").read_text())
    _, records = parse_vp_feed(feed)
    assert records[1].state.current_status == "STOPPED_AT"
    assert records[1].state.current_stop_sequence == 3


def test_parse_bucr_vp_entity_count(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "bucr_vp.json").read_text())
    header_ts, records = parse_vp_feed(feed)
    assert header_ts == pytest.approx(1700010000.0)
    assert len(records) == 2
    assert records[0].trip_id == "bucr_trip_1"
    assert records[0].vehicle_id == "bucr_bus_10"
    assert records[0].state.lat == pytest.approx(9.9350)


def test_parse_vp_skips_entity_without_position() -> None:
    """Entities missing lat/lon must be silently skipped."""
    feed: dict = {
        "header": {"timestamp": 1700000000},
        "entity": [
            # no position field
            {"id": "x", "vehicle": {"trip": {"trip_id": "t1"}, "timestamp": 1700000000}},
            # valid entity
            {
                "id": "y",
                "vehicle": {
                    "trip": {"trip_id": "t2", "route_id": "r1"},
                    "vehicle": {"id": "v1"},
                    "position": {"latitude": 1.0, "longitude": 2.0},
                    "timestamp": 1700000000,
                },
            },
        ],
    }
    _, records = parse_vp_feed(feed)
    assert len(records) == 1
    assert records[0].trip_id == "t2"


# ── P1: TU fixture parsing ────────────────────────────────────────────────────


def test_parse_mbta_tu_entity_count(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "mbta_tu.json").read_text())
    header_ts, records = parse_tu_feed(feed)
    assert header_ts == pytest.approx(1700000000.0)
    assert len(records) == 2


def test_parse_mbta_tu_first_record_stop_updates(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "mbta_tu.json").read_text())
    _, records = parse_tu_feed(feed)
    r = records[0]
    assert r.trip_id == "trip_1"
    assert r.route_id == "route_1"
    assert len(r.stop_time_updates) == 2
    stu = r.stop_time_updates[0]
    assert stu.stop_sequence == 2
    assert stu.stop_id == "s2"
    assert stu.arrival_time == pytest.approx(1700000120.0)
    assert stu.departure_time == pytest.approx(1700000150.0)


def test_parse_mbta_tu_second_entity_one_update(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "mbta_tu.json").read_text())
    _, records = parse_tu_feed(feed)
    assert len(records[1].stop_time_updates) == 1


def test_parse_bucr_tu_entity_count(fixtures_dir: Path) -> None:
    feed = json.loads((fixtures_dir / "bucr_tu.json").read_text())
    header_ts, records = parse_tu_feed(feed)
    assert header_ts == pytest.approx(1700010000.0)
    assert len(records) == 2
    assert records[0].trip_id == "bucr_trip_1"
    assert records[0].stop_time_updates[0].arrival_time == pytest.approx(1700010120.0)


def test_parse_tu_skips_entity_without_trip_id() -> None:
    """TU entities lacking trip_id must be silently skipped."""
    feed: dict = {
        "header": {"timestamp": 1700000000},
        "entity": [
            {"id": "x", "trip_update": {"trip": {}, "stop_time_update": []}},
            {
                "id": "y",
                "trip_update": {
                    "trip": {"trip_id": "t1"},
                    "stop_time_update": [],
                },
            },
        ],
    }
    _, records = parse_tu_feed(feed)
    assert len(records) == 1
    assert records[0].trip_id == "t1"


# ── P1: async pollers (respx mock) ────────────────────────────────────────────


async def test_vp_poller_returns_parsed_result(fixtures_dir: Path) -> None:
    feed_data = json.loads((fixtures_dir / "mbta_vp.json").read_text())
    with respx.mock:
        respx.get("http://test.mbta.example.com/vp.json").mock(
            return_value=httpx.Response(200, json=feed_data)
        )
        poller = VehiclePositionsPoller("http://test.mbta.example.com/vp.json")
        result = await poller.poll()

    assert result.header_timestamp == pytest.approx(1700000000.0)
    assert len(result.records) == 2
    assert result.records[0].trip_id == "trip_1"


async def test_tu_poller_returns_parsed_result(fixtures_dir: Path) -> None:
    feed_data = json.loads((fixtures_dir / "mbta_tu.json").read_text())
    with respx.mock:
        respx.get("http://test.mbta.example.com/tu.json").mock(
            return_value=httpx.Response(200, json=feed_data)
        )
        poller = TripUpdatesPoller("http://test.mbta.example.com/tu.json")
        result = await poller.poll()

    assert result.header_timestamp == pytest.approx(1700000000.0)
    assert len(result.records) == 2
    assert result.records[0].stop_time_updates[1].arrival_time == pytest.approx(1700000300.0)


async def test_vp_poller_passes_custom_headers(fixtures_dir: Path) -> None:
    feed_data = json.loads((fixtures_dir / "mbta_vp.json").read_text())
    with respx.mock:
        route = respx.get("http://test.mbta.example.com/vp.json").mock(
            return_value=httpx.Response(200, json=feed_data)
        )
        poller = VehiclePositionsPoller(
            "http://test.mbta.example.com/vp.json",
            headers={"X-API-Key": "secret"},
        )
        await poller.poll()
    assert route.called
    assert route.calls.last.request.headers["x-api-key"] == "secret"


# ── P1: GtfsStaticLoader ─────────────────────────────────────────────────────


def test_static_loader_loads_dataframes(fixtures_dir: Path) -> None:
    loader = GtfsStaticLoader(fixtures_dir / "mini_gtfs.zip")
    assert len(loader.stops) == 4
    assert len(loader.trips) == 1
    assert len(loader.stop_times) == 4
    assert len(loader.shapes) == 8
    assert len(loader.routes) == 1


def test_static_loader_trip_geometry_shape_point_count(fixtures_dir: Path) -> None:
    loader = GtfsStaticLoader(fixtures_dir / "mini_gtfs.zip")
    shape_points, _ = loader.trip_geometry("trip1")
    assert len(shape_points) == 8


def test_static_loader_trip_geometry_shape_point_values(fixtures_dir: Path) -> None:
    loader = GtfsStaticLoader(fixtures_dir / "mini_gtfs.zip")
    shape_points, _ = loader.trip_geometry("trip1")
    lat, lon, dist = shape_points[0]
    assert lat == pytest.approx(42.3601)
    assert lon == pytest.approx(-71.0589)
    assert dist == pytest.approx(0.0)


def test_static_loader_trip_geometry_ordered_stops(fixtures_dir: Path) -> None:
    loader = GtfsStaticLoader(fixtures_dir / "mini_gtfs.zip")
    _, ordered_stops = loader.trip_geometry("trip1")
    assert len(ordered_stops) == 4
    seqs = [s.stop_sequence for s in ordered_stops]
    assert seqs == sorted(seqs)
    assert ordered_stops[0].stop_id == "s1"
    assert ordered_stops[0].lat == pytest.approx(42.3601)
    assert ordered_stops[0].lon == pytest.approx(-71.0589)
    assert ordered_stops[0].progress_m is None  # assigned in P2


def test_static_loader_trip_geometry_unknown_trip(fixtures_dir: Path) -> None:
    loader = GtfsStaticLoader(fixtures_dir / "mini_gtfs.zip")
    with pytest.raises(ValueError, match="not found"):
        loader.trip_geometry("no_such_trip")


def test_static_loader_missing_zip_raises() -> None:
    with pytest.raises(FileNotFoundError):
        GtfsStaticLoader("/tmp/__etaval_no_such_gtfs__.zip")


# ── P1: ActiveTripTracker ────────────────────────────────────────────────────


def _vp_record(
    trip_id: str,
    vehicle_id: str,
    lat: float,
    epoch: float,
    route_id: str = "r1",
) -> VehiclePositionRecord:
    state = VehicleState(epoch=epoch, lat=lat, lon=-71.0)
    return VehiclePositionRecord(
        trip_id=trip_id,
        route_id=route_id,
        direction_id=0,
        start_date="20240101",
        vehicle_id=vehicle_id,
        state=state,
    )


def test_tracker_creates_trip_on_first_update() -> None:
    tracker = ActiveTripTracker(source="mbta")
    tracker.update([_vp_record("t1", "v1", 42.0, 1000.0)], now=1000.0)

    trips = tracker.active_trips()
    assert len(trips) == 1
    trip = trips[0]
    assert trip.trip_id == "t1"
    assert trip.vehicle_id == "v1"
    assert trip.route_id == "r1"
    assert trip.source == "mbta"
    assert trip.started_at == pytest.approx(1000.0)


def test_tracker_appends_position_history() -> None:
    tracker = ActiveTripTracker(source="mbta")
    tracker.update([_vp_record("t1", "v1", 42.00, 1000.0)], now=1000.0)
    tracker.update([_vp_record("t1", "v1", 42.01, 1015.0)], now=1015.0)
    tracker.update([_vp_record("t1", "v1", 42.02, 1030.0)], now=1030.0)

    history = tracker.position_history(("t1", "v1"))
    assert len(history) == 3
    assert history[0].lat == pytest.approx(42.00)
    assert history[2].lat == pytest.approx(42.02)


def test_tracker_two_trips_independent() -> None:
    tracker = ActiveTripTracker(source="mbta")
    tracker.update(
        [_vp_record("t1", "v1", 42.0, 1000.0), _vp_record("t2", "v2", 43.0, 1000.0)],
        now=1000.0,
    )
    assert len(tracker.active_trips()) == 2
    assert len(tracker.position_history(("t1", "v1"))) == 1
    assert len(tracker.position_history(("t2", "v2"))) == 1


def test_tracker_started_at_fixed_on_first_sight() -> None:
    """started_at must not change on subsequent updates."""
    tracker = ActiveTripTracker(source="mbta")
    tracker.update([_vp_record("t1", "v1", 42.0, 1000.0)], now=1000.0)
    tracker.update([_vp_record("t1", "v1", 42.1, 1015.0)], now=1015.0)

    trip = tracker.get_trip(("t1", "v1"))
    assert trip is not None
    assert trip.started_at == pytest.approx(1000.0)


def test_tracker_expire_removes_stale_trips() -> None:
    tracker = ActiveTripTracker(source="mbta")
    tracker.update([_vp_record("t1", "v1", 42.0, 1000.0)], now=1000.0)
    tracker.update([_vp_record("t2", "v2", 43.0, 1200.0)], now=1200.0)

    # t1 last seen at 1000, t2 at 1200.
    # At now=1350, max_age_s=150: t1 age=350 > 150 → stale; t2 age=150 (not >) → alive.
    tracker.expire(now=1350.0, max_age_s=150.0)

    active = tracker.active_trips()
    assert len(active) == 1
    assert active[0].trip_id == "t2"
    # position history must also be gone
    assert tracker.position_history(("t1", "v1")) == []


def test_tracker_expire_keeps_recently_updated_trips() -> None:
    tracker = ActiveTripTracker(source="mbta")
    tracker.update([_vp_record("t1", "v1", 42.0, 1000.0)], now=1000.0)
    tracker.expire(now=1050.0, max_age_s=300.0)
    assert len(tracker.active_trips()) == 1


def test_tracker_get_trip_returns_none_for_unknown() -> None:
    tracker = ActiveTripTracker(source="mbta")
    assert tracker.get_trip(("unknown", "vehicle")) is None
