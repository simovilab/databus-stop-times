"""test_feeds — YAML → SourceConfig round-trip tests (P0).

Tests for:
- load_named_source("mbta") and ("bucr") round-trip via shipped YAML files
- list_sources() returning both shipped sources
- load_source_config() from an arbitrary path
- load_named_source() raising FileNotFoundError for unknown names
"""

from __future__ import annotations

from pathlib import Path

import pytest

from etaval.feeds.source_config import list_sources, load_named_source, load_source_config

# ── shipped YAML round-trips ──────────────────────────────────────────────────


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


# ── list_sources ──────────────────────────────────────────────────────────────


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


# ── load_source_config from arbitrary path ────────────────────────────────────


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


# ── error cases ───────────────────────────────────────────────────────────────


def test_load_named_source_unknown_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_named_source("__nonexistent_source__")


def test_load_source_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_source_config("/tmp/__etaval_no_such_file__.yaml")
