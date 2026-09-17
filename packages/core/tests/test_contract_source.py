"""Tests for contracts.source and the val-YAML adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from stoptimes.core.contracts.adapters import source_from_val_yaml
from stoptimes.core.contracts.source import (
    Archive,
    ArchiveKind,
    FeedFormat,
    FeedKind,
    GroundTruthDetector,
    RealtimeFeed,
    Source,
    dump_source,
    load_source,
)

# The val source YAMLs live here today; a future track (C2) may move them to
# examples/agencies/. Discover whichever exists so this test survives that move.
_REPO_ROOT = Path(__file__).parents[3]
_CANDIDATE_DIRS = [
    _REPO_ROOT / "packages" / "val" / "src" / "stoptimes" / "val" / "config" / "sources",
    _REPO_ROOT / "examples" / "agencies",
]


def _sources_dir() -> Path:
    for candidate in _CANDIDATE_DIRS:
        if candidate.is_dir() and list(candidate.glob("*.yaml")):
            return candidate
    pytest.skip("no val source YAMLs found in either known location")


@pytest.mark.parametrize("name", ["mbta", "bucr"])
def test_val_yaml_loads_via_adapter(name):
    directory = _sources_dir()
    path = directory / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    with path.open() as fh:
        data = yaml.safe_load(fh)

    source = source_from_val_yaml(data)

    assert source.id == name
    assert source.gtfs_static_url == data["gtfs_static_url"]
    vp = source.feed(FeedKind.VEHICLE_POSITIONS)
    tu = source.feed(FeedKind.TRIP_UPDATES)
    assert vp is not None and vp.url == data["vehicle_positions_url"]
    assert tu is not None and tu.url == data["trip_updates_url"]
    assert vp.format == FeedFormat.JSON
    assert vp.poll_interval_s == data["poll_interval_s"]
    assert source.detector.arrival_radius_m == data.get("arrival_radius_m", 40.0)
    assert source.detector.dwell_s == data.get("dwell_s", 0.0)
    assert source.detector.shape_threshold_m == data.get("shape_threshold_m", 30.0)


def test_all_yaml_files_in_sources_dir_load():
    directory = _sources_dir()
    for path in directory.glob("*.yaml"):
        with path.open() as fh:
            data = yaml.safe_load(fh)
        source_from_val_yaml(data)  # must not raise


def _source(**overrides) -> Source:
    base = dict(
        id="mbta",
        name="MBTA",
        gtfs_static_url="https://cdn.mbta.com/MBTA_GTFS.zip",
        realtime_feeds=[
            RealtimeFeed(kind=FeedKind.VEHICLE_POSITIONS, url="https://x/vp.json"),
            RealtimeFeed(kind=FeedKind.TRIP_UPDATES, url="https://x/tu.json"),
        ],
    )
    base.update(overrides)
    return Source(**base)


def test_round_trip_yaml(tmp_path):
    source = _source()
    path = dump_source(source, tmp_path / "mbta.yaml")
    assert load_source(path) == source


def test_rejects_duplicate_feed_kind():
    with pytest.raises(ValidationError):
        _source(
            realtime_feeds=[
                RealtimeFeed(kind=FeedKind.VEHICLE_POSITIONS, url="https://x/1"),
                RealtimeFeed(kind=FeedKind.VEHICLE_POSITIONS, url="https://x/2"),
            ]
        )


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Source.model_validate({**_source().model_dump(), "surprise": 1})


def test_rejects_bad_id():
    with pytest.raises(ValidationError):
        _source(id="Not Valid!")


def test_auth_env_never_carries_secret_values():
    feed = RealtimeFeed(
        kind=FeedKind.TRIP_UPDATES,
        url="https://x/tu.json",
        auth_env={"x-api-key": "MBTA_API_KEY"},
    )
    # The contract only stores an env var *name*; nothing here should look
    # like a live secret value (heuristic: no long opaque token strings).
    assert feed.auth_env == {"x-api-key": "MBTA_API_KEY"}
    assert feed.auth_env["x-api-key"].isupper()


def test_archive_kind_inferred_by_uri_shape():
    local = Archive(uri="/var/data/mbta")
    assert local.kind == ArchiveKind.LOCAL
    s3 = Archive(kind=ArchiveKind.S3, uri="s3://transit/feeds/mbta")
    assert s3.kind == ArchiveKind.S3


def test_detector_defaults_match_domain_model():
    detector = GroundTruthDetector()
    assert detector.arrival_radius_m == 40.0
    assert detector.dwell_s == 0.0
    assert detector.shape_threshold_m == 30.0


def test_agency_profile_keys_survive_adapter():
    from stoptimes.core.contracts.adapters import source_from_val_yaml

    data = {
        "name": "demo",
        "gtfs_static_url": "feed.zip",
        "vehicle_positions_url": "vp.json",
        "trip_updates_url": "tu.json",
        "poll_interval_s": 10,
        "timezone": "America/Costa_Rica",
        "holiday_region": "CR",
        "serving": {"primary_model": "a", "fallback_model": "b"},
    }
    source = source_from_val_yaml(data)
    assert source.timezone == "America/Costa_Rica"
    assert source.holiday_region == "CR"
    assert source.serving.primary_model == "a"
    assert source.serving.fallback_model == "b"
