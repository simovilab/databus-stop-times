from datetime import datetime

import pytest
from pydantic import ValidationError

from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    HorizonScore,
    ModelFormat,
    Provenance,
    artifact_dir,
    read_metadata,
    version_mismatches,
    write_metadata,
)


def _meta(**overrides):
    base = dict(
        artifact_id="xgboost_mbta_20260818_120020",
        family="xgboost",
        features=["distance_to_stop", "hour"],
        params={"max_depth": 5},
        provenance=Provenance(
            created_at=datetime(2026, 8, 18, 12, 0, 20),
            lib_versions={"xgboost": "3.1.2", "scikit-learn": "1.7.2"},
        ),
        scores_by_horizon=[HorizonScore(horizon_min_s=0, horizon_max_s=300, n=10, mae_s=31.5)],
    )
    base.update(overrides)
    return ArtifactMetadata(**base)


def test_round_trip(tmp_path):
    meta = _meta()
    directory = artifact_dir(tmp_path, meta.artifact_id)
    write_metadata(meta, directory)
    assert read_metadata(directory) == meta
    assert not list(directory.glob(".artifact.*"))


def test_model_filename_follows_format():
    assert _meta().model_filename == "model.pkl"
    assert _meta(format=ModelFormat.ONNX).model_filename == "model.onnx"


@pytest.mark.parametrize("bad_id", ["", "../escape", "a/b", " spaced"])
def test_artifact_id_rejects_path_like_values(bad_id):
    with pytest.raises(ValidationError):
        _meta(artifact_id=bad_id)


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        ArtifactMetadata.model_validate({**_meta().model_dump(), "surprise": 1})


def test_version_mismatches_compares_major_minor():
    meta = _meta()
    assert version_mismatches(meta, {"xgboost": "3.1.9", "scikit-learn": "1.7.0"}) == []
    assert version_mismatches(meta, {"xgboost": "3.0.5"}) == ["xgboost: built with 3.1.2, installed 3.0.5"]
    assert version_mismatches(meta, {}) == []


# --- sidecar assets (schema version 2) -------------------------------------------


def test_assets_round_trip_and_default_to_empty(tmp_path):
    from stoptimes.core.contracts.artifact import ArtifactAsset, write_asset

    assert _meta().assets == {}
    directory = artifact_dir(tmp_path, "with-assets")
    asset = write_asset(directory, "segment_stats.bin", b"abc")
    meta = _meta(
        artifact_id="with-assets",
        assets={"segment_stats": asset},
        timezone="America/New_York",
        holiday_region="US_MA",
    )
    write_metadata(meta, directory)
    back = read_metadata(directory)
    assert back == meta
    assert back.assets["segment_stats"] == ArtifactAsset(file="segment_stats.bin", sha256=asset.sha256)
    assert back.timezone == "America/New_York" and back.schema_version == 2
    assert not list(directory.glob(".asset.*"))


def test_version_one_artifact_json_still_loads(tmp_path):
    meta = _meta()
    directory = artifact_dir(tmp_path, meta.artifact_id)
    write_metadata(meta, directory)
    import json

    path = directory / "artifact.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = 1
    for key in ("assets", "timezone", "holiday_region"):
        raw.pop(key)
    path.write_text(json.dumps(raw))
    old = read_metadata(directory)
    assert old.schema_version == 1 and old.assets == {} and old.timezone is None


@pytest.mark.parametrize("bad_file", ["", "../x", "a/b", "artifact.json", "card.md", ".hidden"])
def test_asset_file_must_be_a_plain_name(bad_file):
    from stoptimes.core.contracts.artifact import ArtifactAsset

    with pytest.raises(ValidationError):
        ArtifactAsset(file=bad_file, sha256="0" * 64)


def test_asset_sha256_must_be_hex():
    from stoptimes.core.contracts.artifact import ArtifactAsset

    with pytest.raises(ValidationError):
        ArtifactAsset(file="x.bin", sha256="nothex")


def test_read_asset_bytes_verifies_hash_and_presence(tmp_path):
    from stoptimes.core.contracts.artifact import read_asset_bytes, write_asset

    asset = write_asset(tmp_path, "a.bin", b"payload")
    assert read_asset_bytes(tmp_path, "a", asset) == b"payload"
    (tmp_path / "a.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        read_asset_bytes(tmp_path, "a", asset)
    (tmp_path / "a.bin").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        read_asset_bytes(tmp_path, "a", asset)
