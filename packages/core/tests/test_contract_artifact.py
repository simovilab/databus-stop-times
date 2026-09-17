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
