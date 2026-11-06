"""`stoptimes models publish`: a directory artifact becomes an MLflow model version.

Runs on a local sqlite MLflow store, no network."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime

import pytest
from stoptimes.cli.main import app
from typer.testing import CliRunner

pytest.importorskip("mlflow")
runner = CliRunner()


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    from stoptimes.core import registry as registry_pkg
    from stoptimes.core.registry.mlflow_backend import MlflowRegistry

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    reg = MlflowRegistry(
        uri,
        artifact_location=str(tmp_path / "mlartifacts"),
        cache_dir=tmp_path / "cache",
        list_ttl_s=0,
    )
    monkeypatch.setitem(registry_pkg._MLFLOW_REGISTRIES, uri, reg)
    return reg


def _artifact(root, artifact_id="xgb-live", source_id="mbta"):
    from stoptimes.core.contracts.artifact import (
        ArtifactMetadata,
        ModelScope,
        Provenance,
        write_metadata,
    )

    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="fake",
        scope=ModelScope(source_id=source_id),
        provenance=Provenance(created_at=datetime.now(UTC)),
    )
    write_metadata(meta, root / artifact_id)
    (root / artifact_id / meta.model_filename).write_bytes(pickle.dumps({"stub": True}))


def test_publish_creates_a_version_and_sets_the_alias(tmp_path, target):
    _artifact(tmp_path / "reg")
    args = [
        "models",
        "publish",
        "xgb-live",
        "--registry",
        str(tmp_path / "reg"),
        "--to",
        target.uri,
    ]
    result = runner.invoke(app, [*args, "--alias", "staging", "--json"])
    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row == {
        "name": "stoptimes-mbta",
        "version": 1,
        "artifact_id": "xgb-live",
        "aliases": ["staging"],
    }
    assert target.resolve("stoptimes-mbta@staging").artifact_id == "xgb-live"


def test_publish_refuses_a_non_mlflow_target_and_unknown_artifacts(tmp_path, target):
    _artifact(tmp_path / "reg")
    bad = runner.invoke(
        app,
        ["models", "publish", "xgb-live", "--registry", str(tmp_path / "reg"), "--to", "/tmp/x"],
    )
    assert bad.exit_code == 2
    missing = runner.invoke(
        app, ["models", "publish", "nope", "--registry", str(tmp_path / "reg"), "--to", target.uri]
    )
    assert missing.exit_code == 1
