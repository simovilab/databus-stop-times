"""Serving from the MLflow registry backend: the same selection, loading and refusals as from a
directory (docs/MLFLOW_INTEGRATION.md §0 D2). Runs on a local sqlite MLflow store, no network."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.contracts.artifact import ArtifactMetadata, ModelScope, Provenance, ScopeLevel, write_metadata
from stoptimes.serve.model_loading import (
    ALLOW_VERSION_DRIFT_ENV,
    ArtifactVersionDriftError,
    iter_registry_artifacts,
    load_artifact_model,
    resolve_artifact,
)

pytest.importorskip("mlflow")


class _StubModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["x"].to_numpy(dtype=float) * 3


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    from stoptimes.core.registry.mlflow_backend import MlflowRegistry

    return MlflowRegistry(
        f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "mlartifacts"),
        cache_dir=tmp_path / "cache",
        list_ttl_s=0.0,
    )


def _publish(registry, artifact_id, *, metric=50.0, scope=None, lib_versions=None) -> ArtifactMetadata:
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="fake",
        scope=scope or ModelScope(source_id="agency"),
        features=["x"],
        provenance=Provenance(created_at=datetime.now(timezone.utc), lib_versions=lib_versions or {}),
        metrics={"test_mae_seconds": metric},
    )
    with registry.staged(artifact_id) as directory:
        write_metadata(meta, directory)
        with open(directory / meta.model_filename, "wb") as fh:
            pickle.dump(_StubModel(), fh)
    return meta


def test_selection_and_loading_work_from_mlflow(registry):
    _publish(registry, "worse", metric=99.0)
    _publish(registry, "route22", metric=999.0, scope=ModelScope(level=ScopeLevel.ROUTE, route_ids=["22"]))
    _publish(registry, "better", metric=10.0)
    assert [m.artifact_id for m in iter_registry_artifacts(registry)] == ["better", "route22", "worse"]
    assert resolve_artifact(registry, source_id="agency").artifact_id == "better"
    chosen = resolve_artifact(registry, source_id="agency", route_id="22")
    assert chosen.artifact_id == "route22"
    model = load_artifact_model(chosen, registry)
    assert list(model.predict(pd.DataFrame({"x": [1.0, 2.0]}))) == [3.0, 6.0]
    assert model.artifact_id == "route22"


def test_drift_refusal_still_applies_to_mlflow_artifacts(registry, monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr("stoptimes.serve.model_loading._installed_lib_versions", lambda: {"sklearn": "1.9.0"})
    meta = _publish(registry, "stale", lib_versions={"sklearn": "1.2.0"})
    with pytest.raises(ArtifactVersionDriftError):
        load_artifact_model(meta, registry)
