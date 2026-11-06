"""`mlflow:<model>@<alias>` / `mlflow:<model>/<version>` predictors (MLFLOW_INTEGRATION §0 D9).

Runs on a local sqlite MLflow store, no network."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    ModelScope,
    Provenance,
    write_metadata,
)
from stoptimes.core.registry import MLFLOW_ENV, RegistryError, mlflow_registry
from stoptimes.val.engine.assembly import is_known_predictor_id, resolve_predictor_factory
from stoptimes.val.predictors.mlflow_models import build_mlflow_predictor, mlflow_predictor_ids

pytest.importorskip("mlflow")


class _StubModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A sqlite MLflow registry that `$STOPTIMES_MLFLOW` points at."""
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    from stoptimes.core import registry as registry_pkg
    from stoptimes.core.registry.mlflow_backend import MlflowRegistry

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    reg = MlflowRegistry(
        uri,
        artifact_location=str(tmp_path / "mlartifacts"),
        cache_dir=tmp_path / "cache",
        list_ttl_s=0.0,
    )
    monkeypatch.setitem(registry_pkg._MLFLOW_REGISTRIES, uri, reg)
    monkeypatch.setenv(MLFLOW_ENV, uri)
    monkeypatch.delenv("STOPTIMES_REGISTRY", raising=False)
    return reg


def _publish(registry, artifact_id: str, source_id: str = "mbta") -> ArtifactMetadata:
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="fake",
        scope=ModelScope(source_id=source_id),
        features=["x"],
        provenance=Provenance(created_at=datetime.now(UTC)),
    )
    with registry.staged(artifact_id, overwrite=True) as directory:
        write_metadata(meta, directory)
        with open(directory / meta.model_filename, "wb") as fh:
            pickle.dump(_StubModel(), fh)
    return meta


def test_ids_list_aliases_then_versions_newest_first(registry):
    _publish(registry, "a")
    _publish(registry, "b")
    registry.set_alias("stoptimes-mbta", "staging", 2)
    registry.set_alias("stoptimes-mbta", "production", 1)
    assert mlflow_predictor_ids(registry.listing(refresh=True)) == [
        "mlflow:stoptimes-mbta@production",
        "mlflow:stoptimes-mbta@staging",
        "mlflow:stoptimes-mbta/2",
        "mlflow:stoptimes-mbta/1",
    ]


def test_alias_is_pinned_to_the_version_it_had_when_the_run_started(registry):
    _publish(registry, "first")
    registry.set_alias("stoptimes-mbta", "staging", 1)
    predictor = build_mlflow_predictor("stoptimes-mbta@staging")
    assert predictor.predictor_id == "mlflow:stoptimes-mbta@staging"
    assert predictor.resolved_version == "stoptimes-mbta/1"

    _publish(registry, "second")
    registry.set_alias("stoptimes-mbta", "staging", 2)
    assert predictor._static_artifact.artifact_id == "first"
    assert build_mlflow_predictor("stoptimes-mbta@staging")._static_artifact.artifact_id == "second"


def test_version_refs_and_the_assembly_prefix(registry):
    _publish(registry, "only")
    assert is_known_predictor_id("mlflow:stoptimes-mbta/1")
    assert not is_known_predictor_id("mlflow:")
    predictor = resolve_predictor_factory("mlflow:stoptimes-mbta/1")()
    assert predictor.predictor_id == "mlflow:stoptimes-mbta/1"
    assert predictor._static_artifact.artifact_id == "only"
    assert predictor._candidates == [predictor._static_artifact]


def test_unknown_alias_raises(registry):
    _publish(registry, "only")
    with pytest.raises(RegistryError):
        build_mlflow_predictor("stoptimes-mbta@nope")


def test_no_server_configured(monkeypatch):
    monkeypatch.delenv(MLFLOW_ENV, raising=False)
    monkeypatch.delenv("STOPTIMES_REGISTRY", raising=False)
    assert mlflow_registry() is None
    with pytest.raises(RegistryError, match=MLFLOW_ENV):
        build_mlflow_predictor("stoptimes-mbta@staging")


def test_registry_env_mlflow_uri_is_the_fallback(registry, monkeypatch):
    monkeypatch.delenv(MLFLOW_ENV, raising=False)
    monkeypatch.setenv("STOPTIMES_REGISTRY", f"mlflow+{registry.tracking_uri}")
    assert mlflow_registry() is registry


# ── API ────────────────────────────────────────────────────────────────────────


async def _predictors(tmp_path: Path) -> list[str]:
    from stoptimes.val.api.app import create_app
    from stoptimes.val.api.runs import RunManager

    app = create_app(run_manager=RunManager(data_dir=tmp_path / "data"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/predictors")
    assert resp.status_code == 200
    return resp.json()


async def test_api_lists_mlflow_predictors(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_MODEL_REGISTRY_DIR", str(tmp_path / "empty"))
    _publish(registry, "only")
    registry.set_alias("stoptimes-mbta", "staging", 1)
    ids = await _predictors(tmp_path)
    assert "gtfsrt" in ids
    assert ids[-2:] == ["mlflow:stoptimes-mbta@staging", "mlflow:stoptimes-mbta/1"]


async def test_api_still_answers_when_mlflow_fails(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_MODEL_REGISTRY_DIR", str(tmp_path / "empty"))

    def _down(**_: object) -> list:
        raise RegistryError("server unreachable")

    monkeypatch.setattr(registry, "listing", _down)
    ids = await _predictors(tmp_path)
    assert "gtfsrt" in ids
    assert not any(pid.startswith("mlflow:") for pid in ids)


async def test_api_does_not_wait_forever_on_a_slow_server(registry, tmp_path, monkeypatch):
    import time

    from stoptimes.val.api import app as app_module

    monkeypatch.setenv("STOPTIMES_MODEL_REGISTRY_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(registry, "listing", lambda **_: time.sleep(1.0) or [])
    monkeypatch.setattr(app_module, "MLFLOW_LIST_TIMEOUT_S", 0.05)
    started = time.monotonic()
    assert await app_module._mlflow_predictor_ids(0.05) == []
    assert time.monotonic() - started < 0.9
