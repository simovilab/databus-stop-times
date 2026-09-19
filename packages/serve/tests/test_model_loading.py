"""Model resolution (`select_artifact` over `artifact.json` files, never a
pickle) and BYO `ModelRef` loaders (pickle / onnx / callable / http)."""

from __future__ import annotations

import importlib.util
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.contracts.artifact import ArtifactMetadata, ModelScope, Provenance, ScopeLevel, write_metadata
from stoptimes.core.contracts.model import ModelRef, ModelRefKind
from stoptimes.core.contracts.selection import SelectionPolicy
from stoptimes.serve.model_loading import (
    ALLOW_UNPRODUCIBLE_FEATURES_ENV,
    ALLOW_VERSION_DRIFT_ENV,
    ArtifactVersionDriftError,
    CallableModelLoader,
    HttpModelLoader,
    OnnxModelLoader,
    PickleModelLoader,
    UnproducibleFeaturesError,
    check_feature_producibility,
    check_version_drift,
    iter_registry_artifacts,
    load_artifact_model,
    load_model_ref,
    resolve_artifact,
)


class _StubModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["x"].to_numpy(dtype=float)


def _write_artifact(registry_dir, artifact_id, *, family="fake", metric=50.0, scope=None, fmt=None):
    from stoptimes.core.contracts.artifact import ModelFormat

    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family=family,
        format=fmt or ModelFormat.PICKLE,
        scope=scope or ModelScope(),
        features=["x"],
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
        metrics={"test_mae_seconds": metric},
    )
    directory = registry_dir / artifact_id
    write_metadata(meta, directory)
    with open(directory / meta.model_filename, "wb") as fh:
        pickle.dump(_StubModel(), fh)
    return meta


def test_iter_registry_artifacts_skips_dirs_without_metadata(tmp_path):
    _write_artifact(tmp_path, "a1")
    (tmp_path / "not-an-artifact").mkdir()
    metas = iter_registry_artifacts(tmp_path)
    assert [m.artifact_id for m in metas] == ["a1"]


def test_iter_registry_artifacts_empty_when_dir_missing(tmp_path):
    assert iter_registry_artifacts(tmp_path / "nope") == []


def test_resolve_artifact_picks_best_global_by_metric(tmp_path):
    _write_artifact(tmp_path, "worse", metric=99.0)
    _write_artifact(tmp_path, "better", metric=10.0)
    chosen = resolve_artifact(tmp_path, source_id="agency")
    assert chosen.artifact_id == "better"


def test_resolve_artifact_honors_route_scope_over_global(tmp_path):
    _write_artifact(tmp_path, "global", metric=5.0)
    _write_artifact(
        tmp_path,
        "route22",
        metric=999.0,
        scope=ModelScope(level=ScopeLevel.ROUTE, route_ids=["22"]),
    )
    chosen = resolve_artifact(tmp_path, source_id="agency", route_id="22")
    assert chosen.artifact_id == "route22"


def test_resolve_artifact_hard_override_by_artifact_id(tmp_path):
    _write_artifact(tmp_path, "global", metric=5.0)
    _write_artifact(tmp_path, "pinned", metric=999.0)
    chosen = resolve_artifact(tmp_path, source_id="agency", artifact_id="pinned")
    assert chosen.artifact_id == "pinned"


def test_resolve_artifact_pin_via_policy(tmp_path):
    _write_artifact(tmp_path, "global", metric=5.0)
    _write_artifact(tmp_path, "pinned", metric=999.0)
    policy = SelectionPolicy(pins={"global": "pinned"})
    chosen = resolve_artifact(tmp_path, source_id="agency", policy=policy)
    assert chosen.artifact_id == "pinned"


def test_load_artifact_model_duck_types_missing_attrs(tmp_path):
    meta = _write_artifact(tmp_path, "a1", family="fake")
    model = load_artifact_model(meta, tmp_path)
    assert model.family == "fake"
    assert model.required_features == ["x"]
    assert model.artifact_id == "a1"
    preds = model.predict(pd.DataFrame({"x": [1.0, 2.0]}))
    assert list(preds) == [1.0, 2.0]


def test_load_artifact_model_rejects_non_pickle_format(tmp_path):
    from stoptimes.core.contracts.artifact import ModelFormat

    meta = _write_artifact(tmp_path, "a1", fmt=ModelFormat.ONNX)
    with pytest.raises(NotImplementedError):
        load_artifact_model(meta, tmp_path)


def test_pickle_model_loader(tmp_path):
    path = tmp_path / "model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(_StubModel(), fh)
    ref = ModelRef(kind=ModelRefKind.PICKLE, location=str(path), family="byo", required_features=["x"])
    model = PickleModelLoader().load(ref)
    assert list(model.predict(pd.DataFrame({"x": [3.0]}))) == [3.0]
    assert model.family == "byo"


def test_callable_model_loader(monkeypatch):
    import sys
    import types

    module = types.ModuleType("_fake_callable_model_mod")

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        return X["x"].to_numpy(dtype=float) * 2.0

    module.predict_fn = predict_fn
    sys.modules["_fake_callable_model_mod"] = module

    ref = ModelRef(kind=ModelRefKind.CALLABLE, location="_fake_callable_model_mod:predict_fn", required_features=["x"])
    model = CallableModelLoader().load(ref)
    assert list(model.predict(pd.DataFrame({"x": [3.0]}))) == [6.0]


def test_http_model_loader_and_predict(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"predictions": [1.0, 2.0]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)

    ref = ModelRef(
        kind=ModelRefKind.HTTP,
        location="https://example.test/predict",
        required_features=["x"],
        headers={"X-Static": "1"},
        auth_env={"Authorization": "STOPTIMES_TEST_TOKEN"},
    )
    monkeypatch.setenv("STOPTIMES_TEST_TOKEN", "secret-token")
    model = HttpModelLoader().load(ref)
    preds = model.predict(pd.DataFrame({"x": [1.0, 2.0]}))
    assert list(preds) == [1.0, 2.0]
    assert calls["headers"]["X-Static"] == "1"
    assert calls["headers"]["Authorization"] == "secret-token"


def test_http_model_mismatched_row_count_raises(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"predictions": [1.0]}

    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeResponse())

    ref = ModelRef(kind=ModelRefKind.HTTP, location="https://example.test/predict", required_features=["x"])
    model = HttpModelLoader().load(ref)
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"x": [1.0, 2.0]}))


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime not installed")
def test_onnx_model_loader_smoke():
    pytest.skip("onnxruntime present but no fixture .onnx model is shipped; loader import path covered elsewhere")


def test_onnx_model_loader_missing_dependency_raises_clear_error(monkeypatch):
    if importlib.util.find_spec("onnxruntime") is not None:
        pytest.skip("onnxruntime is installed; this test covers the absent-dependency path")
    ref = ModelRef(kind=ModelRefKind.ONNX, location="/dev/null", required_features=["x"])
    with pytest.raises(ImportError, match="onnxruntime"):
        OnnxModelLoader().load(ref)


def test_load_model_ref_dispatches_by_kind(tmp_path):
    path = tmp_path / "model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(_StubModel(), fh)
    ref = ModelRef(kind=ModelRefKind.PICKLE, location=str(path))
    model = load_model_ref(ref)
    assert list(model.predict(pd.DataFrame({"x": [5.0]}))) == [5.0]


# ---------------------------------------------------------------------------
# Version drift refusal (U7): a stale artifact must be refused, not loaded
# quietly, when a library that deserializes the model has drifted.
# ---------------------------------------------------------------------------


def _write_artifact_with_lib_versions(registry_dir, artifact_id, lib_versions, *, fmt=None):
    from stoptimes.core.contracts.artifact import ModelFormat

    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="fake",
        format=fmt or ModelFormat.PICKLE,
        features=["x"],
        provenance=Provenance(created_at=datetime.now(timezone.utc), lib_versions=lib_versions),
        metrics={"test_mae_seconds": 1.0},
    )
    directory = registry_dir / artifact_id
    write_metadata(meta, directory)
    with open(directory / meta.model_filename, "wb") as fh:
        pickle.dump(_StubModel(), fh)
    return meta


def test_check_version_drift_noop_when_lib_versions_absent(monkeypatch):
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.0.0", "xgboost": "1.0.0"},
    )
    meta = _meta_no_provenance_versions()
    check_version_drift(meta)  # must not raise: absence of info is not evidence of drift


def _meta_no_provenance_versions():
    from stoptimes.core.contracts.artifact import ModelFormat

    return ArtifactMetadata(
        artifact_id="legacy1",
        family="fake",
        format=ModelFormat.PICKLE,
        features=["x"],
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
        metrics={"test_mae_seconds": 1.0},
    )


def test_check_version_drift_raises_on_load_critical_mismatch(monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.9.0"},
    )
    meta = _meta_no_provenance_versions().model_copy(
        update={"provenance": Provenance(created_at=datetime.now(timezone.utc), lib_versions={"sklearn": "1.2.0"})}
    )
    with pytest.raises(ArtifactVersionDriftError, match="sklearn"):
        check_version_drift(meta)


def test_check_version_drift_ignores_non_critical_library(monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"numpy": "2.0.0"},
    )
    meta = _meta_no_provenance_versions().model_copy(
        update={"provenance": Provenance(created_at=datetime.now(timezone.utc), lib_versions={"numpy": "1.20.0"})}
    )
    check_version_drift(meta)  # numpy is not load-critical for PICKLE: warns, never refuses


def test_check_version_drift_no_mismatch_passes(monkeypatch):
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.2.5"},
    )
    meta = _meta_no_provenance_versions().model_copy(
        update={"provenance": Provenance(created_at=datetime.now(timezone.utc), lib_versions={"sklearn": "1.2.0"})}
    )
    check_version_drift(meta)  # same major.minor: no refusal


def test_check_version_drift_override_param_allows_load(monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.9.0"},
    )
    meta = _meta_no_provenance_versions().model_copy(
        update={"provenance": Provenance(created_at=datetime.now(timezone.utc), lib_versions={"sklearn": "1.2.0"})}
    )
    check_version_drift(meta, allow_override=True)  # explicit override: no raise


def test_check_version_drift_env_override_all(monkeypatch):
    monkeypatch.setenv(ALLOW_VERSION_DRIFT_ENV, "all")
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.9.0"},
    )
    meta = _meta_no_provenance_versions().model_copy(
        update={"provenance": Provenance(created_at=datetime.now(timezone.utc), lib_versions={"sklearn": "1.2.0"})}
    )
    check_version_drift(meta)


def test_check_version_drift_env_override_by_artifact_id(monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.9.0"},
    )
    meta = _meta_no_provenance_versions().model_copy(
        update={"provenance": Provenance(created_at=datetime.now(timezone.utc), lib_versions={"sklearn": "1.2.0"})}
    )
    monkeypatch.setenv(ALLOW_VERSION_DRIFT_ENV, "other-artifact, legacy1")
    check_version_drift(meta)  # legacy1 is named in the allow-list

    monkeypatch.setenv(ALLOW_VERSION_DRIFT_ENV, "other-artifact")
    with pytest.raises(ArtifactVersionDriftError):
        check_version_drift(meta)  # legacy1 is not named: still refused


def test_load_artifact_model_refuses_stale_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.9.0"},
    )
    meta = _write_artifact_with_lib_versions(tmp_path, "a1", {"sklearn": "1.2.0"})
    with pytest.raises(ArtifactVersionDriftError):
        load_artifact_model(meta, tmp_path)


def test_load_artifact_model_override_param_loads_stale_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "1.9.0"},
    )
    meta = _write_artifact_with_lib_versions(tmp_path, "a1", {"sklearn": "1.2.0"})
    model = load_artifact_model(meta, tmp_path, allow_version_drift=True)
    assert model.family == "fake"


def test_load_artifact_model_loads_when_no_lib_versions_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stoptimes.serve.model_loading._installed_lib_versions",
        lambda: {"sklearn": "9.9.9", "xgboost": "9.9.9"},
    )
    meta = _write_artifact_with_lib_versions(tmp_path, "a1", {})
    model = load_artifact_model(meta, tmp_path)
    assert model.family == "fake"


# ---------------------------------------------------------------------------
# Feature-producibility refusal (docs/PLAN.md §7.4): an artifact declaring a
# feature the default serving pipeline cannot compute must be refused before
# serving starts, not silently served on a missing (None) input.
# ---------------------------------------------------------------------------


def _write_artifact_with_features(registry_dir, artifact_id, features):
    from stoptimes.core.contracts.artifact import ModelFormat

    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="fake",
        format=ModelFormat.PICKLE,
        features=features,
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
        metrics={"test_mae_seconds": 1.0},
    )
    directory = registry_dir / artifact_id
    write_metadata(meta, directory)
    with open(directory / meta.model_filename, "wb") as fh:
        pickle.dump(_StubModel(), fh)
    return meta


def test_check_feature_producibility_passes_for_producible_registry_features():
    meta = ArtifactMetadata(
        artifact_id="a1",
        family="fake",
        features=["hour", "distance_to_stop", "trip_id"],
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
    )
    check_feature_producibility(meta)  # must not raise


def test_check_feature_producibility_ignores_names_unknown_to_the_registry():
    meta = ArtifactMetadata(
        artifact_id="a1",
        family="fake",
        features=["x"],  # not a core.features.registry entry -- no evidence either way
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
    )
    check_feature_producibility(meta)  # must not raise


def test_check_feature_producibility_refuses_a_train_only_registered_feature():
    meta = ArtifactMetadata(
        artifact_id="a1",
        family="fake",
        features=["shape_progress"],
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
    )
    with pytest.raises(UnproducibleFeaturesError, match="shape_progress"):
        check_feature_producibility(meta)


def test_check_feature_producibility_override_param_allows_load():
    meta = ArtifactMetadata(
        artifact_id="a1",
        family="fake",
        features=["shape_progress"],
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
    )
    check_feature_producibility(meta, allow_override=True)  # explicit override: no raise


def test_check_feature_producibility_env_override_by_artifact_id(monkeypatch):
    monkeypatch.delenv(ALLOW_UNPRODUCIBLE_FEATURES_ENV, raising=False)
    meta = ArtifactMetadata(
        artifact_id="a1",
        family="fake",
        features=["shape_progress"],
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
    )
    monkeypatch.setenv(ALLOW_UNPRODUCIBLE_FEATURES_ENV, "other-artifact, a1")
    check_feature_producibility(meta)  # a1 is named in the allow-list

    monkeypatch.setenv(ALLOW_UNPRODUCIBLE_FEATURES_ENV, "other-artifact")
    with pytest.raises(UnproducibleFeaturesError):
        check_feature_producibility(meta)  # a1 is not named: still refused


def test_load_artifact_model_refuses_artifact_requiring_unproducible_feature(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_UNPRODUCIBLE_FEATURES_ENV, raising=False)
    meta = _write_artifact_with_features(tmp_path, "a1", ["shape_progress"])
    with pytest.raises(UnproducibleFeaturesError, match="shape_progress"):
        load_artifact_model(meta, tmp_path)


def test_load_artifact_model_override_param_loads_despite_unproducible_feature(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_UNPRODUCIBLE_FEATURES_ENV, raising=False)
    meta = _write_artifact_with_features(tmp_path, "a1", ["shape_progress"])
    model = load_artifact_model(meta, tmp_path, allow_unproducible_features=True)
    assert model.family == "fake"


def test_load_artifact_model_loads_normally_for_producible_features(tmp_path):
    meta = _write_artifact_with_features(tmp_path, "a1", ["distance_to_stop", "hour"])
    model = load_artifact_model(meta, tmp_path)
    assert model.family == "fake"
