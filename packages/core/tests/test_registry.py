"""Contract tests for `stoptimes.core.registry`.

The `backend` fixture is parametrized so every backend (filesystem today, MLflow in M2) must pass
the same suite; backend-specific behavior lives in the classes at the bottom.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    Provenance,
    artifact_dir,
    write_metadata,
)
from stoptimes.core.registry import (
    REGISTRY_ENV,
    ArtifactExistsError,
    FilesystemRegistry,
    RegistryBackend,
    RegistryError,
    open_registry,
)


def _meta(artifact_id: str = "xgb-live", **overrides) -> ArtifactMetadata:
    base = dict(
        artifact_id=artifact_id,
        family="xgboost",
        features=["distance_to_stop", "hour"],
        provenance=Provenance(created_at=datetime(2026, 11, 6, 12, 0, 0), lib_versions={"xgboost": "3.1.2"}),
    )
    base.update(overrides)
    return ArtifactMetadata(**base)


def _publish(registry: RegistryBackend, meta: ArtifactMetadata, *, card: str | None = "# card", overwrite=False):
    with registry.staged(meta.artifact_id, overwrite=overwrite) as tmp:
        write_metadata(meta, tmp)
        (tmp / "model.pkl").write_bytes(b"model-bytes")
        if card is not None:
            (tmp / "card.md").write_text(card, encoding="utf-8")


def mlflow_registry(tmp_path: Path, monkeypatch, **kwargs):
    """An `MlflowRegistry` on a throwaway local sqlite store: no server, no network."""
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    from stoptimes.core.registry.mlflow_backend import MlflowRegistry

    return MlflowRegistry(
        f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "mlartifacts"),
        cache_dir=tmp_path / "cache",
        list_ttl_s=kwargs.pop("list_ttl_s", 0.0),
        **kwargs,
    )


@pytest.fixture(params=["filesystem", "mlflow"])
def backend(request, tmp_path: Path, monkeypatch) -> RegistryBackend:
    if request.param == "filesystem":
        return FilesystemRegistry(tmp_path / "registry")
    if request.param == "mlflow":
        return mlflow_registry(tmp_path, monkeypatch)
    raise AssertionError(request.param)


class TestContract:
    def test_satisfies_protocol(self, backend):
        assert isinstance(backend, RegistryBackend)

    def test_empty_registry_lists_nothing(self, backend):
        assert backend.list() == []
        assert backend.get("missing") is None
        assert backend.exists("missing") is False
        assert backend.card("missing") is None

    def test_published_artifact_round_trips(self, backend):
        meta = _meta()
        _publish(backend, meta, card="# xgb-live\n")
        assert [m.artifact_id for m in backend.list()] == ["xgb-live"]
        assert backend.get("xgb-live") == meta
        assert backend.exists("xgb-live")
        assert backend.card("xgb-live") == "# xgb-live\n"

    def test_list_is_sorted_by_artifact_id(self, backend):
        for aid in ["b", "c", "a"]:
            _publish(backend, _meta(aid))
        assert [m.artifact_id for m in backend.list()] == ["a", "b", "c"]

    def test_local_root_holds_the_artifact_files(self, backend):
        meta = _meta()
        _publish(backend, meta)
        directory = artifact_dir(backend.local_root(meta), meta.artifact_id)
        assert (directory / "model.pkl").read_bytes() == b"model-bytes"
        assert (directory / "artifact.json").is_file()

    def test_existing_artifact_is_not_overwritten_by_default(self, backend):
        _publish(backend, _meta(params={"v": 1}))
        with pytest.raises(ArtifactExistsError):
            _publish(backend, _meta(params={"v": 2}))
        assert backend.get("xgb-live").params == {"v": 1}

    def test_overwrite_replaces_the_artifact(self, backend):
        _publish(backend, _meta(params={"v": 1}))
        _publish(backend, _meta(params={"v": 2}), overwrite=True)
        assert backend.get("xgb-live").params == {"v": 2}

    def test_failed_write_publishes_nothing(self, backend):
        with pytest.raises(RuntimeError, match="boom"):
            with backend.staged("half-written") as tmp:
                write_metadata(_meta("half-written"), tmp)
                raise RuntimeError("boom")
        assert backend.exists("half-written") is False
        assert backend.list() == []

    def test_failed_overwrite_keeps_the_old_artifact(self, backend):
        _publish(backend, _meta(params={"v": 1}))
        with pytest.raises(RuntimeError):
            with backend.staged("xgb-live", overwrite=True) as tmp:
                write_metadata(_meta(params={"v": 2}), tmp)
                raise RuntimeError("boom")
        assert backend.get("xgb-live").params == {"v": 1}


class TestFilesystem:
    def test_malformed_entry_is_skipped_and_reported(self, tmp_path):
        reg = FilesystemRegistry(tmp_path)
        _publish(reg, _meta("good"))
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "artifact.json").write_text("{not json", encoding="utf-8")
        skipped: list[str] = []
        assert [m.artifact_id for m in reg.list(on_skip=lambda name, exc: skipped.append(name))] == ["good"]
        assert skipped == ["bad"]

    def test_malformed_entry_raises_when_strict(self, tmp_path):
        reg = FilesystemRegistry(tmp_path)
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "artifact.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(Exception):
            reg.list(strict=True)

    def test_directories_without_metadata_and_staging_dirs_are_ignored(self, tmp_path):
        reg = FilesystemRegistry(tmp_path)
        (tmp_path / "legacy-pickles").mkdir()
        (tmp_path / ".train-x-abc").mkdir()
        assert reg.list() == []

    def test_local_root_is_the_registry_root(self, tmp_path):
        reg = FilesystemRegistry(tmp_path)
        assert reg.local_root(_meta()) == tmp_path
        assert reg.uri == str(tmp_path)

    def test_no_staging_dir_left_behind(self, tmp_path):
        reg = FilesystemRegistry(tmp_path)
        _publish(reg, _meta())
        assert sorted(p.name for p in tmp_path.iterdir()) == ["xgb-live"]


class TestOpenRegistry:
    def test_backend_passes_through(self, tmp_path):
        reg = FilesystemRegistry(tmp_path)
        assert open_registry(reg) is reg

    def test_path_and_string_open_a_directory(self, tmp_path):
        assert open_registry(tmp_path).root == tmp_path
        assert open_registry(str(tmp_path)).root == tmp_path

    def test_none_uses_env_then_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "from-env"))
        assert open_registry().root == tmp_path / "from-env"
        monkeypatch.delenv(REGISTRY_ENV)
        monkeypatch.setenv("STOPTIMES_MODEL_REGISTRY_DIR", str(tmp_path / "from-config"))
        from stoptimes.core.config import reset_config

        reset_config()  # get_config() is a process-wide singleton
        try:
            assert open_registry().root == (tmp_path / "from-config").resolve()
        finally:
            monkeypatch.delenv("STOPTIMES_MODEL_REGISTRY_DIR")
            reset_config()

    def test_mlflow_uri_opens_one_shared_mlflow_backend(self):
        pytest.importorskip("mlflow")
        from stoptimes.core.registry.mlflow_backend import MlflowRegistry

        reg = open_registry("mlflow+https://ml.example.org")
        assert isinstance(reg, MlflowRegistry)
        assert reg.tracking_uri == "https://ml.example.org"
        assert reg.uri == "mlflow+https://ml.example.org"
        assert open_registry("mlflow+https://ml.example.org") is reg  # cache and downloads shared

    def test_mlflow_uri_needs_a_tracking_uri(self):
        with pytest.raises(RegistryError, match="missing tracking URI"):
            open_registry("mlflow+")


class TestMlflow:
    def test_versions_live_in_one_registered_model_per_agency(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        _publish(reg, _meta("a", scope={"source_id": "mbta"}))
        _publish(reg, _meta("b"))  # no source -> the global model
        assert [(v.name, v.version, v.artifact_id) for v in reg.versions()] == [
            ("stoptimes-global", 1, "b"),
            ("stoptimes-mbta", 1, "a"),
        ]

    def test_version_tags_carry_selection_fields(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        meta = _meta(scope={"source_id": "mbta"}, metrics={"test_mae_seconds": 95.6})
        _publish(reg, meta)
        mv = reg._client().get_model_version("stoptimes-mbta", "1")
        assert mv.tags["stoptimes.artifact_id"] == "xgb-live"
        assert mv.tags["stoptimes.source_id"] == "mbta"
        assert mv.tags["stoptimes.test_mae_s"] == "95.600"
        assert mv.run_id  # every publish records a run

    def test_the_publish_run_carries_params_metrics_and_card(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        _publish(reg, _meta(scope={"source_id": "mbta"}, params={"max_depth": 8}, metrics={"test_mae_seconds": 95.6}))
        client = reg._client()
        run = client.get_run(reg.resolve("stoptimes-mbta/1").run_id)
        assert run.data.params["model.max_depth"] == "8"
        assert run.data.metrics["test_mae_seconds"] == 95.6
        assert run.data.tags["stoptimes.kind"] == "train"
        assert "card.md" in {a.path for a in client.list_artifacts(run.info.run_id)}

    def test_overwrite_adds_a_version_and_latest_wins(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        _publish(reg, _meta(params={"v": 1}))
        _publish(reg, _meta(params={"v": 2}), overwrite=True)
        assert [v.version for v in reg.versions()] == [1, 2]
        assert reg.get("xgb-live").params == {"v": 2}
        assert reg.get_version("stoptimes-global", 1).meta.params == {"v": 1}

    def test_old_version_downloads_its_own_files(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        _publish(reg, _meta(params={"v": 1}))
        _publish(reg, _meta(params={"v": 2}), overwrite=True)
        old = reg.get_version("stoptimes-global", 1).meta
        root = reg.local_root(old)
        from stoptimes.core.contracts.artifact import read_metadata

        assert read_metadata(artifact_dir(root, "xgb-live")).params == {"v": 1}

    def test_aliases_resolve_and_move(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        _publish(reg, _meta(params={"v": 1}))
        _publish(reg, _meta(params={"v": 2}), overwrite=True)
        reg.set_alias("stoptimes-global", "production", 1)
        assert reg.resolve("stoptimes-global@production").meta.params == {"v": 1}
        assert reg.resolve("stoptimes-global/2").meta.params == {"v": 2}
        assert reg.resolve("xgb-live").version == 2
        assert {v.version: v.aliases for v in reg.versions()} == {1: ("production",), 2: ()}
        reg.set_alias("stoptimes-global", "production", 2)  # promote; moving it back is the rollback
        assert reg.resolve("stoptimes-global@production").version == 2
        reg.delete_alias("stoptimes-global", "production")
        with pytest.raises(RegistryError):
            reg.resolve("stoptimes-global@production")

    def test_download_is_cached(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        meta = _meta()
        _publish(reg, meta)
        first = reg.local_root(meta)
        import mlflow

        def boom(*a, **k):
            raise AssertionError("downloaded twice")

        monkeypatch.setattr(mlflow.artifacts, "download_artifacts", boom)
        assert reg.local_root(meta) == first

    def test_metadata_too_big_for_a_tag_is_fetched_from_the_files(self, tmp_path, monkeypatch):
        import stoptimes.core.registry.mlflow_backend as backend_mod

        monkeypatch.setattr(backend_mod, "MAX_TAG_VALUE", 10)
        reg = mlflow_registry(tmp_path, monkeypatch)
        meta = _meta()
        _publish(reg, meta)
        mv = reg._client().get_model_version("stoptimes-global", "1")
        assert "stoptimes.metadata" not in mv.tags
        assert reg.get("xgb-live") == meta

    def test_foreign_model_versions_are_ignored(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch)
        _publish(reg, _meta())
        client = reg._client()
        client.create_registered_model("stoptimes-other")  # matches the prefix, not a stoptimes artifact
        assert [m.artifact_id for m in reg.list()] == ["xgb-live"]

    def test_listing_is_cached_for_the_ttl(self, tmp_path, monkeypatch):
        reg = mlflow_registry(tmp_path, monkeypatch, list_ttl_s=3600)
        assert reg.list() == []
        calls = []
        real = reg._search
        monkeypatch.setattr(reg, "_search", lambda **kw: calls.append(kw) or real(**kw))
        assert reg.list() == []
        assert calls == []  # served from the cache
        _publish(reg, _meta())  # a publish invalidates it
        assert [m.artifact_id for m in reg.list()] == ["xgb-live"]

    def test_publish_restores_mlflow_global_uris(self, tmp_path, monkeypatch):
        import mlflow

        reg = mlflow_registry(tmp_path, monkeypatch)
        before = (mlflow.get_tracking_uri(), mlflow.get_registry_uri())
        _publish(reg, _meta())
        assert (mlflow.get_tracking_uri(), mlflow.get_registry_uri()) == before

    def test_pyfunc_predicts_on_a_feature_frame(self, tmp_path, monkeypatch):
        import pickle

        import mlflow
        import pandas as pd

        reg = mlflow_registry(tmp_path, monkeypatch)
        meta = _meta(features=["distance_to_stop"])
        with reg.staged(meta.artifact_id) as tmp:
            write_metadata(meta, tmp)
            (tmp / "model.pkl").write_bytes(pickle.dumps(_Doubler()))
        root = reg.local_root(meta)  # the version's files, as MLflow stored them
        model = mlflow.pyfunc.load_model(str(root.parent))
        frame = pd.DataFrame({"distance_to_stop": [1.0, 2.5], "ignored": [0, 0]})
        assert list(model.predict(frame)) == [2.0, 5.0]


class _Doubler:
    def predict(self, frame):
        return frame["distance_to_stop"].to_numpy() * 2


class TestEnsureRegisteredModel:
    """Basic-auth servers answer 403 (not 404) for a model a non-admin can't see, including a new one."""

    @staticmethod
    def _client(get_code: str | None, create_code: str | None = None):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import ErrorCode

        created: list[str] = []

        class _Client:
            def get_registered_model(self, name):
                if get_code:
                    raise MlflowException("nope", error_code=ErrorCode.Value(get_code))

            def create_registered_model(self, name):
                if create_code:
                    raise MlflowException("nope", error_code=ErrorCode.Value(create_code))
                created.append(name)

        return _Client(), created

    def _ensure(self, client):
        pytest.importorskip("mlflow")
        from stoptimes.core.registry.mlflow_backend import MlflowRegistry

        MlflowRegistry._ensure_registered_model(client, "stoptimes-mbta")

    def test_existing_model_is_left_alone(self):
        client, created = self._client(None)
        self._ensure(client)
        assert created == []

    @pytest.mark.parametrize("code", ["RESOURCE_DOES_NOT_EXIST", "PERMISSION_DENIED"])
    def test_missing_or_unreadable_model_is_created(self, code):
        client, created = self._client(code)
        self._ensure(client)
        assert created == ["stoptimes-mbta"]

    def test_someone_elses_model_gives_a_clear_error(self):
        from stoptimes.core.registry import RegistryError

        client, _ = self._client("PERMISSION_DENIED", create_code="RESOURCE_ALREADY_EXISTS")
        with pytest.raises(RegistryError, match="belongs to another user"):
            self._ensure(client)

    def test_other_errors_propagate(self):
        from mlflow.exceptions import MlflowException

        client, created = self._client("INTERNAL_ERROR")
        with pytest.raises(MlflowException):
            self._ensure(client)
        assert created == []
