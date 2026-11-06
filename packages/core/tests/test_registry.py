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


@pytest.fixture(params=["filesystem"])
def backend(request, tmp_path: Path) -> RegistryBackend:
    if request.param == "filesystem":
        return FilesystemRegistry(tmp_path / "registry")
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

    def test_mlflow_uri_is_refused_until_the_backend_exists(self):
        with pytest.raises(RegistryError, match="MLflow"):
            open_registry("mlflow+https://ml.example.org")
