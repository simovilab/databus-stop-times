"""Tests for stoptimes.core.api's model-registry read verbs (list/card/promote)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from stoptimes.core.api import list_models, model_card, promote_model
from stoptimes.core.contracts.artifact import ArtifactMetadata, Provenance, write_metadata


def _write_artifact(registry: Path, artifact_id: str, *, mae: float = 42.0) -> Path:
    directory = registry / artifact_id
    directory.mkdir(parents=True, exist_ok=True)
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family="historical_mean",
        provenance=Provenance(created_at=datetime.now(timezone.utc)),
        metrics={"test_mae_seconds": mae},
    )
    write_metadata(meta, directory)
    (directory / "card.md").write_text(f"# {artifact_id}\n\nmae={mae}\n", encoding="utf-8")
    return directory


def test_list_models_empty_registry(tmp_path):
    assert list_models(registry=tmp_path / "nope") == []


def test_list_models_reads_artifact_json(tmp_path):
    registry = tmp_path / "registry"
    _write_artifact(registry, "a1", mae=10.0)
    _write_artifact(registry, "a2", mae=5.0)

    rows = list_models(registry=registry)
    assert {r["artifact_id"] for r in rows} == {"a1", "a2"}
    assert all(r["family"] == "historical_mean" for r in rows)

    filtered = list_models(registry=registry, family="nonexistent")
    assert filtered == []


def test_model_card_reads_and_raises(tmp_path):
    registry = tmp_path / "registry"
    _write_artifact(registry, "a1")

    text = model_card("a1", registry=registry)
    assert "a1" in text

    with pytest.raises(FileNotFoundError):
        model_card("missing", registry=registry)


def test_promote_model_writes_pin(tmp_path):
    registry = tmp_path / "registry"
    _write_artifact(registry, "a1")

    sources_dir = tmp_path / "agencies"
    sources_dir.mkdir()
    (sources_dir / "demo.yaml").write_text("name: demo\n", encoding="utf-8")

    profile_path = promote_model(
        "a1", as_pin="global", source="demo", registry=registry, sources_dir=sources_dir
    )
    assert profile_path == sources_dir / "demo.yaml"

    import yaml

    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert data["serving"]["selection"]["pins"]["global"] == "a1"


def test_promote_model_raises_for_missing_artifact(tmp_path):
    sources_dir = tmp_path / "agencies"
    sources_dir.mkdir()
    (sources_dir / "demo.yaml").write_text("name: demo\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        promote_model("nope", as_pin="global", source="demo", registry=tmp_path / "registry", sources_dir=sources_dir)
