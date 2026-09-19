"""A model refused for library drift must read as an operational message.

U7 made a load-critical version drift a refusal rather than a warning, which
is right -- but `serve` only caught `NotImplementedError`, so the refusal
reached the operator as a raw traceback. Found by U5 while writing the
registry-lifecycle guide. The whole point of the refusal is to tell someone
what to do at 2am, so how it *surfaces* is part of the feature.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

pytest.importorskip("pandas")

_MODELS_FIXTURES = Path(__file__).resolve().parents[2] / "models" / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("STOPTIMES_ALLOW_VERSION_DRIFT", raising=False)


def _sources_dir(tmp_path: Path) -> Path:
    sources = tmp_path / "agencies"
    sources.mkdir(exist_ok=True)
    (sources / "demo.yaml").write_text(
        "name: demo\n"
        "gtfs_static_url: unused.zip\n"
        "vehicle_positions_url: http://example.invalid/vp.json\n"
        "trip_updates_url: http://example.invalid/tu.json\n"
        "poll_interval_s: 15\n"
        "timezone: UTC\n",
        encoding="utf-8",
    )
    return sources


def _trained_registry(tmp_path: Path) -> tuple[Path, str]:
    """Train one real artifact, then forge a load-critical version drift in it."""
    experiment = {
        "name": "drift-demo",
        "data": {"datasets": ["sample"]},
        "target": "time_to_arrival_seconds",
        "model": {"family": "historical_mean", "params": {}},
        "features": ["route_id", "stop_sequence", "hour"],
        "split": {"kind": "time", "train_fraction": 0.7, "test_fraction": 0.3},
        "seed": 42,
        "outputs": {"registry_dir": str(tmp_path / "registry")},
        "baselines": [],
    }
    path = tmp_path / "drift.experiment.yaml"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    result = runner.invoke(app, ["train", "run", str(path), "--dataset-dir", str(_MODELS_FIXTURES)])
    assert result.exit_code == 0, result.output

    registry = tmp_path / "registry"
    artifact_dir = next(d for d in registry.iterdir() if (d / "artifact.json").is_file())
    meta_path = artifact_dir / "artifact.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # a major.minor away from anything that could plausibly be installed
    meta["provenance"]["lib_versions"]["sklearn"] = "0.1.0"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return registry, artifact_dir.name


def _serve(registry: Path, sources: Path, artifact_id: str):
    return runner.invoke(
        app,
        ["serve", "--source", "demo", "--sources-dir", str(sources),
         "--registry", str(registry), "--model", artifact_id, "--once"],
    )


def test_a_drifted_model_is_refused_with_an_actionable_message(tmp_path):
    registry, artifact_id = _trained_registry(tmp_path)
    result = _serve(registry, _sources_dir(tmp_path), artifact_id)

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"a refusal must not reach the operator as a traceback: {result.exception!r}"
    )
    output = result.output
    assert "sklearn" in output, output                       # what drifted
    assert "0.1.0" in output, output                         # the version it was built with
    assert "re-export" in output.lower(), output             # what to do about it
    assert "STOPTIMES_ALLOW_VERSION_DRIFT" in output, output  # how to override


def test_the_override_lets_a_known_good_model_run(tmp_path, monkeypatch):
    registry, artifact_id = _trained_registry(tmp_path)
    monkeypatch.setenv("STOPTIMES_ALLOW_VERSION_DRIFT", "all")

    result = _serve(registry, _sources_dir(tmp_path), artifact_id)

    # It gets past loading; whether the feed has entities depends on the
    # (unreachable) example feed, which is not what this test is about.
    assert "STOPTIMES_ALLOW_VERSION_DRIFT" not in result.output, result.output


def _artifact_with_unproducible_feature(tmp_path: Path) -> tuple[Path, str]:
    """Train one real artifact, then declare a feature only a training-side
    dataset builder can materialize."""
    registry, artifact_id = _trained_registry(tmp_path)
    meta_path = registry / artifact_id / "artifact.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # undo the drift `_trained_registry` forged: this test is about features
    meta["provenance"]["lib_versions"].pop("sklearn", None)
    meta["features"] = [*meta.get("features", []), "shape_progress"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return registry, artifact_id


def test_a_feature_serving_cannot_compute_is_refused_not_imputed(tmp_path):
    """The §7.4 defect: a missing feature used to arrive as None and be
    imputed into confident nonsense. It must be refused at load, and -- like
    every other refusal -- reach the operator as a message."""
    registry, artifact_id = _artifact_with_unproducible_feature(tmp_path)
    result = _serve(registry, _sources_dir(tmp_path), artifact_id)

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"a refusal must not reach the operator as a traceback: {result.exception!r}"
    )
    assert "shape_progress" in result.output, result.output
    assert "STOPTIMES_ALLOW_UNPRODUCIBLE_FEATURES" in result.output, result.output


def test_the_unproducible_override_lets_a_custom_pipeline_through(tmp_path, monkeypatch):
    registry, artifact_id = _artifact_with_unproducible_feature(tmp_path)
    monkeypatch.setenv("STOPTIMES_ALLOW_UNPRODUCIBLE_FEATURES", "all")

    result = _serve(registry, _sources_dir(tmp_path), artifact_id)

    assert "STOPTIMES_ALLOW_UNPRODUCIBLE_FEATURES" not in result.output, result.output
