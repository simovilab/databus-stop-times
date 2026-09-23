"""Tests for `stoptimes train run` and `stoptimes models {list,card,promote}`.

New file (T3 owns it) rather than editing the shared `test_cli.py`, whose
`test_train_run_is_planned` / `test_models_card_is_planned` cover the
no-argument case only (still true: `train run` / `models card` with no
argument stay "planned"). These tests cover the real behavior once an
experiment path / artifact id is given.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

_MODELS_FIXTURES = Path(__file__).resolve().parents[2] / "models" / "tests" / "fixtures"


def _write_experiment(tmp_path: Path, **overrides) -> Path:
    experiment = {
        "name": "cli-smoke",
        "data": {"datasets": ["sample"]},
        "target": "time_to_arrival_seconds",
        "model": {"family": "historical_mean", "params": {}},
        "features": ["route_id", "stop_sequence", "hour"],
        "split": {"kind": "time", "train_fraction": 0.7, "test_fraction": 0.3},
        "seed": 42,
        "outputs": {"registry_dir": str(tmp_path / "registry")},
        "baselines": ["schedule", "constant_speed"],
    }
    experiment.update(overrides)
    path = tmp_path / "experiment.yaml"
    import yaml

    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return path


# ── train run ────────────────────────────────────────────────────────────────


def test_train_run_dry_run_prints_plan_without_training(tmp_path):
    experiment_path = _write_experiment(tmp_path)
    result = runner.invoke(app, ["train", "run", str(experiment_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["name"] == "cli-smoke"
    assert plan["model"]["family"] == "historical_mean"
    assert not (tmp_path / "registry").exists()


def test_train_run_trains_and_writes_artifact(tmp_path):
    experiment_path = _write_experiment(tmp_path)
    result = runner.invoke(
        app,
        ["train", "run", str(experiment_path), "--dataset-dir", str(_MODELS_FIXTURES)],
    )
    assert result.exit_code == 0, result.output
    assert "cli-smoke" in result.output

    registry_dir = tmp_path / "registry"
    assert (registry_dir / "cli-smoke" / "artifact.json").exists()
    assert (registry_dir / "cli-smoke" / "model.pkl").exists()
    assert (registry_dir / "cli-smoke" / "card.md").exists()


def test_train_run_refuses_to_overwrite_without_flag(tmp_path):
    experiment_path = _write_experiment(tmp_path)
    first = runner.invoke(app, ["train", "run", str(experiment_path), "--dataset-dir", str(_MODELS_FIXTURES)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["train", "run", str(experiment_path), "--dataset-dir", str(_MODELS_FIXTURES)])
    assert second.exit_code == 1
    assert "already exists" in second.output
    # a CLI remedy only -- `overwrite=True` is the library spelling (docs/PLAN.md §7.5)
    assert "--overwrite" in second.output
    assert "overwrite=True" not in second.output

    third = runner.invoke(
        app,
        ["train", "run", str(experiment_path), "--dataset-dir", str(_MODELS_FIXTURES), "--overwrite"],
    )
    assert third.exit_code == 0, third.output


def test_train_run_with_explicit_registry_option(tmp_path):
    experiment_path = _write_experiment(tmp_path)
    other_registry = tmp_path / "elsewhere"
    result = runner.invoke(
        app,
        [
            "train", "run", str(experiment_path),
            "--dataset-dir", str(_MODELS_FIXTURES),
            "--registry", str(other_registry),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (other_registry / "cli-smoke" / "artifact.json").exists()
    assert not (tmp_path / "registry" / "cli-smoke").exists()


# ── models list / card / promote ────────────────────────────────────────────


def _train_one(tmp_path: Path, name: str = "cli-smoke") -> Path:
    registry_dir = tmp_path / "registry"
    experiment_path = _write_experiment(tmp_path, name=name)
    result = runner.invoke(app, ["train", "run", str(experiment_path), "--dataset-dir", str(_MODELS_FIXTURES)])
    assert result.exit_code == 0, result.output
    return registry_dir


def test_models_list_json_shows_trained_artifact(tmp_path):
    registry_dir = _train_one(tmp_path)
    result = runner.invoke(app, ["models", "list", "--registry", str(registry_dir), "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 1
    assert rows[0]["artifact_id"] == "cli-smoke"
    assert rows[0]["family"] == "historical_mean"


def test_models_list_filters_by_family(tmp_path):
    registry_dir = _train_one(tmp_path)
    result = runner.invoke(
        app, ["models", "list", "--registry", str(registry_dir), "--family", "xgb", "--json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_models_list_empty_registry_is_not_an_error(tmp_path):
    result = runner.invoke(app, ["models", "list", "--registry", str(tmp_path / "nope")])
    assert result.exit_code == 0
    assert "no artifacts found" in result.output


def test_models_card_with_id_renders_card(tmp_path):
    registry_dir = _train_one(tmp_path)
    result = runner.invoke(app, ["models", "card", "cli-smoke", "--registry", str(registry_dir)])
    assert result.exit_code == 0, result.output
    assert "historical_mean" in result.output
    assert "Scores by horizon" in result.output



def test_models_card_unknown_id_errors(tmp_path):
    result = runner.invoke(app, ["models", "card", "nope", "--registry", str(tmp_path)])
    assert result.exit_code == 1
    assert "no card.md" in result.output


def test_models_promote_refuses_missing_artifact(tmp_path):
    result = runner.invoke(
        app,
        [
            "models", "promote", "nope",
            "--as", "global",
            "--source", "bucr",
            "--registry", str(tmp_path / "registry"),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_models_promote_writes_pin_into_agency_profile(tmp_path):
    registry_dir = _train_one(tmp_path)
    sources_dir = tmp_path / "agencies"
    sources_dir.mkdir()
    (sources_dir / "bucr.yaml").write_text(
        "name: bucr\ngtfs_static_url: http://example/feed.zip\n", encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "models", "promote", "cli-smoke",
            "--as", "global",
            "--source", "bucr",
            "--registry", str(registry_dir),
            "--sources-dir", str(sources_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    import yaml

    data = yaml.safe_load((sources_dir / "bucr.yaml").read_text())
    assert data["serving"]["selection"]["pins"]["global"] == "cli-smoke"
    assert data["gtfs_static_url"] == "http://example/feed.zip"
