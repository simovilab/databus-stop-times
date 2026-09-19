"""Tests for `stoptimes models promote` / `rollback` / `pins` (U7).

`stoptimes models list`/`card`/`promote` basics are covered by
`test_cli_train.py` (a different track's file). This file is U7's: it covers
what `promote` does beyond "write a pin" -- retaining the previous value so
`rollback` can undo it -- and the `pins` command that shows what is live.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

_MODELS_FIXTURES = Path(__file__).resolve().parents[2] / "models" / "tests" / "fixtures"


def _write_experiment(tmp_path: Path, name: str) -> Path:
    experiment = {
        "name": name,
        "data": {"datasets": ["sample"]},
        "target": "time_to_arrival_seconds",
        "model": {"family": "historical_mean", "params": {}},
        "features": ["route_id", "stop_sequence", "hour"],
        "split": {"kind": "time", "train_fraction": 0.7, "test_fraction": 0.3},
        "seed": 42,
        "outputs": {"registry_dir": str(tmp_path / "registry")},
        "baselines": ["schedule", "constant_speed"],
    }
    path = tmp_path / f"{name}.experiment.yaml"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return path


def _train(tmp_path: Path, name: str) -> Path:
    registry_dir = tmp_path / "registry"
    experiment_path = _write_experiment(tmp_path, name)
    result = runner.invoke(app, ["train", "run", str(experiment_path), "--dataset-dir", str(_MODELS_FIXTURES)])
    assert result.exit_code == 0, result.output
    return registry_dir


def _write_profile(tmp_path: Path, source: str = "bucr") -> Path:
    sources_dir = tmp_path / "agencies"
    sources_dir.mkdir(exist_ok=True)
    profile_path = sources_dir / f"{source}.yaml"
    profile_path.write_text("name: bucr\ngtfs_static_url: http://example/feed.zip\n", encoding="utf-8")
    return sources_dir


def _promote(registry_dir, sources_dir, artifact_id, *, as_pin="global", source="bucr", json_output=True):
    args = [
        "models", "promote", artifact_id,
        "--as", as_pin,
        "--source", source,
        "--registry", str(registry_dir),
        "--sources-dir", str(sources_dir),
    ]
    if json_output:
        args.append("--json")
    return runner.invoke(app, args)


def test_promote_records_history_sidecar(tmp_path):
    registry_dir = _train(tmp_path, "m1")
    sources_dir = _write_profile(tmp_path)

    result = _promote(registry_dir, sources_dir, "m1")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["promotions_for_pin"] == 1

    history_path = Path(payload["history_path"])
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert [e["artifact_id"] for e in history["pins"]["global"]] == ["m1"]


def test_promote_twice_appends_to_history(tmp_path):
    registry_dir = _train(tmp_path, "m1")
    _train(tmp_path, "m2")
    sources_dir = _write_profile(tmp_path)

    _promote(registry_dir, sources_dir, "m1")
    result = _promote(registry_dir, sources_dir, "m2")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["promotions_for_pin"] == 2

    data = yaml.safe_load((sources_dir / "bucr.yaml").read_text())
    assert data["serving"]["selection"]["pins"]["global"] == "m2"


def test_rollback_restores_previous_artifact(tmp_path):
    registry_dir = _train(tmp_path, "m1")
    _train(tmp_path, "m2")
    sources_dir = _write_profile(tmp_path)

    _promote(registry_dir, sources_dir, "m1")
    _promote(registry_dir, sources_dir, "m2")

    result = runner.invoke(
        app,
        [
            "models", "rollback",
            "--as", "global",
            "--source", "bucr",
            "--registry", str(registry_dir),
            "--sources-dir", str(sources_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["restored_artifact_id"] == "m1"

    data = yaml.safe_load((sources_dir / "bucr.yaml").read_text())
    assert data["serving"]["selection"]["pins"]["global"] == "m1"


def test_rollback_with_no_prior_promotion_refuses(tmp_path):
    registry_dir = _train(tmp_path, "m1")
    sources_dir = _write_profile(tmp_path)
    _promote(registry_dir, sources_dir, "m1")

    result = runner.invoke(
        app,
        [
            "models", "rollback",
            "--as", "global",
            "--source", "bucr",
            "--registry", str(registry_dir),
            "--sources-dir", str(sources_dir),
        ],
    )
    assert result.exit_code == 1
    assert "nothing to roll back to" in result.output


def test_rollback_missing_profile_errors(tmp_path):
    result = runner.invoke(
        app,
        [
            "models", "rollback",
            "--as", "global",
            "--source", "nope",
            "--sources-dir", str(tmp_path / "agencies"),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_rollback_warns_if_restored_artifact_since_deleted(tmp_path):
    registry_dir = _train(tmp_path, "m1")
    _train(tmp_path, "m2")
    sources_dir = _write_profile(tmp_path)

    _promote(registry_dir, sources_dir, "m1")
    _promote(registry_dir, sources_dir, "m2")

    import shutil

    shutil.rmtree(registry_dir / "m1")

    result = runner.invoke(
        app,
        [
            "models", "rollback",
            "--as", "global",
            "--source", "bucr",
            "--registry", str(registry_dir),
            "--sources-dir", str(sources_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no longer exists" in result.output


def test_pins_shows_live_artifact_and_promotion_count(tmp_path):
    registry_dir = _train(tmp_path, "m1")
    _train(tmp_path, "m2")
    sources_dir = _write_profile(tmp_path)

    _promote(registry_dir, sources_dir, "m1")
    _promote(registry_dir, sources_dir, "m2")

    result = runner.invoke(
        app,
        ["models", "pins", "--source", "bucr", "--sources-dir", str(sources_dir), "--json"],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert rows == [{"pin": "global", "artifact_id": "m2", "promotions": 2}]


def test_pins_empty_profile_reports_no_pins(tmp_path):
    sources_dir = _write_profile(tmp_path)
    result = runner.invoke(app, ["models", "pins", "--source", "bucr", "--sources-dir", str(sources_dir)])
    assert result.exit_code == 0
    assert "no pins set" in result.output
