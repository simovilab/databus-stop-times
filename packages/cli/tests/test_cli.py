import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from stoptimes.cli.main import app

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _uv_run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["uv", "run", "--no-sync", "stoptimes", *args],
        cwd=str(cwd or _REPO_ROOT),
        capture_output=True,
        text=True,
        env=full_env,
    )


# ── top-level ─────────────────────────────────────────────────────────────────


def test_help_lists_every_area():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("data", "record", "train", "models", "serve", "diagnose", "val"):
        assert name in result.output


def test_planned_command_exits_2_and_names_phase():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_val_is_mounted():
    result = runner.invoke(app, ["val", "--help"])
    assert result.exit_code == 0


# ── data ──────────────────────────────────────────────────────────────────────


def test_data_help():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for name in ("add", "inspect", "arrivals"):
        assert name in result.output


def test_data_add_is_planned():
    result = runner.invoke(app, ["data", "add"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_data_inspect_is_planned():
    result = runner.invoke(app, ["data", "inspect"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_data_arrivals_is_planned():
    result = runner.invoke(app, ["data", "arrivals"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


# ── record ────────────────────────────────────────────────────────────────────


def test_record_help():
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    for name in ("start", "compact"):
        assert name in result.output


def test_record_start_is_planned():
    result = runner.invoke(app, ["record", "start"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_record_compact_passthrough_help_reaches_argparse():
    """`--help` should reach the real argparse parser in stoptimes.record.compaction.cli."""
    result = _uv_run("record", "compact", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLI entrypoint for the compaction package" in result.stdout


# ── train ─────────────────────────────────────────────────────────────────────


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    for name in ("run", "ui", "legacy-all", "seed-baseline"):
        assert name in result.output


def test_train_run_is_planned():
    result = runner.invoke(app, ["train", "run"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_train_ui_is_planned():
    result = runner.invoke(app, ["train", "ui"])
    assert result.exit_code == 2
    assert "Phase 3" in result.output


def test_train_legacy_all_passthrough_help_reaches_argparse():
    result = _uv_run("train", "legacy-all", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Train ETA prediction models" in result.stdout


def test_train_seed_baseline_passthrough_runs_underlying_module(tmp_path):
    """seed-baseline has no argparse of its own; verify the passthrough actually
    reaches stoptimes.train.seed_baseline_model.main() end to end."""
    result = _uv_run(
        "train", "seed-baseline",
        env={"MODEL_REGISTRY_DIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Seeded model" in result.stdout


# ── models ────────────────────────────────────────────────────────────────────


def test_models_help():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    for name in ("check", "card"):
        assert name in result.output


def test_models_card_is_planned():
    result = runner.invoke(app, ["models", "card"])
    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_models_check_passthrough_help_reaches_argparse():
    result = _uv_run("models", "check", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Check model registry status" in result.stdout


# ── serve / diagnose ──────────────────────────────────────────────────────────


def test_serve_is_planned_phase2_track_t4():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 2
    assert "track T4" in result.output


def test_diagnose_is_planned_phase3_track_u3():
    result = runner.invoke(app, ["diagnose"])
    assert result.exit_code == 2
    assert "Phase 3" in result.output
    assert "track U3" in result.output


# ── val (mounted etaval CLI + api/tui launchers) ─────────────────────────────


def test_val_help_lists_existing_and_new_commands():
    result = runner.invoke(app, ["val", "--help"])
    assert result.exit_code == 0
    for name in ("version", "sources", "run", "report", "api", "tui"):
        assert name in result.output


def test_val_api_help():
    result = runner.invoke(app, ["val", "api", "--help"])
    assert result.exit_code == 0


def test_val_tui_help_reaches_argparse():
    """stoptimes.val.tui.app.main() parses sys.argv with argparse; --help should
    reach it through the passthrough context settings."""
    result = _uv_run("val", "tui", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--api-url" in result.stdout
