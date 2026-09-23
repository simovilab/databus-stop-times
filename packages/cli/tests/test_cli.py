import json
import os
import subprocess
import sys
from pathlib import Path

from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parents[3]


_WORKSPACE_SRC_DIRS = [str(p) for p in sorted(_REPO_ROOT.glob("packages/*/src"))]


def _uv_run(
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Spawn a fresh interpreter running the real CLI assembler, the same code
    path the installed `stoptimes` console script uses.

    Sets `PYTHONPATH` explicitly to every workspace member's `src/` directory
    instead of relying solely on the venv's editable-install `.pth` files:
    on at least one macOS dev machine, those `.pth` files intermittently carry
    the Finder "hidden" (UF_HIDDEN) flag, which cpython's `site.addpackage`
    silently skips, making a freshly spawned interpreter fail to import
    `stoptimes` even though the current test process (already past site
    initialization) sees it fine. Setting `PYTHONPATH` sidesteps `site.py`'s
    `.pth` handling entirely and is harmless where the venv is normal (CI).
    """
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    existing = full_env.get("PYTHONPATH")
    src_dirs = [*_WORKSPACE_SRC_DIRS, existing] if existing else _WORKSPACE_SRC_DIRS
    full_env["PYTHONPATH"] = os.pathsep.join(src_dirs)
    return subprocess.run(
        [sys.executable, "-m", "stoptimes.cli.main", *args],
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


def test_planned_commands_are_all_gone():
    """Phase 3 built the last two placeholders (`diagnose`, `train ui`). Nothing
    in the command tree may still answer with the `planned()` stub."""
    for argv in (["diagnose", "--help"], ["train", "ui", "--help"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, argv
        assert "planned" not in result.output.lower(), argv


def test_val_is_mounted():
    result = runner.invoke(app, ["val", "--help"])
    assert result.exit_code == 0


# ── data ──────────────────────────────────────────────────────────────────────


def test_data_help():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for name in ("add", "inspect", "arrivals"):
        assert name in result.output


def test_data_add_requires_arguments():
    result = runner.invoke(app, ["data", "add"])
    assert result.exit_code != 0
    assert "Phase 2" not in result.output


def test_data_inspect_requires_a_name():
    result = runner.invoke(app, ["data", "inspect"])
    assert result.exit_code != 0
    assert "Phase 2" not in result.output


def test_data_arrivals_requires_arguments():
    """Implemented by T2 + the orchestrator's wiring; missing options must be
    a usage error, not a "planned" placeholder."""
    result = runner.invoke(app, ["data", "arrivals"])
    assert result.exit_code != 0
    assert "planned" not in result.output

def test_record_help():
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    for name in ("start", "compact"):
        assert name in result.output


def test_record_start_requires_a_source():
    """`record start` is implemented; without --source it must say so, not
    claim the command is unbuilt."""
    result = runner.invoke(app, ["record", "start"])
    assert result.exit_code != 0
    assert "--source" in result.output
    assert "planned" not in result.output


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



def test_train_ui_is_wired_to_the_builder():
    """`train ui --help` is real help now, and names the file it writes."""
    result = runner.invoke(app, ["train", "ui", "--help"])
    assert result.exit_code == 0
    assert "planned" not in result.output.lower()


def test_train_legacy_all_passthrough_reaches_train_all_models():
    """`train legacy-all --help` reaches stoptimes.train.train_all_models' argparse."""
    result = _uv_run("train", "legacy-all", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Train ETA prediction models" in result.stdout


def test_train_seed_baseline_help_does_not_run_the_seed(tmp_path):
    result = _uv_run("train", "seed-baseline", "--help", env={"MODEL_REGISTRY_DIR": str(tmp_path)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Seeded model" not in result.stdout
    assert not any(tmp_path.iterdir())


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
    for name in ("check", "card", "migrate"):
        assert name in result.output



def test_models_check_passthrough_help_reaches_argparse():
    result = _uv_run("models", "check", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Check model registry status" in result.stdout


def test_models_migrate_help():
    result = runner.invoke(app, ["models", "migrate", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--overwrite" in result.output
    assert "--json" in result.output


def test_models_command_module_does_not_import_pandas_or_sklearn():
    """`stoptimes.cli.commands.models` must import `migrate` lazily inside
    the command body, not at module level -- otherwise every `stoptimes`
    invocation (not just `models migrate`) would pay pandas/sklearn import
    cost. (`stoptimes.cli.main` itself already eagerly mounts `val`, which
    pulls in pandas independently of this command's own hygiene -- that's
    out of this track's scope -- so this checks the models module in
    isolation rather than the whole assembled CLI.)"""
    full_env = dict(os.environ)
    existing = full_env.get("PYTHONPATH")
    full_env["PYTHONPATH"] = os.pathsep.join(
        [*_WORKSPACE_SRC_DIRS, existing] if existing else _WORKSPACE_SRC_DIRS
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import stoptimes.cli.commands.models, sys; "
            "assert 'pandas' not in sys.modules, sorted(sys.modules); "
            "assert 'sklearn' not in sys.modules; "
            "print('OK')",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        env=full_env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def _write_legacy_artifact(legacy_dir, key: str):
    import pickle

    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / f"{key}.pkl").write_bytes(pickle.dumps({"coef": [1, 2, 3]}))
    meta = {
        "model_type": "diagnostic_test",
        "dataset": "unit_test",
        "route_id": None,
        "features": [],
        "n_samples": 10,
        "train_samples": 8,
        "test_samples": 2,
        "metrics": {"test_mae_seconds": 1.0},
        "model_key": key,
        "saved_at": "2026-08-18T11:57:00.478753",
        "model_path": f"{key}.pkl",
    }
    (legacy_dir / f"{key}_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_models_migrate_end_to_end(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    _write_legacy_artifact(legacy_dir, "diag_key")

    result = runner.invoke(app, ["models", "migrate", str(legacy_dir), str(registry_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [i["key"] for i in payload["migrated"]] == ["diag_key"]
    assert (registry_dir / "diag_key" / "artifact.json").exists()
    assert (registry_dir / "diag_key" / "model.pkl").exists()


def test_models_migrate_dry_run_writes_nothing(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    _write_legacy_artifact(legacy_dir, "diag_key")

    result = runner.invoke(app, ["models", "migrate", str(legacy_dir), str(registry_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "would migrate: 1" in result.output
    assert not registry_dir.exists()


def test_models_migrate_exits_1_on_missing_legacy_dir(tmp_path):
    result = runner.invoke(
        app, ["models", "migrate", str(tmp_path / "nope"), str(tmp_path / "registry")]
    )
    assert result.exit_code != 0


# ── serve / diagnose ──────────────────────────────────────────────────────────


def test_serve_requires_a_source():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "--source" in result.output
    assert "track T4" not in result.output
    assert "planned" not in result.output


def test_diagnose_runs_and_reports_rather_than_failing():
    """A feed with nothing in it is a successful run reporting blockers, not a
    crash and not a non-zero exit -- that distinction is the whole point of the
    command for a newcomer."""
    result = runner.invoke(app, ["diagnose", "--json"])
    assert result.exit_code == 0, result.output
    assert "planned" not in result.output.lower()


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


def _app_without_val(monkeypatch):
    """Assemble a fresh CLI with `stoptimes.val` made to look uninstalled."""
    import importlib.util

    import typer
    from stoptimes.cli.commands import val as val_commands

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "stoptimes.val":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    bare = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
    val_commands.mount(bare)
    return bare


def test_val_without_the_extra_hints_instead_of_a_usage_error(monkeypatch):
    """A correct command line for a part that isn't installed must say so.

    Before this, `val` collapsed into one no-argument command when the extra was
    missing, so click rejected `val report <file>` with "Got unexpected extra
    argument(s)" (exit 2) -- a usage error about a command line that is in fact
    correct. Found by the clean-machine gate; see docs/PLAN.md §7.5."""
    bare = _app_without_val(monkeypatch)

    result = runner.invoke(bare, ["val", "report", "scores.parquet"])
    assert result.exit_code == 1, result.output
    assert "unexpected extra argument" not in result.output.lower()
    assert "needs the 'val' part" in result.output


def test_val_without_the_extra_hints_for_the_bare_group(monkeypatch):
    bare = _app_without_val(monkeypatch)

    result = runner.invoke(bare, ["val"])
    assert result.exit_code == 1, result.output
    assert "needs the 'val' part" in result.output


def test_install_hints_name_a_command_that_works(monkeypatch):
    """`stoptimes` isn't on PyPI, and `uv sync --extra ...` only works from
    `packages/cli` -- neither belongs in a hint aimed at someone who installed
    from a clone."""
    bare = _app_without_val(monkeypatch)

    result = runner.invoke(bare, ["val", "run"])
    assert "uv pip install './packages/cli[val]'" in result.output
    assert "pip install 'stoptimes[" not in result.output


def test_implemented_commands_never_claim_to_be_planned():
    """Every command that Phase 2 built must give a real usage error, not the
    placeholder text from before it existed."""
    # invoked with required arguments missing: must be a real usage error
    for argv in (["data", "add"], ["data", "arrivals"], ["record", "start"], ["record", "status"],
                 ["train", "run"], ["models", "card"], ["serve"]):
        result = runner.invoke(app, argv)
        assert result.exit_code != 0, argv
        assert "planned" not in result.output.lower(), argv

    # the two Phase 3 commands take no required arguments, so --help is the check
    for argv in (["train", "ui", "--help"], ["diagnose", "--help"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, argv
        assert "planned" not in result.output.lower(), argv
