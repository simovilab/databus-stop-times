"""`stoptimes train` — build datasets and train models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from stoptimes.cli._util import PASSTHROUGH, planned, require, run_module

app = typer.Typer(help="Build datasets and train models.", no_args_is_help=True)


@app.command("run")
def train_run(
    experiment_path: Optional[Path] = typer.Argument(
        None,
        exists=True,
        dir_okay=False,
        help="experiment.yaml (stoptimes.core.contracts.experiment.Experiment).",
    ),
    registry: Optional[Path] = typer.Option(
        None, "--registry", help="Artifact registry directory. Defaults to the experiment's outputs.registry_dir."
    ),
    dataset_dir: Optional[Path] = typer.Option(
        None,
        "--dataset-dir",
        help="Directory holding named datasets (overrides STOPTIMES_DATASET_DIR / ./datasets).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved plan (data, features, split, model, outputs) and exit."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing artifact with the same id instead of refusing."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the trained artifact's id/family/metrics as JSON."
    ),
) -> None:
    """Train from an experiment.yaml: resolve data, build features, split, fit
    the model plus its baselines, score by horizon, and write an artifact +
    card. `--dry-run` resolves and prints the plan without touching data.

    `--json` (ignored with `--dry-run`, which always prints the plan as JSON):
    ``{"artifact_id": str, "family": str, "metrics": {...}}``.
    """
    if experiment_path is None:
        typer.secho(
            "✗ `stoptimes train run` needs an experiment file; see examples/experiments/.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    require("stoptimes.train", "train")
    from stoptimes.core.contracts.experiment import load_experiment
    from stoptimes.train.experiment import ArtifactExistsError, plan_experiment, run_experiment

    experiment = load_experiment(experiment_path)

    if dry_run:
        typer.echo(json.dumps(plan_experiment(experiment), indent=2))
        return

    try:
        meta = run_experiment(
            experiment,
            registry_dir=registry,
            dataset_dir=dataset_dir,
            overwrite=overwrite,
        )
    except ArtifactExistsError as exc:
        typer.secho(f"✗ {exc}", fg="red", err=True)
        raise typer.Exit(1) from exc

    if json_output:
        typer.echo(json.dumps({"artifact_id": meta.artifact_id, "family": meta.family, "metrics": meta.metrics}))
        return

    mae = meta.metrics.get("test_mae_seconds")
    mae_str = f"{mae:.1f}s" if isinstance(mae, (int, float)) else "n/a"
    typer.echo(f"trained {meta.artifact_id} ({meta.family}), test_mae_seconds={mae_str}")


@app.command("ui")
def train_ui() -> None:
    """Interactive experiment builder (TUI). Writes experiment.yaml."""
    planned("train ui", "Phase 3 · track U1 (experiment TUI)")


@app.command("legacy-all", context_settings=PASSTHROUGH)
def train_legacy_all(ctx: typer.Context) -> None:
    """Train every model family on a prepared dataset (migrated gtfs-eta trainer)."""
    require("stoptimes.train", "train")
    run_module("stoptimes.train.train_all_models", "stoptimes train legacy-all", ctx.args)


@app.command("seed-baseline")
def train_seed_baseline() -> None:
    """Seed the synthetic global baseline model into the registry (migrated gtfs-eta helper).

    A regular command, not a passthrough: the underlying script takes no arguments
    and runs on import as __main__, so passing `--help` through would execute it.
    """
    require("stoptimes.train", "train")
    from stoptimes.train.seed_baseline_model import main

    main()
