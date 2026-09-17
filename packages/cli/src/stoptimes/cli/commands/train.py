"""`stoptimes train` — build datasets and train models."""

from __future__ import annotations

import typer
from stoptimes.cli._util import PASSTHROUGH, planned, require, run_module

app = typer.Typer(help="Build datasets and train models.", no_args_is_help=True)


@app.command("run")
def train_run() -> None:
    """Train from an experiment.yaml (model, features, hyperparameters, split)."""
    planned("train run", "Phase 2 · track T3 (experiments)")


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
