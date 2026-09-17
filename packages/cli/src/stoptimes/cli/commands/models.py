"""`stoptimes models` — list, inspect and promote trained models."""

from __future__ import annotations

import typer

from stoptimes.cli._util import PASSTHROUGH, planned, require, run_module

app = typer.Typer(help="List, inspect and promote trained models.", no_args_is_help=True)


@app.command("check", context_settings=PASSTHROUGH)
def models_check(ctx: typer.Context) -> None:
    """Check the model registry (migrated gtfs-eta check_registry)."""
    require("stoptimes.models", "core")
    run_module("stoptimes.models.check_registry", "stoptimes models check", ctx.args)


@app.command("card")
def models_card() -> None:
    """Show a model card (data, features, scores by horizon)."""
    planned("models card", "Phase 2 · track T3 (artifacts + cards)")
