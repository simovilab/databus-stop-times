"""`stoptimes` — one command for the whole toolkit.

Every sub-command imports its implementation lazily, so `stoptimes serve` never
loads training dependencies, and a missing optional part produces an install hint
instead of a traceback.

Commands marked "planned" are placeholders: they name the phase in docs/PLAN.md
that implements them and exit with status 2.
"""

from __future__ import annotations

import importlib
import runpy
import sys

import typer

from stoptimes.cli._util import PASSTHROUGH, planned, require

app = typer.Typer(help="stoptimes — train, serve and validate transit arrival-time models on GTFS data.",
                  no_args_is_help=True)

# --------------------------------------------------------------------------- data
data_app = typer.Typer(help="Load and inspect GTFS static + realtime data.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("add")
def data_add() -> None:
    """Register GTFS static / realtime / AVL sources (protobuf, JSON, parquet, CSV)."""
    planned("data add", "Phase 2 · track T1 (data layer)")


@data_app.command("inspect")
def data_inspect() -> None:
    """Summarize coverage of registered data (trips matched, arrivals present, gaps)."""
    planned("data inspect", "Phase 2 · track T1 (data layer)")


@data_app.command("arrivals")
def data_arrivals() -> None:
    """Derive actual stop arrivals from vehicle positions (map-matching)."""
    planned("data arrivals", "Phase 2 · track T2 (arrival derivation)")


# ------------------------------------------------------------------------- record
record_app = typer.Typer(help="Record live GTFS-RT feeds into a compacted archive.", no_args_is_help=True)
app.add_typer(record_app, name="record")


@record_app.command("start")
def record_start() -> None:
    """Poll a GTFS-RT feed and spool it to the archive (standalone, no Django/Celery)."""
    planned("record start", "Phase 2 · track T5 (recorder)")


@record_app.command("compact", context_settings=PASSTHROUGH)
def record_compact(ctx: typer.Context) -> None:
    """Compact spooled data into the curated archive (existing implementation)."""
    require("stoptimes.record", "record")
    _run_module("stoptimes.record.compaction.cli", "stoptimes record compact", ctx.args)


# -------------------------------------------------------------------------- train
train_app = typer.Typer(help="Build datasets and train models.", no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("run")
def train_run() -> None:
    """Train from an experiment.yaml (model, features, hyperparameters, split)."""
    planned("train run", "Phase 2 · track T3 (experiments)")


@train_app.command("ui")
def train_ui() -> None:
    """Interactive experiment builder (TUI). Writes experiment.yaml."""
    planned("train ui", "Phase 3 · track U1 (experiment TUI)")


@train_app.command("legacy-all", context_settings=PASSTHROUGH)
def train_legacy_all(ctx: typer.Context) -> None:
    """Train every model family on a prepared dataset (migrated gtfs-eta trainer)."""
    require("stoptimes.train", "train")
    _run_module("stoptimes.train.train_all_models", "stoptimes train legacy-all", ctx.args)


@train_app.command("seed-baseline", context_settings=PASSTHROUGH)
def train_seed_baseline(ctx: typer.Context) -> None:
    """Seed a baseline model into the registry (migrated gtfs-eta helper)."""
    require("stoptimes.train", "train")
    _run_module("stoptimes.train.seed_baseline_model", "stoptimes train seed-baseline", ctx.args)


# ------------------------------------------------------------------------- models
models_app = typer.Typer(help="List, inspect and promote trained models.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("check", context_settings=PASSTHROUGH)
def models_check(ctx: typer.Context) -> None:
    """Check the model registry (migrated gtfs-eta check_registry)."""
    require("stoptimes.models", "core")
    _run_module("stoptimes.models.check_registry", "stoptimes models check", ctx.args)


@models_app.command("card")
def models_card() -> None:
    """Show a model card (data, features, scores by horizon)."""
    planned("models card", "Phase 2 · track T3 (artifacts + cards)")


# -------------------------------------------------------------------------- serve
@app.command("serve")
def serve() -> None:
    """Run inference and publish GTFS-RT TripUpdates (HTTP, file, MQTT, ...)."""
    planned("serve", "Phase 2 · track T4 (serving pipeline)")


# ----------------------------------------------------------------------- diagnose
@app.command("diagnose")
def diagnose() -> None:
    """Report GTFS / GTFS-RT feed quality problems before training on them."""
    planned("diagnose", "Phase 3 · track U3 (feed diagnostics)")


# ---------------------------------------------------------------------------- val
def _mount_val() -> None:
    try:
        module = importlib.import_module("stoptimes.val.cli.main")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not missing.startswith("stoptimes"):
            raise

        @app.command("val", help="Validate predictions (not installed).")
        def val_missing() -> None:
            require("stoptimes.val", "val")
        return
    app.add_typer(module.app, name="val", help="Validate arrival predictions against ground truth (etaval).")


def _run_module(module: str, prog: str, args: list[str]) -> None:
    sys.argv = [prog, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


_mount_val()

if __name__ == "__main__":
    app()
