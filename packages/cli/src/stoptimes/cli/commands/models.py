"""`stoptimes models` — list, inspect and promote trained models."""

from __future__ import annotations

import json
from pathlib import Path

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


@app.command("migrate")
def models_migrate(
    legacy_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Legacy gtfs-eta registry directory (*.pkl + *_meta.json pairs).",
    ),
    registry_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        help="Destination artifact registry directory (contracts/artifact.py layout).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would happen without writing anything."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-migrate artifacts that already exist at the destination."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Migrate a legacy gtfs-eta model registry (`{key}.pkl` + `{key}_meta.json`)
    into the `stoptimes.core.contracts.artifact` layout.

    Idempotent: an artifact already present at the destination is skipped
    unless `--overwrite` is given. Never raises on one bad artifact (e.g. one
    needing an uninstalled optional library like xgboost/torch) -- that
    artifact is reported skipped and the rest of the run continues. Exits 1
    if any artifact failed outright.
    """
    require("stoptimes.models", "core")
    from stoptimes.models.compat.migrate import MigrationItem, migrate

    report = migrate(legacy_dir, registry_dir, dry_run=dry_run, overwrite=overwrite)

    def _item_dict(item: MigrationItem) -> dict:
        return {
            "key": item.key,
            "artifact_id": item.artifact_id,
            "status": item.status,
            "reason": item.reason,
            "warnings": list(item.warnings),
        }

    if json_output:
        payload = {
            "migrated": [_item_dict(i) for i in report.migrated],
            "planned": [_item_dict(i) for i in report.planned] if dry_run else [],
            "skipped": [_item_dict(i) for i in report.skipped],
            "failed": [_item_dict(i) for i in report.failed],
            "orphans": [_item_dict(i) for i in report.orphans],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        _print_summary(report, dry_run=dry_run)

    if report.failed:
        raise typer.Exit(1)


def _print_summary(report, *, dry_run: bool) -> None:
    verb = "would migrate" if dry_run else "migrated"
    n_migrated = len(report.planned) if dry_run else len(report.migrated)

    typer.echo(f"{verb}: {n_migrated}")
    typer.echo(f"skipped:  {len(report.skipped)}")
    typer.echo(f"failed:   {len(report.failed)}")
    typer.echo(f"orphans:  {len(report.orphans)}")

    for item in report.skipped:
        typer.echo(f"  skip   {item.key}: {item.reason}")
    for item in report.failed:
        typer.secho(f"  fail   {item.key}: {item.reason}", fg="red")
    for item in report.orphans:
        typer.echo(f"  orphan {item.key}: {item.reason}")
    for item in report.items:
        for warning in item.warnings:
            typer.echo(f"  note   {item.key}: {warning}")
