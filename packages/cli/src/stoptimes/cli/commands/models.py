"""`stoptimes models` — list, inspect and promote trained models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from stoptimes.cli._util import PASSTHROUGH, planned, print_table, require, run_module

app = typer.Typer(help="List, inspect and promote trained models.", no_args_is_help=True)

_DEFAULT_REGISTRY = Path("artifacts/models")


@app.command("check", context_settings=PASSTHROUGH)
def models_check(ctx: typer.Context) -> None:
    """Check the model registry (migrated gtfs-eta check_registry)."""
    require("stoptimes.models", "core")
    run_module("stoptimes.models.check_registry", "stoptimes models check", ctx.args)


@app.command("list")
def models_list(
    registry: Path = typer.Option(_DEFAULT_REGISTRY, "--registry", help="Artifact registry directory."),
    source: Optional[str] = typer.Option(
        None, "--source", help="Filter to this Source.id (contracts.artifact.ModelScope.source_id)."
    ),
    route: Optional[str] = typer.Option(None, "--route", help="Filter to artifacts scoped to this route."),
    family: Optional[str] = typer.Option(None, "--family", help="Filter to this model family."),
    json_output: bool = typer.Option(False, "--json", help="Print as JSON instead of a table."),
) -> None:
    """List artifacts in a registry directory (contracts.artifact layout):
    scope, headline metric, and creation time.

    `--json`: a list of ``{"artifact_id", "family", "scope", "routes", "metric", "created"}``.
    """
    require("stoptimes.models", "core")
    from stoptimes.core.contracts.artifact import ArtifactMetadata, METADATA_FILENAME, read_metadata

    rows: list[dict] = []
    if registry.exists():
        for entry in sorted(registry.iterdir()):
            if not (entry / METADATA_FILENAME).exists():
                continue
            try:
                meta: ArtifactMetadata = read_metadata(entry)
            except Exception as exc:  # noqa: BLE001 - one bad artifact shouldn't kill the listing
                typer.secho(f"  skip {entry.name}: could not read artifact.json ({exc!r})", fg="yellow", err=True)
                continue
            if family and meta.family != family:
                continue
            if source and meta.scope.source_id not in (None, source):
                continue
            if route and route not in meta.scope.route_ids:
                continue
            rows.append(
                {
                    "artifact_id": meta.artifact_id,
                    "family": meta.family,
                    "scope": meta.scope.level.value,
                    "routes": meta.scope.route_ids,
                    "metric": meta.metrics.get("test_mae_seconds"),
                    "created": meta.provenance.created_at.isoformat(),
                }
            )

    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return

    table_rows = []
    for row in rows:
        mae = row["metric"]
        mae_str = f"{mae:.1f}" if isinstance(mae, (int, float)) else "n/a"
        scope = row["scope"] if not row["routes"] else f"{row['scope']}:{','.join(row['routes'])}"
        table_rows.append([row["artifact_id"], row["family"], scope, mae_str, row["created"]])

    print_table(
        f"models: {registry}",
        ["artifact id", "family", "scope", "mae_s", "created"],
        table_rows,
        empty_message=f"(no artifacts found under {registry})",
    )


@app.command("card")
def models_card(
    artifact_id: Optional[str] = typer.Argument(None, help="Artifact id (registry subdirectory name)."),
    registry: Path = typer.Option(_DEFAULT_REGISTRY, "--registry", help="Artifact registry directory."),
    json_output: bool = typer.Option(
        False, "--json", help="Print {'artifact_id', 'card_path', 'card'} as JSON instead of raw markdown."
    ),
) -> None:
    """Show a model card (data, features, scores by horizon).

    `--json`: ``{"artifact_id": str, "card_path": str, "card": str}`` (the
    card's raw markdown text, unparsed).
    """
    if artifact_id is None:
        typer.secho(
            "✗ `stoptimes models card` needs an artifact id; run `stoptimes models list`.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    require("stoptimes.models", "core")
    from stoptimes.core.contracts.artifact import artifact_dir

    card_path = artifact_dir(registry, artifact_id) / "card.md"
    if not card_path.exists():
        typer.secho(f"✗ no card.md for artifact {artifact_id!r} at {card_path}", fg="red", err=True)
        raise typer.Exit(1)
    text = card_path.read_text(encoding="utf-8")
    if json_output:
        typer.echo(json.dumps({"artifact_id": artifact_id, "card_path": str(card_path), "card": text}))
        return
    typer.echo(text)


@app.command("promote")
def models_promote(
    artifact_id: str = typer.Argument(..., help="Artifact id to promote."),
    as_pin: str = typer.Option(
        ...,
        "--as",
        help='Pin key (contracts.selection.SelectionPolicy.pins): "global", "route:<id>", '
        '"route:<id>/<direction>", or "trip:<id>".',
    ),
    source: str = typer.Option(..., "--source", help="Agency id whose profile file to update."),
    registry: Path = typer.Option(_DEFAULT_REGISTRY, "--registry", help="Artifact registry directory."),
    sources_dir: Optional[Path] = typer.Option(
        None, "--sources-dir", help="Agency profile directory. Defaults to STOPTIMES_SOURCES or ./examples/agencies."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the pin as JSON."),
) -> None:
    """Pin `artifact_id` into `source`'s agency profile (`serving.selection.pins[as_pin]`),
    per the `stoptimes.core.contracts.selection` contract. Refuses if the
    artifact doesn't exist in `registry`.

    `--json`: ``{"artifact_id": str, "pin": str, "source": str, "profile_path": str}``.
    """
    require("stoptimes.models", "core")
    import yaml
    from stoptimes.core.contracts.artifact import METADATA_FILENAME, artifact_dir, read_metadata

    target_dir = artifact_dir(registry, artifact_id)
    if not (target_dir / METADATA_FILENAME).exists():
        typer.secho(f"✗ artifact {artifact_id!r} not found at {target_dir}", fg="red", err=True)
        raise typer.Exit(1)
    read_metadata(target_dir)  # fail loudly if artifact.json is corrupt, before touching the profile

    profile_dir = sources_dir or Path(os.environ.get("STOPTIMES_SOURCES", "examples/agencies"))
    profile_path = profile_dir / f"{source}.yaml"
    if not profile_path.exists():
        typer.secho(f"✗ agency profile not found: {profile_path}", fg="red", err=True)
        raise typer.Exit(1)

    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    serving = data.setdefault("serving", {})
    selection = serving.setdefault("selection", {})
    pins = selection.setdefault("pins", {})
    pins[as_pin] = artifact_id
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    if json_output:
        typer.echo(
            json.dumps(
                {"artifact_id": artifact_id, "pin": as_pin, "source": source, "profile_path": str(profile_path)}
            )
        )
        return
    typer.echo(f"promoted {artifact_id!r} -> {profile_path} serving.selection.pins[{as_pin!r}]")


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
