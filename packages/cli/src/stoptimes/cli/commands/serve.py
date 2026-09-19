"""`stoptimes serve` — run inference and publish GTFS-RT TripUpdates.

Every import that isn't `typer`/`pathlib`/`json` is deferred into the
function body so `stoptimes serve --help` (and any other subcommand's
`--help`) stays fast and never pulls in pandas/pydantic/the pipeline.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path

import typer

DEFAULT_SOURCES_DIR = "./examples/agencies"


def serve(
    source: str = typer.Option(None, "--source", help="Agency id, e.g. 'mbta' (examples/agencies/<id>.yaml)."),
    sources_dir: Path = typer.Option(
        None, "--sources-dir", help="Directory of agency YAMLs (default: --sources-dir -> STOPTIMES_SOURCES -> ./examples/agencies)."
    ),
    registry: Path = typer.Option(None, "--registry", help="Model registry directory (one artifact.json per subdir)."),
    model: str = typer.Option(None, "--model", help="Artifact id, or a BYO spec: pickle:PATH | onnx:PATH | callable:module:func | http(s)://URL."),
    artifact: str = typer.Option(None, "--artifact", help="Alias for --model."),
    static: Path = typer.Option(
        None,
        "--static",
        help="GTFS static feed (.zip or URL) used to map-match vehicle positions onto stops. "
        "Defaults to the source's gtfs_static_url. Without geometry only feeds that already name "
        "a stop (TripUpdates) can be served.",
    ),
    pipeline: str = typer.Option(None, "--pipeline", help="Custom Pipeline: 'module:attr' or 'path/to/file.py:attr'."),
    sink: str = typer.Option(None, "--sink", help="file:PATH | http(s)://... | mqtt://host/topic"),
    format: str = typer.Option("json", "--format", help="Feed encoding: pb | json."),
    once: bool = typer.Option(False, "--once", help="Produce one feed and exit (the Phase 2 e2e gate uses this)."),
    interval: float = typer.Option(None, "--interval", help="Seconds between runs; loops until interrupted."),
    json_output: bool = typer.Option(False, "--json", help="Print a machine-readable run summary."),
) -> None:
    """Run the serving pipeline and publish GTFS-RT TripUpdates (file, HTTP, MQTT).

    `--json`: one line per run (repeated if `--interval` loops):
    ``{"source", "artifact_id", "generated_at", "trip_update_count", "errors", "published"}``.
    """
    if source is None:
        typer.secho(
            "✗ `stoptimes serve` needs --source <id>; see examples/agencies/ for the agency files.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    if format not in ("pb", "json"):
        typer.secho(f"✗ --format must be 'pb' or 'json', got {format!r}", fg="red", err=True)
        raise typer.Exit(2)

    if once and interval is not None:
        typer.secho("✗ pass --once or --interval, not both", fg="red", err=True)
        raise typer.Exit(2)

    # Everything past this point needs the real dependency stack.
    import time
    from datetime import datetime, timezone

    from stoptimes.cli._util import require

    require("stoptimes.serve", "serve")

    import yaml
    from stoptimes.core.contracts.adapters import source_from_val_yaml
    from stoptimes.core.contracts.pipeline import PipelineContext
    from stoptimes.serve.model_loading import (
        ArtifactVersionDriftError,
        load_artifact_model,
        load_model_ref,
        resolve_artifact,
    )
    from stoptimes.serve.pipeline import build_default_pipeline, load_pipeline as load_user_pipeline
    from stoptimes.serve.sinks import sink_from_uri

    src_dir = sources_dir or Path(
        __import__("os").environ.get("STOPTIMES_SOURCES", DEFAULT_SOURCES_DIR)
    )
    source_path = Path(src_dir) / f"{source}.yaml"
    if not source_path.is_file():
        typer.secho(f"✗ no agency file for --source {source!r}: {source_path} does not exist", fg="red", err=True)
        raise typer.Exit(1)
    agency = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    src = source_from_val_yaml(agency)

    # Geometry for the `match` stage. Without it a vehicle position cannot be
    # turned into "which stop is next and how far away" -- serving would be
    # limited to feeds that already name a stop.
    static_loader = None
    static_location = str(static) if static else (src.gtfs_static_url or None)
    if static_location:
        from stoptimes.core.readers import read_static

        try:
            static_loader, _static_tables = read_static(static_location, src.id)
        except Exception as exc:  # a bad static feed must not be a silent downgrade
            typer.secho(
                f"⚠ could not read the static feed {static_location!r} ({exc}); serving without "
                "map-matching — only stop-bearing feeds (TripUpdates) will produce predictions",
                fg="yellow",
                err=True,
            )

    model_choice = artifact or model
    resolved_model = None
    resolved_artifact_id = None
    if model_choice is not None and (
        model_choice.startswith("pickle:")
        or model_choice.startswith("onnx:")
        or model_choice.startswith("callable:")
        or model_choice.startswith("http://")
        or model_choice.startswith("https://")
    ):
        from stoptimes.core.contracts.model import ModelRef, ModelRefKind

        kind_str, _, location = model_choice.partition(":")
        if kind_str in ("http", "https"):
            ref = ModelRef(kind=ModelRefKind.HTTP, location=model_choice)
        else:
            ref = ModelRef(kind=ModelRefKind(kind_str), location=location)
        resolved_model = load_model_ref(ref)
    elif registry is not None:
        meta = resolve_artifact(
            registry,
            source_id=src.id,
            policy=src.serving.selection,
            artifact_id=model_choice,
        )
        if meta is not None:
            resolved_artifact_id = meta.artifact_id
            try:
                resolved_model = load_artifact_model(meta, registry)
            except (NotImplementedError, ArtifactVersionDriftError) as exc:
                # A refused model is an operational event, not a crash: the
                # message already says what drifted and how to override, so
                # print it and exit rather than dumping a traceback at whoever
                # is looking at this during an incident.
                typer.secho(f"✗ {exc}", fg="red", err=True)
                raise typer.Exit(1) from exc
        elif model_choice is not None:
            typer.secho(f"✗ artifact {model_choice!r} not found in registry {registry}", fg="red", err=True)
            raise typer.Exit(1)

    if pipeline is not None:
        run_pipeline = load_user_pipeline(pipeline)
    else:
        run_pipeline = build_default_pipeline()

    active_sink = sink_from_uri(sink) if sink else None

    def run_once() -> dict:
        ctx = PipelineContext(
            source=src,
            static_feed=static_loader,
            model=resolved_model,
            now=datetime.now(timezone.utc),
            config={"sink": active_sink, "format": format},
        )
        run_pipeline.run(ctx)
        trip_updates = ctx.state.get("trip_updates", [])
        errors = ctx.state.get("errors", [])
        return {
            "source": src.id,
            "artifact_id": resolved_artifact_id,
            "generated_at": ctx.now.isoformat(),
            "trip_update_count": len(trip_updates),
            "errors": errors,
            "published": ctx.state.get("published", False),
        }

    def report(summary: dict) -> None:
        if json_output:
            typer.echo(jsonlib.dumps(summary))
        else:
            typer.echo(
                f"serve[{summary['source']}]: {summary['trip_update_count']} trip update(s), "
                f"{len(summary['errors'])} error(s), published={summary['published']}"
            )
            for err in summary["errors"]:
                typer.secho(f"  ! {err['stage']} trip={err['trip_id']}: {err['error']}", fg="yellow", err=True)

    if interval is None:
        report(run_once())
        return

    try:
        while True:
            report(run_once())
            time.sleep(interval)
    except KeyboardInterrupt:
        raise typer.Exit(0)
