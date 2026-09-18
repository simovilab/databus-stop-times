"""`stoptimes record` — record live GTFS-RT feeds into a compacted archive."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from stoptimes.cli._util import PASSTHROUGH, planned, require, run_module

app = typer.Typer(help="Record live GTFS-RT feeds into a compacted archive.", no_args_is_help=True)

DEFAULT_SOURCES_DIR = "./examples/agencies"


def _sources_dir(sources_dir: Optional[str]) -> Path:
    import os

    if sources_dir:
        return Path(sources_dir)
    env = os.environ.get("STOPTIMES_SOURCES")
    if env:
        return Path(env)
    return Path(DEFAULT_SOURCES_DIR)


def _resolve_source(source_id: str, sources_dir: Optional[str]):
    """Load a `Source` by id, trying the contract's own YAML shape first and
    falling back to the flat val-source shape. See docs/CONTRACTS.md §1."""
    import yaml
    from stoptimes.core.contracts.adapters import source_from_val_yaml
    from stoptimes.core.contracts.source import load_source

    path = _sources_dir(sources_dir) / f"{source_id}.yaml"
    if not path.exists():
        typer.secho(f"✗ No source config found at {path}", fg="red", err=True)
        raise typer.Exit(1)
    try:
        return load_source(path)
    except Exception:
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return source_from_val_yaml(data)


def _apply_out(source, out: Optional[str]):
    if not out:
        return source
    from stoptimes.core.contracts.adapters import source_from_archive_uri

    return source_from_archive_uri(source, out)


@app.command("start")
def record_start(
    source_id: Optional[str] = typer.Option(
        None, "--source", help="Source id (matches examples/agencies/<id>.yaml)."
    ),
    out: Optional[str] = typer.Option(
        None,
        "--out",
        help="Archive root: a local path or s3://bucket/prefix. Overrides the source's configured archive.",
    ),
    feeds: Optional[str] = typer.Option(
        None,
        "--feeds",
        help="Comma-separated feed kinds to poll (vp,tu). Default: every feed the source configures.",
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", help="Stop automatically after this many seconds. Default: run until SIGINT/SIGTERM."
    ),
    spool_dir: Path = typer.Option(
        Path("./spool"), "--spool-dir", help="Local DuckDB spool directory."
    ),
    sources_dir: Optional[str] = typer.Option(
        None,
        "--sources-dir",
        help="Directory of <id>.yaml source configs (default: $STOPTIMES_SOURCES or ./examples/agencies).",
    ),
    status_dir: Optional[str] = typer.Option(
        None, "--status-dir", help="Where to write <source>.json/.txt/.events.log status snapshots."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Poll and parse but never write to the spool or archive."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the run summary as JSON instead of text."
    ),
) -> None:
    """Poll a Source's realtime feeds and spool them to the archive (standalone, no Django/Celery).

    `--json`: ``{"stopped_reason": str, "stats": {feed_kind: {"polls", "written",
    "skipped_unchanged", "errors"}}}``.
    """
    if source_id is None:
        # Bare `record start` stays a stub message (docs/PLAN.md) rather than
        # a bare "missing option" error -- the shared CLI smoke test
        # (packages/cli/tests/test_cli.py::test_record_start_is_planned)
        # invokes it with no args and expects the planned-command contract.
        # See docs/notes/p2-t5.md.
        typer.secho(
            "✗ `stoptimes record start` needs --source <id>; see examples/agencies/.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    require("stoptimes.record", "record")
    import asyncio

    from stoptimes.core.contracts.source import FeedKind
    from stoptimes.record.recorder import Recorder, RecorderConfig, run_recorder

    source = _apply_out(_resolve_source(source_id, sources_dir), out)

    feed_kinds = None
    if feeds:
        alias = {"vp": FeedKind.VEHICLE_POSITIONS, "tu": FeedKind.TRIP_UPDATES}
        feed_kinds = []
        for token in feeds.split(","):
            token = token.strip().lower()
            if token in alias:
                feed_kinds.append(alias[token])
            else:
                try:
                    feed_kinds.append(FeedKind(token))
                except ValueError:
                    typer.secho(f"✗ Unknown feed kind: {token!r}", fg="red", err=True)
                    raise typer.Exit(1) from None

    spool_dir.mkdir(parents=True, exist_ok=True)
    config = RecorderConfig(
        source=source,
        spool_dir=spool_dir,
        feed_kinds=feed_kinds,
        dry_run=dry_run,
        duration_s=duration,
        status_dir=status_dir,
    )
    recorder = Recorder(config)

    try:
        result = asyncio.run(run_recorder(recorder))
    except Exception as exc:  # a real failure — non-zero exit
        typer.secho(f"✗ record start failed: {exc}", fg="red", err=True)
        raise typer.Exit(1) from exc

    if json_output:
        import json as jsonlib

        payload = {
            "stopped_reason": result.stopped_reason,
            "stats": {
                str(kind): {
                    "polls": stats.polls,
                    "written": stats.written,
                    "skipped_unchanged": stats.skipped_unchanged,
                    "errors": stats.errors,
                }
                for kind, stats in result.stats.items()
            },
        }
        typer.echo(jsonlib.dumps(payload))
        return

    typer.echo(f"stopped ({result.stopped_reason})")
    for kind, stats in result.stats.items():
        typer.echo(
            f"  {kind}: polls={stats.polls} written={stats.written} "
            f"skipped_unchanged={stats.skipped_unchanged} errors={stats.errors}"
        )


@app.command("compact", context_settings=PASSTHROUGH)
def record_compact(ctx: typer.Context) -> None:
    """Compact spooled data into the curated archive (existing implementation)."""
    require("stoptimes.record", "record")
    run_module("stoptimes.record.compaction.cli", "stoptimes record compact", ctx.args)


@app.command("status")
def record_status(
    source_id: Optional[str] = typer.Option(None, "--source", help="Source id."),
    spool_dir: Path = typer.Option(Path("./spool"), "--spool-dir"),
    sources_dir: Optional[str] = typer.Option(None, "--sources-dir"),
    status_dir: Optional[str] = typer.Option(None, "--status-dir"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Recorder status: spool size, last flush, last poll per feed, archive size.

    Generic replacement for SIMOVI's ops-only `simovi-status` script — no
    agency hostnames or paths baked in; everything comes from `--source` /
    `--spool-dir` / `--status-dir`.

    `--json`: ``{"source", "vehicle_positions_spool", "trip_updates_spool",
    "last_status", "archive_bytes"}``; the two spool keys are each
    ``{"rows", "oldest_ts", "newest_ts", "bytes"}`` or null if unstarted.
    """
    if source_id is None:
        typer.secho(
            "✗ `stoptimes record status` needs --source <id>; see examples/agencies/.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    require("stoptimes.record", "record")
    import json as jsonlib

    from stoptimes.core.contracts.source import FeedKind
    from stoptimes.record import status as status_mod
    from stoptimes.record.recorder import spool_path_for
    from stoptimes.record.storage.spool import Spool
    from stoptimes.record.storage.tu_spool import TripUpdateSpool

    source = _resolve_source(source_id, sources_dir)

    report: dict = {"source": source_id}
    for kind, spool_cls, key in (
        (FeedKind.VEHICLE_POSITIONS, Spool, "vehicle_positions"),
        (FeedKind.TRIP_UPDATES, TripUpdateSpool, "trip_updates"),
    ):
        path = Path(spool_path_for(spool_dir, source_id, kind))
        if path.exists():
            stats = spool_cls(str(path), must_exist=True).stats()
            report[f"{key}_spool"] = {
                "rows": stats["rows"],
                "oldest_ts": str(stats["oldest_ts"]) if stats["oldest_ts"] else None,
                "newest_ts": str(stats["newest_ts"]) if stats["newest_ts"] else None,
                "bytes": stats["bytes"],
            }
        else:
            report[f"{key}_spool"] = None

    report["last_status"] = status_mod.read(source_id, status_dir)

    archive_size = None
    if source.archive is not None and not source.archive.uri.startswith("s3://"):
        root = Path(source.archive.uri)
        if root.exists():
            archive_size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    report["archive_bytes"] = archive_size

    if as_json:
        typer.echo(jsonlib.dumps(report, indent=2, default=str))
        return

    typer.echo(f"source: {source_id}")
    for key in ("vehicle_positions_spool", "trip_updates_spool"):
        spool = report[key]
        if spool is None:
            typer.echo(f"  {key}: (not started)")
        else:
            typer.echo(
                f"  {key}: rows={spool['rows']} bytes={spool['bytes']} "
                f"oldest={spool['oldest_ts']} newest={spool['newest_ts']}"
            )
    last = report["last_status"]
    if last:
        typer.echo(f"  last_status: {last}")
    if archive_size is not None:
        typer.echo(f"  archive_bytes: {archive_size}")
