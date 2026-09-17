"""`stoptimes record` — record live GTFS-RT feeds into a compacted archive."""

from __future__ import annotations

import typer

from stoptimes.cli._util import PASSTHROUGH, planned, require, run_module

app = typer.Typer(help="Record live GTFS-RT feeds into a compacted archive.", no_args_is_help=True)


@app.command("start")
def record_start() -> None:
    """Poll a GTFS-RT feed and spool it to the archive (standalone, no Django/Celery)."""
    planned("record start", "Phase 2 · track T5 (recorder)")


@app.command("compact", context_settings=PASSTHROUGH)
def record_compact(ctx: typer.Context) -> None:
    """Compact spooled data into the curated archive (existing implementation)."""
    require("stoptimes.record", "record")
    run_module("stoptimes.record.compaction.cli", "stoptimes record compact", ctx.args)
