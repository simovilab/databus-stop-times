"""`stoptimes data` — load and inspect GTFS static + realtime data."""

from __future__ import annotations

import typer

from stoptimes.cli._util import planned

app = typer.Typer(help="Load and inspect GTFS static + realtime data.", no_args_is_help=True)


@app.command("add")
def data_add() -> None:
    """Register GTFS static / realtime / AVL sources (protobuf, JSON, parquet, CSV)."""
    planned("data add", "Phase 2 · track T1 (data layer)")


@app.command("inspect")
def data_inspect() -> None:
    """Summarize coverage of registered data (trips matched, arrivals present, gaps)."""
    planned("data inspect", "Phase 2 · track T1 (data layer)")


@app.command("arrivals")
def data_arrivals() -> None:
    """Derive actual stop arrivals from vehicle positions (map-matching)."""
    planned("data arrivals", "Phase 2 · track T2 (arrival derivation)")
