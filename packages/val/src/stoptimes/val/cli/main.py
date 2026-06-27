"""Typer CLI. Phase P6.

Commands: version, sources, run, report.

Design
------
Typer command functions are intentionally thin: they validate CLI args, build
concrete dependencies (download GTFS zip, construct pollers / feed-source /
predictors / detectors), then delegate to the async core ``_run_validation``.

``_run_validation`` accepts already-constructed deps and delegates to
``run_engine``.  Tests exercise it directly with ``ReplayFeedSource`` +
``SqliteStorage(":memory:")`` — fully offline, no network, no sleeping —
mirroring how ``test_engine_replay.py`` tests the engine.

Route filter
------------
``_RouteFilterLiveSource`` wraps a ``LiveFeedSource`` and drops VP records
whose ``route_id`` is not in the target set.  It subclasses ``LiveFeedSource``
so ``isinstance(feed_source, LiveFeedSource)`` in ``run_engine`` stays True,
preserving the live-mode sleep and wall-clock duration check.
``LiveFeedSource.__init__`` is bypassed via ``object.__init__``.

GTFS zip download
-----------------
The ``run`` command downloads the GTFS static zip synchronously (``httpx.Client``)
before entering the async loop, writes it to a ``NamedTemporaryFile``, passes
the path to ``GtfsStaticLoader``, then deletes the temp file.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

import etaval
from etaval.engine.assembly import (
    DEFAULT_PREDICTOR_IDS,
    PREDICTOR_REGISTRY,
    _RouteFilterLiveSource,
    build_detectors,
    download_gtfs,
    parse_duration,
)
from etaval.engine.runtime import (
    LiveFeedSource,
    ReplayFeedSource,
    RunSummary,
    run_engine,
)
from etaval.feeds.gtfsrt_json import (
    TripUpdatesPoller,
    VehiclePositionsPoller,
)
from etaval.feeds.source_config import list_sources, load_named_source
from etaval.feeds.static import GtfsStaticLoader
from etaval.scoring.metrics import compute_metrics
from etaval.storage.parquet_export import export_results, load_results_parquet
from etaval.storage.sqlite import SqliteStorage

app = typer.Typer(help="etaval — ETA Prediction Validation Tool", no_args_is_help=True)

# ``parse_duration`` and ``_RouteFilterLiveSource`` are imported from
# ``etaval.engine.assembly`` above so the CLI and the future API share one
# implementation.  They remain importable from this module for backward compat
# (e.g. ``from etaval.cli.main import parse_duration`` in tests).


# ── Core validation function (testable without network) ───────────────────────


async def _run_validation(
    feed_source: LiveFeedSource | ReplayFeedSource,
    static_loader: GtfsStaticLoader,
    predictors: list[Any],
    detectors: list[Any],
    storage: SqliteStorage,
    *,
    primary_method: str = "shape_distance",
    duration_s: float | None = None,
    max_ticks: int | None = None,
    poll_interval_s: float = 15.0,
    source_name: str = "",
) -> RunSummary:
    """Thin async core that delegates to :func:`~etaval.engine.runtime.run_engine`.

    Keeping all dependency construction *outside* this function lets tests
    call it directly with ``ReplayFeedSource`` + ``SqliteStorage(":memory:")``
    for fully offline coverage (no network, no sleeping).

    Args:
        feed_source:     Live or replay async feed-source iterator.
        static_loader:   Loaded GTFS static zip data.
        predictors:      Configured predictor instances to run each tick.
        detectors:       Configured ground-truth detector instances.
        storage:         Open SQLite storage for persistence.
        primary_method:  Detector method used for scoring (default: shape_distance).
        duration_s:      Wall-clock cap in seconds (live mode only; ``None`` = unlimited).
        max_ticks:       Tick count cap (live or replay; ``None`` = unlimited).
        poll_interval_s: Sleep between live ticks (ignored in replay mode).
        source_name:     Source label forwarded to the trip tracker.

    Returns:
        :class:`~etaval.engine.runtime.RunSummary` with counts.
    """
    return await run_engine(
        feed_source,
        static_loader,
        predictors,
        detectors,
        storage,
        primary_method=primary_method,
        duration_s=duration_s,
        max_ticks=max_ticks,
        poll_interval_s=poll_interval_s,
        source_name=source_name,
    )


# ── CLI app ───────────────────────────────────────────────────────────────────


@app.callback()
def _main() -> None:
    """etaval — ETA Prediction Validation Tool."""


@app.command()
def version() -> None:
    """Print the etaval version."""
    typer.echo(etaval.__version__)


# ── sources ───────────────────────────────────────────────────────────────────


@app.command()
def sources() -> None:
    """List all configured GTFS-RT sources."""
    # Console created here (not at module level) so CliRunner captures output.
    console = Console(file=sys.stdout, no_color=True)
    names = list_sources()
    if not names:
        typer.echo("No sources configured.")
        return
    table = Table(title="Configured Sources", show_lines=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("poll_s", justify="right")
    table.add_column("VP URL")
    table.add_column("TU URL")
    table.add_column("Static URL")
    for name in names:
        cfg = load_named_source(name)
        table.add_row(
            cfg.name,
            str(cfg.poll_interval_s),
            cfg.vehicle_positions_url,
            cfg.trip_updates_url,
            cfg.gtfs_static_url,
        )
    console.print(table)


# ── run ───────────────────────────────────────────────────────────────────────


@app.command()
def run(
    source: str = typer.Option(..., "--source", help="Named source (e.g. mbta, bucr)"),
    route: list[str] | None = typer.Option(  # noqa: B008
        None, "--route", help="Filter to route ID(s); repeat for multiple"
    ),
    predictor: list[str] | None = typer.Option(  # noqa: B008
        None, "--predictor", help="Predictor ID(s); repeat for multiple (default: both)"
    ),
    duration: str = typer.Option("30m", "--duration", help="Run duration, e.g. 30m / 90s / 1h"),
    out: Path | None = typer.Option(None, "--out", help="Parquet output path"),  # noqa: B008
    db: Path | None = typer.Option(None, "--db", help="SQLite path (default: in-memory)"),  # noqa: B008
) -> None:
    """Run headless ETA validation against a live GTFS-RT source.

    Downloads the GTFS static zip, polls VP and TU feeds, scores predictions
    against detected actual arrivals, persists to SQLite, and (optionally)
    exports results to a Parquet file for reporting.
    """
    # ── Parse duration ────────────────────────────────────────────────────────
    try:
        duration_s = parse_duration(duration)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # ── Load source config ────────────────────────────────────────────────────
    try:
        cfg = load_named_source(source)
    except FileNotFoundError:
        typer.echo(
            f"Error: source {source!r} not found. Run 'etaval sources' to list available.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    # ── Resolve predictor list ────────────────────────────────────────────────
    predictor_ids: list[str] = list(predictor) if predictor else DEFAULT_PREDICTOR_IDS
    unknown = [pid for pid in predictor_ids if pid not in PREDICTOR_REGISTRY]
    if unknown:
        typer.echo(f"Error: unknown predictor IDs: {unknown!r}", err=True)
        typer.echo(f"Known predictor IDs: {sorted(PREDICTOR_REGISTRY)}", err=True)
        raise typer.Exit(code=1)
    predictors: list[Any] = [PREDICTOR_REGISTRY[pid]() for pid in predictor_ids]

    # ── Download GTFS static zip (synchronous; before asyncio.run) ────────────
    typer.echo(f"Downloading GTFS static zip from {cfg.gtfs_static_url} …")
    try:
        tmp_path = download_gtfs(cfg)
    except httpx.HTTPError as exc:
        typer.echo(f"Error downloading GTFS static zip: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        try:
            static_loader = GtfsStaticLoader(tmp_path)
        except Exception as exc:
            typer.echo(f"Error loading GTFS static data: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Build pollers + live feed source ──────────────────────────────────────
    headers: dict[str, str] = dict(cfg.headers)
    vp_poller = VehiclePositionsPoller(cfg.vehicle_positions_url, headers=headers)
    tu_poller = TripUpdatesPoller(cfg.trip_updates_url, headers=headers)
    live_source = LiveFeedSource(vp_poller, tu_poller)

    # Route filter wraps the live source; run_engine still sees a LiveFeedSource
    # (via inheritance) so live-mode sleep and duration-check are preserved.
    feed_source: LiveFeedSource
    if route:
        feed_source = _RouteFilterLiveSource(live_source, set(route))
    else:
        feed_source = live_source

    # ── Build detectors from source config ────────────────────────────────────
    detectors: list[Any] = build_detectors(cfg)

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str = str(db) if db is not None else ":memory:"

    # ── Run ───────────────────────────────────────────────────────────────────
    route_display = list(route) if route else "all"
    typer.echo(
        f"Running: source={source!r}, duration={duration}, "
        f"predictors={predictor_ids}, routes={route_display}"
    )

    with SqliteStorage(db_path) as storage:
        summary: RunSummary = asyncio.run(
            _run_validation(
                feed_source,
                static_loader,
                predictors,
                detectors,
                storage,
                primary_method="shape_distance",
                duration_s=duration_s,
                poll_interval_s=float(cfg.poll_interval_s),
                source_name=source,
            )
        )

        typer.echo(
            f"Done. Ticks={summary.ticks} Samples={summary.samples_saved} "
            f"Arrivals={summary.arrivals_saved} Results={summary.results_saved}"
        )

        # ── Parquet export (inside the storage context) ────────────────────────
        if out is not None:
            results = storage.load_results()
            if results:
                export_results(results, out)
                typer.echo(f"Exported {len(results)} results → {out}")
            else:
                typer.echo("No results to export (0 ValidationResults scored).")


# ── report ────────────────────────────────────────────────────────────────────


@app.command()
def report(
    path: Path = typer.Argument(..., help="Parquet file of ValidationResults"),  # noqa: B008
) -> None:
    """Render a metrics report from a Parquet results file.

    Prints three sections using rich tables:

    1. Per-predictor MAE / RMSE / MAPE / bias / count.
    2. Error vs horizon bucket (< 2 m, 2–5 m, 5–10 m, > 10 m).
    3. Head-to-head win-rate (when two or more predictors are present).
    """
    # Console created here so CliRunner captures output in tests.
    console = Console(file=sys.stdout, no_color=True)

    if not path.exists():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(code=1)

    try:
        results = load_results_parquet(path)
    except Exception as exc:
        typer.echo(f"Error loading {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not results:
        typer.echo("No results in file (0 rows).")
        return

    metrics = compute_metrics(results)

    # ── 1. Per-predictor table ────────────────────────────────────────────────
    pp_table = Table(title="Per-Predictor Metrics", show_lines=True)
    pp_table.add_column("Predictor", no_wrap=True)
    pp_table.add_column("MAE (s)", justify="right")
    pp_table.add_column("RMSE (s)", justify="right")
    pp_table.add_column("MAPE (%)", justify="right")
    pp_table.add_column("Bias (s)", justify="right")
    pp_table.add_column("Count", justify="right")
    for m in metrics.per_predictor:
        pp_table.add_row(
            m.predictor_id,
            f"{m.mae:.1f}",
            f"{m.rmse:.1f}",
            f"{m.mape:.1f}",
            f"{m.bias:.1f}",
            str(m.count),
        )
    console.print(pp_table)

    # ── 2. Horizon bucket table ───────────────────────────────────────────────
    if metrics.horizon_buckets:
        hb_table = Table(title="Error vs Horizon Bucket", show_lines=True)
        hb_table.add_column("Bucket")
        hb_table.add_column("Predictor")
        hb_table.add_column("MAE (s)", justify="right")
        hb_table.add_column("Bias (s)", justify="right")
        hb_table.add_column("Count", justify="right")
        for b in metrics.horizon_buckets:
            hb_table.add_row(
                b.label,
                b.predictor_id,
                f"{b.mae:.1f}",
                f"{b.bias:.1f}",
                str(b.count),
            )
        console.print(hb_table)

    # ── 3. Head-to-head ───────────────────────────────────────────────────────
    if metrics.head_to_head:
        h2h_table = Table(title="Head-to-Head Win Rate", show_lines=True)
        h2h_table.add_column("Predictor")
        h2h_table.add_column("Wins", justify="right")
        h2h_table.add_column("Total", justify="right")
        h2h_table.add_column("Win %", justify="right")
        for w in metrics.head_to_head:
            h2h_table.add_row(
                w.predictor_id,
                str(w.wins),
                str(w.total),
                f"{w.win_fraction * 100:.1f}",
            )
        console.print(h2h_table)


if __name__ == "__main__":
    app()
