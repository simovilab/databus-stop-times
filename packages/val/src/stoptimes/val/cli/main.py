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
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

import stoptimes.val
from stoptimes.val.backtest import walk_forward
from stoptimes.val.engine.assembly import (
    DEFAULT_PREDICTOR_IDS,
    PREDICTOR_REGISTRY,
    _attach_static_loader,
    _RouteFilterLiveSource,
    build_detectors,
    build_replay_run,
    download_gtfs,
    parse_duration,
)
from stoptimes.val.engine.runtime import (
    LiveFeedSource,
    ReplayFeedSource,
    RunSummary,
    run_engine,
)
from stoptimes.core.realtime.gtfsrt_json import (
    TripUpdatesPoller,
    VehiclePositionsPoller,
)
from stoptimes.val.feeds.source_config import list_sources, load_named_source
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.val.figures import render_all_figures
from stoptimes.val.provenance import RunProvenance, build_run_provenance, write_provenance_sidecar
from stoptimes.val.scoring.metrics import compute_metrics
from stoptimes.val.scoring.sensitivity import sensitivity_from_persisted_arrivals
from stoptimes.val.storage.parquet_export import export_results, load_results_parquet
from stoptimes.val.storage.sqlite import SqliteStorage

app = typer.Typer(help="etaval — ETA Prediction Validation Tool", no_args_is_help=True)

# ``parse_duration`` and ``_RouteFilterLiveSource`` are imported from
# ``stoptimes.val.engine.assembly`` above so the CLI and the future API share one
# implementation.  They remain importable from this module for backward compat
# (e.g. ``from stoptimes.val.cli.main import parse_duration`` in tests).


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
    seed: int | None = None,
    provenance: RunProvenance | None = None,
) -> RunSummary:
    """Thin async core that delegates to :func:`~stoptimes.val.engine.runtime.run_engine`.

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
        seed:            Optional run seed, threaded through to RunSummary/provenance.
        provenance:      Optional pre-built RunProvenance, threaded through unchanged.

    Returns:
        :class:`~stoptimes.val.engine.runtime.RunSummary` with counts.
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
        seed=seed,
        provenance=provenance,
    )


# ── CLI app ───────────────────────────────────────────────────────────────────


@app.callback()
def _main() -> None:
    """etaval — ETA Prediction Validation Tool."""


@app.command()
def version() -> None:
    """Print the etaval version."""
    typer.echo(stoptimes.val.__version__)


# ── sources ───────────────────────────────────────────────────────────────────


@app.command()
def sources(
    sources_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--sources-dir",
        help=(
            "Directory of agency profile YAMLs to search exclusively "
            "(default: STOPTIMES_SOURCES, then ./examples/agencies)"
        ),
    ),
) -> None:
    """List all configured GTFS-RT sources."""
    # Console created here (not at module level) so CliRunner captures output.
    console = Console(file=sys.stdout, no_color=True)
    names = list_sources(config_dir=sources_dir)
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
        cfg = load_named_source(name, config_dir=sources_dir)
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
    replay_vp: str | None = typer.Option(
        None, "--replay-vp", help="Archived VP parquet path (dir/file/s3://); enables replay mode"
    ),
    replay_tu: str | None = typer.Option(
        None, "--replay-tu", help="Archived TU parquet path (dir/file/s3://); optional"
    ),
    replay_start: str | None = typer.Option(
        None, "--replay-start", help="Replay window start (epoch seconds or ISO 8601)"
    ),
    replay_end: str | None = typer.Option(
        None, "--replay-end", help="Replay window end (epoch seconds or ISO 8601)"
    ),
    replay_gtfs: Path | None = typer.Option(  # noqa: B008
        None, "--replay-gtfs", help="Static GTFS zip to use for replay"
    ),
    replay_gtfs_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--replay-gtfs-dir",
        help="Directory of dated GTFS zips (YYYY-MM-DD.zip); newest <= first replay tick is used",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Run seed, recorded in provenance (no RNG in the tick loop today)"
    ),
) -> None:
    """Run headless ETA validation against a live GTFS-RT source, or replay an archive.

    Live mode (default): downloads the GTFS static zip, polls VP and TU feeds,
    scores predictions against detected actual arrivals, persists to SQLite,
    and (optionally) exports results to a Parquet file for reporting.

    Replay mode (``--replay-vp`` given): sources snapshots from an archived
    Hive-partitioned Parquet VP/TU feed instead of live HTTP, wall-clock
    independent. Requires ``--replay-gtfs`` or ``--replay-gtfs-dir``.
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

    db_path: str = str(db) if db is not None else ":memory:"

    # ── Replay mode ────────────────────────────────────────────────────────────
    if replay_vp is not None:
        if replay_gtfs is None and replay_gtfs_dir is None:
            typer.echo(
                "Error: --replay-gtfs or --replay-gtfs-dir is required in replay mode", err=True
            )
            raise typer.Exit(code=1)

        try:
            deps = build_replay_run(
                cfg,
                vp_path=replay_vp,
                tu_path=replay_tu,
                start=replay_start,
                end=replay_end,
                routes=list(route) if route else None,
                predictor_ids=predictor_ids,
                static_gtfs_zip=replay_gtfs,
                static_gtfs_dir=replay_gtfs_dir,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            typer.echo(f"Error building replay run: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        route_display = list(route) if route else "all"
        typer.echo(
            f"Replaying: source={source!r}, vp={replay_vp!r}, tu={replay_tu!r}, "
            f"predictors={predictor_ids}, routes={route_display}"
        )

        gtfs_snapshot = str(replay_gtfs) if replay_gtfs is not None else (
            str(replay_gtfs_dir) if replay_gtfs_dir is not None else None
        )
        provenance = build_run_provenance(
            source_name=source,
            source_config=cfg,
            detectors=deps.detectors,
            predictor_ids=predictor_ids,
            seed=seed,
            gtfs_snapshot=gtfs_snapshot,
        )

        with SqliteStorage(db_path) as storage:
            summary = asyncio.run(
                _run_validation(
                    deps.feed_source,
                    deps.static_loader,
                    deps.predictors,
                    deps.detectors,
                    storage,
                    primary_method="shape_distance",
                    source_name=source,
                    seed=seed,
                    provenance=provenance,
                )
            )
            _report_run(summary)
            _export_if_requested(storage, out)
            _write_sidecars_if_requested(summary, out)
        return

    # ── Live mode ──────────────────────────────────────────────────────────────
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

    # Give any predictor that exposes attach_static_loader() the run's loader
    # (e.g. ScheduleBaselinePredictor). See stoptimes.val.engine.assembly.build_live_run
    # for the same wiring used on the assembly-based path.
    _attach_static_loader(predictors, static_loader)

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

    # ── Run ───────────────────────────────────────────────────────────────────
    route_display = list(route) if route else "all"
    typer.echo(
        f"Running: source={source!r}, duration={duration}, "
        f"predictors={predictor_ids}, routes={route_display}"
    )

    provenance = build_run_provenance(
        source_name=source,
        source_config=cfg,
        detectors=detectors,
        predictor_ids=predictor_ids,
        seed=seed,
        gtfs_snapshot=cfg.gtfs_static_url,
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
                seed=seed,
                provenance=provenance,
            )
        )

        _report_run(summary)
        _export_if_requested(storage, out)
        _write_sidecars_if_requested(summary, out)


def _format_provenance_stamp(prov: RunProvenance) -> str:
    """One-line ``etaval sha[dirty] / gtfs-eta sha[dirty] / seed / snapshot`` stamp."""

    def _sha_dirty(sha: str | None, dirty: bool | None) -> str:
        short = sha[:12] if sha else "unknown"
        if dirty is None:
            tag = "?"
        else:
            tag = "dirty" if dirty else "clean"
        return f"{short}[{tag}]"

    etaval_part = _sha_dirty(prov.etaval_git_sha, prov.etaval_dirty)
    gtfs_eta_part = _sha_dirty(prov.gtfs_eta_git_sha, prov.gtfs_eta_dirty)
    seed_part = str(prov.seed) if prov.seed is not None else "none"
    snapshot_part = prov.gtfs_snapshot if prov.gtfs_snapshot is not None else "none"
    return (
        f"Provenance: etaval {etaval_part} / gtfs-eta {gtfs_eta_part} / "
        f"seed={seed_part} / snapshot={snapshot_part}"
    )


def _report_run(summary: RunSummary) -> None:
    """Echo the standard end-of-run summary line + any swallowed-error counts."""
    typer.echo(
        f"Done. Ticks={summary.ticks} Samples={summary.samples_saved} "
        f"Arrivals={summary.arrivals_saved} Results={summary.results_saved}"
    )
    if summary.predictor_errors:
        typer.echo(f"Predictor errors: {summary.predictor_errors}")
    if summary.detector_errors:
        typer.echo(f"Detector errors: {summary.detector_errors}")
    if summary.provenance is not None:
        typer.echo(_format_provenance_stamp(summary.provenance))


def _write_sidecars_if_requested(summary: RunSummary, out: Path | None) -> None:
    """Write ``<out>.provenance.json`` and ``<out>.summary.json``, if *out* given.

    The summary sidecar carries the inference-latency block so a later
    cost/accuracy Pareto analysis can read it without re-running the engine.
    """
    if out is None:
        return
    if summary.provenance is not None:
        write_provenance_sidecar(out, summary.provenance)
        typer.echo(f"Wrote provenance → {out}.provenance.json")
    summary_path = Path(f"{out}.summary.json")
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    typer.echo(f"Wrote summary → {summary_path}")


def _export_if_requested(storage: SqliteStorage, out: Path | None) -> None:
    """Export ValidationResults to Parquet at *out*, if given.

    Must be called inside the storage context (before it closes).
    """
    if out is None:
        return
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
    sensitivity: bool = typer.Option(
        False, "--sensitivity", help="Also render a detector-sensitivity table (requires --db)"
    ),
    db: Path | None = typer.Option(  # noqa: B008
        None, "--db", help="SQLite db from the original 'etaval run' (required for --sensitivity)"
    ),
) -> None:
    """Render a metrics report from a Parquet results file.

    Prints three sections using rich tables:

    1. Per-predictor MAE / RMSE / MAPE / bias / count.
    2. Error vs horizon bucket (< 2 m, 2–5 m, 5–10 m, > 10 m).
    3. Head-to-head win-rate (when two or more predictors are present).

    With ``--sensitivity --db <sqlite>``, also renders a fourth section
    showing how scored MAE shifts per predictor when the ground-truth
    detector method is swapped (see ``stoptimes.val.scoring.sensitivity`` for what
    this can and cannot re-derive from storage). ``--sensitivity`` without
    ``--db`` is declined — see below.
    """
    # Console created here so CliRunner captures output in tests.
    console = Console(file=sys.stdout, no_color=True)

    if sensitivity and db is None:
        typer.echo(
            "Error: --sensitivity requires --db <sqlite path>. Detector sensitivity "
            "re-scores each ground-truth detector method's arrivals, but only "
            "'etaval run --db <path>' persists those per-method arrivals (the engine "
            "saves every configured detector's output, not just the primary one used "
            "for scoring). The Parquet results file alone carries neither raw "
            "position history nor per-method arrivals, so true detector "
            "re-sensitivity cannot be computed from it -- pass --db pointing at the "
            "SQLite database written by the run that produced this report.",
            err=True,
        )
        raise typer.Exit(code=1)

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

    # ── 4. Detector sensitivity (optional) ────────────────────────────────────
    if sensitivity:
        assert db is not None  # guaranteed by the decline-check above
        if not db.exists():
            typer.echo(f"Error: db file not found: {db}", err=True)
            raise typer.Exit(code=1)

        with SqliteStorage(db) as storage:
            db_samples = storage.load_samples()
            db_arrivals = storage.load_arrivals()

        rows = sensitivity_from_persisted_arrivals(db_samples, db_arrivals)
        if not rows:
            typer.echo("No detector sensitivity to report (0 persisted arrivals in --db).")
        else:
            sens_table = Table(title="Detector Sensitivity", show_lines=True)
            sens_table.add_column("Detector", no_wrap=True)
            sens_table.add_column("Predictor", no_wrap=True)
            sens_table.add_column("MAE (s)", justify="right")
            sens_table.add_column("Count", justify="right")
            sens_table.add_column("Δ MAE vs default (s)", justify="right")
            for row in rows:
                if not row.metrics_by_predictor:
                    sens_table.add_row(row.detector_id, "—", "—", "0", "—")
                    continue
                for predictor_id, m in row.metrics_by_predictor.items():
                    delta = row.delta_mae_vs_default
                    delta_str = "nan" if math.isnan(delta) else f"{delta:.1f}"
                    sens_table.add_row(
                        row.detector_id,
                        predictor_id,
                        f"{m.mae:.1f}",
                        str(m.count),
                        delta_str,
                    )
            console.print(sens_table)


# ── walk-forward ──────────────────────────────────────────────────────────────


@app.command("walk-forward")
def walk_forward_cmd(
    out: Path = typer.Argument(..., help="Output JSON path for the WalkForwardResult"),  # noqa: B008
    source: str = typer.Option(..., "--source", help="Named source (e.g. mbta, bucr)"),
    replay_vp: str = typer.Option(
        ..., "--replay-vp", help="Archived VP parquet path (dir/file/s3://)"
    ),
    replay_tu: str | None = typer.Option(
        None, "--replay-tu", help="Archived TU parquet path (dir/file/s3://); optional"
    ),
    replay_start: str = typer.Option(
        ..., "--replay-start", help="Backtest window start (epoch seconds or ISO 8601)"
    ),
    replay_end: str = typer.Option(
        ..., "--replay-end", help="Backtest window end (epoch seconds or ISO 8601)"
    ),
    replay_gtfs: Path | None = typer.Option(  # noqa: B008
        None, "--replay-gtfs", help="Static GTFS zip to use for replay"
    ),
    replay_gtfs_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--replay-gtfs-dir",
        help="Directory of dated GTFS zips (YYYY-MM-DD.zip); newest <= first tick is used",
    ),
    route: list[str] | None = typer.Option(  # noqa: B008
        None, "--route", help="Filter to route ID(s); repeat for multiple"
    ),
    predictor: list[str] | None = typer.Option(  # noqa: B008
        None, "--predictor", help="Predictor ID(s); repeat for multiple (default: both)"
    ),
    windows: int | None = typer.Option(
        None, "--windows", help="Split into this many equal-count windows"
    ),
    width: str | None = typer.Option(
        None, "--width", help="Split into fixed-width windows, e.g. 1h/30m/90s"
    ),
    seed: int = typer.Option(0, "--seed", help="Shared run seed for every window"),
) -> None:
    """Walk-forward backtest: per-window MAE + CI over an archived replay window.

    Splits ``[--replay-start, --replay-end]`` into windows (``--windows`` for
    an equal COUNT, or ``--width`` for a fixed WIDTH — exactly one required),
    runs a fresh replay per window with the SAME ``--seed``, and writes a
    :class:`~stoptimes.val.backtest.WalkForwardResult` (per-window MAE+CI, the
    cross-window aggregate, provenance, and the M6 split_note) to *out*.
    """
    if replay_gtfs is None and replay_gtfs_dir is None:
        typer.echo(
            "Error: --replay-gtfs or --replay-gtfs-dir is required for walk-forward", err=True
        )
        raise typer.Exit(code=1)
    if (windows is None) == (width is None):
        typer.echo("Error: exactly one of --windows or --width is required", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = load_named_source(source)
    except FileNotFoundError:
        typer.echo(
            f"Error: source {source!r} not found. Run 'etaval sources' to list available.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    predictor_ids: list[str] = list(predictor) if predictor else DEFAULT_PREDICTOR_IDS
    unknown = [pid for pid in predictor_ids if pid not in PREDICTOR_REGISTRY]
    if unknown:
        typer.echo(f"Error: unknown predictor IDs: {unknown!r}", err=True)
        typer.echo(f"Known predictor IDs: {sorted(PREDICTOR_REGISTRY)}", err=True)
        raise typer.Exit(code=1)

    try:
        width_s = parse_duration(width) if width is not None else None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Walk-forward: source={source!r}, vp={replay_vp!r}, "
        f"predictors={predictor_ids}, windows={windows}, width={width}"
    )

    try:
        result = asyncio.run(
            walk_forward(
                cfg,
                vp_path=replay_vp,
                tu_path=replay_tu,
                start=replay_start,
                end=replay_end,
                predictor_ids=predictor_ids,
                n_windows=windows,
                width=width_s,
                seed=seed,
                routes=list(route) if route else None,
                static_gtfs_zip=replay_gtfs,
                static_gtfs_dir=replay_gtfs_dir,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error running walk-forward: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    written = result.write_json(out)
    typer.echo(f"Wrote walk-forward result ({len(result.windows)} windows) → {written}")


# ── figures ───────────────────────────────────────────────────────────────────


@app.command("figures")
def figures_cmd(
    parquet: Path = typer.Argument(..., help="Parquet file of ValidationResults"),  # noqa: B008
    out_dir: Path = typer.Option(  # noqa: B008
        ..., "--out-dir", help="Directory to write PNG figures into"
    ),
    summary: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary",
        help="<...>.summary.json sidecar; enables the cost/accuracy Pareto figure",
    ),
) -> None:
    """Render publication figures (300 dpi PNG) from a Parquet results file."""
    if not parquet.exists():
        typer.echo(f"Error: file not found: {parquet}", err=True)
        raise typer.Exit(code=1)
    if summary is not None and not summary.exists():
        typer.echo(f"Error: summary file not found: {summary}", err=True)
        raise typer.Exit(code=1)

    written = render_all_figures(parquet, out_dir, summary_path=summary)
    for name, path in written.items():
        if path is not None:
            typer.echo(f"Wrote {name} → {path}")
        else:
            typer.echo(f"Skipped {name} (no data)")


if __name__ == "__main__":
    app()
