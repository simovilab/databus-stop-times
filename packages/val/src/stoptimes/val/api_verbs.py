"""Python API: `stoptimes val replay`/`report` as functions returning data.

Named `api_verbs.py`, not `api.py` -- `stoptimes.val.api` already exists and
is the web API (REST + WebSocket) behind `stoptimes val api`; this module is
the notebook-facing counterpart Phase 3 track U4 owns, unrelated to it.

Both verbs here are deterministic and offline: `replay_score` scores a
recorded archive (no live network, no wall-clock dependency), and
`score_report` turns the resulting rows into break-down metrics. Together
they are the "replay-score" step of the end-to-end chain in
`examples/notebooks/01_end_to_end.ipynb`.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
from stoptimes.core.domain.models import ValidationResult
from stoptimes.val.engine.assembly import DEFAULT_PREDICTOR_IDS, build_replay_run
from stoptimes.val.engine.runtime import run_engine
from stoptimes.val.feeds.source_config import load_named_source
from stoptimes.val.scoring.metrics import MetricsReport, compute_metrics
from stoptimes.val.storage.parquet_export import load_results_parquet
from stoptimes.val.storage.sqlite import SqliteStorage

__all__ = ["replay_score", "score_report"]

_T = TypeVar("_T")


def _run_in_thread_if_loop_running(fn: Callable[[], _T]) -> _T:
    """Call `fn()` directly, unless an event loop is already running on this
    thread -- Jupyter/IPython kernels run one, plain scripts don't -- in
    which case `fn` (which itself calls `asyncio.run(...)`) is run on a
    worker thread instead.

    `asyncio.run()` raises `RuntimeError: asyncio.run() cannot be called
    from a running event loop` inside a notebook cell; running `fn` on a
    separate thread gives it a thread with no running loop to call
    `asyncio.run()` from, so `replay_score` works identically from a script
    or a notebook cell. `fn` must create anything thread-affine (e.g. a
    `sqlite3` connection) inside itself, not before this call.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()


def replay_score(
    *,
    source: str,
    archive: str | Path,
    static: str | Path | None = None,
    static_dir: str | Path | None = None,
    archive_tu: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    routes: list[str] | None = None,
    predictor_ids: list[str] | None = None,
    sources_dir: str | Path | None = None,
    seed: int | None = None,
    out: str | Path | None = None,
) -> pd.DataFrame:
    """Score prediction(s) from a recorded archive against what happened.

    Equivalent to `stoptimes val replay`. Deterministic: the same archive,
    window, and predictor list always produce the same scores.

    Args:
        source: a named source (`stoptimes val sources` / an
            `examples/agencies/<source>.yaml` file).
        archive: archived VehiclePositions parquet path (dir/file/`s3://`),
            the Hive-partitioned layout `stoptimes.record` writes.
        static: a static GTFS `.zip` to use for the whole replay.
        static_dir: a directory of dated `YYYY-MM-DD.zip` snapshots; the
            newest one on/before the first replay tick's date is used.
            Give exactly one of `static`/`static_dir`.
        archive_tu: archived TripUpdates parquet path; only needed for the
            `gtfsrt` predictor.
        start, end: replay window bounds (`YYYY-MM-DD` or full ISO 8601).
        routes: restrict to these route IDs; `None` means all.
        predictor_ids: predictor IDs to score, e.g. `["baseline:schedule",
            "stoptimes:<artifact-id>"]`; defaults to
            `stoptimes.val.engine.assembly.DEFAULT_PREDICTOR_IDS`.
        sources_dir: agency profile directory to search exclusively.
        seed: run seed, recorded in provenance.
        out: also export the scored rows to this parquet path.

    Returns:
        One row per scored prediction (`ValidationResult` fields as
        columns) -- the same rows `stoptimes val replay --out FILE` writes.

    Example:
        >>> from stoptimes.val.api_verbs import replay_score  # doctest: +SKIP
        >>> scored = replay_score(
        ...     source="demo", archive="archive/", static="gtfs.zip",
        ...     start="2026-09-17", end="2026-09-18",
        ...     predictor_ids=["baseline:schedule"],
        ... )  # doctest: +SKIP
        >>> len(scored) > 0  # doctest: +SKIP
        True

    Raises:
        FileNotFoundError: `source` isn't a known/configured source, or a
            required static snapshot is missing.
        ValueError: neither `static` nor `static_dir` was given, or no VP
            rows matched the archive/window/route filter.
    """
    cfg = load_named_source(source, config_dir=sources_dir)
    ids = list(predictor_ids) if predictor_ids else list(DEFAULT_PREDICTOR_IDS)

    deps = build_replay_run(
        cfg,
        vp_path=archive,
        tu_path=archive_tu,
        start=start,
        end=end,
        routes=routes,
        predictor_ids=ids,
        static_gtfs_zip=static,
        static_gtfs_dir=static_dir,
    )

    def _run_and_load() -> list[ValidationResult]:
        # The SQLite connection and the event loop that drives it must live
        # in the same thread (sqlite3 objects aren't cross-thread-safe), so
        # both are created here, inside whichever thread this function ends
        # up running on -- see `_run_async`.
        with SqliteStorage(":memory:") as storage:
            asyncio.run(
                run_engine(
                    deps.feed_source,
                    deps.static_loader,
                    deps.predictors,
                    deps.detectors,
                    storage,
                    primary_method="shape_distance",
                    source_name=source,
                    seed=seed,
                )
            )
            return storage.load_results()

    results = _run_in_thread_if_loop_running(_run_and_load)

    frame = pd.DataFrame([r.model_dump() for r in results])
    if out is not None:
        from stoptimes.val.storage.parquet_export import export_results

        export_results(results, out)
    return frame


def score_report(results: pd.DataFrame | str | Path, *, tz: str = "UTC") -> MetricsReport:
    """Break scored predictions down by predictor, route, horizon, and time of day.

    Equivalent to the metrics section of `stoptimes val report`.

    Args:
        results: either the DataFrame `replay_score` returns, or a path to a
            parquet file written by `replay_score(..., out=...)` /
            `stoptimes val replay --out`.
        tz: IANA timezone used to bucket `time_of_day` metrics.

    Returns:
        A `stoptimes.val.scoring.metrics.MetricsReport` (per-predictor,
        per-route, per-horizon-bucket, per-time-of-day breakdowns, plus
        win-rate and convergence series).

    Example:
        >>> from stoptimes.val.api_verbs import replay_score, score_report  # doctest: +SKIP
        >>> scored = replay_score(source="demo", archive="archive/", static="gtfs.zip")  # doctest: +SKIP
        >>> report = score_report(scored)  # doctest: +SKIP
        >>> report.per_predictor  # doctest: +SKIP
    """
    if isinstance(results, (str, Path)):
        rows: list[ValidationResult] = load_results_parquet(results)
    else:
        rows = [ValidationResult.model_validate(r) for r in results.to_dict("records")]
    return compute_metrics(rows, tz=tz)
