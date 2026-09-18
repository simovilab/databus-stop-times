"""RunManager — async background run lifecycle for the etaval REST API.

Phase A: REST management only.
Phase B seam: subscribe / unsubscribe / _publish are wired and ready for
WebSocket consumers but not yet called by any REST route.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from stoptimes.val.api.stats_ext import compute_stats_extension
from stoptimes.val.api.telemetry import TelemetryCollector
from stoptimes.core.domain.models import ValidationResult
from stoptimes.val.engine.assembly import LiveRunDeps, build_live_run
from stoptimes.val.engine.runtime import RunSummary, TickEvent, run_engine
from stoptimes.val.feeds.source_config import load_named_source
from stoptimes.val.provenance import RunProvenance, build_run_provenance, write_provenance_sidecar
from stoptimes.val.storage.sqlite import SqliteStorage


@dataclass
class RunState:
    """Live + persisted state for one validation run."""

    id: str
    params: dict
    status: Literal["pending", "running", "done", "error"]
    db_path: str
    started_at: float
    summary: RunSummary | None = None
    provenance: RunProvenance | None = None
    ended_at: float | None = None
    error: str | None = None


class RunManager:
    """Manages the lifecycle of background validation runs.

    Parameters
    ----------
    data_dir:
        Root directory for run data.  Per-run SQLite files land under
        ``<data_dir>/runs/``.  An ``index.json`` in the same directory
        persists metadata across restarts.
    deps_builder:
        Sync callable ``(cfg, predictor_ids, routes) -> LiveRunDeps``.
        Defaults to :func:`build_live_run` (live HTTP).  Pass a replay
        builder in tests to avoid any network calls.

    Phase B note
    ------------
    ``subscribe`` / ``unsubscribe`` / ``_publish`` implement a per-run
    asyncio.Queue broadcast channel.  No REST route consumes these in
    Phase A; Phase B's WebSocket handler will call ``subscribe`` on
    connect and ``unsubscribe`` on disconnect.
    """

    def __init__(
        self,
        data_dir: Path,
        deps_builder: Callable[..., LiveRunDeps] = build_live_run,
        telemetry: TelemetryCollector | None = None,
        sources_dir: str | Path | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._runs_dir = self._data_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._runs_dir / "index.json"
        self._deps_builder = deps_builder
        self._sources_dir = sources_dir
        self.telemetry = telemetry if telemetry is not None else TelemetryCollector()

        # In-memory run states (merged with persisted on init).
        self._runs: dict[str, RunState] = {}
        # Background asyncio tasks keyed by run_id.
        self._tasks: dict[str, asyncio.Task] = {}
        # Phase B broadcast: run_id → list of subscriber queues.
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        # ?stats=1 augmentation, computed once per run and cached in memory.
        self._stats_cache: dict[str, dict[str, Any]] = {}

        self._load_index()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Populate _runs from index.json (survives restarts)."""
        if not self._index_path.exists():
            return
        with self._index_path.open() as fh:
            raw: dict = json.load(fh)
        for run_id, meta in raw.items():
            summary_raw = meta.get("summary")
            summary = (
                RunSummary(
                    ticks=summary_raw.get("ticks", 0),
                    samples_saved=summary_raw.get("samples_saved", 0),
                    arrivals_saved=summary_raw.get("arrivals_saved", 0),
                    results_saved=summary_raw.get("results_saved", 0),
                    predictor_errors=summary_raw.get("predictor_errors", {}),
                    detector_errors=summary_raw.get("detector_errors", {}),
                )
                if summary_raw is not None
                else None
            )
            provenance_raw = meta.get("provenance")
            provenance = (
                RunProvenance.from_dict(provenance_raw) if provenance_raw is not None else None
            )
            self._runs[run_id] = RunState(
                id=run_id,
                params=meta["params"],
                status=meta["status"],
                db_path=meta["db_path"],
                started_at=meta["started_at"],
                summary=summary,
                provenance=provenance,
                ended_at=meta.get("ended_at"),
                error=meta.get("error"),
            )

    def _persist(self) -> None:
        """Atomically write current run states to index.json."""
        index: dict[str, Any] = {}
        for run_id, state in self._runs.items():
            summary_dict: dict | None = None
            if state.summary is not None:
                summary_dict = state.summary.to_dict()
            provenance_dict: dict | None = None
            if state.provenance is not None:
                provenance_dict = state.provenance.to_dict()
            index[run_id] = {
                "params": state.params,
                "status": state.status,
                "summary": summary_dict,
                "provenance": provenance_dict,
                "db_path": state.db_path,
                "started_at": state.started_at,
                "ended_at": state.ended_at,
                "error": state.error,
            }
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(self._index_path)

    # ── Phase B broadcast seam ─────────────────────────────────────────────────

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """Register a new subscriber queue for a run.

        Phase B WebSocket handler calls this on connect.  Each subscriber
        receives a copy of every :class:`TickEvent` published during the run.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        """Remove a subscriber queue (called on WebSocket disconnect)."""
        subs = self._subscribers.get(run_id, [])
        try:
            subs.remove(q)
        except ValueError:
            pass

    async def _publish(self, run_id: str, event: TickEvent) -> None:
        """Publish a TickEvent to all current subscribers for this run."""
        for q in list(self._subscribers.get(run_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow consumer — drop silently

    def _signal_end(self, run_id: str) -> None:
        """Put a ``None`` sentinel into every subscriber queue for *run_id*.

        Called at the end of :meth:`_run_background` so WebSocket handlers
        can detect run termination without relying solely on polling.  If a
        queue is full (slow subscriber), one item is dropped to make room;
        handlers also carry a timeout-based fallback for resilience.
        """
        for q in list(self._subscribers.get(run_id, [])):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass  # handler's timeout-based fallback detects terminal state

    # ── on_tick factory ────────────────────────────────────────────────────────

    def _make_on_tick(self, run_id: str) -> Callable:
        """Return an async on_tick callback bound to *run_id*.

        Updates the in-memory RunState counts every tick and publishes the
        TickEvent to Phase B subscribers.
        """

        async def on_tick(event: TickEvent) -> None:
            state = self._runs.get(run_id)
            if state is not None:
                state.summary = RunSummary(
                    ticks=event.tick,
                    samples_saved=event.samples_saved,
                    arrivals_saved=event.arrivals_saved,
                    results_saved=event.results_saved,
                    predictor_errors=dict(event.predictor_errors),
                    detector_errors=dict(event.detector_errors),
                )
            self.telemetry.record_tick(run_id)
            self.telemetry.record_feed_poll(run_id, ok=True)
            await self._publish(run_id, event)

        return on_tick

    def _make_on_predictor_error(self, run_id: str) -> Callable:
        """Return an async on_predictor_error callback bound to *run_id*.

        Records the exception in telemetry and logs a "warn" event, so a
        predictor exception (which the engine otherwise swallows silently)
        is visible via GET /api/telemetry and /api/telemetry/events.
        """

        async def on_predictor_error(predictor_id: str, exc: BaseException) -> None:
            self.telemetry.record_predictor_error(run_id, predictor_id)
            self.telemetry.log_event(
                "warn",
                source=f"predictor:{predictor_id}",
                message=f"{type(exc).__name__}: {exc}",
                run_id=run_id,
            )

        return on_predictor_error

    # ── Run lifecycle ──────────────────────────────────────────────────────────

    async def start_run(
        self,
        source: str,
        predictor_ids: list[str],
        routes: list[str],
        duration_s: float,
        seed: int | None = None,
    ) -> str:
        """Create and launch a new background validation run.

        Returns the new run_id immediately; the run executes in a background
        asyncio task.
        """
        cfg = load_named_source(source, config_dir=self._sources_dir)
        run_id = str(uuid.uuid4())
        db_path = str(self._runs_dir / f"{run_id}.sqlite")

        state = RunState(
            id=run_id,
            params={
                "source": source,
                "predictors": predictor_ids,
                "routes": routes,
                "duration_s": duration_s,
                "seed": seed,
            },
            status="pending",
            db_path=db_path,
            started_at=time.time(),
        )
        self._runs[run_id] = state
        self._persist()

        task = asyncio.create_task(
            self._run_background(run_id, cfg, predictor_ids, routes, duration_s, seed)
        )
        self._tasks[run_id] = task
        return run_id

    async def _run_background(
        self,
        run_id: str,
        cfg: Any,
        predictor_ids: list[str],
        routes: list[str],
        duration_s: float,
        seed: int | None = None,
    ) -> None:
        """Background task: build deps, run engine, update state on completion."""
        state = self._runs[run_id]
        gtfs_path = None
        storage = None
        try:
            deps: LiveRunDeps = await asyncio.to_thread(
                self._deps_builder,
                cfg,
                predictor_ids or None,
                routes or None,
            )
            gtfs_path = deps.gtfs_path
            storage = SqliteStorage(state.db_path)

            # Build provenance BEFORE the engine starts and persist it right
            # away — so GET /api/runs/{id} exposes it even while "running".
            provenance = build_run_provenance(
                source_name=cfg.name,
                source_config=cfg,
                detectors=deps.detectors,
                predictor_ids=predictor_ids or list(predictor_ids),
                seed=seed,
            )
            state.provenance = provenance
            try:
                write_provenance_sidecar(state.db_path, provenance)
            except Exception:  # noqa: BLE001
                pass  # sidecar is best-effort; never abort the run over it

            state.status = "running"
            self._persist()

            summary = await run_engine(
                feed_source=deps.feed_source,  # type: ignore[arg-type]
                static_loader=deps.static_loader,
                predictors=deps.predictors,
                detectors=deps.detectors,
                storage=storage,
                duration_s=duration_s,
                poll_interval_s=cfg.poll_interval_s,
                source_name=cfg.name,
                on_tick=self._make_on_tick(run_id),
                on_predictor_error=self._make_on_predictor_error(run_id),
                seed=seed,
                provenance=provenance,
            )

            state.summary = summary
            state.status = "done"
            state.ended_at = time.time()

        except Exception as exc:  # noqa: BLE001
            state.status = "error"
            state.error = str(exc)
            state.ended_at = time.time()
            self.telemetry.record_feed_poll(run_id, ok=False)
            self.telemetry.log_event(
                "error",
                source="run",
                message=f"{type(exc).__name__}: {exc}",
                run_id=run_id,
            )

        finally:
            if storage is not None:
                try:
                    storage.close()
                except Exception:  # noqa: BLE001
                    pass
            if gtfs_path is not None:
                try:
                    os.unlink(gtfs_path)
                except OSError:
                    pass
            self._persist()
            # Signal Phase B WebSocket subscribers that this run has ended.
            self._signal_end(run_id)

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> RunState | None:
        """Return the RunState for *run_id*, or None if unknown."""
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunState]:
        """Return all known RunStates (live + persisted), newest first."""
        return sorted(self._runs.values(), key=lambda s: s.started_at, reverse=True)

    def get_stats_extension(
        self, run_id: str, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """Return the cached ``?stats=1`` augmentation for *run_id*, computing it once.

        Callers are responsible for only invoking this for DONE runs — the
        computation itself is a pure function of the run's persisted results
        (plus its summary's inference-latency block) and is deterministic
        given the pinned seed, so it is safe to cache indefinitely in memory.
        """
        cached = self._stats_cache.get(run_id)
        if cached is not None:
            return cached
        state = self._runs.get(run_id)
        summary = state.summary if state is not None else None
        ext = compute_stats_extension(results, summary)
        self._stats_cache[run_id] = ext
        return ext

    async def delete_run(self, run_id: str) -> bool:
        """Cancel running task (if any) and remove the run from the index.

        Also removes the per-run SQLite database if it exists.

        Returns True if the run existed, False if not found.
        """
        if run_id not in self._runs:
            return False

        task = self._tasks.pop(run_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        state = self._runs.pop(run_id)
        try:
            os.unlink(state.db_path)
        except OSError:
            pass
        try:
            os.unlink(f"{state.db_path}.provenance.json")
        except OSError:
            pass

        self._subscribers.pop(run_id, None)
        self._stats_cache.pop(run_id, None)
        self._persist()
        return True
