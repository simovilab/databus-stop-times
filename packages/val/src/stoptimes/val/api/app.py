"""FastAPI application factory and uvicorn runner for the etaval REST API.

Phase A: REST-only.  WebSocket streaming (Phase B) will be mounted as an
additional router on the same app — no structural changes needed here.

Entry points
------------
create_app(run_manager=None) -> FastAPI
    Build and configure the FastAPI application.  Accepts an injected
    RunManager so tests can pass a replay-backed manager without touching
    the HTTP stack.

main()
    Console script entry point — reads host/port from get_settings() and
    launches uvicorn with the factory pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from stoptimes.val.api.config import get_settings
from stoptimes.val.api.runs import RunManager, RunState
from stoptimes.val.api.schemas import MetricsResponse, ResultRow, RunInfo, RunRequest
from stoptimes.val.api.ws import router as ws_router
from stoptimes.val.engine.assembly import (
    DEFAULT_PREDICTOR_IDS,
    PREDICTOR_REGISTRY,
    is_known_predictor_id,
    parse_duration,
)
from stoptimes.val.feeds.source_config import list_sources, load_named_source
from stoptimes.val.scoring.metrics import compute_metrics
from stoptimes.val.storage.sqlite import SqliteStorage

# ── Helpers ────────────────────────────────────────────────────────────────────


def _state_to_info(state: RunState) -> RunInfo:
    """Convert a RunState to the RunInfo response schema."""
    summary_dict: dict | None = None
    if state.summary is not None:
        summary_dict = state.summary.to_dict()
    provenance_dict: dict | None = None
    if state.provenance is not None:
        provenance_dict = state.provenance.to_dict()
    return RunInfo(
        id=state.id,
        params=state.params,
        status=state.status,
        summary=summary_dict,
        provenance=provenance_dict,
        db_path=state.db_path,
        started_at=state.started_at,
        ended_at=state.ended_at,
        error=state.error,
    )


# ── Application factory ────────────────────────────────────────────────────────


def create_app(
    run_manager: RunManager | None = None,
    sources_dir: str | Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        run_manager: Optional pre-constructed RunManager.  When None the
            manager is built from the current settings data_dir.  Inject
            a replay-backed manager for offline tests.
        sources_dir: Directory of agency source-config YAMLs, searched
            exclusively (passed through to ``list_sources``/``load_named_source``
            as ``config_dir``). Defaults to ``get_settings().sources_dir``
            (``ETAVAL_SOURCES_DIR``), which itself defaults to ``None`` — i.e.
            the normal ``sources_search_path()`` order. Pass this explicitly
            (rather than relying on the process's current working directory)
            in tests and in any deployment that doesn't run from the repo root.

    Returns:
        A configured FastAPI instance ready to serve requests.
    """
    settings = get_settings()
    resolved_sources_dir = sources_dir if sources_dir is not None else settings.sources_dir

    app = FastAPI(
        title="stoptimes val REST API",
        description="ETA Prediction Validation — Phase A REST layer",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if run_manager is None:
        run_manager = RunManager(data_dir=Path(settings.data_dir), sources_dir=resolved_sources_dir)

    # Expose on app.state so middleware / lifespan hooks can reach it.
    app.state.run_manager = run_manager

    # ── WebSocket routes (Phase B) ─────────────────────────────────────────────
    app.include_router(ws_router)

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health() -> dict:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/api/sources")
    async def sources() -> list[dict]:
        """List all configured GTFS-RT sources."""
        result = []
        for name in list_sources(config_dir=resolved_sources_dir):
            try:
                cfg = load_named_source(name, config_dir=resolved_sources_dir)
            except Exception:  # noqa: BLE001
                continue
            result.append(
                {
                    "name": cfg.name,
                    "poll_interval_s": cfg.poll_interval_s,
                    "vehicle_positions_url": cfg.vehicle_positions_url,
                    "trip_updates_url": cfg.trip_updates_url,
                    "gtfs_static_url": cfg.gtfs_static_url,
                }
            )
        return result

    @app.get("/api/predictors")
    async def predictors() -> list[str]:
        """Return the IDs of all registered predictors."""
        return list(PREDICTOR_REGISTRY.keys())

    @app.post("/api/runs")
    async def create_run(body: RunRequest) -> dict:
        """Start a new validation run.  Returns ``{run_id}`` immediately."""
        known_sources = list_sources(config_dir=resolved_sources_dir)
        if body.source not in known_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source {body.source!r}. Known: {known_sources}",
            )

        bad_pids = [p for p in body.predictors if not is_known_predictor_id(p)]
        if bad_pids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown predictor(s): {bad_pids}. Known: {list(PREDICTOR_REGISTRY)}",
            )

        duration_s = parse_duration(body.duration)
        predictor_ids = body.predictors if body.predictors else list(DEFAULT_PREDICTOR_IDS)

        run_id = await run_manager.start_run(
            source=body.source,
            predictor_ids=predictor_ids,
            routes=body.routes,
            duration_s=duration_s,
            seed=body.seed,
        )
        return {"run_id": run_id}

    @app.get("/api/runs")
    async def list_runs() -> list[RunInfo]:
        """Return all known runs (live + persisted from index.json)."""
        return [_state_to_info(s) for s in run_manager.list_runs()]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> RunInfo:
        """Return the current state of a run."""
        state = run_manager.get_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        return _state_to_info(state)

    @app.get("/api/runs/{run_id}/metrics", response_model_exclude_unset=True)
    async def get_metrics(
        run_id: str,
        stats: Annotated[int, Query()] = 0,
    ) -> MetricsResponse:
        """Compute and return the full metrics report for a completed run.

        With ``?stats=1`` (only valid for DONE runs), the response is
        augmented with a per-predictor ``mae_ci``/``latency`` and top-level
        ``significance``/``calibration``/``seed`` — see
        :mod:`stoptimes.val.api.stats_ext` for the full contract. The augmentation
        is computed once per run and cached on the RunManager. Without
        ``stats``, the response is byte-identical to the pre-stats contract
        (``response_model_exclude_unset=True`` keeps the three new top-level
        fields out of the payload entirely when they were never set).
        """
        state = run_manager.get_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        storage = SqliteStorage(state.db_path)
        try:
            results = storage.load_results()
        finally:
            storage.close()
        report = compute_metrics(results)
        s = report.summary()

        if not stats:
            return MetricsResponse(**s)

        if state.status != "done":
            raise HTTPException(
                status_code=409,
                detail=f"Run {run_id!r} is not done (status={state.status!r}); "
                "?stats=1 requires a completed run",
            )

        ext = run_manager.get_stats_extension(run_id, results)
        per_predictor_ext = ext["per_predictor_ext"]

        per_predictor = [dict(row) for row in s["per_predictor"]]
        for row in per_predictor:
            pred_ext = per_predictor_ext.get(row.get("predictor_id"), {})
            row["mae_ci"] = pred_ext.get("mae_ci")
            row["latency"] = pred_ext.get("latency")

        return MetricsResponse(
            per_predictor=per_predictor,
            horizon_buckets=s["horizon_buckets"],
            convergence=s["convergence"],
            head_to_head=s["head_to_head"],
            significance=ext["significance"],
            calibration=ext["calibration"],
            seed=ext["seed"],
        )

    @app.get("/api/runs/{run_id}/results")
    async def get_results(
        run_id: str,
        predictor: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100_000)] = 1000,
    ) -> list[ResultRow]:
        """Return raw ValidationResult rows for client-side analytics.

        Query params:
            predictor: filter by predictor_id
            limit:     max rows returned (default 1000)
        """
        state = run_manager.get_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        storage = SqliteStorage(state.db_path)
        try:
            results = storage.load_results(predictor_id=predictor)
        finally:
            storage.close()
        return [ResultRow(**r.model_dump()) for r in results[:limit]]

    @app.get("/api/telemetry")
    async def telemetry() -> dict:
        """Process-wide + per-run telemetry snapshot ("no silent failures")."""
        snap = run_manager.telemetry.snapshot()
        states = {s.id: s for s in run_manager.list_runs()}
        per_run = []
        for run_id, rt in snap["per_run"].items():
            state = states.get(run_id)
            per_run.append(
                {
                    "run_id": run_id,
                    "status": state.status if state is not None else None,
                    "ticks": rt["ticks"],
                    "tick_rate": rt["tick_rate"],
                    "last_tick_age_s": rt["last_tick_age_s"],
                    "feed": rt["feed"],
                    "predictor_errors": rt["predictor_errors"],
                }
            )
        return {
            "api_uptime_s": snap["api_uptime_s"],
            "active_runs": sum(1 for s in states.values() if s.status == "running"),
            "per_run": per_run,
        }

    @app.get("/api/telemetry/events")
    async def telemetry_events(
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> list[dict]:
        """Return the *limit* most-recent telemetry events, newest-last."""
        return run_manager.telemetry.events(limit=limit)

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str) -> dict:
        """Cancel a running run (if active) and remove all its data."""
        found = await run_manager.delete_run(run_id)
        if not found:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        return {"deleted": run_id}

    return app


# ── Console script entry point ─────────────────────────────────────────────────


def main() -> None:
    """Launch the uvicorn server using settings from environment variables."""
    settings = get_settings()
    uvicorn.run(
        "stoptimes.val.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
