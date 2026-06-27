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

from etaval.api.config import get_settings
from etaval.api.runs import RunManager, RunState
from etaval.api.schemas import MetricsResponse, ResultRow, RunInfo, RunRequest
from etaval.engine.assembly import DEFAULT_PREDICTOR_IDS, PREDICTOR_REGISTRY, parse_duration
from etaval.feeds.source_config import list_sources, load_named_source
from etaval.scoring.metrics import compute_metrics
from etaval.storage.sqlite import SqliteStorage

# ── Helpers ────────────────────────────────────────────────────────────────────


def _state_to_info(state: RunState) -> RunInfo:
    """Convert a RunState to the RunInfo response schema."""
    summary_dict: dict | None = None
    if state.summary is not None:
        summary_dict = {
            "ticks": state.summary.ticks,
            "samples_saved": state.summary.samples_saved,
            "arrivals_saved": state.summary.arrivals_saved,
            "results_saved": state.summary.results_saved,
        }
    return RunInfo(
        id=state.id,
        params=state.params,
        status=state.status,
        summary=summary_dict,
        db_path=state.db_path,
        started_at=state.started_at,
        ended_at=state.ended_at,
        error=state.error,
    )


# ── Application factory ────────────────────────────────────────────────────────


def create_app(run_manager: RunManager | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        run_manager: Optional pre-constructed RunManager.  When None the
            manager is built from the current settings data_dir.  Inject
            a replay-backed manager for offline tests.

    Returns:
        A configured FastAPI instance ready to serve requests.
    """
    settings = get_settings()

    app = FastAPI(
        title="etaval REST API",
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
        run_manager = RunManager(data_dir=Path(settings.data_dir))

    # Expose on app.state so middleware / lifespan hooks can reach it.
    app.state.run_manager = run_manager

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health() -> dict:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/api/sources")
    async def sources() -> list[dict]:
        """List all configured GTFS-RT sources."""
        result = []
        for name in list_sources():
            try:
                cfg = load_named_source(name)
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
        known_sources = list_sources()
        if body.source not in known_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source {body.source!r}. Known: {known_sources}",
            )

        bad_pids = [p for p in body.predictors if p not in PREDICTOR_REGISTRY]
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

    @app.get("/api/runs/{run_id}/metrics")
    async def get_metrics(run_id: str) -> MetricsResponse:
        """Compute and return the full metrics report for a completed run."""
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
        return MetricsResponse(**s)

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
        "etaval.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
