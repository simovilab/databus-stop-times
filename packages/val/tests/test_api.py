"""test_api.py — offline REST API tests using httpx.AsyncClient + ASGITransport.

Injects a RunManager backed by ReplayFeedSource so no live HTTP is needed.
The replay builder reuses the snapshot helper from test_engine_replay.py.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from stoptimes.val.api.app import create_app
from stoptimes.val.api.runs import RunManager
from stoptimes.val.engine.assembly import (
    DEFAULT_PREDICTOR_IDS,
    LiveRunDeps,
    build_detectors,
    build_predictors,
)
from stoptimes.val.engine.runtime import ReplayFeedSource
from stoptimes.core.gtfs.static import GtfsStaticLoader

# Reuse the snapshot builder from the engine replay test.
from tests.test_engine_replay import _build_snapshots

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"


# ── Replay deps builder (no HTTP) ─────────────────────────────────────────────


def _replay_deps_builder(cfg, predictor_ids, routes) -> LiveRunDeps:
    """Offline replacement for build_live_run — uses ReplayFeedSource.

    This is the primary test injection seam: the RunManager calls
    ``await asyncio.to_thread(self._deps_builder, cfg, ...)`` for every run,
    so this function executes in a thread pool (safe, all pure Python / stdlib).
    """
    snapshots = _build_snapshots()
    feed_source = ReplayFeedSource(snapshots)
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    ids = predictor_ids if predictor_ids else DEFAULT_PREDICTOR_IDS
    predictors = build_predictors(ids)
    detectors = build_detectors(cfg)

    # Create a throwaway temp file so the run's os.unlink(gtfs_path) call
    # succeeds without error (matches the contract from build_live_run).
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()

    return LiveRunDeps(  # type: ignore[arg-type]
        feed_source=feed_source,  # type: ignore[arg-type]
        static_loader=static_loader,
        predictors=predictors,
        detectors=detectors,
        gtfs_path=Path(tmp.name),
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "etaval_data"


@pytest.fixture
def run_manager(tmp_data_dir: Path) -> RunManager:
    return RunManager(data_dir=tmp_data_dir, deps_builder=_replay_deps_builder)


@pytest.fixture
async def client(run_manager: RunManager):
    app = create_app(run_manager=run_manager)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ── Poll helper ────────────────────────────────────────────────────────────────


async def _wait_done(
    client: httpx.AsyncClient,
    run_id: str,
    timeout: float = 30.0,
) -> dict:
    """Poll GET /api/runs/{run_id} until status is 'done' or 'error'."""
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get(f"/api/runs/{run_id}")
        data = resp.json()
        if data["status"] in ("done", "error"):
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Run {run_id} did not finish within {timeout}s; "
                f"last status={data['status']!r}"
            )
        await asyncio.sleep(0.05)


# ── Basic endpoint smoke tests ─────────────────────────────────────────────────


async def test_health(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_sources_contains_expected(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/sources")
    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()}
    assert "mbta" in names
    assert "bucr" in names


async def test_predictors_contains_expected(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/predictors")
    assert resp.status_code == 200
    ids = resp.json()
    assert "gtfsrt" in ids
    assert "baseline:constant_speed" in ids


# ── Full run lifecycle ─────────────────────────────────────────────────────────


async def test_post_runs_returns_run_id(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert body["run_id"]


async def test_run_completes_with_summary(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]

    data = await _wait_done(client, run_id)
    assert data["status"] == "done", f"Run ended in error: {data.get('error')}"

    summary = data["summary"]
    assert summary["ticks"] > 0
    assert summary["samples_saved"] > 0
    assert summary["arrivals_saved"] > 0
    assert summary["results_saved"] > 0


async def test_get_run_returns_state(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    resp = await client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id


# ── Metrics endpoint ───────────────────────────────────────────────────────────


async def test_metrics_has_per_predictor_mae(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    resp = await client.get(f"/api/runs/{run_id}/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "per_predictor" in body
    assert len(body["per_predictor"]) > 0
    for row in body["per_predictor"]:
        assert "mae" in row
        assert row["mae"] >= 0.0
        assert "count" in row
        assert row["count"] > 0


# ── Results endpoint ───────────────────────────────────────────────────────────


async def test_results_returns_rows(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    resp = await client.get(f"/api/runs/{run_id}/results")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) > 0
    for row in rows:
        assert "trip_key" in row
        assert "predictor_id" in row
        assert "error_s" in row


async def test_results_predictor_filter(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    resp = await client.get(f"/api/runs/{run_id}/results?predictor=gtfsrt")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) > 0
    assert all(r["predictor_id"] == "gtfsrt" for r in rows)


async def test_results_predictor_filter_baseline(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    pid = "baseline:constant_speed"
    resp = await client.get(f"/api/runs/{run_id}/results?predictor={pid}")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) > 0
    assert all(r["predictor_id"] == pid for r in rows)


# ── List and delete ────────────────────────────────────────────────────────────


async def test_list_runs_includes_completed(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    resp = await client.get("/api/runs")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()]
    assert run_id in ids


async def test_delete_removes_run(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 200

    resp = await client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 404


# ── Provenance (A7) ──────────────────────────────────────────────────────────


async def test_run_with_seed_exposes_provenance(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/runs", json={"source": "mbta", "duration": "1m", "seed": 42}
    )
    run_id = resp.json()["run_id"]
    data = await _wait_done(client, run_id)

    assert data["provenance"] is not None
    assert data["provenance"]["seed"] == 42
    assert data["provenance"]["source_name"] == "mbta"
    assert "shape_distance" in data["provenance"]["detectors"]


async def test_run_without_seed_still_has_provenance(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    data = await _wait_done(client, run_id)

    assert data["provenance"] is not None
    assert data["provenance"]["seed"] is None


async def test_provenance_sidecar_written_to_run_dir(
    client: httpx.AsyncClient, run_manager: RunManager
) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    state = run_manager.get_run(run_id)
    assert state is not None
    sidecar = Path(f"{state.db_path}.provenance.json")
    assert sidecar.is_file()


async def test_delete_run_removes_provenance_sidecar(
    client: httpx.AsyncClient, run_manager: RunManager
) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)

    state = run_manager.get_run(run_id)
    assert state is not None
    sidecar = Path(f"{state.db_path}.provenance.json")
    assert sidecar.is_file()

    resp = await client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    assert not sidecar.exists()


# ── Error handling ─────────────────────────────────────────────────────────────


async def test_bad_source_returns_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/runs", json={"source": "nonexistent_source_xyz", "duration": "1m"}
    )
    assert resp.status_code == 400


async def test_bad_predictor_returns_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/runs",
        json={"source": "mbta", "predictors": ["no_such_predictor"], "duration": "1m"},
    )
    assert resp.status_code == 400


async def test_bad_duration_returns_422(client: httpx.AsyncClient) -> None:
    """Malformed duration triggers Pydantic validation → 422."""
    resp = await client.post(
        "/api/runs", json={"source": "mbta", "duration": "notaduration!!"}
    )
    assert resp.status_code == 422


async def test_unknown_run_id_get_returns_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/runs/no-such-id-999")
    assert resp.status_code == 404


async def test_unknown_run_id_metrics_returns_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/runs/no-such-id-999/metrics")
    assert resp.status_code == 404


async def test_unknown_run_id_delete_returns_404(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/api/runs/no-such-id-999")
    assert resp.status_code == 404


# ── Persistence round-trip ─────────────────────────────────────────────────────


async def test_persistence_across_restarts(tmp_data_dir: Path) -> None:
    """A new RunManager on the same data_dir sees runs from the previous manager."""
    mgr1 = RunManager(data_dir=tmp_data_dir, deps_builder=_replay_deps_builder)
    app1 = create_app(run_manager=mgr1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app1),
        base_url="http://test",
    ) as c1:
        resp = await c1.post("/api/runs", json={"source": "mbta", "duration": "1m"})
        run_id = resp.json()["run_id"]
        await _wait_done(c1, run_id)

    # Simulate restart: fresh RunManager, same data_dir.
    mgr2 = RunManager(data_dir=tmp_data_dir, deps_builder=_replay_deps_builder)
    run_ids = {s.id for s in mgr2.list_runs()}
    assert run_id in run_ids, "Completed run should survive RunManager restart"

    state = mgr2.get_run(run_id)
    assert state is not None
    assert state.status == "done"
    assert state.summary is not None
    assert state.summary.results_saved > 0
