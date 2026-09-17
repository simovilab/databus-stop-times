"""test_ws.py — offline WebSocket streaming tests.

Uses ``fastapi.testclient.TestClient`` (Starlette's sync test client) which
supports WebSocket connections and runs the ASGI app — including background
asyncio tasks — in its own thread/event loop.  ``httpx.AsyncClient`` +
``ASGITransport`` does *not* handle WebSockets, hence the different client.

The replay deps builder from test_api.py is reused to keep the run entirely
offline (no live HTTP, no real sleeping).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from stoptimes.val.api.app import create_app
from stoptimes.val.api.runs import RunManager
from tests.test_api import _replay_deps_builder  # offline replay builder

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_test_client(tmp_path: Path) -> TestClient:
    mgr = RunManager(data_dir=tmp_path / "data", deps_builder=_replay_deps_builder)
    app = create_app(run_manager=mgr)
    return TestClient(app)


def _collect_frames(ws, *, max_frames: int = 200) -> list[dict]:
    """Read JSON frames from *ws* until an ``end`` frame arrives."""
    frames: list[dict] = []
    for _ in range(max_frames):
        data = ws.receive_json()
        frames.append(data)
        if data["type"] == "end":
            break
    return frames


def _wait_for_run_done(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    """Poll REST until the run is terminal; return the final RunInfo dict."""
    deadline = time.monotonic() + timeout
    while True:
        resp = client.get(f"/api/runs/{run_id}")
        data = resp.json()
        if data["status"] in ("done", "error"):
            return data
        if time.monotonic() > deadline:
            pytest.fail(f"Run {run_id!r} did not finish within {timeout}s")
        time.sleep(0.05)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_ws_stream_run_collects_ticks_and_ends(tmp_path: Path) -> None:
    """Start a run, connect to its stream immediately, collect all frames.

    Asserts:
    - First frame is ``snapshot``.
    - At least one ``tick`` frame was received.
    - Tick numbers are monotonically increasing starting at 1.
    - At least one tick carries ``new_results``.
    - Vehicle lat/lon are finite numbers.
    - Last frame is ``end`` with ``status == "done"``.
    - ``end`` summary counts match the REST GET /api/runs/{id} response.
    """
    with _make_test_client(tmp_path) as client:
        resp = client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
            frames = _collect_frames(ws)

    assert frames, "No frames received"
    assert frames[0]["type"] == "snapshot", "First frame must be snapshot"

    tick_frames = [f for f in frames if f["type"] == "tick"]
    assert len(tick_frames) >= 1, f"Expected ≥1 tick frame; got {len(tick_frames)}"

    # Tick numbers must be monotonically increasing.
    ticks = [f["data"]["tick"] for f in tick_frames]
    assert ticks == sorted(ticks), f"Tick numbers not monotone: {ticks}"
    assert ticks[0] >= 1, "First tick must be ≥ 1"

    # At least one tick must carry validation results.
    assert any(
        len(f["data"]["new_results"]) > 0 for f in tick_frames
    ), "Expected at least one tick with new_results"

    # Vehicles must have finite lat/lon.
    for f in tick_frames:
        for v in f["data"]["vehicles"]:
            assert v["lat"] is not None and math.isfinite(v["lat"]), (
                f"Non-finite lat in vehicle: {v}"
            )
            assert v["lon"] is not None and math.isfinite(v["lon"]), (
                f"Non-finite lon in vehicle: {v}"
            )

    # End frame assertions.
    end_frame = frames[-1]
    assert end_frame["type"] == "end", f"Last frame must be end, got {end_frame['type']!r}"
    assert end_frame["data"]["status"] == "done"
    assert end_frame["data"]["summary"] is not None
    assert end_frame["data"]["summary"]["results_saved"] > 0


def test_ws_end_summary_matches_rest(tmp_path: Path) -> None:
    """``end`` frame summary must match GET /api/runs/{id} after completion."""
    with _make_test_client(tmp_path) as client:
        resp = client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
        run_id = resp.json()["run_id"]

        with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
            frames = _collect_frames(ws)

        end_frame = next(f for f in frames if f["type"] == "end")

        rest = client.get(f"/api/runs/{run_id}").json()

    assert rest["status"] == "done"
    assert rest["summary"] is not None
    assert (
        rest["summary"]["results_saved"]
        == end_frame["data"]["summary"]["results_saved"]
    ), "end frame results_saved must match REST summary"
    assert (
        rest["summary"]["ticks"]
        == end_frame["data"]["summary"]["ticks"]
    ), "end frame ticks must match REST summary"


def test_ws_unknown_run_id_closes_cleanly(tmp_path: Path) -> None:
    """Connecting to a non-existent run_id must close with code 4404 (no hang)."""
    with _make_test_client(tmp_path) as client:
        with client.websocket_connect("/api/runs/no-such-id-xyz/stream") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_json()
        assert exc_info.value.code == 4404


def test_ws_connect_after_run_done_gets_snapshot_and_end(tmp_path: Path) -> None:
    """Connecting AFTER the run has already finished must yield snapshot + end.

    The snapshot must reflect the final (done) state and no tick frames should
    appear between snapshot and end.
    """
    with _make_test_client(tmp_path) as client:
        resp = client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
        run_id = resp.json()["run_id"]

        # Explicitly wait for the run to finish before connecting.
        final = _wait_for_run_done(client, run_id)
        assert final["status"] == "done"

        with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
            frames = _collect_frames(ws)

    assert frames[0]["type"] == "snapshot"
    assert frames[0]["data"]["status"] == "done", (
        "Snapshot should show done status when connecting after run finished"
    )

    end_frame = frames[-1]
    assert end_frame["type"] == "end"
    assert end_frame["data"]["status"] == "done"
    assert end_frame["data"]["error"] is None

    # No tick frames — run was already done when we connected.
    tick_frames = [f for f in frames if f["type"] == "tick"]
    assert len(tick_frames) == 0, (
        f"Expected no tick frames after-done connect; got {len(tick_frames)}"
    )
