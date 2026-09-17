"""WebSocket streaming endpoint for live run tick events. Phase B.

Frame protocol
--------------
Every frame is a JSON object with a ``type`` key and a ``data`` payload.

``{"type": "snapshot", "data": {...}}``
    Sent immediately after the connection is accepted.  Carries the current
    RunState so late subscribers can determine the run's present position.

``{"type": "tick", "data": {...}}``
    One frame per engine tick while the run is active.  ``data`` is the
    output of :func:`tick_event_to_dict`.

``{"type": "end", "data": {status, summary, error}}``
    Sent once when the run reaches a terminal state (``done`` or ``error``).
    The server closes the connection immediately afterward with code 1000.

Close codes
-----------
4404 — ``run_id`` is unknown (accepted then immediately closed).
1000 — normal closure after the run ends.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from etaval.api.runs import RunManager, RunState
from etaval.engine.runtime import TickEvent

router = APIRouter()

# How long to wait on queue.get() before checking terminal state as a fallback.
_POLL_TIMEOUT_S: float = 0.5


# ── Serialization helpers ─────────────────────────────────────────────────────


def _safe_float(v: float | None) -> float | None:
    """Return *v*, replacing NaN / ±Inf with ``None`` for JSON safety."""
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def tick_event_to_dict(ev: TickEvent) -> dict:
    """Serialize a :class:`~etaval.engine.runtime.TickEvent` to a plain dict.

    * ``tick``, cumulative counts, and ``header_timestamp`` are included as-is.
    * ``new_results`` — each :class:`~etaval.domain.models.ValidationResult`
      is serialized via ``model_dump()``.
    * ``vehicles`` — each :class:`~etaval.engine.runtime.VehiclePing`
      (a frozen dataclass) is converted with ``dataclasses.asdict``; ``lat``,
      ``lon``, and ``progress_m`` are guarded against NaN / ±Inf.
    """
    vehicles = []
    for vp in ev.vehicles:
        d = asdict(vp)
        d["lat"] = _safe_float(d["lat"])
        d["lon"] = _safe_float(d["lon"])
        d["progress_m"] = _safe_float(d.get("progress_m"))
        vehicles.append(d)

    return {
        "tick": ev.tick,
        "header_timestamp": ev.header_timestamp,
        "samples_saved": ev.samples_saved,
        "arrivals_saved": ev.arrivals_saved,
        "results_saved": ev.results_saved,
        "new_results": [r.model_dump() for r in ev.new_results],
        "vehicles": vehicles,
        "predictor_errors": dict(ev.predictor_errors),
        "detector_errors": dict(ev.detector_errors),
    }


def _snapshot_data(state: RunState) -> dict:
    """Minimal JSON-serializable representation of a :class:`~etaval.api.runs.RunState`."""
    summary: dict | None = None
    if state.summary is not None:
        summary = state.summary.to_dict()
    return {
        "run_id": state.id,
        "status": state.status,
        "summary": summary,
        "started_at": state.started_at,
    }


def _end_data(state: RunState) -> dict:
    """Payload for the ``end`` frame."""
    summary: dict | None = None
    if state.summary is not None:
        summary = state.summary.to_dict()
    return {
        "status": state.status,
        "summary": summary,
        "error": state.error,
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@router.websocket("/api/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """Stream live tick events for *run_id* over a WebSocket connection.

    Behavior
    --------
    1. Accept the connection.
    2. If *run_id* is unknown close with code 4404 and return.
    3. Send a ``snapshot`` frame with the current :class:`~etaval.api.runs.RunState`.
    4. If the run is already finished (``done`` / ``error``): send ``end`` and close.
    5. Otherwise subscribe to the run's broadcast queue and forward each
       :class:`~etaval.engine.runtime.TickEvent` as a ``tick`` frame.
    6. When the ``None`` sentinel arrives (or the timeout-based fallback detects
       the run has ended), send ``end`` and close with code 1000.
    7. Always ``unsubscribe`` in a ``finally`` block — handles client disconnect.
    """
    run_manager: RunManager = websocket.app.state.run_manager

    await websocket.accept()

    state = run_manager.get_run(run_id)
    if state is None:
        await websocket.close(code=4404)
        return

    # ── Snapshot ───────────────────────────────────────────────────────────────
    await websocket.send_json({"type": "snapshot", "data": _snapshot_data(state)})

    # ── Already finished? → end immediately ────────────────────────────────────
    if state.status in ("done", "error"):
        await websocket.send_json({"type": "end", "data": _end_data(state)})
        await websocket.close()
        return

    # ── Subscribe and stream ───────────────────────────────────────────────────
    q = run_manager.subscribe(run_id)
    current_state = state
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=_POLL_TIMEOUT_S)
            except TimeoutError:
                # Fallback: sentinel may have been missed (e.g. subscriber
                # joined after _signal_end ran with an empty subscribers list).
                fresh = run_manager.get_run(run_id)
                if fresh is not None:
                    current_state = fresh
                if current_state.status in ("done", "error") and q.empty():
                    break
                continue

            if event is None:
                # None sentinel — run has reached a terminal state.
                fresh = run_manager.get_run(run_id)
                if fresh is not None:
                    current_state = fresh
                break

            # Forward the tick event to the client.
            try:
                await websocket.send_json(
                    {"type": "tick", "data": tick_event_to_dict(event)}
                )
            except WebSocketDisconnect:
                return

    except WebSocketDisconnect:
        return
    finally:
        run_manager.unsubscribe(run_id, q)

    # ── End frame (normal completion) ──────────────────────────────────────────
    try:
        await websocket.send_json({"type": "end", "data": _end_data(current_state)})
        await websocket.close()
    except WebSocketDisconnect:
        pass
