"""Thin, injectable HTTP + WebSocket client for the etaval REST API.

Kept deliberately dumb: it does no caching, retries, or business logic — just
typed-ish wrappers around the REST/WS surface documented in
``etaval/api/app.py`` and ``etaval/api/ws.py``.  Screens depend on this class
(or a fake with the same method signatures) rather than talking to httpx /
websockets directly, so tests can inject a fake client with zero network
access.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets

DEFAULT_API_URL = "http://localhost:8010"


class ApiClient:
    """Async client for the etaval REST + WebSocket API.

    Args:
        api_url: Base URL, e.g. ``"http://localhost:8010"``.  Falls back to
            the ``ETVAL_API_URL`` environment variable, then
            :data:`DEFAULT_API_URL`.
    """

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = (
            api_url or os.environ.get("ETVAL_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.api_url, timeout=10.0)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── REST ─────────────────────────────────────────────────────────────────

    async def get_sources(self) -> list[dict[str, Any]]:
        """GET /api/sources — configured GTFS-RT sources."""
        resp = await self._client.get("/api/sources")
        resp.raise_for_status()
        return resp.json()

    async def get_predictors(self) -> list[str]:
        """GET /api/predictors — registered predictor IDs."""
        resp = await self._client.get("/api/predictors")
        resp.raise_for_status()
        return resp.json()

    async def list_runs(self) -> list[dict[str, Any]]:
        """GET /api/runs — all known runs, newest first."""
        resp = await self._client.get("/api/runs")
        resp.raise_for_status()
        return resp.json()

    async def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/runs — start a new run. Returns ``{"run_id": ...}``."""
        resp = await self._client.post("/api/runs", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """GET /api/runs/{run_id} — current RunInfo."""
        resp = await self._client.get(f"/api/runs/{run_id}")
        resp.raise_for_status()
        return resp.json()

    async def get_metrics(self, run_id: str, stats: int = 0) -> dict[str, Any]:
        """GET /api/runs/{run_id}/metrics — full metrics report.

        ``stats`` is forwarded as a query param for the statistical layer
        (added in a later wave); ``0`` disables it.
        """
        resp = await self._client.get(
            f"/api/runs/{run_id}/metrics", params={"stats": stats}
        )
        resp.raise_for_status()
        return resp.json()

    async def get_results(
        self, run_id: str, predictor: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """GET /api/runs/{run_id}/results — raw ValidationResult rows."""
        params: dict[str, Any] = {"limit": limit}
        if predictor is not None:
            params["predictor"] = predictor
        resp = await self._client.get(f"/api/runs/{run_id}/results", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_telemetry(self) -> dict[str, Any]:
        """GET /api/telemetry — process + per-run telemetry snapshot."""
        resp = await self._client.get("/api/telemetry")
        resp.raise_for_status()
        return resp.json()

    async def get_telemetry_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """GET /api/telemetry/events — newest-last structured event log."""
        resp = await self._client.get("/api/telemetry/events", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    # ── WebSocket ────────────────────────────────────────────────────────────

    def _ws_url(self, run_id: str) -> str:
        ws_base = self.api_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/api/runs/{run_id}/stream"

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded JSON frames from ``/api/runs/{run_id}/stream``.

        Stops iterating after an ``end`` frame or when the socket closes.
        """
        async with websockets.connect(self._ws_url(run_id)) as ws:
            async for raw in ws:
                frame = json.loads(raw)
                yield frame
                if frame.get("type") == "end":
                    break
