"""tests/tui/test_app.py — Textual pilot-based smoke tests for EtavalApp.

Uses ``App.run_test()`` with a :class:`FakeApi` injected in place of the real
:class:`~etaval.tui.client.ApiClient`, so these tests never touch the
network. They assert that all four screens mount and can be switched between
without raising.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from etaval.tui.app import EtavalApp
from etaval.tui.screens import ConsoleScreen, ExplorerScreen, LeaderboardScreen, TelemetryScreen


class FakeApi:
    """In-memory stand-in for ApiClient — same method signatures, no network."""

    def __init__(self) -> None:
        self.created_payload: dict[str, Any] | None = None

    async def aclose(self) -> None:
        return None

    async def get_sources(self) -> list[dict[str, Any]]:
        return [{"name": "mbta"}, {"name": "bucr"}]

    async def get_predictors(self) -> list[str]:
        return ["gtfsrt", "baseline:constant_speed"]

    async def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "run-abc123",
                "status": "done",
                "params": {"source": "mbta"},
                "summary": {"ticks": 10, "samples_saved": 5, "results_saved": 3},
                "db_path": "x.sqlite",
                "started_at": 0.0,
                "ended_at": 1.0,
                "error": None,
            }
        ]

    async def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created_payload = payload
        return {"run_id": "run-xyz789"}

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "done"}

    async def get_metrics(self, run_id: str, stats: int = 0) -> dict[str, Any]:
        return {
            "per_predictor": [{"predictor_id": "gtfsrt", "mae_s": 12.5, "n": 100}],
            "horizon_buckets": [{"bucket": "0-60", "mae_s": 10.0, "n": 50}],
            "convergence": [],
            "head_to_head": [],
        }

    async def get_results(
        self, run_id: str, predictor: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return [
            {
                "predictor_id": "gtfsrt",
                "route_id": "1",
                "stop_id": "s1",
                "error_s": 5.0,
                "horizon_s": 90.0,
            }
        ]

    async def get_telemetry(self) -> dict[str, Any]:
        return {
            "api_uptime_s": 42.0,
            "active_runs": 1,
            "per_run": [
                {
                    "run_id": "run-abc123",
                    "status": "running",
                    "ticks": 3,
                    "tick_rate": 0.1,
                    "last_tick_age_s": 1.2,
                    "feed": {"last_poll_age_s": 1.0, "ok": True},
                    "predictor_errors": {},
                }
            ],
        }

    async def get_telemetry_events(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "ts": 0.0,
                "run_id": "run-abc123",
                "level": "info",
                "source": "startup",
                "message": "api boot",
            }
        ]

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "tick",
            "data": {
                "tick": 1,
                "samples_saved": 1,
                "arrivals_saved": 0,
                "results_saved": 0,
                "new_results": [],
                "vehicles": [],
            },
        }
        yield {"type": "end", "data": {"status": "done", "summary": {}, "error": None}}


@pytest.fixture
def fake_client() -> FakeApi:
    return FakeApi()


async def test_app_mounts_console_screen(fake_client: FakeApi) -> None:
    app = EtavalApp(client=fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConsoleScreen)


async def test_switch_to_leaderboard_loads_runs(fake_client: FakeApi) -> None:
    app = EtavalApp(client=fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, LeaderboardScreen)


async def test_switch_to_explorer_screen(fake_client: FakeApi) -> None:
    app = EtavalApp(client=fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f3")
        await pilot.pause()
        assert isinstance(app.screen, ExplorerScreen)


async def test_switch_to_telemetry_screen_polls(fake_client: FakeApi) -> None:
    app = EtavalApp(client=fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f4")
        await pilot.pause()
        assert isinstance(app.screen, TelemetryScreen)
