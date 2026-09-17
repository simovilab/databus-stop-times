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


def _static_content(widget: Any) -> str:
    """Return the plain-text content last passed to a Static widget's ``.update()``."""
    return str(widget._Static__content)  # noqa: SLF001 — textual stores it name-mangled


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
        return {
            "id": run_id,
            "status": "done",
            "provenance": {
                "etaval_git_sha": "abc123def456",
                "etaval_dirty": True,
                "gtfs_eta_git_sha": "deadbeefcafe",
                "gtfs_eta_dirty": False,
                "gtfs_snapshot": "2026-09-01",
                "source_config_sha256": "0123456789abcdef",
                "seed": 0,
                "created_at": "2026-09-01T00:00:00+00:00",
            },
        }

    async def get_metrics(self, run_id: str, stats: int = 0) -> dict[str, Any]:
        body: dict[str, Any] = {
            "per_predictor": [{"predictor_id": "gtfsrt", "mae_s": 12.5, "n": 100}],
            "horizon_buckets": [{"bucket": "0-60", "mae_s": 10.0, "n": 50}],
            "convergence": [],
            "head_to_head": [],
        }
        if stats:
            body["per_predictor"] = [
                {
                    "predictor_id": "gtfsrt",
                    "mae": 12.5,
                    "mae_s": 12.5,
                    "n": 100,
                    "mae_ci": {"lo": 10.0, "hi": 15.0, "ci": 0.95, "n_boot": 1000},
                    "latency": {"p50_s": 0.01, "p95_s": 0.02, "p99_s": 0.03, "count": 100},
                },
                {
                    "predictor_id": "baseline:constant_speed",
                    "mae": 20.0,
                    "mae_s": 20.0,
                    "n": 90,
                    "mae_ci": {"lo": 18.0, "hi": 22.0, "ci": 0.95, "n_boot": 1000},
                    "latency": None,
                },
            ]
            body["significance"] = [
                {
                    "a": "baseline:constant_speed",
                    "b": "gtfsrt",
                    "test": "wilcoxon",
                    "statistic": 3.0,
                    "p_value": 0.04,
                    "p_adjusted": 0.04,
                    "reject": True,
                    "n": 10,
                    "diff_mean": 7.5,
                }
            ]
            body["calibration"] = [
                {
                    "band_label": "<2m",
                    "band_lo_s": 0.0,
                    "band_hi_s": 120.0,
                    "tolerance_s": 30.0,
                    "n": 40,
                    "nominal_coverage": 0.68,
                    "empirical_coverage": 0.62,
                }
            ]
            body["seed"] = 0
        return body

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


# ── Stats surfaces (CIs, latency, significance, provenance, calibration) ─────────


async def test_leaderboard_detail_renders_stats_surfaces_without_error(
    fake_client: FakeApi,
) -> None:
    """A DONE run's detail panel mounts CI/latency/significance/provenance widgets
    from a ``stats=1`` payload without raising."""
    from textual.widgets import DataTable, Static

    app = EtavalApp(client=fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LeaderboardScreen)

        await screen._open_detail("run-abc123")
        await pilot.pause()

        pred_table = screen.query_one("#predictor-metrics-table", DataTable)
        assert pred_table.row_count == 2

        latency_table = screen.query_one("#latency-table", DataTable)
        assert latency_table.row_count == 2

        significance_text = _static_content(screen.query_one("#significance-block", Static))
        assert "wilcoxon" in str(significance_text)
        assert "p=0.04*" in str(significance_text)

        provenance_str = _static_content(screen.query_one("#provenance-block", Static))
        assert "abc123de*" in provenance_str  # dirty etaval sha, starred
        assert "deadbeef" in provenance_str  # clean gtfs-eta sha, no star
        assert "2026-09-01" in provenance_str


async def test_leaderboard_detail_null_safe_when_stats_missing(fake_client: FakeApi) -> None:
    """A run whose get_run/get_metrics return no provenance/stats renders '—', not a crash."""

    class NoStatsApi(FakeApi):
        async def get_run(self, run_id: str) -> dict[str, Any]:
            return {"id": run_id, "status": "pending"}

    app = EtavalApp(client=NoStatsApi())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LeaderboardScreen)

        await screen._open_detail("run-abc123")
        await pilot.pause()
        # Should not raise; plain (non-stats) metrics still render.
        from textual.widgets import DataTable

        pred_table = screen.query_one("#predictor-metrics-table", DataTable)
        assert pred_table.row_count == 1


async def test_explorer_screen_renders_calibration(fake_client: FakeApi) -> None:
    from textual.widgets import Static

    app = EtavalApp(client=fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f3")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ExplorerScreen)

        screen.set_run("run-abc123")
        await pilot.pause()
        await pilot.pause()

        calibration_text = _static_content(screen.query_one("#calibration-block", Static))
        assert "<2m" in calibration_text
        assert "0.68" in calibration_text
        assert "0.62" in calibration_text
