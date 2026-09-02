"""tests/api/test_telemetry.py — TelemetryCollector unit tests + endpoint tests.

Unit tests exercise TelemetryCollector directly (no network / no engine).
Endpoint tests use the same offline replay-backed TestClient pattern as
tests/test_api.py and tests/test_ws.py.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from etaval.api.app import create_app
from etaval.api.runs import RunManager
from etaval.api.telemetry import TelemetryCollector
from tests.test_api import _replay_deps_builder

# ── TelemetryCollector unit tests ───────────────────────────────────────────────


def test_record_tick_increments_count() -> None:
    tc = TelemetryCollector()
    tc.record_tick("run-1")
    tc.record_tick("run-1")
    snap = tc.snapshot()
    assert snap["per_run"]["run-1"]["ticks"] == 2


def test_record_tick_sets_last_tick_age_small() -> None:
    tc = TelemetryCollector()
    tc.record_tick("run-1")
    snap = tc.snapshot()
    age = snap["per_run"]["run-1"]["last_tick_age_s"]
    assert age is not None
    assert 0 <= age < 1.0


def test_no_ticks_yet_last_tick_age_is_none() -> None:
    tc = TelemetryCollector()
    tc.record_feed_poll("run-1", ok=True)  # touches the run without a tick
    snap = tc.snapshot()
    assert snap["per_run"]["run-1"]["last_tick_age_s"] is None


def test_tick_rate_is_positive_after_ticks() -> None:
    tc = TelemetryCollector()
    for _ in range(5):
        tc.record_tick("run-1")
    snap = tc.snapshot()
    assert snap["per_run"]["run-1"]["tick_rate"] > 0


def test_record_feed_poll_ok_true() -> None:
    tc = TelemetryCollector()
    tc.record_feed_poll("run-1", ok=True)
    snap = tc.snapshot()
    feed = snap["per_run"]["run-1"]["feed"]
    assert feed is not None
    assert feed["ok"] is True
    assert feed["last_poll_age_s"] >= 0


def test_record_feed_poll_ok_false() -> None:
    tc = TelemetryCollector()
    tc.record_feed_poll("run-1", ok=False)
    snap = tc.snapshot()
    assert snap["per_run"]["run-1"]["feed"]["ok"] is False


def test_feed_poll_none_when_never_polled() -> None:
    tc = TelemetryCollector()
    tc.record_tick("run-1")
    snap = tc.snapshot()
    assert snap["per_run"]["run-1"]["feed"] is None


def test_record_predictor_error_counts_per_predictor() -> None:
    tc = TelemetryCollector()
    tc.record_predictor_error("run-1", "gtfsrt")
    tc.record_predictor_error("run-1", "gtfsrt")
    tc.record_predictor_error("run-1", "baseline:constant_speed")
    snap = tc.snapshot()
    errs = snap["per_run"]["run-1"]["predictor_errors"]
    assert errs == {"gtfsrt": 2, "baseline:constant_speed": 1}


def test_ring_buffer_caps_at_maxlen() -> None:
    tc = TelemetryCollector(event_capacity=5)
    for i in range(10):
        tc.log_event("info", "test", f"event-{i}")
    events = tc.events(limit=500)
    assert len(events) == 5
    # Only the most recent 5 survive.
    assert [e["message"] for e in events] == [f"event-{i}" for i in range(5, 10)]


def test_events_newest_last_ordering() -> None:
    tc = TelemetryCollector()
    tc.log_event("info", "a", "first")
    tc.log_event("warn", "b", "second")
    tc.log_event("error", "c", "third")
    events = tc.events(limit=10)
    assert [e["message"] for e in events] == ["first", "second", "third"]


def test_events_limit_truncates_to_newest() -> None:
    tc = TelemetryCollector()
    for i in range(20):
        tc.log_event("info", "src", f"e{i}")
    events = tc.events(limit=3)
    assert [e["message"] for e in events] == ["e17", "e18", "e19"]


def test_log_event_run_id_is_optional() -> None:
    tc = TelemetryCollector()
    tc.log_event("info", "startup", "api boot", run_id=None)
    events = tc.events(limit=10)
    assert events[0]["run_id"] is None
    assert events[0]["level"] == "info"
    assert events[0]["source"] == "startup"


def test_snapshot_includes_api_uptime() -> None:
    tc = TelemetryCollector()
    snap = tc.snapshot()
    assert snap["api_uptime_s"] >= 0


# ── Endpoint tests ───────────────────────────────────────────────────────────────


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


async def test_telemetry_endpoint_empty_state(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_runs"] == 0
    assert data["per_run"] == []
    assert data["api_uptime_s"] >= 0


async def test_telemetry_endpoint_after_run(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]

    # Poll until at least one tick has been recorded via telemetry.
    import asyncio
    import time

    deadline = time.monotonic() + 10.0
    per_run_entry = None
    while time.monotonic() < deadline:
        data = (await client.get("/api/telemetry")).json()
        matches = [r for r in data["per_run"] if r["run_id"] == run_id]
        if matches and matches[0]["ticks"] > 0:
            per_run_entry = matches[0]
            break
        await asyncio.sleep(0.05)

    assert per_run_entry is not None, "Expected telemetry to record ticks for the run"
    assert per_run_entry["tick_rate"] > 0
    assert per_run_entry["feed"] is not None
    assert per_run_entry["feed"]["ok"] is True


async def test_telemetry_events_endpoint_default_limit(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/telemetry/events")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_telemetry_events_endpoint_limit_bounds(client: httpx.AsyncClient) -> None:
    resp_low = await client.get("/api/telemetry/events", params={"limit": 0})
    assert resp_low.status_code == 422

    resp_high = await client.get("/api/telemetry/events", params={"limit": 501})
    assert resp_high.status_code == 422

    resp_ok = await client.get("/api/telemetry/events", params={"limit": 50})
    assert resp_ok.status_code == 200
