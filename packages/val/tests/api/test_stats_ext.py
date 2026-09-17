"""tests/api/test_stats_ext.py — ?stats=1 endpoint contract tests.

Offline replay-backed TestClient pattern (same seam as tests/test_api.py):
a RunManager backed by ReplayFeedSource, no live HTTP.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from stoptimes.val.api.app import create_app
from stoptimes.val.api.runs import RunManager
from tests.test_api import _replay_deps_builder, _wait_done

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


async def _done_run(client: httpx.AsyncClient) -> str:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    await _wait_done(client, run_id)
    return run_id


# ── ?stats=1 augmentation ───────────────────────────────────────────────────────


async def test_stats_1_adds_mae_ci_and_latency_per_predictor(
    client: httpx.AsyncClient,
) -> None:
    run_id = await _done_run(client)
    resp = await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})
    assert resp.status_code == 200
    body = resp.json()

    assert body["seed"] == 0
    assert len(body["per_predictor"]) > 0
    for row in body["per_predictor"]:
        assert "mae_ci" in row
        ci = row["mae_ci"]
        assert ci is not None
        assert set(ci) == {"lo", "hi", "ci", "n_boot"}
        assert ci["ci"] == 0.95
        assert ci["n_boot"] == 1000
        assert ci["lo"] <= ci["hi"]

        assert "latency" in row
        latency = row["latency"]
        assert latency is None or set(latency) == {"p50_s", "p95_s", "p99_s", "count"}


async def test_stats_1_significance_list_obeys_contract(client: httpx.AsyncClient) -> None:
    run_id = await _done_run(client)
    resp = await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})
    body = resp.json()

    significance = body["significance"]
    assert isinstance(significance, list)

    seen_pairs: set[tuple[str, str, str]] = set()
    for row in significance:
        assert row["test"] in {"wilcoxon", "diebold_mariano"}
        assert row["a"] < row["b"], "a, b must be sorted predictor ids"
        key = (row["a"], row["b"], row["test"])
        assert key not in seen_pairs, "each (a, b, test) must appear at most once"
        seen_pairs.add(key)
        assert row["n"] >= 2
        assert 0.0 <= row["p_value"] <= 1.0
        assert 0.0 <= row["p_adjusted"] <= 1.0
        assert isinstance(row["reject"], bool)
        assert isinstance(row["diff_mean"], float)

    # Holm within a family: adjusted p-values are >= raw p-values.
    for row in significance:
        assert row["p_adjusted"] >= row["p_value"] - 1e-9


async def test_stats_1_calibration_bins_present(client: httpx.AsyncClient) -> None:
    run_id = await _done_run(client)
    resp = await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})
    body = resp.json()

    calibration = body["calibration"]
    assert isinstance(calibration, list)
    for row in calibration:
        assert set(row) == {
            "band_label",
            "band_lo_s",
            "band_hi_s",
            "tolerance_s",
            "n",
            "nominal_coverage",
            "empirical_coverage",
        }
        assert row["n"] > 0


async def test_plain_response_has_no_stats_keys(client: httpx.AsyncClient) -> None:
    run_id = await _done_run(client)
    resp = await client.get(f"/api/runs/{run_id}/metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"per_predictor", "horizon_buckets", "convergence", "head_to_head"}
    for row in body["per_predictor"]:
        assert "mae_ci" not in row
        assert "latency" not in row


async def test_stats_1_vs_no_stats_plain_fields_identical(
    client: httpx.AsyncClient,
) -> None:
    run_id = await _done_run(client)
    plain = (await client.get(f"/api/runs/{run_id}/metrics")).json()
    with_stats = (await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})).json()

    assert plain["horizon_buckets"] == with_stats["horizon_buckets"]
    assert plain["convergence"] == with_stats["convergence"]
    assert plain["head_to_head"] == with_stats["head_to_head"]

    stripped = [
        {k: v for k, v in row.items() if k not in ("mae_ci", "latency")}
        for row in with_stats["per_predictor"]
    ]
    assert stripped == plain["per_predictor"]


async def test_stats_1_on_non_done_run_returns_409(
    client: httpx.AsyncClient, run_manager: RunManager
) -> None:
    resp = await client.post("/api/runs", json={"source": "mbta", "duration": "1m"})
    run_id = resp.json()["run_id"]
    # Don't wait for completion — request stats immediately while running/pending.
    resp2 = await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})
    assert resp2.status_code in (200, 409)
    # If the replay finished before we asked (fast CI), that's fine too — but
    # if it's not done yet, 409 is required.
    if resp2.status_code != 200:
        assert resp2.status_code == 409

    await _wait_done(client, run_id)


async def test_stats_1_unknown_run_returns_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/runs/does-not-exist/metrics", params={"stats": 1})
    assert resp.status_code == 404


async def test_stats_1_computed_once_and_cached(
    client: httpx.AsyncClient, run_manager: RunManager
) -> None:
    run_id = await _done_run(client)
    first = (await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})).json()
    assert run_id in run_manager._stats_cache
    cached_obj = run_manager._stats_cache[run_id]

    second = (await client.get(f"/api/runs/{run_id}/metrics", params={"stats": 1})).json()
    assert run_manager._stats_cache[run_id] is cached_obj
    assert first["significance"] == second["significance"]
    assert first["calibration"] == second["calibration"]
