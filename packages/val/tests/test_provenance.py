"""Tests for etaval.provenance — RunProvenance, sidecar I/O, LatencyRecorder."""

from __future__ import annotations

import json
from pathlib import Path

from etaval import provenance as prov_mod
from etaval.domain.models import SourceConfig
from etaval.groundtruth.shape_distance import ShapeDistanceDetector
from etaval.provenance import (
    LatencyRecorder,
    LatencyStats,
    RunProvenance,
    build_run_provenance,
    write_provenance_sidecar,
)

_CFG = SourceConfig(
    name="test-source",
    gtfs_static_url="https://example.com/gtfs.zip",
    vehicle_positions_url="https://example.com/vp",
    trip_updates_url="https://example.com/tu",
    poll_interval_s=15,
)


# ── Defensiveness ────────────────────────────────────────────────────────────


def test_build_run_provenance_never_raises_when_git_is_broken(monkeypatch) -> None:
    """git subprocess failures degrade every git-derived field to None."""

    def _boom(*args, **kwargs):
        raise FileNotFoundError("no git binary")

    monkeypatch.setattr(prov_mod.subprocess, "run", _boom)

    result = build_run_provenance(
        source_name="mbta",
        source_config=_CFG,
        detectors=[ShapeDistanceDetector(shape_threshold_m=30.0)],
        predictor_ids=["gtfsrt"],
        seed=None,
    )

    assert isinstance(result, RunProvenance)
    assert result.etaval_git_sha is None
    assert result.etaval_dirty is None
    assert result.gtfs_eta_git_sha is None
    assert result.gtfs_eta_dirty is None
    # Non-git fields still populate normally.
    assert result.source_name == "mbta"
    assert result.predictor_ids == ["gtfsrt"]


def test_git_rev_parse_head_returns_none_on_nonzero_exit(tmp_path: Path) -> None:
    """A directory that is not a git repo yields returncode != 0 -> None."""
    assert prov_mod._git_rev_parse_head(tmp_path) is None


def test_git_is_dirty_returns_none_on_nonzero_exit(tmp_path: Path) -> None:
    assert prov_mod._git_is_dirty(tmp_path) is None


def test_gtfs_eta_repo_root_defensive_on_import_failure(monkeypatch) -> None:
    """A broken import of GtfsEtaModelPredictor degrades to None, never raises."""
    import builtins

    real_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        if name == "etaval.predictors.gtfs_eta_models":
            raise ImportError("simulated broken import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    assert prov_mod._gtfs_eta_repo_root() is None


def test_build_run_provenance_with_unserializable_config_degrades_hash() -> None:
    """A config that can't be introspected still returns a RunProvenance."""

    class Weird:
        pass

    result = build_run_provenance(
        source_name="x",
        source_config=Weird(),
        detectors=[],
        predictor_ids=[],
        seed=None,
    )
    assert isinstance(result, RunProvenance)
    assert result.source_config_sha256 is None


# ── to_dict / from_dict round-trip ──────────────────────────────────────────


def test_provenance_to_dict_from_dict_round_trip() -> None:
    original = build_run_provenance(
        source_name="mbta",
        source_config=_CFG,
        detectors=[ShapeDistanceDetector(shape_threshold_m=25.0)],
        predictor_ids=["gtfsrt", "baseline:constant_speed"],
        seed=42,
        gtfs_snapshot="s3://bucket/gtfs/2026-01-01.zip",
    )
    data = original.to_dict()
    restored = RunProvenance.from_dict(data)
    assert restored == original


def test_provenance_from_dict_tolerates_missing_keys() -> None:
    restored = RunProvenance.from_dict({})
    assert restored.source_name == ""
    assert restored.predictor_ids == []
    assert restored.detectors == {}
    assert restored.seed is None


# ── source_config_sha256 determinism ────────────────────────────────────────


def test_source_config_sha256_is_stable_and_deterministic() -> None:
    p1 = build_run_provenance(
        source_name="mbta", source_config=_CFG, detectors=[], predictor_ids=[], seed=None
    )
    p2 = build_run_provenance(
        source_name="mbta", source_config=_CFG, detectors=[], predictor_ids=[], seed=None
    )
    assert p1.source_config_sha256 is not None
    assert p1.source_config_sha256 == p2.source_config_sha256


def test_source_config_sha256_changes_with_config() -> None:
    other_cfg = _CFG.model_copy(update={"arrival_radius_m": 999.0})
    p1 = build_run_provenance(
        source_name="mbta", source_config=_CFG, detectors=[], predictor_ids=[], seed=None
    )
    p2 = build_run_provenance(
        source_name="mbta", source_config=other_cfg, detectors=[], predictor_ids=[], seed=None
    )
    assert p1.source_config_sha256 != p2.source_config_sha256


# ── detectors dict ───────────────────────────────────────────────────────────


def test_detectors_captured_with_thresholds() -> None:
    result = build_run_provenance(
        source_name="mbta",
        source_config=_CFG,
        detectors=[ShapeDistanceDetector(shape_threshold_m=30.0)],
        predictor_ids=[],
        seed=None,
    )
    assert "shape_distance" in result.detectors
    assert result.detectors["shape_distance"]["shape_threshold_m"] == 30.0


# ── sidecar I/O ───────────────────────────────────────────────────────────────


def test_write_provenance_sidecar_written_and_readable(tmp_path: Path) -> None:
    prov = build_run_provenance(
        source_name="mbta", source_config=_CFG, detectors=[], predictor_ids=["gtfsrt"], seed=7
    )
    base = tmp_path / "results.parquet"
    sidecar_path = write_provenance_sidecar(base, prov)

    assert sidecar_path == tmp_path / "results.parquet.provenance.json"
    assert sidecar_path.is_file()

    loaded = json.loads(sidecar_path.read_text())
    restored = RunProvenance.from_dict(loaded)
    assert restored == prov


# ── LatencyStats / LatencyRecorder ──────────────────────────────────────────


def test_latency_recorder_percentiles_known_sample() -> None:
    recorder = LatencyRecorder()
    # 0..10 inclusive => 11 samples, indices 0-10.
    for v in range(11):
        recorder.record("p1", float(v))

    stats = recorder.summary()["p1"]
    assert isinstance(stats, LatencyStats)
    assert stats.count == 11
    # p50 of 0..10 (linear interpolation) is 5.0
    assert stats.p50_s == 5.0
    # p95 rank = 0.95 * 10 = 9.5 -> interpolate between 9 and 10
    assert stats.p95_s == 9.5
    # p99 rank = 0.99 * 10 = 9.9 -> interpolate between 9 and 10
    assert abs(stats.p99_s - 9.9) < 1e-9


def test_latency_recorder_empty_predictor_absent_from_summary() -> None:
    recorder = LatencyRecorder()
    assert recorder.summary() == {}


def test_latency_recorder_caps_retained_samples_at_10k() -> None:
    recorder = LatencyRecorder()
    for i in range(10_050):
        recorder.record("p1", float(i))

    # Retained sample list is capped, but the running count is not.
    assert len(recorder._samples["p1"]) == 10_000
    stats = recorder.summary()["p1"]
    assert stats.count == 10_050


def test_latency_stats_to_dict_from_dict_round_trip() -> None:
    stats = LatencyStats(count=5, p50_s=1.0, p95_s=2.0, p99_s=3.0)
    restored = LatencyStats.from_dict(stats.to_dict())
    assert restored == stats
