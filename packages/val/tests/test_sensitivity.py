"""Tests for Phase A5 — etaval/scoring/sensitivity.py.

Coverage
--------
- compute_detector_sensitivity: one DetectorSensitivityRow per detector,
  default detector's own delta_mae_vs_default == 0.0.
- sensitivity_from_persisted_arrivals: same shape, grouped by persisted
  ActualArrival.method instead of re-running detection.
- CLI: `report --sensitivity` without `--db` is declined with an explanatory
  message; with a synthetic sqlite db (built via the offline engine path,
  mirroring tests/test_cli.py) it runs and renders a sensitivity table.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from etaval.cli.main import _run_validation, app
from etaval.domain.models import ActualArrival, PredictionSample, Stop, TripInstance, VehicleState
from etaval.engine.runtime import ReplayFeedSource
from etaval.feeds.static import GtfsStaticLoader
from etaval.groundtruth.position_radius import PositionRadiusDetector
from etaval.groundtruth.shape_distance import ShapeDistanceDetector
from etaval.groundtruth.stop_status import StopStatusDetector
from etaval.predictors.constant_speed import ConstantSpeedPredictor
from etaval.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor
from etaval.scoring.sensitivity import (
    TripContext,
    compute_detector_sensitivity,
    sensitivity_from_persisted_arrivals,
)
from etaval.storage.parquet_export import export_results
from etaval.storage.sqlite import SqliteStorage

# Reuse the offline snapshot builder from test_cli.py so this suite stays
# deterministic and network-free, matching that module's conventions.
from tests.test_cli import _build_snapshots

runner = CliRunner()

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"

_TRIP_KEY = "test:trip1:v1:20240101"

TRIP = TripInstance(
    source="test",
    trip_id="trip1",
    vehicle_id="v1",
    route_id="r1",
    direction_id=0,
    shape_id="shape1",
    start_date="20240101",
    started_at=1000.0,
)


def _sample(
    stop_id: str, predicted: float, made_at: float, predictor_id: str = "p1"
) -> PredictionSample:
    return PredictionSample(
        trip_key=_TRIP_KEY,
        predictor_id=predictor_id,
        stop_id=stop_id,
        stop_sequence=1,
        predicted_arrival_epoch=predicted,
        made_at_epoch=made_at,
    )


def _arrival(stop_id: str, actual: float, method: str) -> ActualArrival:
    return ActualArrival(
        trip_key=_TRIP_KEY,
        stop_id=stop_id,
        stop_sequence=1,
        actual_arrival_epoch=actual,
        method=method,
    )


class _FakeDetector:
    """Detector double: returns fixed ActualArrivals, ignoring its inputs."""

    def __init__(self, method: str, arrivals: list[ActualArrival]) -> None:
        self.method = method
        self._arrivals = arrivals

    def detect(
        self, trip: TripInstance, position_history: list[VehicleState], stops: list[Stop]
    ) -> list[ActualArrival]:
        return list(self._arrivals)


# ---------------------------------------------------------------------------
# compute_detector_sensitivity (full-fidelity, synthetic detectors)
# ---------------------------------------------------------------------------


class TestComputeDetectorSensitivity:
    def _samples(self) -> list[PredictionSample]:
        return [
            _sample("s1", predicted=2010.0, made_at=1900.0),
            _sample("s2", predicted=2110.0, made_at=1900.0),
        ]

    def _trip_contexts(self) -> list[TripContext]:
        return [TripContext(trip=TRIP, position_history=[], stops=[], route_id="r1")]

    def test_one_row_per_detector(self) -> None:
        samples = self._samples()
        detectors = [
            _FakeDetector("shape_distance", [_arrival("s1", 2000.0, "shape_distance")]),
            _FakeDetector("position_radius", [_arrival("s1", 2020.0, "position_radius")]),
        ]
        rows = compute_detector_sensitivity(samples, self._trip_contexts(), detectors)
        assert {r.detector_id for r in rows} == {"shape_distance", "position_radius"}

    def test_default_detector_delta_is_zero(self) -> None:
        samples = self._samples()
        detectors = [
            _FakeDetector("shape_distance", [_arrival("s1", 2000.0, "shape_distance")]),
            _FakeDetector("position_radius", [_arrival("s1", 2020.0, "position_radius")]),
        ]
        rows = compute_detector_sensitivity(
            samples, self._trip_contexts(), detectors, default_detector_id="shape_distance"
        )
        default_row = next(r for r in rows if r.detector_id == "shape_distance")
        assert default_row.delta_mae_vs_default == pytest.approx(0.0)

    def test_non_default_delta_reflects_mae_shift(self) -> None:
        # shape_distance: error = 2010-2000 = 10 -> MAE=10
        # position_radius: error = 2010-2020 = -10, |error|=10 too -> delta 0
        # use a farther arrival to force a nonzero delta
        detectors = [
            _FakeDetector("shape_distance", [_arrival("s1", 2000.0, "shape_distance")]),
            _FakeDetector("position_radius", [_arrival("s1", 1970.0, "position_radius")]),
        ]
        rows = compute_detector_sensitivity(
            [self._samples()[0]], self._trip_contexts(), detectors
        )
        by_id = {r.detector_id: r for r in rows}
        # shape_distance: |2010-2000| = 10
        assert by_id["shape_distance"].metrics_by_predictor["p1"].mae == pytest.approx(10.0)
        # position_radius: |2010-1970| = 40
        assert by_id["position_radius"].metrics_by_predictor["p1"].mae == pytest.approx(40.0)
        assert by_id["position_radius"].delta_mae_vs_default == pytest.approx(30.0)

    def test_metrics_by_predictor_keys_are_predictor_ids(self) -> None:
        samples = [
            _sample("s1", predicted=2010.0, made_at=1900.0, predictor_id="p1"),
            _sample("s1", predicted=2005.0, made_at=1900.0, predictor_id="p2"),
        ]
        detectors = [_FakeDetector("shape_distance", [_arrival("s1", 2000.0, "shape_distance")])]
        rows = compute_detector_sensitivity(samples, self._trip_contexts(), detectors)
        assert set(rows[0].metrics_by_predictor) == {"p1", "p2"}

    def test_empty_detectors_returns_empty_list(self) -> None:
        rows = compute_detector_sensitivity(self._samples(), self._trip_contexts(), [])
        assert rows == []


# ---------------------------------------------------------------------------
# sensitivity_from_persisted_arrivals (storage-backed, restricted)
# ---------------------------------------------------------------------------


class TestSensitivityFromPersistedArrivals:
    def test_one_row_per_persisted_method(self) -> None:
        samples = [_sample("s1", predicted=2010.0, made_at=1900.0)]
        arrivals = [
            _arrival("s1", 2000.0, "shape_distance"),
            _arrival("s1", 1990.0, "position_radius"),
            _arrival("s1", 2005.0, "stop_status"),
        ]
        rows = sensitivity_from_persisted_arrivals(samples, arrivals)
        assert {r.detector_id for r in rows} == {
            "shape_distance",
            "position_radius",
            "stop_status",
        }

    def test_default_row_delta_is_zero(self) -> None:
        samples = [_sample("s1", predicted=2010.0, made_at=1900.0)]
        arrivals = [
            _arrival("s1", 2000.0, "shape_distance"),
            _arrival("s1", 1950.0, "position_radius"),
        ]
        rows = sensitivity_from_persisted_arrivals(samples, arrivals)
        default_row = next(r for r in rows if r.detector_id == "shape_distance")
        assert default_row.delta_mae_vs_default == pytest.approx(0.0)

    def test_empty_arrivals_returns_empty_list(self) -> None:
        assert sensitivity_from_persisted_arrivals([_sample("s1", 2010.0, 1900.0)], []) == []

    def test_no_matching_samples_yields_nan_mae_and_delta(self) -> None:
        """A method whose arrivals never join any sample -> nan MAE / delta."""
        samples = [_sample("s1", predicted=2010.0, made_at=1900.0)]
        arrivals = [
            _arrival("s1", 2000.0, "shape_distance"),
            _arrival("no_such_stop", 1990.0, "position_radius"),
        ]
        rows = sensitivity_from_persisted_arrivals(samples, arrivals)
        pr_row = next(r for r in rows if r.detector_id == "position_radius")
        assert pr_row.metrics_by_predictor == {}
        assert math.isnan(pr_row.delta_mae_vs_default)


# ---------------------------------------------------------------------------
# CLI: report --sensitivity
# ---------------------------------------------------------------------------


def _make_result_parquet(tmp_path: Path) -> Path:
    from etaval.domain.models import ValidationResult

    results = [
        ValidationResult(
            trip_key=_TRIP_KEY,
            predictor_id="gtfsrt",
            stop_id="s2",
            stop_sequence=2,
            route_id="r1",
            error_s=10.0,
            horizon_s=100.0,
            predicted_arrival_epoch=2010.0,
            actual_arrival_epoch=2000.0,
            made_at_epoch=1900.0,
        )
    ]
    out = tmp_path / "results.parquet"
    export_results(results, out)
    return out


def test_report_sensitivity_without_db_declined(tmp_path: Path) -> None:
    out = _make_result_parquet(tmp_path)
    result = runner.invoke(app, ["report", str(out), "--sensitivity"])
    assert result.exit_code != 0
    assert "--db" in result.output
    assert "position history" in result.output or "persist" in result.output


@pytest.mark.asyncio
async def test_report_sensitivity_with_db_runs(tmp_path: Path) -> None:
    """Build a synthetic sqlite db via the offline engine path (mirrors test_cli.py),
    export a parquet report from it, then run `report --sensitivity --db`."""
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    db_path = tmp_path / "run.sqlite3"
    storage = SqliteStorage(db_path)
    predictors = [GtfsRtTripUpdatesPredictor(), ConstantSpeedPredictor()]
    detectors = [
        ShapeDistanceDetector(shape_threshold_m=30.0),
        PositionRadiusDetector(arrival_radius_m=50.0),
        StopStatusDetector(),
    ]
    feed_source = ReplayFeedSource(_build_snapshots())

    await _run_validation(
        feed_source,
        static_loader,
        predictors,
        detectors,
        storage,
        primary_method="shape_distance",
        source_name="test",
    )

    results = storage.load_results()
    assert len(results) > 0, "Engine produced no results — cannot test sensitivity CLI"
    out = tmp_path / "results.parquet"
    export_results(results, out)
    storage.close()

    cli_result = runner.invoke(
        app, ["report", str(out), "--sensitivity", "--db", str(db_path)]
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "Detector Sensitivity" in cli_result.output
    assert "shape_distance" in cli_result.output
