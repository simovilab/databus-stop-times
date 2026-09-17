"""Tests for etaval/figures.py — A9 publication figures.

Headless (matplotlib Agg backend, set at import in stoptimes.val.figures). Builds a
small synthetic ValidationResults DataFrame (matching the Parquet schema from
stoptimes.val.storage.parquet_export) directly rather than running the engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stoptimes.val.cli.main import app
from stoptimes.core.domain.models import ValidationResult
from stoptimes.val.figures import (
    calibration_diagram,
    convergence_curve,
    cost_accuracy_pareto,
    error_distribution,
    error_vs_horizon,
    load_latency_from_summary,
    per_route_mae,
    render_all_figures,
)
from stoptimes.val.storage.parquet_export import export_results

runner = CliRunner()


def _make_results() -> list[ValidationResult]:
    results: list[ValidationResult] = []
    predictors = ["gtfsrt", "baseline:constant_speed"]
    routes = ["r1", "r2"]
    horizons = [30.0, 90.0, 200.0, 400.0, 700.0, 1000.0]
    made_at = 1_700_000_000.0
    for i in range(60):
        pid = predictors[i % 2]
        route = routes[i % 2]
        horizon = horizons[i % len(horizons)]
        error = float(((i * 37) % 21) - 10)  # deterministic spread, signed
        actual_epoch = made_at + horizon
        results.append(
            ValidationResult(
                trip_key=f"trip{i % 5}",
                predictor_id=pid,
                stop_id=f"s{i % 4 + 1}",
                stop_sequence=(i % 4) + 1,
                route_id=route,
                error_s=error,
                horizon_s=horizon,
                dist_remaining_m=float(100 + i),
                stops_remaining=(i % 4),
                predicted_arrival_epoch=actual_epoch + error,
                actual_arrival_epoch=actual_epoch,
                made_at_epoch=made_at,
            )
        )
    return results


@pytest.fixture()
def results_df() -> pd.DataFrame:
    results = _make_results()
    return pd.DataFrame([r.model_dump() for r in results])


@pytest.fixture()
def results_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "r.parquet"
    export_results(_make_results(), path)
    return path


def _assert_nonempty_png(path: Path | None) -> None:
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


class TestPerFigureFunctions:
    def test_error_vs_horizon_writes_png(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        out = error_vs_horizon(results_df, tmp_path / "evh.png")
        _assert_nonempty_png(out)

    def test_error_distribution_writes_png(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        out = error_distribution(results_df, tmp_path / "dist.png")
        _assert_nonempty_png(out)

    def test_convergence_curve_writes_png(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        out = convergence_curve(results_df, tmp_path / "conv.png")
        _assert_nonempty_png(out)

    def test_per_route_mae_writes_png(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        out = per_route_mae(results_df, tmp_path / "route.png")
        _assert_nonempty_png(out)

    def test_calibration_diagram_writes_png(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        out = calibration_diagram(results_df, tmp_path / "calib.png")
        _assert_nonempty_png(out)

    def test_empty_dataframe_returns_none(self, tmp_path: Path) -> None:
        empty = pd.DataFrame()
        assert error_vs_horizon(empty, tmp_path / "a.png") is None
        assert error_distribution(empty, tmp_path / "b.png") is None
        assert convergence_curve(empty, tmp_path / "c.png") is None
        assert per_route_mae(empty, tmp_path / "d.png") is None
        assert calibration_diagram(empty, tmp_path / "e.png") is None

    def test_per_route_mae_none_without_route_column(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"error_s": [1.0, 2.0], "predictor_id": ["a", "b"]})
        assert per_route_mae(df, tmp_path / "f.png") is None


class TestCostAccuracyPareto:
    def test_returns_none_without_latency(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        assert cost_accuracy_pareto(results_df, None, tmp_path / "p.png") is None
        assert cost_accuracy_pareto(results_df, {}, tmp_path / "p2.png") is None

    def test_returns_path_with_latency(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        latency = {
            "gtfsrt": {"count": 10, "p50_s": 0.001, "p95_s": 0.002, "p99_s": 0.003},
            "baseline:constant_speed": {
                "count": 10, "p50_s": 0.0005, "p95_s": 0.001, "p99_s": 0.0015,
            },
        }
        out = cost_accuracy_pareto(results_df, latency, tmp_path / "pareto.png")
        _assert_nonempty_png(out)

    def test_ignores_predictors_absent_from_results(
        self, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        latency = {"unknown_predictor": {"p50_s": 0.5}}
        assert cost_accuracy_pareto(results_df, latency, tmp_path / "p3.png") is None


class TestLoadLatencyFromSummary:
    def test_parses_inference_latency_block(self, tmp_path: Path) -> None:
        summary_path = tmp_path / "r.parquet.summary.json"
        payload = {
            "ticks": 5,
            "samples_saved": 10,
            "arrivals_saved": 2,
            "results_saved": 2,
            "predictor_errors": {},
            "detector_errors": {},
            "provenance": None,
            "inference_latency": {
                "gtfsrt": {"count": 5, "p50_s": 0.001, "p95_s": 0.002, "p99_s": 0.003},
            },
        }
        summary_path.write_text(json.dumps(payload))
        latency = load_latency_from_summary(summary_path)
        assert latency == payload["inference_latency"]

    def test_missing_block_returns_empty(self, tmp_path: Path) -> None:
        summary_path = tmp_path / "r.parquet.summary.json"
        summary_path.write_text(json.dumps({"ticks": 1}))
        assert load_latency_from_summary(summary_path) == {}


class TestRenderAllFigures:
    def test_writes_all_figures_without_summary(
        self, results_parquet: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "figs"
        written = render_all_figures(results_parquet, out_dir)
        expected_names = {
            "error_vs_horizon",
            "error_distribution",
            "convergence_curve",
            "per_route_mae",
            "calibration_diagram",
            "cost_accuracy_pareto",
        }
        assert set(written) == expected_names
        # No summary given -> Pareto skipped, everything else populated.
        assert written["cost_accuracy_pareto"] is None
        for name, path in written.items():
            if name == "cost_accuracy_pareto":
                continue
            _assert_nonempty_png(path)

    def test_writes_pareto_with_summary(self, results_parquet: Path, tmp_path: Path) -> None:
        summary_path = tmp_path / "r.parquet.summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "inference_latency": {
                        "gtfsrt": {"count": 5, "p50_s": 0.001, "p95_s": 0.002, "p99_s": 0.003},
                        "baseline:constant_speed": {
                            "count": 5, "p50_s": 0.0004, "p95_s": 0.001, "p99_s": 0.001,
                        },
                    }
                }
            )
        )
        out_dir = tmp_path / "figs2"
        written = render_all_figures(results_parquet, out_dir, summary_path=summary_path)
        _assert_nonempty_png(written["cost_accuracy_pareto"])


class TestFiguresCli:
    def test_cli_figures_writes_pngs(self, results_parquet: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "cli_figs"
        result = runner.invoke(
            app, ["figures", str(results_parquet), "--out-dir", str(out_dir)]
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "error_vs_horizon.png").exists()
        assert "Skipped cost_accuracy_pareto" in result.output

    def test_cli_figures_with_summary_enables_pareto(
        self, results_parquet: Path, tmp_path: Path
    ) -> None:
        summary_path = tmp_path / "r.parquet.summary.json"
        summary_path.write_text(
            json.dumps({"inference_latency": {"gtfsrt": {"p50_s": 0.001}}})
        )
        out_dir = tmp_path / "cli_figs2"
        result = runner.invoke(
            app,
            [
                "figures",
                str(results_parquet),
                "--out-dir",
                str(out_dir),
                "--summary",
                str(summary_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "cost_accuracy_pareto.png").exists()

    def test_cli_figures_missing_parquet_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["figures", str(tmp_path / "missing.parquet"), "--out-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "not found" in result.output
