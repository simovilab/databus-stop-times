"""Tests for stoptimes.train.experiment: run_experiment end to end against
the tracked sample dataset (packages/models/tests/fixtures/sample.parquet)."""

from __future__ import annotations

from pathlib import Path

import pytest

from stoptimes.core.contracts.artifact import ModelScope, ScopeLevel, artifact_dir, read_metadata
from stoptimes.core.contracts.experiment import (
    DataSelection,
    Experiment,
    ModelSpec,
    OutputSpec,
    SplitStrategy,
)
from stoptimes.train.experiment import (
    ArtifactExistsError,
    plan_experiment,
    resolve_dataset,
    run_experiment,
)

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "models" / "tests" / "fixtures"


def _experiment(name: str = "test-historical-mean", **overrides) -> Experiment:
    defaults = dict(
        name=name,
        data=DataSelection(datasets=["sample"]),
        target="time_to_arrival_seconds",
        model=ModelSpec(family="historical_mean", params={}),
        features=[],
        split=SplitStrategy(kind="time", train_fraction=0.7, test_fraction=0.3),
        seed=42,
        outputs=OutputSpec(registry_dir="unused-in-tests"),
        baselines=["schedule", "constant_speed"],
    )
    defaults.update(overrides)
    return Experiment(**defaults)


def test_resolve_dataset_loads_named_dataset():
    dataset = resolve_dataset(DataSelection(datasets=["sample"]), dataset_dir=_FIXTURES_DIR)
    assert len(dataset.df) > 0
    assert "time_to_arrival_seconds" in dataset.df.columns


def test_resolve_dataset_raises_clear_error_for_sources():
    with pytest.raises(NotImplementedError, match="data catalog"):
        resolve_dataset(DataSelection(sources=["mbta"]))


def test_plan_experiment_is_pure_and_matches_experiment():
    experiment = _experiment()
    plan = plan_experiment(experiment)
    assert plan["name"] == "test-historical-mean"
    assert plan["model"]["family"] == "historical_mean"
    assert plan["data"]["datasets"] == ["sample"]
    assert plan["baselines"] == ["schedule", "constant_speed"]


def test_run_experiment_writes_artifact_with_scores(tmp_path):
    experiment = _experiment(
        features=["route_id", "stop_sequence", "hour"],
    )
    meta = run_experiment(experiment, registry_dir=tmp_path, dataset_dir=_FIXTURES_DIR)

    assert meta.artifact_id == "test-historical-mean"
    assert meta.family == "historical_mean"
    assert "test_mae_seconds" in meta.metrics
    assert meta.metrics["test_mae_seconds"] >= 0
    assert meta.scores_by_horizon
    assert meta.data.n_train > 0 and meta.data.n_test > 0

    target_dir = artifact_dir(tmp_path, "test-historical-mean")
    assert (target_dir / "artifact.json").exists()
    assert (target_dir / "model.pkl").exists()
    assert (target_dir / "card.md").exists()

    card = (target_dir / "card.md").read_text()
    assert "historical_mean" in card
    assert "schedule" in card
    assert "constant_speed" in card

    reloaded = read_metadata(target_dir)
    assert reloaded.artifact_id == meta.artifact_id


def test_run_experiment_refuses_to_overwrite_silently(tmp_path):
    experiment = _experiment(features=["route_id", "stop_sequence", "hour"])
    run_experiment(experiment, registry_dir=tmp_path, dataset_dir=_FIXTURES_DIR)

    with pytest.raises(ArtifactExistsError):
        run_experiment(experiment, registry_dir=tmp_path, dataset_dir=_FIXTURES_DIR)

    # overwrite=True is allowed to replace it.
    meta = run_experiment(experiment, registry_dir=tmp_path, dataset_dir=_FIXTURES_DIR, overwrite=True)
    assert meta.artifact_id == "test-historical-mean"


def test_run_experiment_is_deterministic_given_seed(tmp_path):
    experiment = _experiment(features=["route_id", "stop_sequence", "hour"])
    meta1 = run_experiment(experiment, registry_dir=tmp_path / "a", dataset_dir=_FIXTURES_DIR)
    meta2 = run_experiment(experiment, registry_dir=tmp_path / "b", dataset_dir=_FIXTURES_DIR)
    assert meta1.metrics["test_mae_seconds"] == pytest.approx(meta2.metrics["test_mae_seconds"])
    assert meta1.data.n_train == meta2.data.n_train


def test_run_experiment_with_route_scope(tmp_path):
    experiment = _experiment(
        name="route-scoped",
        data=DataSelection(datasets=["sample"], routes=["Red"]),
        features=["route_id", "stop_sequence", "hour"],
    )
    meta = run_experiment(experiment, registry_dir=tmp_path, dataset_dir=_FIXTURES_DIR)
    assert meta.scope.level == ScopeLevel.ROUTE
    assert meta.scope.route_ids == ["Red"]


def test_run_experiment_xgboost_family(tmp_path):
    pytest.importorskip("xgboost")
    experiment = _experiment(
        name="xgb-smoke",
        model=ModelSpec(family="xgb", params={"n_estimators": 5, "max_depth": 2}),
        features=["distance_to_stop", "hour", "is_peak_hour"],
    )
    meta = run_experiment(experiment, registry_dir=tmp_path, dataset_dir=_FIXTURES_DIR)
    assert meta.family == "xgb"
    assert meta.metrics["test_mae_seconds"] >= 0
