"""Experiment tracking (docs/MLFLOW_INTEGRATION.md §0 D6): records, the best-effort writer, and the
publish run. Runs against a local sqlite MLflow store in tmp_path, no network."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    DataWindow,
    HorizonScore,
    ModelScope,
    Provenance,
)
from stoptimes.core.tracking import (
    TRACKING_ENV,
    RunRecord,
    Tracker,
    get_tracker,
    metric_key,
    tracking_uri,
    training_record,
)

pytest.importorskip("mlflow")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any default artifact root lands here, never in the repo
    return Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}")


def _meta() -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_id="xgb-live",
        family="xgboost",
        scope=ModelScope(source_id="mbta"),
        features=["distance_to_stop", "stops_ahead"],
        params={"max_depth": 8, "learning_rate": 0.06},
        data=DataWindow(dataset="mbta-live", n_train=100, n_test=10),
        provenance=Provenance(created_at=datetime.now(UTC), git_sha="abc123", seed=42),
        metrics={"test_mae_seconds": 93.2, "test_rmse_seconds": None},
        scores_by_horizon=[
            HorizonScore(horizon_min_s=0, horizon_max_s=120, n=50, mae_s=30.0),
            HorizonScore(horizon_min_s=600, n=5, mae_s=150.0),
        ],
    )


def _runs(tracker: Tracker, experiment: str):
    from mlflow import MlflowClient

    client = MlflowClient(tracker.uri)
    exp = client.get_experiment_by_name(experiment)
    assert exp is not None
    return client, client.search_runs([exp.experiment_id])


def test_tracking_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv(TRACKING_ENV, raising=False)
    assert tracking_uri() is None and get_tracker() is None
    monkeypatch.setenv(TRACKING_ENV, "mlflow+https://ml.example.org")
    assert tracking_uri() == "https://ml.example.org"
    assert tracking_uri("") is None  # an explicit '' turns it off despite the env
    assert get_tracker("http://other").uri == "http://other"


def test_metric_keys_are_mlflow_safe():
    assert (
        metric_key("mlflow:stoptimes-mbta@staging", "mae_s")
        == "mlflow_stoptimes-mbta_staging.mae_s"
    )


def test_training_record_has_params_horizons_and_baseline_children():
    record = training_record(
        _meta(), card="# card", baselines={"historical_mean": {"test_mae_seconds": 150.0}}
    )
    assert record.experiment == "stoptimes/mbta"
    assert record.params["model.max_depth"] == 8
    assert record.params["features"] == "distance_to_stop,stops_ahead"
    assert record.metrics["h0-120s.mae_s"] == 30.0
    assert record.metrics["h600s-inf.n"] == 5.0
    assert record.tags["mlflow.source.git.commit"] == "abc123"
    assert set(record.texts) == {"artifact.json", "card.md"}
    [child] = record.children
    assert child.metrics == {"test_mae_seconds": 150.0}


def test_tracker_writes_the_run_and_its_children(tracker):
    record = training_record(
        _meta(), card="# card", baselines={"historical_mean": {"test_mae_seconds": 150.0}}
    )
    run_id = tracker.log(record)
    assert run_id is not None
    client, runs = _runs(tracker, "stoptimes/mbta")
    by_id = {r.info.run_id: r for r in runs}
    parent = by_id[run_id]
    assert parent.info.status == "FINISHED"
    assert parent.data.params["model.learning_rate"] == "0.06"
    assert parent.data.metrics["test_mae_seconds"] == 93.2
    assert "test_rmse_seconds" not in parent.data.metrics  # None is skipped, not logged as NaN
    assert {a.path for a in client.list_artifacts(run_id)} == {"artifact.json", "card.md"}
    [child] = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == run_id]
    assert child.data.metrics["test_mae_seconds"] == 150.0


def test_many_params_and_metrics_are_split_into_batches(tracker):
    record = RunRecord(
        experiment="stoptimes/big",
        run_name="big",
        params={f"p{i}": i for i in range(250)},
        metrics={f"m{i}": float(i) for i in range(2500)},
        tags={f"t{i}": "x" for i in range(150)},
    )
    run_id = tracker.log(record)
    _, [run] = _runs(tracker, "stoptimes/big")
    assert run.info.run_id == run_id
    assert len(run.data.params) == 250 and len(run.data.metrics) == 2500
    assert sum(k.startswith("t") for k in run.data.tags) == 150


def test_logging_into_an_existing_run(tracker):
    first = tracker.log(RunRecord(experiment="stoptimes/x", run_name="publish", metrics={"a": 1.0}))
    again = tracker.log(
        RunRecord(experiment="stoptimes/x", run_name="ignored", metrics={"b": 2.0}), run_id=first
    )
    assert again == first
    _, [run] = _runs(tracker, "stoptimes/x")
    assert run.data.metrics == {"a": 1.0, "b": 2.0}


def test_failures_are_warnings_never_errors(caplog):
    broken = Tracker("http://127.0.0.1:9")  # nothing listens on the discard port

    class _Refusing:
        def get_experiment_by_name(self, name):
            raise ConnectionError("refused")

    broken._client_obj = _Refusing()
    assert broken.log(RunRecord(experiment="stoptimes/x", run_name="r")) is None
    assert "the run itself is unaffected" in caplog.text


def test_run_url_points_at_the_mlflow_ui():
    tracker = Tracker("https://ml.example.org/")
    tracker._experiments["stoptimes-val/mbta"] = "7"
    assert (
        tracker.run_url("stoptimes-val/mbta", "abc")
        == "https://ml.example.org/#/experiments/7/runs/abc"
    )
    assert tracker.run_url("unknown", "abc") is None
    assert Tracker("file:///tmp/x").run_url("stoptimes-val/mbta", "abc") is None
