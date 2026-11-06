"""`run_experiment(..., tracker=)` logs the training run to MLflow (D6), best-effort.

Local sqlite MLflow stores in tmp_path, no network."""

from __future__ import annotations

import pytest
from stoptimes.core.tracking import Tracker

from tests.test_experiment import _FIXTURES_DIR
from tests.test_experiment import _experiment as _base_experiment

pytest.importorskip("mlflow")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.chdir(tmp_path)  # default artifact roots land here, never in the repo


def _experiment():
    return _base_experiment(features=["route_id", "stop_sequence", "hour"])


def _runs(uri: str, experiment: str):
    from mlflow import MlflowClient

    client = MlflowClient(uri)
    exp = client.get_experiment_by_name(experiment)
    assert exp is not None, experiment
    return client, client.search_runs([exp.experiment_id])


def test_filesystem_registry_gets_a_tracked_run_with_baseline_children(tmp_path):
    from stoptimes.train.experiment import run_experiment

    uri = f"sqlite:///{tmp_path / 'track.db'}"
    meta = run_experiment(
        _experiment(),
        registry_dir=tmp_path / "reg",
        dataset_dir=_FIXTURES_DIR,
        tracker=Tracker(uri),
    )
    client, runs = _runs(uri, "stoptimes/global")
    [parent] = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    assert parent.data.tags["stoptimes.artifact_id"] == meta.artifact_id
    assert parent.data.metrics["test_mae_seconds"] == pytest.approx(
        meta.metrics["test_mae_seconds"]
    )
    assert {"artifact.json", "card.md", "experiment.json"} <= {
        a.path for a in client.list_artifacts(parent.info.run_id)
    }
    children = sorted(
        r.data.params["family"]
        for r in runs
        if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    )
    assert children == ["constant_speed", "schedule"]


def test_mlflow_registry_on_the_same_server_reuses_the_publish_run(tmp_path):
    from stoptimes.core.registry.mlflow_backend import MlflowRegistry
    from stoptimes.train.experiment import run_experiment

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MlflowRegistry(
        uri, artifact_location=str(tmp_path / "art"), cache_dir=tmp_path / "cache", list_ttl_s=0
    )
    meta = run_experiment(
        _experiment(), registry_dir=registry, dataset_dir=_FIXTURES_DIR, tracker=Tracker(uri)
    )
    version = registry.resolve(meta.artifact_id)
    client, runs = _runs(uri, "stoptimes/global")
    top = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    assert [r.info.run_id for r in top] == [version.run_id]  # one record: the version's run
    assert "experiment.json" in {a.path for a in client.list_artifacts(version.run_id)}
    assert sum(r.data.tags.get("mlflow.parentRunId") == version.run_id for r in runs) == 2


def test_an_unreachable_tracker_does_not_fail_training(tmp_path, caplog):
    from stoptimes.train.experiment import run_experiment

    tracker = Tracker("http://127.0.0.1:9")

    class _Down:
        def get_experiment_by_name(self, name):
            raise ConnectionError("refused")

    tracker._client_obj = _Down()
    meta = run_experiment(
        _experiment(), registry_dir=tmp_path / "reg", dataset_dir=_FIXTURES_DIR, tracker=tracker
    )
    assert (tmp_path / "reg" / meta.artifact_id / "artifact.json").is_file()
    assert "the run itself is unaffected" in caplog.text
