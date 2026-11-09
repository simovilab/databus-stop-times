"""``train run`` on a live frame: calendar split, early stopping, sidecar assets in the artifact."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.contracts.artifact import read_metadata
from stoptimes.core.contracts.experiment import Experiment
from stoptimes.core.features.live import LIVE_FEATURES, LiveAssets, assemble_live_features
from stoptimes.train.experiment import ArtifactExistsError, run_experiment
from stoptimes.train.live_frame import write_live_assets

from .test_live_frame import TZ, VAL, dataset, feed, observations, records  # noqa: F401  (fixtures)


@pytest.fixture()
def frame_dir(tmp_path, dataset):  # noqa: F811
    frame = dataset.frame
    frame.to_parquet(tmp_path / "live.parquet", index=False)
    write_live_assets(tmp_path / "live.assets", dataset)
    return tmp_path


def _experiment(**over):
    spec = {
        "name": "xgb-live-test",
        "data": {"datasets": ["live"]},
        "target": "time_to_arrival_seconds",
        "model": {"family": "xgboost", "params": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.3,
                                                  "early_stopping_rounds": 5, "n_jobs": 2}},
        "features": list(LIVE_FEATURES),
        "split": {"kind": "time", "train_fraction": 0.9, "test_fraction": 0.1},
        "baselines": [],
    }
    spec.update(over)
    return Experiment.model_validate(spec)


def test_train_run_writes_artifact_with_verified_assets(frame_dir, tmp_path):
    registry = tmp_path / "registry"
    meta = run_experiment(_experiment(), registry_dir=registry, dataset_dir=frame_dir)
    directory = registry / meta.artifact_id
    back = read_metadata(directory)
    assert back.features == list(LIVE_FEATURES) and back.timezone == TZ and back.holiday_region == "US_MA"
    assert set(back.assets) == {"segment_stats", "route_vocabulary"}
    for name, asset in back.assets.items():
        from stoptimes.core.contracts.artifact import read_asset_bytes

        read_asset_bytes(directory, name, asset)  # hash verified
    assert back.metrics["val_mae_seconds"] == back.metrics["test_mae_seconds"] > 0
    assert back.metrics["best_iteration"] >= 0
    assert [h.horizon_min_s for h in back.scores_by_horizon][0] == 0
    assert back.data.n_train > 0 and back.data.n_test > 0
    assert "VALIDATION days" in (directory / "card.md").read_text()
    with pytest.raises(ArtifactExistsError):
        run_experiment(_experiment(), registry_dir=registry, dataset_dir=frame_dir)
    run_experiment(_experiment(), registry_dir=registry, dataset_dir=frame_dir, overwrite=True)


def test_trained_artifact_serves_through_the_loader_and_live_assembly(frame_dir, tmp_path, dataset):  # noqa: F811
    from stoptimes.serve.model_loading import load_artifact_assets, load_artifact_model

    registry = tmp_path / "registry"
    meta = run_experiment(_experiment(), registry_dir=registry, dataset_dir=frame_dir)
    model = load_artifact_model(meta, registry, allow_unproducible_features=True)  # serving inputs: F4b
    assets = load_artifact_assets(meta, registry)
    assert isinstance(assets, LiveAssets) and assets.segment_stats.n_obs == dataset.assets.segment_stats.n_obs
    val = dataset.frame[dataset.frame.split == "val"]
    pred = model.predict(val[meta.features].head(20))
    assert pred.shape == (20,) and np.isfinite(pred).all()
    assert model.required_features == meta.features


def test_early_stopping_uses_only_the_val_days(frame_dir, tmp_path, dataset):  # noqa: F811
    registry = tmp_path / "registry"
    meta = run_experiment(_experiment(), registry_dir=registry, dataset_dir=frame_dir)
    with open(registry / meta.artifact_id / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    val = dataset.frame[dataset.frame.split == "val"]
    mae = float(np.abs(model.predict(val[meta.features]) - val["time_to_arrival_seconds"]).mean())
    assert mae == pytest.approx(meta.metrics["val_mae_seconds"], rel=1e-5)
    assert set(val["day"]) == {d.replace("-", "") for d in VAL}


def test_missing_validation_days_is_an_error(frame_dir, tmp_path):
    frame = pd.read_parquet(frame_dir / "live.parquet")
    frame[frame.split == "train"].to_parquet(frame_dir / "live.parquet", index=False)
    with pytest.raises(ValueError, match="train and val"):
        run_experiment(_experiment(), registry_dir=tmp_path / "r", dataset_dir=frame_dir)


def test_tampered_asset_blocks_training(frame_dir, tmp_path):
    (frame_dir / "live.assets" / "route_vocabulary.json").write_text('{"route_ids": ["zz"]}')
    with pytest.raises(ValueError, match="manifest hash"):
        run_experiment(_experiment(), registry_dir=tmp_path / "r", dataset_dir=frame_dir)


def test_routes_make_a_route_specific_model(frame_dir, tmp_path, dataset):  # noqa: F811
    exp = _experiment(data={"datasets": ["live"], "routes": ["rA"]})
    meta = run_experiment(exp, registry_dir=tmp_path / "r", dataset_dir=frame_dir)
    assert meta.scope.level.value == "route" and meta.scope.route_ids == ["rA"]
    rA = dataset.frame[dataset.frame.route_id == "rA"]
    assert meta.data.n_train == int((rA.split == "train").sum())
    assert meta.data.n_test == int((rA.split == "val").sum())
    everything = run_experiment(
        _experiment(name="all"), registry_dir=tmp_path / "r", dataset_dir=frame_dir
    )
    assert everything.scope.level.value == "global" and everything.data.n_train > meta.data.n_train


def test_unknown_route_is_an_error(frame_dir, tmp_path):
    with pytest.raises(ValueError, match=r"routes \['nope'\]"):
        run_experiment(
            _experiment(data={"datasets": ["live"], "routes": ["nope"]}),
            registry_dir=tmp_path / "r",
            dataset_dir=frame_dir,
        )
