"""Train an artifact on a live-feature frame (``stoptimes data frame --live``).

``stoptimes train run`` delegates here when the experiment's dataset has a sibling
``<frame>.assets/`` directory (written by ``write_live_assets``).  What differs from the
generic path:

* the split is the frame's own ``split`` column - ``train`` days fit the model, ``val`` days
  drive early stopping and the reported error (calendar split; the experiment's ``split``
  fractions are ignored);
* features are passed through with their missing values (gradient boosting handles NaN, so a
  cold start stays a cold start instead of dropping the row);
* the artifact carries the segment-statistics and route-vocabulary assets, the agency
  timezone and holiday region the temporal features need.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import (
    ArtifactAsset,
    ArtifactMetadata,
    DataWindow,
    ModelFormat,
    ModelScope,
    Provenance,
    ScopeLevel,
    artifact_dir,
    sha256_hex,
    write_asset,
    write_metadata,
)
from stoptimes.core.contracts.experiment import DataSelection, Experiment
from stoptimes.core.features.live import LIVE_FEATURES
from stoptimes.train.live_frame import LIVE_ASSETS_MANIFEST, TARGET_COLUMN

__all__ = ["HORIZON_EDGES_S", "locate_live_frame", "run_live_experiment"]

HORIZON_EDGES_S: tuple[float, ...] = (0, 120, 300, 600, 1200, 1800, float("inf"))
"""[0,2) [2,5) [5,10) [10,20) [20,30) 30+ minutes - the study's horizon buckets."""

_DEFAULT_ROUNDS_CAP = 3000
_DEFAULT_EARLY_STOPPING = 50


def locate_live_frame(selection: DataSelection, dataset_dir: str | Path | None) -> tuple[Path, Path] | None:
    """``(frame parquet, assets dir)`` when the selected dataset is a live frame, else ``None``."""
    from stoptimes.core.catalog import Catalog

    names = [*selection.datasets, *selection.sources]
    for name in names:
        entry = Catalog().get(name)
        if entry is not None:
            candidates = [Path(entry.location)]
        elif dataset_dir is not None:
            candidates = [Path(dataset_dir) / f"{name}.parquet"]
        else:
            try:
                from stoptimes.models.common.data import _default_dataset_search_paths

                candidates = [Path(p) for p in _default_dataset_search_paths(name)]
            except ImportError:  # pragma: no cover - models is a hard dependency of train
                candidates = []
        for frame in candidates:
            assets = frame.with_suffix(".assets")
            if frame.exists() and (assets / LIVE_ASSETS_MANIFEST).is_file():
                return frame, assets
    return None


def _horizon_scores(y: np.ndarray, pred: np.ndarray) -> list:
    from stoptimes.train.experiment import _horizon_scores as generic

    return generic(y, pred, edges=HORIZON_EDGES_S)


def _split_params(params: dict[str, Any]) -> tuple[dict[str, Any], int, int, int | None]:
    rest = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
    cap = int(rest.pop("n_estimators", _DEFAULT_ROUNDS_CAP))
    patience = int(rest.pop("early_stopping_rounds", _DEFAULT_EARLY_STOPPING))
    n_jobs = rest.pop("n_jobs", None)
    return rest, cap, patience, None if n_jobs is None else int(n_jobs)


def _verified_assets(assets_dir: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name, spec in manifest["assets"].items():
        data = (assets_dir / spec["file"]).read_bytes()
        if sha256_hex(data) != spec["sha256"]:
            raise ValueError(f"live asset {name!r} in {assets_dir} does not match the manifest hash")
        out[name] = data
    return out


def _fit(
    train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict[str, Any], seed: int
) -> Any:
    from xgboost import XGBRegressor

    rest, cap, patience, n_jobs = _split_params(params)
    rest.setdefault("objective", "reg:absoluteerror")
    model = XGBRegressor(
        n_estimators=cap,
        early_stopping_rounds=patience,
        eval_metric="mae",
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs if n_jobs is not None else 4,
        **rest,
    )
    model.fit(
        train[features],
        train[TARGET_COLUMN].to_numpy(np.float32),
        eval_set=[(val[features], val[TARGET_COLUMN].to_numpy(np.float32))],
        verbose=False,
    )
    return model


def run_live_experiment(
    experiment: Experiment,
    frame_path: Path,
    assets_dir: Path,
    *,
    registry_dir: str | Path | None = None,
    overwrite: bool = False,
) -> ArtifactMetadata:
    """Fit on the frame's train days, early-stop and score on its val days, write the artifact."""
    from stoptimes.models.common.metrics import compute_all_metrics
    from stoptimes.models.common.provenance import git_sha, lib_versions
    from stoptimes.train.experiment import ArtifactExistsError, _sanitize_artifact_id

    registry_root = Path(registry_dir) if registry_dir is not None else Path(experiment.outputs.registry_dir)
    artifact_id = _sanitize_artifact_id(experiment.name)
    target_dir = artifact_dir(registry_root, artifact_id)
    if target_dir.exists() and not overwrite:
        raise ArtifactExistsError(f"artifact {artifact_id!r} already exists at {target_dir}")
    if experiment.model.family not in ("xgboost", "xgb"):
        raise NotImplementedError(f"live frames train the 'xgboost' family, got {experiment.model.family!r}")

    manifest = json.loads((assets_dir / LIVE_ASSETS_MANIFEST).read_text(encoding="utf-8"))
    features = list(experiment.features) if experiment.features else list(LIVE_FEATURES)
    unknown = [f for f in features if f not in LIVE_FEATURES]
    if unknown:
        raise ValueError(f"not live features: {unknown}")
    frame = pd.read_parquet(frame_path)
    train, val = frame[frame["split"] == "train"], frame[frame["split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("a live frame needs both train and val days (val drives early stopping)")
    assets = _verified_assets(assets_dir, manifest)
    needed = {a for f in features for a in _feature_assets(f)}
    absent = sorted(needed - set(assets))
    if absent:
        raise ValueError(f"features {features} need assets {absent} missing from {assets_dir}")

    model = _fit(train, val, features, dict(experiment.model.params), experiment.seed)
    model.required_features = features
    pred = np.asarray(model.predict(val[features]), dtype=float)
    y_val = val[TARGET_COLUMN].to_numpy(float)
    metrics = compute_all_metrics(y_val, pred, prefix="test_")
    scalar = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    scalar["val_mae_seconds"] = scalar["test_mae_seconds"]
    scalar["best_iteration"] = float(model.best_iteration)

    versions = lib_versions()
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family=experiment.model.family,
        target=TARGET_COLUMN,
        format=ModelFormat.PICKLE,
        scope=ModelScope(
            level=ScopeLevel.GLOBAL, source_id=experiment.data.sources[0] if experiment.data.sources else None
        ),
        features=features,
        params={k: v for k, v in experiment.model.params.items() if isinstance(v, (int, float, str, bool))},
        data=DataWindow(
            dataset=frame_path.stem,
            dataset_sha256=None,
            sources=list(experiment.data.sources),
            start=_day_start(manifest["train_days"][0]),
            end=_day_start(manifest["val_days"][-1]),
            n_samples=len(train) + len(val),
            n_train=len(train),
            n_test=len(val),
        ),
        provenance=Provenance(
            created_at=datetime.now(timezone.utc),
            git_sha=git_sha(),
            python_version=versions.get("python"),
            lib_versions=versions,
            seed=experiment.seed,
        ),
        metrics=scalar,
        scores_by_horizon=_horizon_scores(y_val, pred),
        tags=list(experiment.outputs.tags),
        assets={
            name: ArtifactAsset(file=manifest["assets"][name]["file"], sha256=sha256_hex(data))
            for name, data in assets.items()
        },
        timezone=manifest["timezone"],
        holiday_region=manifest.get("holiday_region"),
    )
    notes = [
        f"trained on {len(manifest['train_days'])} service days "
        f"({manifest['train_days'][0]}..{manifest['train_days'][-1]}), early-stopped and scored on "
        f"{len(manifest['val_days'])} validation days ({manifest['val_days'][0]}..{manifest['val_days'][-1]}).",
        "the test_* metrics are the VALIDATION days used for early stopping (no test day was read); "
        "a held-out test day is scored by `stoptimes val`.",
        "training rows use segment statistics fitted out-of-fold by day; the shipped segment_stats asset "
        "is fitted on all training days.",
        f"best iteration {model.best_iteration} of at most {_split_params(dict(experiment.model.params))[1]}.",
    ]
    _write_live_artifact(model, meta, assets, registry_root, target_dir, notes)
    return meta


def _feature_assets(name: str) -> tuple[str, ...]:
    from stoptimes.core.features.registry import get

    return get(name).assets


def _day_start(day: str) -> datetime:
    return datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)


def _write_live_artifact(
    model: Any,
    meta: ArtifactMetadata,
    assets: dict[str, bytes],
    registry_root: Path,
    target_dir: Path,
    notes: list[str],
) -> None:
    import os

    from stoptimes.train.experiment import render_card

    registry_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=registry_root, prefix=f".train-{meta.artifact_id}-"))
    try:
        with open(tmp_dir / meta.model_filename, "wb") as fh:
            pickle.dump(model, fh)
        for name, data in assets.items():
            write_asset(tmp_dir, meta.assets[name].file, data)
        write_metadata(meta, tmp_dir)
        (tmp_dir / "card.md").write_text(render_card(meta, baseline_metrics={}, notes=notes), encoding="utf-8")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
