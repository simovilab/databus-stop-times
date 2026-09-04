"""
Segment-sequence LSTM/GRU trainer (roadmap 5.2).

`train_lstm` mirrors `xgb.train.train_xgboost`'s contract: same
dataset_name/route_id/pre_split/save_model conventions, same
`compute_all_metrics(prefix="val_"|"test_")` metric keys, same
`clip_predictions` post-processing, and the same
`ModelKey.generate` -> `get_registry().save_model` path so provenance
stamping (git SHA, lib versions, dataset hash) applies uniformly across
model families.

Unlike XGBoost, training here operates on *sequences* (one per trip
instance) rather than independent rows -- see `models/lstm/data.py` for the
grouping/scaling pipeline and `models/lstm/model.py` for the causal
unidirectional RNN. Requires the `.venv-dl` interpreter (torch); see
`models/lstm/README.md`.
"""

from __future__ import annotations

import copy
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))

from common.data import load_dataset
from common.keys import ModelKey
from common.metrics import compute_all_metrics
from common.registry import get_registry
from common.utils import clip_predictions, print_metrics_table, train_test_summary

from lstm.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    Preprocessor,
    apply_preprocessor,
    build_sequences,
    coerce_feature_dtypes,
    fit_preprocessor,
    iter_batches,
    pad_batch,
)
from lstm.model import SegmentSequenceModel, masked_loss

# Target/feature sanity bounds, matching common.data.ETADataset.clean_data
# defaults -- applied here too since train_lstm's pre_split path can bypass
# ETADataset.clean_data entirely (e.g. the day-grouped split used by
# scratchpad/hpo_xgboost.py-style callers reads the parquet directly).
_MAX_ETA_SECONDS = 3600 * 2
_MIN_DISTANCE_M = 10.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    try:
        # CPU-only LSTM/GRU training does not hit the CUDA-only nondeterministic
        # ops (e.g. atomic scatter-add on GPU) that this flag guards against,
        # so it's safe to request here. Guarded regardless: some torch builds
        # raise RuntimeError for ops without a deterministic CPU kernel, and a
        # trainer must never hard-fail on a determinism nicety.
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def _clean_bounds(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    df = df.dropna(subset=[target_col])
    df = df[(df[target_col] >= 0) & (df[target_col] <= _MAX_ETA_SECONDS)]
    if "distance_to_stop" in df.columns:
        df = df[df["distance_to_stop"] >= _MIN_DISTANCE_M]
    return df


def _flatten_masked(preds: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = mask.astype(bool)
    return target[m], preds[m]


_VALID_TARGET_TRANSFORMS = ("none", "log1p", "standardize")


def _fit_target_stats(train_seqs: List, target_transform: str) -> Dict[str, float]:
    """Fit whatever parameters `target_transform` needs, from TRAIN targets
    only (never val/test -- that would leak)."""
    if target_transform not in _VALID_TARGET_TRANSFORMS:
        raise ValueError(f"target_transform must be one of {_VALID_TARGET_TRANSFORMS}, got {target_transform!r}")
    if target_transform == "standardize":
        all_t = np.concatenate([s.target for s in train_seqs if s.target is not None])
        mean = float(np.mean(all_t))
        std = float(np.std(all_t))
        if std < 1e-6:
            std = 1.0
        return {"mean": mean, "std": std}
    return {}


def _forward_target(y: np.ndarray, target_transform: str, stats: Dict[str, float]) -> np.ndarray:
    """Map raw seconds -> the space the network is trained/predicts in."""
    if target_transform == "log1p":
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    if target_transform == "standardize":
        return (y - stats["mean"]) / stats["std"]
    return y


def _invert_target(y: np.ndarray, target_transform: str, stats: Dict[str, float]) -> np.ndarray:
    """Inverse of `_forward_target`: model-space -> raw seconds."""
    if target_transform == "log1p":
        return np.expm1(y)
    if target_transform == "standardize":
        return y * stats["std"] + stats["mean"]
    return y


def _transform_sequence_targets(sequences: List, target_transform: str, stats: Dict[str, float]) -> List:
    """Return copies of `sequences` with `.target` mapped through
    `_forward_target`. Used ONLY to build the training loss batches -- eval
    keeps raw-seconds targets and instead inverts the model's predictions,
    so metrics are always reported in real seconds."""
    out = []
    for s in sequences:
        if s.target is None:
            out.append(s)
            continue
        out.append(replace(s, target=_forward_target(s.target, target_transform, stats).astype(np.float32)))
    return out


def _jsonable_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """compute_all_metrics can return numpy scalar types (e.g. float32, from
    our torch->numpy predictions); the registry writes metadata straight to
    JSON, which chokes on those. Cast to plain python floats."""
    return {k: float(v) for k, v in metrics.items()}


@torch.no_grad()
def _run_eval(
    model: SegmentSequenceModel,
    sequences,
    batch_size: int,
    target_transform: str = "none",
    target_stats: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over `sequences` in eval mode, returning flattened
    (unmasked-only) (y_true, y_pred, row_pos) arrays in a single pass.

    `sequences[i].target` is assumed to be raw seconds (never
    transform-space) -- the model's raw output is inverted via
    `_invert_target` before being returned, so `y_true`/`y_pred` are always
    comparable in real seconds regardless of `target_transform`."""
    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_row_pos: List[np.ndarray] = []
    for batch in iter_batches(sequences, batch_size=batch_size, shuffle=False):
        preds = model(batch.x, batch.lengths).numpy()
        mask_np = batch.mask.numpy()
        y_np = batch.y.numpy()
        y_true, y_pred = _flatten_masked(preds, y_np, mask_np)
        y_pred = _invert_target(y_pred, target_transform, target_stats or {})
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_row_pos.append(batch.row_pos[mask_np])
    if not all_true:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_row_pos)


class LSTMSegmentModel:
    """Picklable wrapper around a trained `SegmentSequenceModel`.

    The registry pickles whatever `save_model` is given. A live nn.Module
    generally *can* be pickled, but it drags along autograd/device state
    that is fragile across torch versions and never what we want persisted.
    Instead this wrapper stores the plain `state_dict` (CPU tensors) plus
    the config needed to reconstruct the network, the fitted `Preprocessor`
    (medians + StandardScaler), and enough metadata for a standalone
    `predict(df)` -- no training-time objects (optimizer, live tensors on a
    non-CPU device, etc.) are retained.
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        config: Dict,
        preprocessor: Preprocessor,
        service_offset_h: float,
        batch_size: int = 64,
        target_transform: str = "none",
        target_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        self.state_dict = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.config = dict(config)
        self.preprocessor = preprocessor
        self.service_offset_h = service_offset_h
        self.batch_size = batch_size
        self.target_transform = target_transform
        self.target_stats = dict(target_stats or {})

    def _build_network(self) -> SegmentSequenceModel:
        net = SegmentSequenceModel(
            n_features=len(self.preprocessor.feature_cols),
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            rnn_type=self.config["rnn_type"],
        )
        net.load_state_dict(self.state_dict)
        net.eval()
        return net

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict per-row ETAs (seconds) for `df`, returned in `df`'s
        original row order (positional, independent of `df.index`)."""
        if len(df) == 0:
            return np.array([])

        work = coerce_feature_dtypes(df, self.preprocessor.feature_cols)
        sequences = build_sequences(
            work,
            feature_cols=self.preprocessor.feature_cols,
            target_col=None,
            service_offset_h=self.service_offset_h,
        )
        sequences = apply_preprocessor(sequences, self.preprocessor)

        net = self._build_network()
        out = np.full(len(df), np.nan, dtype=np.float64)
        with torch.no_grad():
            for batch in iter_batches(sequences, batch_size=self.batch_size, shuffle=False):
                preds = net(batch.x, batch.lengths).numpy()
                preds = _invert_target(preds, self.target_transform, self.target_stats)
                mask_np = batch.mask.numpy()
                rp = batch.row_pos[mask_np]
                out[rp] = preds[mask_np]

        if np.isnan(out).any():
            # Should not happen (build_sequences covers every input row),
            # but never silently return NaN ETAs.
            out = np.nan_to_num(out, nan=0.0)

        return clip_predictions(out)


def train_lstm(
    dataset_name: str = "sample_dataset",
    route_id: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    rnn_type: str = "lstm",
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    loss: str = "huber",
    target_transform: str = "none",
    service_offset_h: float = 7,
    save_model: bool = True,
    pre_split: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
    seed: int = 42,
) -> Dict:
    """
    Train and evaluate the segment-sequence LSTM/GRU model.

    Args:
        dataset_name: Dataset name (used when pre_split is None, and for
            provenance/registry bookkeeping either way).
        route_id: Optional route filter (only used when pre_split is None).
        hidden_size, num_layers, dropout, rnn_type: model architecture.
        learning_rate, batch_size, max_epochs, patience, weight_decay,
            grad_clip, loss: optimization hyperparameters. Early stopping is
            on validation MAE (seconds) with `patience` epochs of no
            improvement; the best-val-MAE weights are restored before
            scoring test.
        target_transform: "none" (default), "log1p", or "standardize". The
            network is trained/predicts in the transformed space; all
            reported metrics (val/test MAE, R², etc.) and early stopping are
            always computed in raw seconds -- predictions are inverted
            before comparison. Stats needed for the transform (e.g.
            standardize's mean/std) are fit on TRAIN targets only and
            persisted on the saved model for serving-time inversion.
        service_offset_h: hours subtracted from vp_ts before taking the
            calendar date, to fold cross-midnight trips onto the correct
            service day when grouping rows into sequences.
        save_model: whether to persist via the model registry.
        pre_split: optional (train_df, val_df, test_df) DataFrames -- when
            given, no dataset file is read at all.
        seed: random seed (python/numpy/torch); same seed -> same metrics.

    Returns:
        {"model": LSTMSegmentModel, "metrics": {...}, "metadata": {...}}
    """
    _seed_everything(seed)
    start_time = time.time()

    if pre_split is not None:
        train_df, val_df, test_df = (df.copy() for df in pre_split)
    else:
        dataset = load_dataset(dataset_name)
        dataset.clean_data()
        if route_id is not None:
            df = dataset.df
            df_filtered = df[df["route_id"] == route_id].copy()
            if len(df_filtered) == 0:
                raise ValueError(f"No data found for route {route_id}")
            dataset.df = df_filtered
        train_df, val_df, test_df = dataset.temporal_split()

    train_df = _clean_bounds(coerce_feature_dtypes(train_df))
    val_df = _clean_bounds(coerce_feature_dtypes(val_df))
    test_df = _clean_bounds(coerce_feature_dtypes(test_df))

    train_test_summary(train_df, test_df, val_df)

    train_seqs_raw = build_sequences(train_df, FEATURE_COLUMNS, TARGET_COLUMN, service_offset_h)
    val_seqs_raw = build_sequences(val_df, FEATURE_COLUMNS, TARGET_COLUMN, service_offset_h)
    test_seqs_raw = build_sequences(test_df, FEATURE_COLUMNS, TARGET_COLUMN, service_offset_h)

    if not train_seqs_raw:
        raise ValueError("No training sequences after grouping -- check service_offset_h / data")

    preprocessor = fit_preprocessor(train_seqs_raw, FEATURE_COLUMNS)
    train_seqs = apply_preprocessor(train_seqs_raw, preprocessor)
    val_seqs = apply_preprocessor(val_seqs_raw, preprocessor) if val_seqs_raw else []
    test_seqs = apply_preprocessor(test_seqs_raw, preprocessor) if test_seqs_raw else []

    target_stats = _fit_target_stats(train_seqs, target_transform)
    train_seqs_loss = _transform_sequence_targets(train_seqs, target_transform, target_stats)

    n_features = len(preprocessor.feature_cols)
    model = SegmentSequenceModel(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    rng = np.random.RandomState(seed)
    best_val_mae = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    epochs_since_improve = 0
    epochs_trained = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in iter_batches(train_seqs_loss, batch_size=batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad()
            preds = model(batch.x, batch.lengths)
            batch_loss = masked_loss(preds, batch.y, batch.mask, loss=loss)
            batch_loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        epochs_trained = epoch + 1

        if val_seqs:
            y_true, y_pred, _ = _run_eval(model, val_seqs, batch_size, target_transform, target_stats)
            val_mae = float(np.mean(np.abs(y_true - clip_predictions(y_pred))))
        else:
            val_mae = float("nan")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs_trained - 1

    wall_clock_seconds = time.time() - start_time

    # Validation metrics (full metric set, on best weights).
    if val_seqs:
        y_val, val_preds, _ = _run_eval(model, val_seqs, batch_size, target_transform, target_stats)
        val_preds = clip_predictions(val_preds)
        val_metrics = _jsonable_metrics(compute_all_metrics(y_val, val_preds, prefix="val_"))
        print_metrics_table(val_metrics, "Validation")
    else:
        val_metrics = {}

    # Test metrics.
    if test_seqs:
        y_test, test_preds, _ = _run_eval(model, test_seqs, batch_size, target_transform, target_stats)
        test_preds = clip_predictions(test_preds)
        test_metrics = _jsonable_metrics(compute_all_metrics(y_test, test_preds, prefix="test_"))
        print_metrics_table(test_metrics, "Test")
    else:
        test_metrics = {}

    config = dict(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
        loss=loss,
    )
    wrapped_model = LSTMSegmentModel(
        state_dict=best_state,
        config=config,
        preprocessor=preprocessor,
        service_offset_h=service_offset_h,
        batch_size=batch_size,
        target_transform=target_transform,
        target_stats=target_stats,
    )

    metadata = {
        "model_type": "lstm",
        "dataset": dataset_name,
        "route_id": route_id,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "rnn_type": rnn_type,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "loss": loss,
        "target_transform": target_transform,
        "target_stats": target_stats,
        "service_offset_h": service_offset_h,
        "n_features": n_features,
        "features": list(preprocessor.feature_cols),
        "seed": seed,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_mae_seconds": best_val_mae if val_seqs else None,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "test_sequences": len(test_seqs),
        "n_samples": len(train_df) + len(val_df) + len(test_df),
        "train_wall_clock_seconds": wall_clock_seconds,
        "metrics": {**val_metrics, **test_metrics},
    }

    if save_model:
        model_key = ModelKey.generate(
            model_type="lstm",
            dataset_name=dataset_name,
            feature_groups=["sequence"],
            route_id=route_id,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        registry = get_registry()
        registry.save_model(model_key, wrapped_model, metadata)
        metadata["model_key"] = model_key

    return {
        "model": wrapped_model,
        "metrics": metadata["metrics"],
        "metadata": metadata,
    }


if __name__ == "__main__":
    _ = train_lstm(dataset_name="sample_dataset", max_epochs=20, patience=5)
