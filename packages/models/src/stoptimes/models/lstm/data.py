"""
Sequence construction, feature scaling, and padded-tensor batching for the
segment-sequence LSTM/GRU trainer (roadmap 5.2).

A "sequence" is one trip instance: all segment rows for a single
(trip_id, service_date) pair, ordered by their position along the trip.
`service_date` folds a `vp_ts` timestamp back onto the *service day* it
belongs to (buses running past midnight still belong to the prior service
day), via `service_date = (vp_ts - Timedelta(hours=service_offset_h)).date()`.
This mirrors the day-grouped split used elsewhere in the project (see
`scratchpad/hpo_xgboost.py::day_grouped_split`) and is what makes a reused
`trip_id` (recurring on multiple days) resolve to *distinct* sequences.

Within a sequence, rows are ordered by `segment_index` (the position of the
segment within the trip), with `vp_ts` then `stop_sequence` as stable
tie-breakers so ordering is fully deterministic even if `segment_index` ever
repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence as SequenceType, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# The 20 numeric, online-serveable feature columns shared by both segment
# corpora (MBTA + bUCR). `bearing_diff` is dtype object with None-holes in
# the bUCR corpus; callers must coerce it (see `coerce_feature_dtypes`)
# before it reaches `build_sequences`.
FEATURE_COLUMNS: List[str] = [
    "distance_to_stop", "distance_to_next_stop", "shape_distance_to_stop",
    "shape_progress", "cross_track_error", "progress_ratio", "stops_ahead",
    "current_speed_kmh", "bearing_to_stop", "bearing_diff", "is_at_stop",
    "hour", "day_of_week", "is_weekend", "is_holiday", "is_peak_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

TARGET_COLUMN = "time_to_arrival_seconds"


@dataclass
class SequenceData:
    """One trip-instance sequence: raw (unscaled) feature/target arrays plus
    the positional row indices (into the source DataFrame, 0-based, as it
    was passed to `build_sequences`) each timestep came from."""

    key: Tuple[str, object]           # (trip_id, service_date)
    row_pos: np.ndarray               # (seq_len,) int64
    features: np.ndarray              # (seq_len, n_features) float64/32
    target: Optional[np.ndarray]      # (seq_len,) float32, or None


@dataclass
class Preprocessor:
    """Scaling/imputation fit on TRAIN rows only. Persisted with the model
    so val/test/serving inference reuses the exact same medians + scaler
    (never refit on val/test -- that would leak)."""

    feature_cols: List[str]
    medians: Dict[str, float]
    scaler: StandardScaler


def coerce_feature_dtypes(df: pd.DataFrame, feature_cols: SequenceType[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Return a copy of `df` with any object-dtype feature columns (notably
    `bearing_diff` in the bUCR corpus, which carries `None`s) coerced to
    numeric via `pd.to_numeric(errors="coerce")`. Non-numeric values become
    NaN and are handled by the train-fit median imputation downstream."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "bool":
            df[col] = pd.to_numeric(df[col], errors="coerce") if df[col].dtype == object else df[col].astype(float)
    # bool columns (is_at_stop, is_weekend, is_holiday, is_peak_hour) -> float
    for col in feature_cols:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(float)
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: SequenceType[str] = FEATURE_COLUMNS,
    target_col: Optional[str] = TARGET_COLUMN,
    service_offset_h: float = 7,
) -> List[SequenceData]:
    """
    Group `df` into per-trip-instance sequences.

    Args:
        df: segment corpus rows (any subset/order; positional index is what
            matters for `row_pos`, not `df.index`).
        feature_cols: numeric feature columns to extract (already coerced --
            see `coerce_feature_dtypes`).
        target_col: target column name, or None if `df` has no target
            (prediction-time use).
        service_offset_h: hours to subtract from `vp_ts` before taking the
            calendar date, so trips that run past local midnight still fold
            onto the prior service day. MBTA-style default 7; bUCR uses 6.

    Returns:
        List of `SequenceData`, one per (trip_id, service_date), sorted
        deterministically by that key. Rows within a sequence are ordered by
        (segment_index, vp_ts, stop_sequence).
    """
    if len(df) == 0:
        return []

    work = df.reset_index(drop=True).copy()
    work["_row_pos"] = np.arange(len(work))
    work["_service_date"] = (
        pd.to_datetime(work["vp_ts"]) - pd.Timedelta(hours=service_offset_h)
    ).dt.date

    available_features = [c for c in feature_cols if c in work.columns]
    has_target = target_col is not None and target_col in work.columns

    sequences: List[SequenceData] = []
    for key, g in work.groupby(["trip_id", "_service_date"], sort=True):
        g = g.sort_values(["segment_index", "vp_ts", "stop_sequence"], kind="mergesort")
        feats = g[available_features].to_numpy(dtype=np.float64)
        tgt = g[target_col].to_numpy(dtype=np.float64) if has_target else None
        sequences.append(
            SequenceData(
                key=key,
                row_pos=g["_row_pos"].to_numpy(dtype=np.int64),
                features=feats,
                target=tgt,
            )
        )
    return sequences


def fit_preprocessor(sequences: List[SequenceData], feature_cols: SequenceType[str] = FEATURE_COLUMNS) -> Preprocessor:
    """Fit per-feature medians (for NaN imputation) and a StandardScaler on
    TRAIN sequences only. Never call this on val/test data."""
    available = [c for c in feature_cols]
    if not sequences:
        raise ValueError("Cannot fit preprocessor on zero sequences")

    stacked = np.concatenate([s.features for s in sequences], axis=0)
    medians = np.nanmedian(stacked, axis=0)
    # A feature that is all-NaN in train has an undefined median; fall back to 0.
    medians = np.where(np.isnan(medians), 0.0, medians)

    imputed = np.where(np.isnan(stacked), medians, stacked)
    scaler = StandardScaler()
    scaler.fit(imputed)

    return Preprocessor(
        feature_cols=list(available),
        medians={c: float(m) for c, m in zip(available, medians)},
        scaler=scaler,
    )


def apply_preprocessor(sequences: List[SequenceData], preprocessor: Preprocessor) -> List[SequenceData]:
    """Return new `SequenceData` with features median-imputed + scaled per
    `preprocessor` (fit on train). Never mutates a sequence's mid-sequence
    length/order -- NaNs are imputed in place, rows are never dropped."""
    medians = np.array([preprocessor.medians[c] for c in preprocessor.feature_cols])
    out: List[SequenceData] = []
    for s in sequences:
        feat = s.features
        feat = np.where(np.isnan(feat), medians, feat)
        feat = preprocessor.scaler.transform(feat)
        out.append(replace(s, features=feat.astype(np.float32)))
    return out


@dataclass
class PaddedBatch:
    x: torch.Tensor        # (batch, max_len, n_features) float32
    mask: torch.Tensor     # (batch, max_len) bool
    y: torch.Tensor        # (batch, max_len) float32 (zeros where mask is False)
    lengths: torch.Tensor  # (batch,) int64
    row_pos: np.ndarray    # (batch, max_len) int64, -1 where padded


def pad_batch(sequences: List[SequenceData]) -> PaddedBatch:
    """Pad a list of `SequenceData` into batch tensors. Padded positions get
    mask=False, y=0.0, row_pos=-1 -- padding must never be mistaken for a
    real (zero-target) row downstream."""
    if not sequences:
        raise ValueError("Cannot pad an empty batch")

    lengths = [len(s.features) for s in sequences]
    max_len = max(lengths)
    n_feat = sequences[0].features.shape[1]
    batch = len(sequences)

    x = np.zeros((batch, max_len, n_feat), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=bool)
    y = np.zeros((batch, max_len), dtype=np.float32)
    row_pos = np.full((batch, max_len), -1, dtype=np.int64)

    for i, s in enumerate(sequences):
        L = len(s.features)
        x[i, :L] = s.features
        mask[i, :L] = True
        row_pos[i, :L] = s.row_pos
        if s.target is not None:
            y[i, :L] = s.target

    return PaddedBatch(
        x=torch.from_numpy(x),
        mask=torch.from_numpy(mask),
        y=torch.from_numpy(y),
        lengths=torch.tensor(lengths, dtype=torch.int64),
        row_pos=row_pos,
    )


def iter_batches(
    sequences: List[SequenceData],
    batch_size: int,
    shuffle: bool = False,
    rng: Optional[np.random.RandomState] = None,
):
    """Yield `PaddedBatch` chunks of `sequences`, optionally shuffled with a
    supplied (seeded) RandomState for determinism."""
    n = len(sequences)
    order = np.arange(n)
    if shuffle:
        (rng or np.random.RandomState()).shuffle(order)

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield pad_batch([sequences[i] for i in idx])
