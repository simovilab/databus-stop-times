"""
Shared preprocessing utilities for all model families (roadmap 5.3).

Centralizes two apples-to-apples concerns for the model leaderboard:

  1. NaN handling — one policy implementation (drop / impute / error).
     Imputation statistics are always fit on the dataframe you hand in and
     nothing else; callers MUST fit on train and pass the resulting stats
     into val/test calls, never recompute stats from the evaluation split.
  2. Prediction clipping — one canonical bound (0s .. 7200s / 2h), applied
     identically by every family, since predictions are seconds-to-arrival
     and negative or wildly-large values are physically impossible.

Every one of ewma, historical_mean, polyreg_distance, polyreg_time, and xgb
routes through this module so the leaderboard compares like with like.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _compute_impute_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Median for numeric columns, mode for everything else (bool included)."""
    stats: Dict[str, Any] = {}
    for col in columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            mode = series.mode(dropna=True)
            stats[col] = bool(mode.iloc[0]) if not mode.empty else False
        elif pd.api.types.is_numeric_dtype(series):
            median = series.median()
            stats[col] = float(median) if pd.notna(median) else 0.0
        else:
            mode = series.mode(dropna=True)
            stats[col] = mode.iloc[0] if not mode.empty else "missing"
    return stats


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    handle_nan: str = "drop",
    impute_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Index, Dict[str, Any]]:
    """
    Apply one NaN-handling policy to ``df[feature_cols]``.

    Args:
        df: Source dataframe — train, val, test, or a single-row serving
            frame, this function does not care which.
        feature_cols: Feature columns to prepare. Columns absent from `df`
            are silently skipped (mirrors the existing "available features"
            behavior in each family).
        handle_nan:
            - "drop": rows with any NaN in the feature columns are removed.
            - "impute": NaN is filled with per-column statistics (median
              for numeric/bool columns, mode otherwise).
            - "error": raises ValueError if any NaN is present.
        impute_stats: Statistics to use for "impute" mode. Pass the `stats`
            dict returned from calling this function on the TRAINING split
            so val/test are imputed with train-only statistics — never
            their own. If None, stats are computed from `df` itself; only
            do this for the split you are fitting on (i.e. train).

    Returns:
        (X, kept_index, stats)
          X: DataFrame of the available feature columns after the policy
             is applied.
          kept_index: index (relative to `df`) of the rows retained in X.
             For "impute" and "error" this is `df.index` unchanged (no
             rows are ever dropped); for "drop" it is the surviving subset
             — use it to align `y` (`y.loc[kept_index]`).
          stats: the impute stats actually used. Empty for "drop"/"error".
             When `impute_stats` was supplied, this is exactly that dict
             (so callers can confirm they got train stats, not fresh ones).
    """
    available = [c for c in feature_cols if c in df.columns]

    if handle_nan == "error":
        if df[available].isna().any().any():
            raise ValueError(
                "NaN values found in feature columns and handle_nan='error'"
            )
        return df[available].copy(), df.index, {}

    if handle_nan == "drop":
        mask = ~df[available].isna().any(axis=1)
        kept_index = df.index[mask]
        return df.loc[kept_index, available].copy(), kept_index, {}

    if handle_nan == "impute":
        stats = (
            impute_stats
            if impute_stats is not None
            else _compute_impute_stats(df, available)
        )
        X = df[available].copy()
        for col in available:
            if col in stats and X[col].isna().any():
                X[col] = X[col].fillna(stats[col])
        return X, df.index, stats

    raise ValueError(
        f"Unknown handle_nan: {handle_nan!r} (expected 'drop', 'impute', or 'error')"
    )


def clip_predictions(
    predictions: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 7200.0,
) -> np.ndarray:
    """
    Canonical prediction clipping, shared by every model family.

    Predictions are seconds-to-arrival: negative values are physically
    impossible, hence min_value=0.0. Values beyond max_value=7200.0
    (2 hours) are treated as unreliable extrapolation rather than genuine
    ETAs and are capped rather than left to blow up MAE/RMSE. All five
    families (ewma, historical_mean, polyreg_distance, polyreg_time, xgb)
    must call this with the same bounds — differing bounds per family
    would silently bias the leaderboard toward whichever family clips
    more (or less) aggressively.
    """
    return np.clip(predictions, min_value, max_value)


def assert_same_row_set(
    index_a: pd.Index,
    index_b: pd.Index,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    """
    Assert two families were scored on the identical set of rows.

    This is the check that makes "every model trained on identical splits"
    provable instead of assumed: if one family silently drops NaN rows
    that another imputes, it gets scored on a smaller, easier subset and
    the leaderboard comparison is not apples-to-apples. Raises
    AssertionError with counts and a few example differing indices.
    """
    set_a, set_b = set(index_a), set(index_b)
    if set_a != set_b:
        only_a = sorted(set_a - set_b)[:5]
        only_b = sorted(set_b - set_a)[:5]
        raise AssertionError(
            f"Row sets differ between {name_a!r} ({len(set_a)} rows) and "
            f"{name_b!r} ({len(set_b)} rows): "
            f"{len(set_a - set_b)} row(s) only in {name_a!r} (e.g. {only_a}), "
            f"{len(set_b - set_a)} row(s) only in {name_b!r} (e.g. {only_b})"
        )
