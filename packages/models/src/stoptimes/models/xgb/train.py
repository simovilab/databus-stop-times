"""
XGBoost Regression Model - Time & Spatial Features
Fits a gradient boosted tree model with temporal, spatial, and
kinematic features for ETA prediction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).parent.parent))

from common.data import load_dataset
from common.metrics import compute_all_metrics
from common.keys import ModelKey
from common.registry import get_registry
from common.utils import print_metrics_table, train_test_summary
from common.preprocessing import LegacyPickleDefaults, legacy_impute_stats, prepare_features, clip_predictions


def _make_one_hot_encoder() -> OneHotEncoder:
    """Return a OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class XGBTimeModel(LegacyPickleDefaults):
    """
    Gradient boosted tree model (XGBoost) with temporal/spatial features.
    
    Features (all serveable online by eta_service.estimator):
    - Core: distance_to_stop
    - Temporal: hour, day_of_week, is_weekend, is_peak_hour + cyclical sin/cos
    - Spatial: distance_to_next_stop, shape_distance_to_stop, shape_progress,
      cross_track_error, progress_ratio, stops_ahead
    - Kinematic: current_speed_kmh, bearing_to_stop, bearing_diff
    - Status: is_at_stop
    """

    _POST_PICKLE_DEFAULTS = {"impute_stats": dict, "handle_nan": lambda: "drop"}


    def __init__(
        self,
        max_depth: int = 5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        early_stopping_rounds: Optional[int] = None,
        include_temporal: bool = True,
        include_spatial: bool = True,
        handle_nan: str = "drop",  # 'drop', 'impute', or 'error'
        random_state: int = 42,
        n_jobs: int = 4,
    ) -> None:
        """
        Initialize model.

        Args:
            max_depth: Maximum tree depth
            n_estimators: Number of boosting stages
            learning_rate: Learning rate (eta)
            subsample: Row subsample ratio per tree
            colsample_bytree: Column subsample ratio per tree
            include_temporal: Include time-of-day features
            include_spatial: Include spatial/kinematic/status features
            handle_nan: How to handle NaN - 'drop', 'impute', or 'error'
            random_state: Random seed
            n_jobs: Number of parallel threads
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.best_iteration: Optional[int] = None
        self.include_temporal = include_temporal
        self.include_spatial = include_spatial
        self.handle_nan = handle_nan
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: Optional[XGBRegressor] = None
        self.feature_cols: Optional[List[str]] = None
        self.available_features: Optional[List[str]] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.categorical_features: List[str] = []
        self.numeric_features: List[str] = []
        self.impute_stats: Dict = {}

    def _get_feature_groups(self) -> Dict[str, List[str]]:
        """Define feature groups (all computable online for serving parity)."""
        return {
            "core": ["distance_to_stop"],
            "spatial_numeric": (
                [
                    "distance_to_next_stop", "shape_distance_to_stop", "shape_progress",
                    "cross_track_error", "progress_ratio", "stops_ahead",
                    "current_speed_kmh", "bearing_to_stop", "bearing_diff", "is_at_stop",
                ]
                if self.include_spatial
                else []
            ),
            "temporal": (
                [
                    "hour", "day_of_week", "is_weekend", "is_peak_hour",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                ]
                if self.include_temporal
                else []
            ),
        }

    def _build_preprocessor(self) -> None:
        """Create a ColumnTransformer to handle numeric + categorical features."""
        if not self.feature_cols:
            raise ValueError("Feature columns must be set before building the preprocessor")

        transformers = []
        numeric_features = [
            col for col in self.feature_cols if col not in self.categorical_features
        ]
        self.numeric_features = numeric_features

        if numeric_features:
            transformers.append((
                "numeric",
                "passthrough",
                numeric_features,
            ))

        if self.categorical_features:
            transformers.append((
                "categorical",
                _make_one_hot_encoder(),
                self.categorical_features,
            ))

        self.preprocessor = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,
        )

    def _clean_data(
        self,
        df: pd.DataFrame,
        target_col: str = "time_to_arrival_seconds",
    ) -> pd.DataFrame:
        """
        Clean data and handle NaN values.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Cleaned dataframe
        """
        # Get all potential features
        feature_groups = self._get_feature_groups()
        all_features: List[str] = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)

        # Keep features that are present and not mostly-NaN (>30% drops the feature)
        available: List[str] = []
        for feat in all_features:
            if feat not in df.columns:
                continue
            if df[feat].isna().sum() / len(df) <= 0.3:
                available.append(feat)

        # Store available features
        self.available_features = available

        # Unify NaN handling with the other families via the shared
        # policy implementation. Imputation stats are fit HERE, on
        # whatever `df` is passed in — callers must only pass the
        # training split so val/test never leak their own statistics
        # (see predict(), which reuses self.impute_stats instead of
        # recomputing from the eval split).
        X_feat, kept_index, stats = prepare_features(
            df, available, handle_nan=self.handle_nan
        )
        if self.handle_nan == "impute":
            self.impute_stats = stats
        df_clean = df.loc[kept_index].copy()
        df_clean[available] = X_feat
        # Target NaN is expected to already be filtered out upstream by
        # ETADataset.clean_data(drop_missing_target=True), but guard anyway.
        df_clean = df_clean.dropna(subset=[target_col])

        # Always keep a train-fit fallback available for predict()-time
        # safety, regardless of handle_nan policy (predict() must return
        # one row per input row and can never drop).
        if not self.impute_stats:
            _, _, self.impute_stats = prepare_features(df, available, handle_nan="impute")

        # Validate no NaN remains
        remaining_nan = df_clean[available + [target_col]].isna().sum().sum()
        if remaining_nan > 0:
            raise ValueError(f"{remaining_nan} NaN values remain after cleaning!")

        return df_clean

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "time_to_arrival_seconds",
        eval_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Train model.

        Args:
            train_df: Training dataframe
            target_col: Target column name
            eval_df: Optional validation dataframe. Only used when
                early_stopping_rounds is set; the tree count then grows until
                validation MAE stops improving instead of a fixed n_estimators.
        """
        # Clean data
        train_clean = self._clean_data(train_df, target_col)

        # Build feature list from available features
        feature_groups = self._get_feature_groups()
        self.feature_cols = []

        group_order = [
            "core",
            "spatial_numeric",
            "spatial_categorical",
            "temporal",
        ]
        for group in group_order:
            for feat in feature_groups.get(group, []):
                if feat in self.available_features and feat not in self.feature_cols:
                    self.feature_cols.append(feat)

        if not self.feature_cols:
            raise ValueError("No features available after cleaning!")

        self.categorical_features = []
        self._build_preprocessor()

        # Prepare data
        X = train_clean[self.feature_cols]
        y = train_clean[target_col].values

        X_transformed = self.preprocessor.fit_transform(X)

        # Optional early-stopping eval set (transformed through the fitted
        # preprocessor; mirrors handle_nan='drop' on the eval frame).
        use_early_stopping = (
            self.early_stopping_rounds is not None and eval_df is not None
        )
        eval_set = None
        if use_early_stopping:
            eval_clean = eval_df.dropna(subset=self.feature_cols + [target_col])
            X_eval = self.preprocessor.transform(eval_clean[self.feature_cols])
            y_eval = eval_clean[target_col].values
            eval_set = [(X_eval, y_eval)]

        # Create and fit model
        xgb_kwargs = dict(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        if use_early_stopping:
            xgb_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            xgb_kwargs["eval_metric"] = "mae"

        self.model = XGBRegressor(**xgb_kwargs)
        if use_early_stopping:
            self.model.fit(X_transformed, y, eval_set=eval_set, verbose=False)
            self.best_iteration = getattr(self.model, "best_iteration", None)
        else:
            self.model.fit(X_transformed, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ETAs.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Array of predicted ETAs in seconds
        """
        if self.model is None:
            raise ValueError("Model not trained")

        if self.feature_cols is None or self.preprocessor is None:
            raise ValueError("Feature/preprocessor not set")

        # Check for missing features
        missing = [f for f in self.feature_cols if f not in X.columns]
        if missing:
            raise ValueError(f"Missing features in input: {missing}")

        # predict() must always return one prediction per input row (other
        # families' evaluation/leaderboard code assumes len(preds) ==
        # len(X)), so NaN is always filled here with TRAIN-fit stats —
        # never dropped, never re-derived from X itself. This replaces the
        # old behavior where 'impute' recomputed medians from the eval
        # split (a stats leak) and 'drop' silently filled with a magic 0.0
        # instead of actually dropping.
        if self.handle_nan == "error":
            if X[self.feature_cols].isna().any().any():
                raise ValueError("NaN values found and handle_nan='error'")
            X_pred = X[self.feature_cols].copy()
        else:
            X_pred, _, _ = prepare_features(
                X, self.feature_cols, handle_nan="impute",
                impute_stats=legacy_impute_stats(self.impute_stats, X, self.feature_cols)
            )

        X_transformed = self.preprocessor.transform(X_pred[self.feature_cols])
        predictions = self.model.predict(X_transformed)

        return clip_predictions(predictions)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances as reported by XGBoost."""
        if self.model is None or self.preprocessor is None:
            return {}

        if not hasattr(self.model, "feature_importances_"):
            return {}

        importances = self.model.feature_importances_
        feature_names = self.preprocessor.get_feature_names_out()

        # Map back to feature names
        importance = {
            feat: float(imp)
            for feat, imp in zip(feature_names, importances)
        }

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def train_xgboost(
    dataset_name: str = "sample_dataset",
    route_id: Optional[str] = None,
    max_depth: int = 5,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    early_stopping_rounds: Optional[int] = None,
    include_temporal: bool = True,
    include_spatial: bool = True,
    handle_nan: str = "drop",
    test_size: float = 0.2,
    save_model: bool = True,
    pre_split: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
) -> Dict:
    """
    Train and evaluate XGBoost time model.
    
    Args:
        dataset_name: Dataset name
        route_id: Optional route ID for route-specific training
        max_depth: Maximum tree depth
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate (eta)
        subsample: Row subsample ratio per tree
        colsample_bytree: Column subsample ratio per tree
        include_temporal: Include temporal features
        include_spatial: Include spatial/kinematic/status features
        handle_nan: 'drop', 'impute', or 'error'
        test_size: Test set fraction (test); val is fixed at 0.1
        save_model: Save to registry
        pre_split: Optional (train, val, test) DataFrames to reuse
        
    Returns:
        Dictionary with model, metrics, metadata
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    dataset.clean_data()

    # Filter by route if specified
    if route_id is not None:
        df = dataset.df
        df_filtered = df[df["route_id"] == route_id].copy()

        if len(df_filtered) == 0:
            raise ValueError(f"No data found for route {route_id}")

        dataset.df = df_filtered

    # Split data (train / val / test)
    if pre_split is not None:
        train_df, val_df, test_df = (df.copy() for df in pre_split)
    else:
        train_df, val_df, test_df = dataset.temporal_split(
            train_frac=1 - test_size - 0.1,
            val_frac=0.1,
        )

    train_test_summary(train_df, test_df, val_df)

    # Train model
    model = XGBTimeModel(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        early_stopping_rounds=early_stopping_rounds,
        include_temporal=include_temporal,
        include_spatial=include_spatial,
        handle_nan=handle_nan,
    )
    if early_stopping_rounds is not None:
        model.fit(train_df, eval_df=val_df)
    else:
        model.fit(train_df)

    # Validation
    y_val = val_df["time_to_arrival_seconds"].values
    val_preds = model.predict(val_df)
    val_metrics = compute_all_metrics(y_val, val_preds, prefix="val_")
    print_metrics_table(val_metrics, "Validation")

    # Test
    y_test = test_df["time_to_arrival_seconds"].values
    test_preds = model.predict(test_df)
    test_metrics = compute_all_metrics(y_test, test_preds, prefix="test_")
    print_metrics_table(test_metrics, "Test")

    # Metadata
    metadata = {
        "model_type": "xgboost",
        "dataset": dataset_name,
        "route_id": route_id,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "gamma": gamma,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": model.best_iteration,
        "include_temporal": include_temporal,
        "include_spatial": include_spatial,
        "handle_nan": handle_nan,
        "n_features": len(model.feature_cols) if model.feature_cols else 0,
        "features": model.feature_cols,
        "seed": model.random_state,
        "n_samples": len(train_df) + len(val_df) + len(test_df),
        "n_trips": dataset.df["trip_id"].nunique() if route_id else None,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "metrics": {**val_metrics, **test_metrics},
    }

    # Save
    if save_model:
        feature_groups = []
        if include_temporal:
            feature_groups.append("temporal")
        if include_spatial:
            feature_groups.append("spatial")

        model_key = ModelKey.generate(
            model_type="xgboost",
            dataset_name=dataset_name,
            feature_groups=feature_groups,
            route_id=route_id,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            handle_nan=handle_nan,
        )

        registry = get_registry()
        registry.save_model(model_key, model, metadata)
        metadata["model_key"] = model_key

    return {
        "model": model,
        "metrics": metadata["metrics"],
        "metadata": metadata,
    }


if __name__ == "__main__":
    # Example: basic global model
    _ = train_xgboost(
        dataset_name="sample_dataset",
        include_temporal=True,
        handle_nan="drop",
    )
