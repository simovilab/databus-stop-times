"""
Polynomial Regression Model - Time Enhanced
Fits polynomial features with temporal, spatial, and kinematic features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from common.data import load_dataset
from common.metrics import compute_all_metrics
from common.keys import ModelKey
from common.registry import get_registry
from common.utils import print_metrics_table, train_test_summary
from common.preprocessing import LegacyPickleDefaults, legacy_impute_stats, prepare_features, clip_predictions


def _make_one_hot_encoder() -> OneHotEncoder:
    """Instantiate OneHotEncoder compatible with sklearn>=1.2 (sparse_output) and older."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


class PolyRegTimeModel(LegacyPickleDefaults):
    """
    Enhanced polynomial regression with temporal/spatial features.
    
    Features (all serveable online by stoptimes.serve.estimator):
    - Core: distance_to_stop (polynomialized)
    - Temporal: hour, day_of_week, is_weekend, is_peak_hour + cyclical sin/cos
    - Spatial: distance_to_next_stop, shape_distance_to_stop, shape_progress,
      cross_track_error, progress_ratio, stops_ahead
    - Kinematic: current_speed_kmh, bearing_to_stop, bearing_diff
    - Status: is_at_stop
    """

    _POST_PICKLE_DEFAULTS = {"impute_stats": dict, "handle_nan": lambda: "drop"}


    def __init__(self,
                 poly_degree: int = 2,
                 alpha: float = 1.0,
                 include_temporal: bool = True,
                 include_spatial: bool = True,
                 handle_nan: str = 'drop'):  # 'drop', 'impute', or 'error'
        """
        Initialize model.

        Args:
            poly_degree: Polynomial degree for distance
            alpha: Ridge regularization strength
            include_temporal: Include time-of-day features
            include_spatial: Include spatial/kinematic/status features
            handle_nan: How to handle NaN - 'drop', 'impute', or 'error'
        """
        self.poly_degree = poly_degree
        self.alpha = alpha
        self.include_temporal = include_temporal
        self.include_spatial = include_spatial
        self.handle_nan = handle_nan
        self.model = None
        self.feature_cols = None
        self.available_features = None
        self.categorical_features: List[str] = []
        self.numeric_features: List[str] = []
        self.impute_stats: Dict = {}

    def _get_feature_groups(self) -> Dict[str, List[str]]:
        """Define feature groups (all computable online for serving parity)."""
        return {
            'core': ['distance_to_stop'],
            'spatial_numeric': [
                'distance_to_next_stop', 'shape_distance_to_stop', 'shape_progress',
                'cross_track_error', 'progress_ratio', 'stops_ahead',
                'current_speed_kmh', 'bearing_to_stop', 'bearing_diff', 'is_at_stop',
            ] if self.include_spatial else [],
            'temporal': [
                'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            ] if self.include_temporal else [],
        }
    
    def _clean_data(self, df: pd.DataFrame, 
                   target_col: str = 'time_to_arrival_seconds') -> pd.DataFrame:
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
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)

        # Keep features that are present and not mostly-NaN (>30% drops the feature)
        available = []
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
    
    def _create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline."""
        if not self.feature_cols:
            raise ValueError("Feature columns must be set before building pipeline")

        transformers = []

        if 'distance_to_stop' in self.feature_cols:
            transformers.append((
                'poly_distance',
                Pipeline([
                    ('poly', PolynomialFeatures(
                        degree=self.poly_degree,
                        include_bias=False
                    )),
                    ('scale', StandardScaler())
                ]),
                ['distance_to_stop']
            ))

        numeric_others = [
            col for col in self.numeric_features
            if col != 'distance_to_stop'
        ]
        if numeric_others:
            transformers.append((
                'scale_numeric',
                StandardScaler(),
                numeric_others
            ))

        if self.categorical_features:
            transformers.append((
                'encode_categorical',
                _make_one_hot_encoder(),
                self.categorical_features
            ))

        return Pipeline([
            ('features', ColumnTransformer(
                transformers,
                remainder='drop',
                sparse_threshold=0.0
            )),
            ('ridge', Ridge(alpha=self.alpha))
        ])
    
    def fit(self, train_df: pd.DataFrame, 
            target_col: str = 'time_to_arrival_seconds'):
        """
        Train model.
        
        Args:
            train_df: Training dataframe
            target_col: Target column name
        """
        # Clean data
        train_clean = self._clean_data(train_df, target_col)
        
        # Build feature list from available features
        feature_groups = self._get_feature_groups()
        self.feature_cols = []
        
        group_order = [
            'core',
            'spatial_numeric',
            'temporal',
        ]
        for group in group_order:
            for feat in feature_groups.get(group, []):
                if feat in self.available_features and feat not in self.feature_cols:
                    self.feature_cols.append(feat)
        
        if not self.feature_cols:
            raise ValueError("No features available after cleaning!")

        self.categorical_features = []
        self.numeric_features = list(self.feature_cols)

        # Prepare data
        X = train_clean[self.feature_cols]
        y = train_clean[target_col].values

        # Create and fit model
        self.model = self._create_pipeline()
        self.model.fit(X, y)

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
        
        if self.feature_cols is None:
            raise ValueError("Feature columns not set")
        
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
        if self.handle_nan == 'error':
            if X[self.feature_cols].isna().any().any():
                raise ValueError("NaN values found and handle_nan='error'")
            X_pred = X[self.feature_cols].copy()
        else:
            X_pred, _, _ = prepare_features(
                X, self.feature_cols, handle_nan="impute",
                impute_stats=legacy_impute_stats(self.impute_stats, X, self.feature_cols)
            )

        predictions = self.model.predict(X_pred[self.feature_cols])
        
        return clip_predictions(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature coefficients (approximate importance)."""
        if self.model is None:
            return {}
        
        coefs = self.model.named_steps['ridge'].coef_
        transformer = self.model.named_steps['features']
        feature_names = transformer.get_feature_names_out()
        
        importance = {
            name: abs(coef)
            for name, coef in zip(feature_names, coefs)
        }
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def train_polyreg_time(dataset_name: str = "sample_dataset",
                      route_id: Optional[str] = None,
                      poly_degree: int = 2,
                      alpha: float = 1.0,
                      include_temporal: bool = True,
                      include_spatial: bool = True,
                      handle_nan: str = 'drop',
                      test_size: float = 0.2,
                      save_model: bool = True,
                      pre_split: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None) -> Dict:
    """
    Train and evaluate polynomial regression time model.
    
    Args:
        dataset_name: Dataset name
        route_id: Optional route ID for route-specific training
        poly_degree: Polynomial degree for distance
        alpha: Ridge regularization
        include_temporal: Include temporal features
        include_spatial: Include spatial/kinematic/status features
        handle_nan: 'drop', 'impute', or 'error'
        test_size: Test set fraction
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
        df_filtered = df[df['route_id'] == route_id].copy()

        if len(df_filtered) == 0:
            raise ValueError(f"No data found for route {route_id}")

        dataset.df = df_filtered

    if pre_split is not None:
        train_df, val_df, test_df = (df.copy() for df in pre_split)
    else:
        train_df, val_df, test_df = dataset.temporal_split(
            train_frac=1-test_size-0.1,
            val_frac=0.1
        )

    train_test_summary(train_df, test_df, val_df)

    # Train model
    model = PolyRegTimeModel(
        poly_degree=poly_degree,
        alpha=alpha,
        include_temporal=include_temporal,
        include_spatial=include_spatial,
        handle_nan=handle_nan
    )
    model.fit(train_df)

    # Validation
    y_val = val_df['time_to_arrival_seconds'].values
    val_preds = model.predict(val_df)
    val_metrics = compute_all_metrics(y_val, val_preds, prefix="val_")
    print_metrics_table(val_metrics, "Validation")

    # Test
    y_test = test_df['time_to_arrival_seconds'].values
    test_preds = model.predict(test_df)
    test_metrics = compute_all_metrics(y_test, test_preds, prefix="test_")
    print_metrics_table(test_metrics, "Test")

    importance = model.get_feature_importance()

    # Metadata
    metadata = {
        'model_type': 'polyreg_time',
        'dataset': dataset_name,
        'route_id': route_id,
        'poly_degree': poly_degree,
        'alpha': alpha,
        'include_temporal': include_temporal,
        'include_spatial': include_spatial,
        'handle_nan': handle_nan,
        'n_features': len(model.feature_cols) if model.feature_cols else 0,
        'features': model.feature_cols,
        'n_samples': len(train_df) + len(val_df) + len(test_df),
        'n_trips': dataset.df['trip_id'].nunique() if route_id else None,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'metrics': {**val_metrics, **test_metrics}
    }
    
    # Save
    if save_model:
        feature_groups = []
        if include_temporal:
            feature_groups.append('temporal')
        if include_spatial:
            feature_groups.append('spatial')
        model_key = ModelKey.generate(
            model_type='polyreg_time',
            dataset_name=dataset_name,
            feature_groups=feature_groups,
            route_id=route_id,
            degree=poly_degree,
            handle_nan=handle_nan
        )
        
        registry = get_registry()
        registry.save_model(model_key, model, metadata)
        metadata['model_key'] = model_key

    return {
        'model': model,
        'metrics': metadata['metrics'],
        'metadata': metadata
    }


if __name__ == "__main__":
    # Example: Train with different configurations
    
    # Basic model - drop NaN
    result1 = train_polyreg_time(
        poly_degree=2,
        include_temporal=True,
        handle_nan='drop'
    )

    # With imputation
    result2 = train_polyreg_time(
        poly_degree=2,
        include_temporal=True,
        handle_nan='impute'
    )
