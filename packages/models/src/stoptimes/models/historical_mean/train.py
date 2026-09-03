"""
Historical Mean Baseline Model
Predicts ETA based on historical average travel times grouped by route, stop, and time features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.data import load_dataset, prepare_features_target
from common.metrics import compute_all_metrics
from common.keys import ModelKey
from common.registry import get_registry
from common.utils import print_metrics_table, train_test_summary
from common.preprocessing import prepare_features, clip_predictions


class HistoricalMeanModel:
    """
    Baseline model using historical mean ETAs.
    
    Groups data by route, stop, and temporal features, then computes
    mean ETA for each group. At prediction time, looks up the appropriate group mean.
    """
    
    def __init__(self, group_by: List[str] = ['route_id', 'stop_sequence', 'hour'],
                 handle_nan: str = 'drop'):
        """
        Initialize model.

        Args:
            group_by: List of columns to group by for computing means
            handle_nan: How to handle NaN in group_by columns - 'drop',
                'impute', or 'error' (see common.preprocessing.prepare_features)
        """
        self.group_by = group_by
        self.handle_nan = handle_nan
        self.lookup_table = None
        self.global_mean = None
        self.feature_cols = None
        self.impute_stats: Dict = {}

    def fit(self, train_df: pd.DataFrame, target_col: str = 'time_to_arrival_seconds'):
        """
        Train model by computing historical means.

        Args:
            train_df: Training dataframe with features and target
            target_col: Name of target column
        """
        # Store feature columns for later
        self.feature_cols = self.group_by

        # Unify NaN handling with the other families: drop/impute/error on
        # the group_by columns, fitting imputation stats on TRAIN only.
        X_train, kept_index, stats = prepare_features(
            train_df, self.group_by, handle_nan=self.handle_nan
        )
        self.impute_stats = stats or self.impute_stats
        train_clean = train_df.loc[kept_index].copy()
        train_clean[self.group_by] = X_train

        # Compute global mean as fallback
        self.global_mean = train_clean[target_col].mean()

        # Compute group means
        self.lookup_table = train_clean.groupby(self.group_by)[target_col].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()

        # Predict-time fallback stats: always available (train-fit, never
        # from the evaluation split) regardless of handle_nan, so predict()
        # can safely fill residual NaN without ever dropping a row (this
        # model must return exactly one prediction per input row so
        # downstream evaluation/leaderboard code stays aligned with y).
        if not self.impute_stats:
            _, _, self.impute_stats = prepare_features(
                train_df, self.group_by, handle_nan="impute"
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ETAs for input data.
        
        Args:
            X: DataFrame with features matching group_by columns
            
        Returns:
            Array of predicted ETAs in seconds
        """
        if self.lookup_table is None:
            raise ValueError("Model not trained. Call fit() first.")

        # predict() must always return one prediction per input row (other
        # families' evaluation/leaderboard code assumes len(preds) ==
        # len(X)), so NaN in group_by columns is always filled here with
        # train-fit stats — never dropped, never re-derived from X itself.
        if self.handle_nan == "error":
            if X[[c for c in self.group_by if c in X.columns]].isna().any().any():
                raise ValueError("NaN values found and handle_nan='error'")
            X_clean = X[self.group_by]
        else:
            X_clean, _, _ = prepare_features(
                X, self.group_by, handle_nan="impute", impute_stats=self.impute_stats
            )

        # Merge with lookup table
        merged = X_clean[self.group_by].merge(
            self.lookup_table,
            on=self.group_by,
            how='left'
        )

        # Use group mean, fallback to global mean
        predictions = merged['mean'].fillna(self.global_mean).values

        return clip_predictions(predictions)
    
    def get_coverage(self, X: pd.DataFrame) -> float:
        """
        Get fraction of predictions with matching historical data.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Coverage ratio (0-1)
        """
        merged = X[self.group_by].merge(
            self.lookup_table,
            on=self.group_by,
            how='left',
            indicator=True
        )
        
        return (merged['_merge'] == 'both').mean()


def train_historical_mean(dataset_name: str = "sample_dataset",
                         route_id: Optional[str] = None,
                         group_by: List[str] = ['route_id', 'stop_sequence', 'hour'],
                         handle_nan: str = 'drop',
                         test_size: float = 0.2,
                         save_model: bool = True,
                         pre_split: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None) -> Dict:
    """
    Train and evaluate historical mean model.
    
    Args:
        dataset_name: Name of dataset in datasets/ directory
        route_id: Optional route ID for route-specific training
        group_by: List of columns to group by
        test_size: Fraction of data for testing
        save_model: Whether to save to registry
        pre_split: Optional (train, val, test) DataFrames to reuse
        
    Returns:
        Dictionary with model, metrics, and metadata
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
    model = HistoricalMeanModel(group_by=group_by, handle_nan=handle_nan)
    model.fit(train_df)

    # Evaluate on validation set
    y_val = val_df['time_to_arrival_seconds'].values
    val_preds = model.predict(val_df)
    val_metrics = compute_all_metrics(y_val, val_preds, prefix="val_")
    print_metrics_table(val_metrics, "Validation Metrics")
    val_coverage = model.get_coverage(val_df)

    # Evaluate on test set
    y_test = test_df['time_to_arrival_seconds'].values
    test_preds = model.predict(test_df)
    test_metrics = compute_all_metrics(y_test, test_preds, prefix="test_")
    print_metrics_table(test_metrics, "Test Metrics")
    test_coverage = model.get_coverage(test_df)

    # Prepare metadata
    metadata = {
        'model_type': 'historical_mean',
        'dataset': dataset_name,
        'route_id': route_id,
        'group_by': group_by,
        'handle_nan': handle_nan,
        'n_samples': len(train_df) + len(val_df) + len(test_df),
        'n_trips': dataset.df['trip_id'].nunique() if route_id else None,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'unique_groups': len(model.lookup_table),
        'global_mean_eta': float(model.global_mean),
        'test_coverage': float(test_coverage),
        'metrics': {**val_metrics, **test_metrics}
    }
    
    # Save model
    if save_model:
        model_key = ModelKey.generate(
            model_type='historical_mean',
            dataset_name=dataset_name,
            feature_groups=['temporal', 'route'],
            route_id=route_id
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
    # Train with different grouping strategies
    
    # Basic: route + stop + hour
    result1 = train_historical_mean(
        group_by=['route_id', 'stop_sequence', 'hour']
    )
    
    # With day of week
    result2 = train_historical_mean(
        group_by=['route_id', 'stop_sequence', 'hour', 'day_of_week']
    )
    
    # With peak hour indicator
    result3 = train_historical_mean(
        group_by=['route_id', 'stop_sequence', 'is_peak_hour']
    )
