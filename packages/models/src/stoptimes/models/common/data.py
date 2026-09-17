"""
Data loading and preprocessing utilities for ETA prediction models.
Handles dataset splitting, feature engineering, and train/test preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from datetime import datetime, timedelta

import warnings

from .splits import fraction_split, service_day


class ETADataset:
    """
    Manages ETA prediction datasets with consistent preprocessing and splitting.
    
    Expected columns from VP dataset builder:
    - Identifiers: trip_id, route_id, vehicle_id, stop_id, stop_sequence
    - Position: vp_lat, vp_lon, vp_bearing, distance_to_stop (+ stop_lat/lon, vp_ts)
    - Spatial: distance_to_next_stop, shape_distance_to_stop, shape_progress,
      cross_track_error, progress_ratio, stops_ahead
    - Kinematic: current_speed_kmh, bearing_to_stop, bearing_diff
    - Status: is_at_stop
    - Temporal: hour, day_of_week, is_weekend, is_holiday, is_peak_hour,
      hour_sin, hour_cos, dow_sin, dow_cos
    - Target: actual_arrival, time_to_arrival_seconds (+ *_computed / *_stopped_at)

    All non-target groups are *serveable* — every feature is computable online by
    stoptimes.serve.estimator from a live VehiclePosition + scheduled stops, so the
    trained models stay in sync with serving.
    """

    FEATURE_GROUPS = {
        'identifiers': ['trip_id', 'route_id', 'vehicle_id', 'stop_id', 'stop_sequence'],
        'position': ['vp_lat', 'vp_lon', 'vp_bearing', 'distance_to_stop'],
        'spatial': ['distance_to_next_stop', 'shape_distance_to_stop', 'shape_progress',
                    'cross_track_error', 'progress_ratio', 'stops_ahead'],
        'kinematic': ['current_speed_kmh', 'bearing_to_stop', 'bearing_diff'],
        'status': ['is_at_stop'],
        'temporal': ['hour', 'day_of_week', 'is_weekend', 'is_holiday', 'is_peak_hour',
                     'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
        'target': ['time_to_arrival_seconds']
    }
    
    def __init__(self, data_path: str):
        """
        Initialize dataset from parquet file.
        
        Args:
            data_path: Path to parquet file from VP dataset builder
        """
        self.data_path = Path(data_path)
        self.df = pd.read_parquet(data_path)
        self.df['vp_ts'] = pd.to_datetime(self.df['vp_ts'])
        
        # Store original size
        self.original_size = len(self.df)
        
    def clean_data(self, 
                   drop_missing_target: bool = True,
                   max_eta_seconds: float = 3600 * 2,  # 2 hours
                   min_distance: float = 10.0) -> 'ETADataset':
        """
        Clean dataset by removing invalid rows.
        
        Args:
            drop_missing_target: Remove rows without valid ETA target
            max_eta_seconds: Maximum reasonable ETA (filter outliers)
            min_distance: Minimum distance to stop (meters) to keep
            
        Returns:
            Self for chaining
        """
        initial_rows = len(self.df)
        
        if drop_missing_target:
            self.df = self.df.dropna(subset=['time_to_arrival_seconds'])
        
        # Filter outliers
        self.df = self.df[
            (self.df['time_to_arrival_seconds'] >= 0) &
            (self.df['time_to_arrival_seconds'] <= max_eta_seconds)
        ]
        
        # Filter too-close stops (likely already passed)
        if 'distance_to_stop' in self.df.columns:
            self.df = self.df[self.df['distance_to_stop'] >= min_distance]

        return self
    
    def get_features(self, feature_groups: List[str]) -> List[str]:
        """
        Get list of feature columns from specified groups.
        
        Args:
            feature_groups: List of group names (e.g., ['temporal', 'position'])
            
        Returns:
            List of column names that exist in the dataset
        """
        features = []
        for group in feature_groups:
            if group in self.FEATURE_GROUPS:
                features.extend(self.FEATURE_GROUPS[group])
        
        # Return only columns that exist
        return [f for f in features if f in self.df.columns]
    
    def temporal_split(self,
                       train_frac: float = 0.7,
                       val_frac: float = 0.15,
                       group_cols: Tuple[str, ...] = ('trip_id',),
                       purge: bool = False,
                       service_offset_h: Optional[int] = None,
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by time (avoids data leakage).

        Delegates to `stoptimes.models.common.splits.fraction_split` (roadmap 3.2):
        rows are sorted by `vp_ts` with a stable, deterministic secondary
        key (independent of input row order), cut at the given fractions,
        then any group in `group_cols` that straddles a split boundary is
        purged from the earlier split so no `trip_id` (by default) leaks
        across train/val/test. See `splits.py` for the full report this
        produces (day-independent here since this is the fraction-based
        split, not the calendar-boundary `day_grouped_split`).

        Args:
            train_frac: Fraction for training
            val_frac: Fraction for validation (remainder goes to test)
            group_cols: Columns identifying a group that must not straddle
                a split boundary (default: trip_id).
            purge: Whether to purge group overlap. **Defaults to False**, which
                reproduces the historical behaviour of this method, because
                purging on `trip_id` alone is actively wrong for MBTA-shaped
                data -- see the warning below. Opt in deliberately.
            service_offset_h: Local service-day boundary in hours (MBTA 7,
                bUCR 6). Required when `purge=True` unless the caller has
                already put a per-day column in `group_cols`; a service-day
                column is derived and appended to `group_cols` so that groups
                are per-*run*, not per-schedule-id.

        Warning:
            MBTA `trip_id` is a reused *schedule* identifier: the segment corpus
            has 1,130 distinct trip_ids across 153,498 rows and 16 service days,
            so nearly every trip_id recurs in every window. Purging on
            `('trip_id',)` alone therefore collapses the MBTA validation split
            from 32,534 rows to 189. The unit of leakage is a trip *instance*
            -- `(trip_id, service_day)` -- which is what `service_offset_h`
            constructs. bUCR is unaffected either way (3,061 distinct trip_ids
            for 17,499 rows, near-zero reuse), but pass the offset regardless.

        Returns:
            train_df, val_df, test_df
        """
        df = self.df
        if purge and service_offset_h is not None:
            day_col = '_service_day'
            df = df.copy()
            df[day_col] = service_day(df['vp_ts'], offset_hours=service_offset_h)
            if day_col not in group_cols:
                group_cols = (*group_cols, day_col)

        result = fraction_split(
            df,
            ts_col='vp_ts',
            train_frac=train_frac,
            val_frac=val_frac,
            group_cols=group_cols,
            purge=purge,
        )
        self.last_split_report = result.report
        train_df, val_df, test_df = result.as_tuple()
        if purge:
            dropped = len(df) - (len(train_df) + len(val_df) + len(test_df))
            if dropped > 0.05 * len(df):
                warnings.warn(
                    f"temporal_split purged {dropped:,} of {len(df):,} rows "
                    f"({dropped / len(df):.1%}) on group_cols={group_cols}. That is "
                    "usually the reused-schedule-id hazard, not real leakage -- pass "
                    "service_offset_h so groups are per trip instance.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for col in ('_service_day',):
            for part in (train_df, val_df, test_df):
                if col in part.columns:
                    part.drop(columns=col, inplace=True)
        return train_df, val_df, test_df
    
    def route_split(self,
                    test_routes: Optional[List[str]] = None,
                    test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by route for cross-route generalization testing.
        
        Args:
            test_routes: Specific routes for test set, or None to sample
            test_frac: Fraction of routes for testing if test_routes is None
            
        Returns:
            train_df, test_df
        """
        if test_routes is None:
            routes = self.df['route_id'].unique()
            n_test = max(1, int(len(routes) * test_frac))
            test_routes = np.random.choice(routes, size=n_test, replace=False)
        
        test_df = self.df[self.df['route_id'].isin(test_routes)].copy()
        train_df = self.df[~self.df['route_id'].isin(test_routes)].copy()
        
        print(f"Route split: {len(test_routes)} test routes")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, test_df
    
    def get_route_stats(self) -> pd.DataFrame:
        """Get statistics per route."""
        return self.df.groupby('route_id').agg({
            'time_to_arrival_seconds': ['count', 'mean', 'std'],
            'distance_to_stop': ['mean', 'std'],
            'trip_id': 'nunique'
        }).round(2)
    
    def summary(self) -> Dict:
        """Get dataset summary statistics."""
        return {
            'total_samples': len(self.df),
            'date_range': (self.df['vp_ts'].min(), self.df['vp_ts'].max()),
            'routes': self.df['route_id'].nunique(),
            'trips': self.df['trip_id'].nunique(),
            'vehicles': self.df['vehicle_id'].nunique(),
            'stops': self.df['stop_id'].nunique(),
            'eta_mean_minutes': self.df['time_to_arrival_seconds'].mean() / 60,
            'eta_std_minutes': self.df['time_to_arrival_seconds'].std() / 60,
            'at_stop_frac': float(self.df['is_at_stop'].mean()) if 'is_at_stop' in self.df.columns else 'N/A'
        }


def load_dataset(dataset_name: str = "sample_dataset") -> ETADataset:
    """
    Load dataset from datasets directory.
    
    Args:
        dataset_name: Name of dataset (without .parquet extension)
        
    Returns:
        ETADataset instance
    """
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    data_path = datasets_dir / f"{dataset_name}.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    return ETADataset(str(data_path))


def prepare_features_target(df: pd.DataFrame, 
                            feature_cols: List[str],
                            target_col: str = 'time_to_arrival_seconds',
                            fill_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target, handle missing values.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        fill_na: Whether to fill NaN values
        
    Returns:
        X (features), y (target)
    """
    # Get features that exist
    available_features = [f for f in feature_cols if f in df.columns]
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    if fill_na:
        # Fill numeric features with median, boolean with False
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].fillna(False)
            elif X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
    
    return X, y