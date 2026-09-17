"""
stoptimes.models -- ETA prediction model families.

Comprehensive modeling framework for transit ETA prediction.

Example usage:
    >>> from stoptimes.models import get_registry
    >>> registry = get_registry()
    >>> best_key = registry.get_best_model()

Training orchestration (``train_all_models``) lives in ``stoptimes.train``,
not here: ``core <- models <- serve <- train`` is the required dependency
direction (see docs/PLAN.md §1), so models must never import train, even
lazily. Import it directly where you need it:

    >>> from stoptimes.train.train_all_models import train_all_models
"""

__version__ = "1.0.0"
__author__ = "SIMOVI Team"

# Core utilities
from .common.data import load_dataset, ETADataset, prepare_features_target
from .common.registry import get_registry, ModelRegistry
from .common.keys import ModelKey, PredictionKey
from .common.metrics import (
    compute_all_metrics,
    mae_minutes,
    rmse_minutes,
    within_threshold
)
from .common.utils import (
    format_seconds,
    haversine_distance,
    clip_predictions,
    setup_logging
)

# Training functions
from .historical_mean.train import train_historical_mean, HistoricalMeanModel
from .polyreg_distance.train import train_polyreg_distance, PolyRegDistanceModel
from .polyreg_time.train import train_polyreg_time, PolyRegTimeModel
from .ewma.train import train_ewma, EWMAModel

# Prediction functions
from .historical_mean.predict import predict_eta as predict_historical_mean
from .polyreg_distance.predict import predict_eta as predict_polyreg_distance
from .polyreg_time.predict import predict_eta as predict_polyreg_time
from .ewma.predict import predict_eta as predict_ewma

# Evaluation and training orchestration (train_all_models) both moved to
# stoptimes.train -- models must not depend on train, not even lazily (see
# module docstring). There is deliberately no train_all_models symbol, PEP
# 562 accessor, or other back-reference to stoptimes.train anywhere in this
# package.


__all__ = [
    # Core utilities
    'load_dataset',
    'ETADataset',
    'prepare_features_target',
    'get_registry',
    'ModelRegistry',
    'ModelKey',
    'PredictionKey',
    'compute_all_metrics',
    'mae_minutes',
    'rmse_minutes',
    'within_threshold',
    'format_seconds',
    'haversine_distance',
    'clip_predictions',
    'setup_logging',
    
    # Models
    'HistoricalMeanModel',
    'PolyRegDistanceModel',
    'PolyRegTimeModel',
    'EWMAModel',
    
    # Training
    'train_historical_mean',
    'train_polyreg_distance',
    'train_polyreg_time',
    'train_ewma',

    # Prediction
    'predict_historical_mean',
    'predict_polyreg_distance',
    'predict_polyreg_time',
    'predict_ewma',
    
]


# Package info
MODELS = {
    'historical_mean': {
        'description': 'Historical average baseline',
        'typical_mae': '2-4 minutes',
        'features': ['route_id', 'stop_sequence', 'temporal']
    },
    'polyreg_distance': {
        'description': 'Polynomial regression on distance',
        'typical_mae': '1.5-3 minutes',
        'features': ['distance_to_stop']
    },
    'polyreg_time': {
        'description': 'Polynomial regression with time features',
        'typical_mae': '1-2.5 minutes',
        'features': ['distance_to_stop', 'temporal', 'operational']
    },
    'ewma': {
        'description': 'Exponentially weighted moving average',
        'typical_mae': '1.5-3 minutes',
        'features': ['route_id', 'stop_sequence', 'temporal'],
        'online_learning': True
    }
}


def list_models():
    """List available model types."""
    print("\nAvailable Model Types:")
    print("=" * 70)
    for name, info in MODELS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Typical MAE: {info['typical_mae']}")
        print(f"  Features: {', '.join(info['features'])}")
        if info.get('online_learning'):
            print(f"  Online Learning: Yes")
    print("\n" + "=" * 70)


def quick_start_guide():
    """Print quick start guide."""
    guide = """
    ETA Prediction Models - Quick Start
    ====================================

    1. Train baselines (from stoptimes.train, not stoptimes.models):
       >>> from stoptimes.train.train_all_models import train_all_models
       >>> results = train_all_models("sample_dataset")

    2. Compare models:
       >>> from stoptimes.train.evaluation.leaderboard import quick_compare
       >>> from stoptimes.models import get_registry
       >>> registry = get_registry()
       >>> model_keys = registry.list_models()['model_key'].tolist()
       >>> comparison = quick_compare(model_keys)

    3. Load and use best model:
       >>> best_key = registry.get_best_model(metric='test_mae_seconds')
       >>> model = registry.load_model(best_key)
       >>> predictions = model.predict(your_data)

    4. Make single prediction:
       >>> from stoptimes.models import predict_polyreg_time
       >>> result = predict_polyreg_time(
       ...     model_key=best_key,
       ...     distance_to_stop=1500.0,
       ...     hour=8,
       ...     is_peak_hour=True
       ... )
       >>> print(f"ETA: {result['eta_formatted']}")

    For more details, see packages/models/README.md
    """
    print(guide)