"""Pytest coverage for the ``shape=`` precomputed-distance hook in estimator.py.

The vendored copy this hook was ported from (databus's backend/gtfs-eta/)
carries no tests of its own, so these are new: they pin the three code
paths added to ``estimate_stop_times`` — precomputed per-stop distance,
``ShapePolyline`` projection fallback, and the untouched haversine proxy
when neither is supplied.
"""

import numpy as np
import pandas as pd
import pytest

import models.common.registry as registry_module
from eta_service.estimator import estimate_stop_times
from feature_engineering.spatial import ShapePolyline
from models.polyreg_distance.train import PolyRegDistanceModel

MODEL_KEY = "shape_hook_test_model"


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    """Point the registry singleton at a throwaway dir seeded with one global model."""
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setattr(registry_module, "_registry", None)

    model = PolyRegDistanceModel(degree=1, alpha=1.0, route_specific=False)
    distances = np.linspace(0, 3000, 50)
    train_df = pd.DataFrame(
        {
            "distance_to_stop": distances,
            "time_to_arrival_seconds": distances / 4.5,
        }
    )
    model.fit(train_df)

    registry = registry_module.get_registry()
    registry.save_model(
        MODEL_KEY,
        model,
        metadata={
            "model_type": "polyreg_distance",
            "route_id": None,
            "metrics": {"test_mae_seconds": 15.0},
        },
    )
    yield registry
    monkeypatch.setattr(registry_module, "_registry", None)


def _vehicle_position(lat=0.0, lon=0.0):
    """A minimal VehiclePosition dict at the given coordinates."""
    return {
        "vehicle_id": "veh1",
        "route": "unknown",
        "lat": lat,
        "lon": lon,
        "speed": 5.0,
        "timestamp": "2026-01-01T12:00:00Z",
    }


def test_precomputed_shape_distance_overrides_haversine(seeded_registry):
    """A stop-level ``shape_distance_to_stop`` wins over the haversine proxy."""
    vehicle_position = _vehicle_position(lat=0.0, lon=0.0)
    stops = [
        {
            "stop_id": "s1",
            "stop_sequence": 1,
            "lat": 0.0,
            "lon": 0.01,  # haversine distance here is ~1113 m
            "shape_distance_to_stop": 5000.0,
        }
    ]

    result = estimate_stop_times(vehicle_position, stops, max_stops=3)

    assert not result.get("error")
    pred = result["predictions"][0]
    assert pred["distance_to_stop_m"] == pytest.approx(5000.0, abs=1.0)


def test_shape_polyline_projection_used_without_precomputed_distance(seeded_registry):
    """A ``shape=`` ShapePolyline is projected when no per-stop distance is given."""
    vehicle_position = _vehicle_position(lat=0.0, lon=0.0)
    shape = ShapePolyline([(0.0, 0.0), (0.0, 0.02), (0.0, 0.05)])
    stops = [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.04}]

    result = estimate_stop_times(vehicle_position, stops, max_stops=3, shape=shape)

    assert not result.get("error")
    pred = result["predictions"][0]
    # Vehicle is at the shape start (progress 0), stop projects near lon=0.04,
    # i.e. distance-along-shape ~= haversine(0,0 -> 0,0.04) ~= 4453 m.
    assert pred["distance_to_stop_m"] == pytest.approx(4453.0, rel=0.05)


def test_no_shape_input_keeps_haversine_proxy(seeded_registry):
    """With no shape and no precomputed distance, behavior is unchanged (haversine)."""
    vehicle_position = _vehicle_position(lat=0.0, lon=0.0)
    stops = [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.01}]

    result = estimate_stop_times(vehicle_position, stops, max_stops=3)

    assert not result.get("error")
    pred = result["predictions"][0]
    assert pred["distance_to_stop_m"] == pytest.approx(1113.0, rel=0.05)


def test_precomputed_distance_wins_even_with_shape_also_supplied(seeded_registry):
    """Precomputed per-stop distance takes priority over a supplied ShapePolyline."""
    vehicle_position = _vehicle_position(lat=0.0, lon=0.0)
    shape = ShapePolyline([(0.0, 0.0), (0.0, 0.05)])
    stops = [
        {
            "stop_id": "s1",
            "stop_sequence": 1,
            "lat": 0.0,
            "lon": 0.01,
            "shape_distance_to_stop": 2500.0,
        }
    ]

    result = estimate_stop_times(vehicle_position, stops, max_stops=3, shape=shape)

    assert not result.get("error")
    pred = result["predictions"][0]
    assert pred["distance_to_stop_m"] == pytest.approx(2500.0, abs=1.0)
