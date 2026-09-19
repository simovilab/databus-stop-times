"""Characterization tests for `estimate_stop_times`, written before rebuilding
it on the pipeline (Phase 3, U6).

These pin *current*, real-production observable behavior on realistic inputs
so the rebuild (a thin adapter over `stoptimes.serve.pipeline`, with model
choice routed through `core.contracts.selection.select_artifact`) can be
checked against them unchanged. They complement the existing
`test_shape_hook.py` (shape hook precedence), `test_served_model_pin.py`
(pin/fallback), and `test_stops_ahead_alignment.py` (stops_ahead indexing) --
this file adds monotonicity, the ewma-retirement error, and the remaining
error/None paths not already pinned elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stoptimes.models.common.registry as registry_module
from stoptimes.models.polyreg_distance.train import PolyRegDistanceModel
from stoptimes.serve.estimator import estimate_stop_times

MODEL_KEY = "characterization_test_model"


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    """One global polyreg_distance model whose predicted ETA is linear in
    distance -- makes monotonicity trivially checkable without depending on
    a specific trained model's exact curve.
    """
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setattr(registry_module, "_registry", None)

    model = PolyRegDistanceModel(degree=1, alpha=1.0, route_specific=False)
    distances = np.linspace(0, 5000, 100)
    train_df = pd.DataFrame(
        {
            "distance_to_stop": distances,
            "time_to_arrival_seconds": distances / 5.0,  # 5 m/s
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
            "metrics": {"test_mae_seconds": 2.0},
        },
    )
    yield registry
    monkeypatch.setattr(registry_module, "_registry", None)


def _vehicle_position():
    return {
        "vehicle_id": "veh1",
        "route": "unknown",
        "lat": 0.0,
        "lon": 0.0,
        "speed": 5.0,
        "timestamp": "2026-01-01T12:00:00Z",
    }


def _stops_at_distances(distances_m: list[float]) -> list[dict]:
    """Stops placed at approximately the given haversine distances (m) east
    of the vehicle, in increasing order -- ``upcoming_stops`` order matters
    (see ``test_stops_ahead_alignment.py``)."""
    return [
        {
            "stop_id": f"stop_{i}",
            "stop_sequence": i + 1,
            "lat": 0.0,
            "lon": d / 111_320.0,
        }
        for i, d in enumerate(distances_m)
    ]


def test_eta_is_monotonically_increasing_with_distance(seeded_registry):
    """Farther stops must never get an earlier ETA than nearer ones -- the
    minimal sanity property any served model must satisfy."""
    stops = _stops_at_distances([500.0, 1500.0, 3000.0, 4500.0])

    result = estimate_stop_times(_vehicle_position(), stops, model_key=MODEL_KEY, max_stops=4)

    assert not result.get("error"), result
    etas = [p["eta_seconds"] for p in result["predictions"]]
    assert all(e is not None for e in etas)
    assert etas == sorted(etas)
    assert etas[0] < etas[-1]


def test_eta_timestamps_are_monotonically_increasing_with_distance(seeded_registry):
    stops = _stops_at_distances([500.0, 1500.0, 3000.0])

    result = estimate_stop_times(_vehicle_position(), stops, model_key=MODEL_KEY, max_stops=3)

    timestamps = [p["eta_timestamp"] for p in result["predictions"]]
    assert timestamps == sorted(timestamps)


def test_missing_vehicle_position_returns_error_and_empty_predictions():
    result = estimate_stop_times(None, [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.0}])

    assert result["error"] == "Missing vehicle position or stops"
    assert result["predictions"] == []
    assert result["model_key"] is None
    assert result["vehicle_id"] == "unknown"


def test_missing_stops_returns_error_and_empty_predictions():
    result = estimate_stop_times(_vehicle_position(), [])

    assert result["error"] == "Missing vehicle position or stops"
    assert result["predictions"] == []


def test_invalid_model_key_reports_metadata_load_failure():
    result = estimate_stop_times(
        _vehicle_position(),
        [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.01}],
        model_key="this_model_key_does_not_exist_12345",
    )

    assert result["error"] is not None
    assert "Failed to load model metadata" in result["error"]
    assert result["predictions"] == []


def test_ewma_model_type_is_retired_and_errors_per_stop(seeded_registry):
    """ewma is kept for offline ablation only; served requests must fail
    loudly per stop rather than silently serving a dominated model."""
    registry_module.get_registry().registry[MODEL_KEY]["model_type"] = "ewma"
    # load_metadata reads the *_meta.json file, not the in-memory index, so
    # patch the metadata actually read at serve time too.
    import json

    meta_path = registry_module.get_registry()._resolve_artifact_path(
        registry_module.get_registry().registry[MODEL_KEY]["meta_path"]
    )
    meta = json.loads(meta_path.read_text())
    meta["model_type"] = "ewma"
    meta_path.write_text(json.dumps(meta))

    result = estimate_stop_times(
        _vehicle_position(),
        [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.01}],
        model_key=MODEL_KEY,
        max_stops=1,
    )

    assert not result.get("error")  # the top-level call succeeds...
    pred = result["predictions"][0]
    assert pred["eta_seconds"] is None
    assert "ewma is retired" in pred["error"]


def test_model_scope_is_route_when_metadata_names_a_specific_route(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setattr(registry_module, "_registry", None)

    model = PolyRegDistanceModel(degree=1, alpha=1.0, route_specific=False)
    distances = np.linspace(0, 3000, 50)
    model.fit(
        pd.DataFrame({"distance_to_stop": distances, "time_to_arrival_seconds": distances / 4.5})
    )
    registry = registry_module.get_registry()
    registry.save_model(
        "route_scoped_model",
        model,
        metadata={
            "model_type": "polyreg_distance",
            "route_id": "42",
            "metrics": {"test_mae_seconds": 5.0},
        },
    )

    result = estimate_stop_times(
        _vehicle_position(),
        [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.01}],
        route_id="42",
        model_key="route_scoped_model",
        max_stops=1,
    )

    assert not result.get("error")
    assert result["model_scope"] == "route"

    monkeypatch.setattr(registry_module, "_registry", None)
