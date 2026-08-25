"""Pytest coverage for the served-model pin that keeps the databus path on
the BUCR corpus model.

See ``core.config.AGENCY_MODEL_DEFAULTS`` / ``ETA_MODEL_KEY`` and the
"SMART MODEL SELECTION" block in ``eta_service.estimator.estimate_stop_times``:
when the active agency has a pinned model, it's used outright and
``ModelRegistry.get_best_model``'s metric-only ranking (which has no
dataset/agency awareness) is never consulted -- these tests seed a
registry with a deliberately "better" (lower MAE) MBTA-looking model
alongside the BUCR-looking one(s) to prove the pin, not the ranking,
decides.
"""

import numpy as np
import pandas as pd
import pytest

import models.common.registry as registry_module
from eta_service import estimator as estimator_module
from eta_service.estimator import estimate_stop_times
from models.polyreg_distance.train import PolyRegDistanceModel

PRIMARY_KEY = "xgboost_bucr_segments_corpus_pin_test"
FALLBACK_KEY = "polyreg_distance_bucr_segments_corpus_pin_test"
MBTA_KEY = "polyreg_distance_mbta_bus_pin_test"


def _fit_polyreg():
    """A cheap-to-fit stand-in model (real corpus models are pickle-heavy)."""
    model = PolyRegDistanceModel(degree=1, alpha=1.0, route_specific=False)
    distances = np.linspace(0, 3000, 50)
    train_df = pd.DataFrame(
        {
            "distance_to_stop": distances,
            "time_to_arrival_seconds": distances / 4.5,
        }
    )
    model.fit(train_df)
    return model


def _vehicle_position():
    return {
        "vehicle_id": "veh1",
        "route": "unknown",
        "lat": 0.0,
        "lon": 0.0,
        "speed": 5.0,
        "timestamp": "2026-01-01T12:00:00Z",
    }


def _stops():
    return [{"stop_id": "s1", "stop_sequence": 1, "lat": 0.0, "lon": 0.01}]


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    """Throwaway registry dir seeded with a BUCR fallback model and an MBTA
    model that has a deliberately better (lower) test_mae_seconds, so any
    test proving the pin wins is also proving metric ranking lost.
    """
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setattr(registry_module, "_registry", None)

    registry = registry_module.get_registry()

    registry.save_model(
        FALLBACK_KEY,
        _fit_polyreg(),
        metadata={
            "model_type": "polyreg_distance",
            "route_id": None,
            "dataset": "bucr_segments_corpus",
            "metrics": {"test_mae_seconds": 35.5},
        },
    )
    registry.save_model(
        MBTA_KEY,
        _fit_polyreg(),
        metadata={
            "model_type": "polyreg_distance",
            "route_id": None,
            "dataset": "mbta_bus_222_15_37_1d",
            "metrics": {"test_mae_seconds": 1.0},  # "best" by metric on purpose
        },
    )
    yield registry
    monkeypatch.setattr(registry_module, "_registry", None)


@pytest.fixture
def bucr_agency(monkeypatch):
    """Point estimator._config at the bucr agency (pinned served model)."""
    monkeypatch.setattr(estimator_module._config, "_agency_override", "bucr")
    monkeypatch.setattr(estimator_module._config, "_served_model_key_override", None)
    monkeypatch.setattr(estimator_module._config, "_served_model_fallback_override", None)


@pytest.fixture
def mbta_agency(monkeypatch):
    """Point estimator._config at the mbta agency (no pin configured)."""
    monkeypatch.setattr(estimator_module._config, "_agency_override", "mbta")
    monkeypatch.setattr(estimator_module._config, "_served_model_key_override", None)
    monkeypatch.setattr(estimator_module._config, "_served_model_fallback_override", None)


def test_bucr_pin_falls_back_to_corpus_polyreg_when_primary_missing(seeded_registry, bucr_agency, monkeypatch):
    """Primary (xgboost corpus) key isn't in this throwaway registry, so the
    pin must resolve to the corpus polyreg_distance fallback -- and NOT the
    better-scoring MBTA model.
    """
    # bucr_agency leaves the real production primary/fallback keys in effect
    # (from AGENCY_MODEL_DEFAULTS); override the fallback to this test's
    # throwaway key since the real one isn't seeded into tmp_path either.
    monkeypatch.setattr(estimator_module._config, "_served_model_fallback_override", FALLBACK_KEY)

    result = estimate_stop_times(_vehicle_position(), _stops(), max_stops=3)

    assert not result.get("error")
    assert result["model_key"] == FALLBACK_KEY
    assert result["model_scope"] == "pinned_fallback"


def test_bucr_pin_prefers_primary_over_fallback_and_mbta(seeded_registry, bucr_agency, monkeypatch):
    """When the primary pinned key exists, it wins over both the fallback
    and the better-scoring MBTA model.
    """
    seeded_registry.save_model(
        PRIMARY_KEY,
        _fit_polyreg(),
        metadata={
            "model_type": "xgboost",
            "route_id": None,
            "dataset": "bucr_segments_corpus",
            "metrics": {"test_mae_seconds": 33.4},
        },
    )
    monkeypatch.setattr(estimator_module._config, "_served_model_key_override", PRIMARY_KEY)
    monkeypatch.setattr(estimator_module._config, "_served_model_fallback_override", FALLBACK_KEY)

    result = estimate_stop_times(_vehicle_position(), _stops(), max_stops=3)

    assert not result.get("error")
    assert result["model_key"] == PRIMARY_KEY
    assert result["model_scope"] == "pinned"


def test_bucr_pin_never_falls_back_to_mbta_model(seeded_registry, bucr_agency):
    """Even with only the MBTA model in the registry (no BUCR keys at all),
    the pin must error out rather than silently serving the MBTA model.
    """
    seeded_registry.delete_model(FALLBACK_KEY)

    result = estimate_stop_times(_vehicle_position(), _stops(), max_stops=3)

    assert result.get("error")
    assert result["model_key"] is None
    assert result["predictions"] == []


def test_mbta_agency_is_unaffected_by_the_pin(seeded_registry, mbta_agency):
    """An agency with no configured pin (e.g. mbta) keeps the pre-existing
    get_best_model metric ranking, unchanged.
    """
    result = estimate_stop_times(_vehicle_position(), _stops(), max_stops=3)

    assert not result.get("error")
    assert result["model_key"] == MBTA_KEY
