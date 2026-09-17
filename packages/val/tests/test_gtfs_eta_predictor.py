"""Tests for GtfsEtaModelPredictor (etaval/predictors/gtfs_eta_models.py).

Mocks ``estimate_stop_times`` rather than exercising a real trained model —
this predictor's own job is just the etaval <-> eta_service payload bridge:
building the vehicle_position/upcoming_stops dicts, calling through, and
converting the result back into StopPrediction objects (or skipping stops the
model couldn't score).
"""

from __future__ import annotations

from unittest.mock import patch

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.predictors.gtfs_eta_models import GtfsEtaModelPredictor

TRIP = TripInstance(
    source="test",
    trip_id="t1",
    vehicle_id="v1",
    route_id="222",
    direction_id=0,
    shape_id="s1",
    start_date="20260701",
    started_at=1_000_000.0,
)

STATE = VehicleState(
    epoch=1_000_100.0,
    lat=42.1,
    lon=-71.0,
    progress_m=200.0,
    speed=5.0,
    current_status="IN_TRANSIT_TO",
)


def _stop(stop_id: str, stop_sequence: int) -> Stop:
    return Stop(stop_id=stop_id, stop_sequence=stop_sequence, lat=42.2, lon=-71.1)


class TestGtfsEtaModelPredictor:
    def test_predictor_id_includes_model_type(self):
        assert GtfsEtaModelPredictor("xgboost").predictor_id == "gtfs_eta:xgboost"
        assert GtfsEtaModelPredictor("polyreg_time").predictor_id == "gtfs_eta:polyreg_time"

    def test_no_upcoming_stops_returns_empty_without_calling_estimator(self):
        predictor = GtfsEtaModelPredictor("xgboost")
        with patch("stoptimes.val.predictors.gtfs_eta_models.estimate_stop_times") as mock_est:
            result = predictor.predict(TRIP, [], STATE, None)
        assert result == []
        mock_est.assert_not_called()

    def test_builds_stop_predictions_from_estimator_result(self):
        predictor = GtfsEtaModelPredictor("xgboost")
        stops = [_stop("s1", 1), _stop("s2", 2)]
        fake_result = {
            "predictions": [
                {"stop_id": "s1", "stop_sequence": 1, "eta_seconds": 120.0,
                 "distance_to_stop_m": 500.0},
                {"stop_id": "s2", "stop_sequence": 2, "eta_seconds": 300.0,
                 "distance_to_stop_m": 1200.0},
            ]
        }
        with patch(
            "stoptimes.val.predictors.gtfs_eta_models.estimate_stop_times", return_value=fake_result
        ) as mock_est:
            preds = predictor.predict(TRIP, stops, STATE, None)

        assert len(preds) == 2
        assert preds[0].stop_id == "s1"
        assert preds[0].predicted_arrival_epoch == STATE.epoch + 120.0
        assert preds[0].made_at_epoch == STATE.epoch
        assert preds[0].dist_remaining_m == 500.0
        assert preds[0].stops_remaining == 2
        assert preds[1].stops_remaining == 1

        # Verify the payload passed through to the estimator.
        call_kwargs = mock_est.call_args.kwargs
        assert call_kwargs["route_id"] == "222"
        assert call_kwargs["trip_id"] == "t1"
        assert call_kwargs["model_type"] == "xgboost"
        assert call_kwargs["vehicle_position"]["vehicle_id"] == "v1"
        assert call_kwargs["upcoming_stops"] == [
            {"stop_id": "s1", "stop_sequence": 1, "lat": 42.2, "lon": -71.1},
            {"stop_id": "s2", "stop_sequence": 2, "lat": 42.2, "lon": -71.1},
        ]

    def test_skips_stops_with_no_eta(self):
        predictor = GtfsEtaModelPredictor("xgboost")
        stops = [_stop("s1", 1)]
        fake_result = {
            "predictions": [
                {"stop_id": "s1", "stop_sequence": 1, "eta_seconds": None, "error": "no model"},
            ]
        }
        with patch(
            "stoptimes.val.predictors.gtfs_eta_models.estimate_stop_times", return_value=fake_result
        ):
            preds = predictor.predict(TRIP, stops, STATE, None)
        assert preds == []

    def test_explicit_model_key_passed_through(self):
        predictor = GtfsEtaModelPredictor(
            "polyreg_time", model_key="polyreg_time_x_y_global_v1"
        )
        with patch(
            "stoptimes.val.predictors.gtfs_eta_models.estimate_stop_times",
            return_value={"predictions": []},
        ) as mock_est:
            predictor.predict(TRIP, [_stop("s1", 1)], STATE, None)
        assert mock_est.call_args.kwargs["model_key"] == "polyreg_time_x_y_global_v1"


# ---------------------------------------------------------------------------
# M5 — _resolve_model_registry_dir() resolution order
# ---------------------------------------------------------------------------


class TestResolveModelRegistryDir:
    def test_env_var_always_wins(self, tmp_path, monkeypatch):
        """Step 1: MODEL_REGISTRY_DIR env var wins unconditionally, unverified."""
        monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "does-not-exist"))
        result = GtfsEtaModelPredictor._resolve_model_registry_dir()
        assert result == tmp_path / "does-not-exist"

    def test_falls_back_when_env_var_unset(self, monkeypatch):
        """With no env var, resolution proceeds through the remaining steps
        without raising -- result is either a real dir or None, never an
        exception."""
        monkeypatch.delenv("MODEL_REGISTRY_DIR", raising=False)
        result = GtfsEtaModelPredictor._resolve_model_registry_dir()
        assert result is None or result.exists()

    def test_dist_info_step_never_raises_when_package_missing(self, monkeypatch):
        """Step 3 (dist-info) degrades to None if gtfs-eta isn't an installed
        distribution -- defensive, never raises."""
        import importlib.metadata

        from stoptimes.val.predictors.gtfs_eta_models import _from_installed_dist_info

        def _raise_not_found(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "distribution", _raise_not_found)
        assert _from_installed_dist_info() is None

    def test_dist_info_step_parses_direct_url_json(self, tmp_path, monkeypatch):
        """Step 3: a monkeypatched dist-info direct_url.json pointing at a dir
        with a real registry.json is trusted."""
        import json

        from stoptimes.val.predictors.gtfs_eta_models import _from_installed_dist_info

        fake_source_root = tmp_path / "gtfs-eta-src"
        trained_dir = fake_source_root / "models" / "trained"
        trained_dir.mkdir(parents=True)
        (trained_dir / "registry.json").write_text("{}")

        class _FakeDist:
            def read_text(self, filename):
                assert filename == "direct_url.json"
                return json.dumps({"url": f"file://{fake_source_root}"})

        import importlib.metadata

        monkeypatch.setattr(
            importlib.metadata, "distribution", lambda name: _FakeDist()
        )
        result = _from_installed_dist_info()
        assert result == trained_dir

    def test_dist_info_step_rejects_dir_without_registry(self, tmp_path, monkeypatch):
        """A resolved directory lacking registry.json is not trusted -- falls
        through to None rather than returning an empty/wrong directory."""
        import json

        from stoptimes.val.predictors.gtfs_eta_models import _from_installed_dist_info

        empty_root = tmp_path / "empty-gtfs-eta"
        (empty_root / "models" / "trained").mkdir(parents=True)

        class _FakeDist:
            def read_text(self, filename):
                return json.dumps({"url": f"file://{empty_root}"})

        import importlib.metadata

        monkeypatch.setattr(
            importlib.metadata, "distribution", lambda name: _FakeDist()
        )
        assert _from_installed_dist_info() is None

    def test_resolver_chain_never_raises(self, monkeypatch):
        """Even if every individual resolver misbehaves, the chain degrades
        to None instead of propagating an exception."""
        import stoptimes.val.predictors.gtfs_eta_models as mod

        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(mod, "_from_env", _boom)
        monkeypatch.setattr(mod, "_from_source_tree", _boom)
        monkeypatch.setattr(mod, "_from_installed_dist_info", _boom)
        monkeypatch.setattr(mod, "_from_installed_models_package", _boom)
        assert GtfsEtaModelPredictor._resolve_model_registry_dir() is None
