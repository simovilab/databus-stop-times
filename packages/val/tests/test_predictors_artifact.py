"""test_predictors_artifact — StoptimesArtifactPredictor (T6 deliverable 3).

Builds a fake `core.contracts.artifact` registry on disk (tmp_path, no real
model files) and checks artifact-id vs. scope resolution, and the bridge to
GtfsEtaModelPredictor for migrated artifacts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    ModelScope,
    Provenance,
    ScopeLevel,
    write_metadata,
)
from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.predictors.artifact import StoptimesArtifactPredictor

TRIP_GLOBAL = TripInstance(
    source="test",
    trip_id="t1",
    vehicle_id="v1",
    route_id="99",
    direction_id=0,
    shape_id="s1",
    start_date="20260701",
    started_at=1_000_000.0,
)

TRIP_ROUTE_22 = TripInstance(
    source="test",
    trip_id="t2",
    vehicle_id="v2",
    route_id="22",
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


def _stops() -> list[Stop]:
    return [Stop(stop_id="s1", stop_sequence=1, lat=42.2, lon=-71.1)]


def _write_artifact(
    registry_dir: Path,
    artifact_id: str,
    *,
    family: str = "xgboost",
    scope: ModelScope | None = None,
    metrics: dict | None = None,
    legacy: dict | None = None,
) -> None:
    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family=family,
        scope=scope or ModelScope(),
        provenance=Provenance(created_at=datetime.now(UTC)),
        metrics=metrics or {"test_mae_seconds": 30.0},
        legacy=legacy,
    )
    directory = registry_dir / artifact_id
    write_metadata(meta, directory)


# ── Resolution by exact artifact_id ───────────────────────────────────────────


def test_exact_artifact_id_is_used_for_any_trip(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "xgb-global-v1", legacy={"model_key": "xgboost_global"})
    predictor = StoptimesArtifactPredictor("xgb-global-v1", registry_dir=tmp_path)

    assert predictor.predictor_id == "stoptimes:xgb-global-v1"
    assert predictor._resolve(TRIP_GLOBAL).artifact_id == "xgb-global-v1"
    assert predictor._resolve(TRIP_ROUTE_22).artifact_id == "xgb-global-v1"


# ── Resolution by scope (source id) via core.contracts.selection ─────────────


def test_scope_ref_selects_route_specific_over_global(tmp_path: Path) -> None:
    _write_artifact(
        tmp_path,
        "global-model",
        scope=ModelScope(level=ScopeLevel.GLOBAL),
        metrics={"test_mae_seconds": 50.0},
        legacy={"model_key": "global_key"},
    )
    _write_artifact(
        tmp_path,
        "route22-model",
        scope=ModelScope(level=ScopeLevel.ROUTE, route_ids=["22"]),
        metrics={"test_mae_seconds": 20.0},
        legacy={"model_key": "route22_key"},
    )

    predictor = StoptimesArtifactPredictor("test", registry_dir=tmp_path)

    assert predictor._resolve(TRIP_ROUTE_22).artifact_id == "route22-model"
    assert predictor._resolve(TRIP_GLOBAL).artifact_id == "global-model"


def test_scope_ref_returns_none_when_no_candidate_matches(tmp_path: Path) -> None:
    predictor = StoptimesArtifactPredictor("empty-source", registry_dir=tmp_path)
    assert predictor._resolve(TRIP_GLOBAL) is None
    assert predictor.predict(TRIP_GLOBAL, _stops(), STATE) == []


# ── predict() bridges to GtfsEtaModelPredictor for migrated artifacts ────────


def test_predict_bridges_to_gtfs_eta_model_predictor(tmp_path: Path) -> None:
    _write_artifact(
        tmp_path,
        "xgb-global-v1",
        family="xgboost",
        legacy={"model_key": "xgboost_global"},
    )
    predictor = StoptimesArtifactPredictor("xgb-global-v1", registry_dir=tmp_path)

    fake_result = {
        "predictions": [
            {"stop_id": "s1", "stop_sequence": 1, "eta_seconds": 90.0, "distance_to_stop_m": 400.0},
        ]
    }
    with patch(
        "stoptimes.val.predictors.gtfs_eta_models.estimate_stop_times", return_value=fake_result
    ) as mock_est:
        preds = predictor.predict(TRIP_GLOBAL, _stops(), STATE, None)

    assert len(preds) == 1
    assert preds[0].predicted_arrival_epoch == STATE.epoch + 90.0
    assert mock_est.call_args.kwargs["model_type"] == "xgboost"
    assert mock_est.call_args.kwargs["model_key"] == "xgboost_global"


def test_predict_empty_upcoming_stops_short_circuits(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "xgb-global-v1", legacy={"model_key": "xgboost_global"})
    predictor = StoptimesArtifactPredictor("xgb-global-v1", registry_dir=tmp_path)
    assert predictor.predict(TRIP_GLOBAL, [], STATE, None) == []


def test_predict_raises_for_native_artifact_without_legacy_bridge(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "native-v1", legacy=None)
    predictor = StoptimesArtifactPredictor("native-v1", registry_dir=tmp_path)

    with pytest.raises(RuntimeError, match="core.features.registry"):
        predictor.predict(TRIP_GLOBAL, _stops(), STATE, None)


def test_missing_registry_dir_yields_no_candidates(tmp_path: Path) -> None:
    predictor = StoptimesArtifactPredictor("mbta", registry_dir=tmp_path / "does-not-exist")
    assert predictor._resolve(TRIP_GLOBAL) is None
