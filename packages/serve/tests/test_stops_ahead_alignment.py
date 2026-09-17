"""Regression test for the train/serve stops_ahead off-by-one (roadmap 1.2).

Root cause: ``stoptimes.train.datasets.dataset_builder`` (the loop building
``mbta_bus_222_15_37_*`` stop-level datasets) labels the closest upcoming
stop to the vehicle with ``stops_ahead=0`` (see
``stops_ahead = int(stop_order_arr[j] - closest_stop_order)`` where ``j``
starts at ``closest_idx``). ``estimate_stop_times`` instead labelled the
first entry of ``upcoming_stops`` (the same "closest upcoming stop")
with ``stops_ahead=idx + 1``, i.e. 1. Every stop handed to the model was
therefore one horizon-step further than the model was trained to expect
for that physical situation. Isolated offline (see roadmap 1.2 write-up),
this off-by-one alone is a real but modest contributor (~2-4s mean/median
~0s across a 284-row sample) -- the dominant driver of the measured
+71s served bias turned out to be ``shape_distance_to_stop`` /
``cross_track_error`` / ``shape_progress`` silently defaulting to the
haversine proxy (0 cross-track) when no ``shape=`` is supplied, compounded
with ``is_at_stop`` defaulting to False. This test only pins the
stops_ahead alignment; it is fixed here because it's an unambiguous bug,
not because it's the whole story.

This test seeds a throwaway registry with a spy model that returns the
``stops_ahead`` value it was fed (as seconds) so the served value can be
asserted directly, independent of any real trained model.
"""

import numpy as np
import pandas as pd
import pytest

import stoptimes.models.common.registry as registry_module
import common.registry as bare_registry_module  # noqa: E402
from stoptimes.serve.estimator import estimate_stop_times

MODEL_KEY = "stops_ahead_alignment_test_model"


class _StopsAheadSpyModel:
    """Fake xgboost-shaped model: echoes back the stops_ahead it was fed.

    Picklable (module-level class, no closures) so it round-trips through
    the registry's plain ``pickle.dump``/``pickle.load`` just like a real
    model would.
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["stops_ahead"].to_numpy(dtype=float)


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    """Point the registry singleton at a throwaway dir seeded with one global model.

    PYTHONPATH=.:models makes ``stoptimes.models.common.registry`` and ``common.registry``
    two *separate* imported modules, each with its own module-level
    ``_registry`` singleton (the latter is what ``models/xgb/predict.py``'s
    ``from common.registry import get_registry`` actually resolves to at
    prediction time). Both must be reset, or a stale singleton left behind
    by another test module (e.g. test_shape_hook.py, which only resets the
    former) makes ``estimate_stop_times`` silently fail to find this
    fixture's seeded model when tests run in the same session.
    """
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setattr(registry_module, "_registry", None)
    monkeypatch.setattr(bare_registry_module, "_registry", None)

    registry = registry_module.get_registry()
    registry.save_model(
        MODEL_KEY,
        _StopsAheadSpyModel(),
        metadata={
            "model_type": "xgboost",
            "route_id": None,
            "metrics": {"test_mae_seconds": 1.0},
        },
    )
    yield registry
    monkeypatch.setattr(registry_module, "_registry", None)
    monkeypatch.setattr(bare_registry_module, "_registry", None)


def _vehicle_position():
    return {
        "vehicle_id": "veh1",
        "route": "unknown",
        "lat": 0.0,
        "lon": 0.0,
        "speed": 0.0,
        "timestamp": "2026-01-01T12:00:00Z",
    }


def _stops(n):
    # Evenly spaced stops so distances/order are well defined; content
    # doesn't matter for this test, only stops_ahead does.
    return [
        {"stop_id": f"s{i}", "stop_sequence": i + 1, "lat": 0.0, "lon": 0.001 * (i + 1)}
        for i in range(n)
    ]


def test_first_upcoming_stop_gets_stops_ahead_zero(seeded_registry):
    """The nearest upcoming stop must be served as stops_ahead=0.

    This matches the training-side convention (the closest stop to the
    vehicle is ``stops_ahead=0`` in the dataset builder). Before the fix,
    this was served as stops_ahead=1.
    """
    result = estimate_stop_times(
        _vehicle_position(), _stops(3), model_key=MODEL_KEY, max_stops=3
    )

    assert not result.get("error")
    predicted_stops_ahead = [p["eta_seconds"] for p in result["predictions"]]
    assert predicted_stops_ahead == [0.0, 1.0, 2.0]


def test_stops_ahead_matches_upcoming_stops_index(seeded_registry):
    """More generally: served stops_ahead must equal the stop's position
    in upcoming_stops (0-indexed), for any window size."""
    result = estimate_stop_times(
        _vehicle_position(), _stops(5), model_key=MODEL_KEY, max_stops=5
    )

    assert not result.get("error")
    served_stops_ahead = [p["eta_seconds"] for p in result["predictions"]]
    assert served_stops_ahead == [0.0, 1.0, 2.0, 3.0, 4.0]
