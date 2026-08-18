"""GtfsEtaModelPredictor — wraps gtfs-eta's trained ETA models for live validation.

Bridges etaval's ``Predictor`` protocol to ``eta_service.estimator.estimate_stop_times``
from the sibling ``gtfs-eta`` repo, so any model trained there (historical_mean, ewma,
polyreg_distance, polyreg_time, xgboost) can be scored against real arrivals the same
way ``gtfsrt`` and ``baseline:constant_speed`` are.

Requires ``gtfs-eta`` importable (see ``[tool.uv.sources]`` in pyproject.toml — it's
wired as a sibling path dependency, same pattern as gtfs-rt-pipeline -> gtfs-django).

Two environment quirks, both worked around below rather than assumed fixed upstream:

1. On at least one dev machine (Python 3.14 + uv-managed venv), the editable
   install's ``.pth``-based import hook silently never fires at interpreter
   startup — confirmed independent of ``uv`` by invoking the venv's python
   binary directly with a stripped env. This module falls back to an explicit
   ``sys.path`` append for ``this-file-is-still-in-the-source-tree`` cases
   (e.g. editable install, or pytest's own rootdir-based import), the same
   pattern this sibling repo's own scripts already use for its internal
   cross-package imports (see e.g. ``models/train_all_models.py``,
   ``models/*/predict.py``).
2. ``etaval``'s own console-script entrypoint needs a *non-editable* install to
   work around (1) — which means this file itself gets physically copied into
   etaval's venv (e.g. ``.venv/lib/python3.14/site-packages/etaval/predictors/``).
   At that point ``Path(__file__)`` says nothing about where the *source* tree
   lives, so any fixed ``parents[N]`` guess is next to useless (a first attempt
   at exactly that silently computed ``.venv/lib/python3.14/gtfs-eta`` — plausible
   looking, structurally wrong, and *not caught by any test* since it never
   raises, just returns an empty registry). ``core.config._find_project_root()``
   has the identical problem for the same reason: walking up from the
   *installed* copy of ``core/config.py`` finds *etaval's* pyproject.toml first,
   not gtfs-eta's, so it can't be trusted here either.

   The only sound anchor in a non-editable install is ``MODEL_REGISTRY_DIR``
   set directly in the environment — not derived from any file's location.
   ``_resolve_model_registry_dir`` below tries the one in-repo guess that's
   valid *only* while running from source, but verifies it against a real
   ``registry.json`` before trusting it, and otherwise leaves the decision to
   an externally-set ``MODEL_REGISTRY_DIR`` (see README) rather than silently
   defaulting to a wrong path.

``predictor_id`` is ``f"gtfs_eta:{model_type}"`` (e.g. ``"gtfs_eta:xgboost"``,
``"gtfs_eta:polyreg_time"``) so runs can distinguish which trained model produced
each prediction. Smart model selection (route-specific model preferred over global,
best by ``test_mae_seconds``) is delegated entirely to ``estimate_stop_times``.
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from etaval.domain.models import Stop, TripInstance, VehicleState
from etaval.predictors.base import StopPrediction

# Valid only when this file is still in-source (editable install, or pytest's
# rootdir-based import) -- see point 2 in the module docstring for why a
# non-editable install can't use this at all.
_SOURCE_TREE_GTFS_ETA_ROOT = Path(__file__).resolve().parents[3] / "gtfs-eta"


def _resolve_model_registry_dir() -> Path | None:
    """Return a verified gtfs-eta models/trained/ dir, or None if unverifiable.

    Only trusts ``_SOURCE_TREE_GTFS_ETA_ROOT`` after confirming it actually
    contains a real ``registry.json`` -- never returns a plausible-looking
    guess that turns out to point at an empty directory.
    """
    candidate = _SOURCE_TREE_GTFS_ETA_ROOT / "models" / "trained"
    if (candidate / "registry.json").exists():
        return candidate
    return None


if "MODEL_REGISTRY_DIR" not in os.environ:
    _resolved = _resolve_model_registry_dir()
    if _resolved is not None:
        os.environ["MODEL_REGISTRY_DIR"] = str(_resolved)
    else:
        warnings.warn(
            "MODEL_REGISTRY_DIR is not set and gtfs-eta's models/trained/ "
            "could not be located automatically (this file is likely running "
            "from a non-editable install, where its own path says nothing "
            "about the source tree). gtfs_eta:* predictors will report 'No "
            "trained models found' until you export MODEL_REGISTRY_DIR "
            "pointing at gtfs-eta's models/trained/ directory -- see README.",
            RuntimeWarning,
            stacklevel=2,
        )

try:
    from eta_service.estimator import estimate_stop_times
except ModuleNotFoundError:
    if str(_SOURCE_TREE_GTFS_ETA_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_TREE_GTFS_ETA_ROOT))
    from eta_service.estimator import estimate_stop_times


class GtfsEtaModelPredictor:
    """Serves predictions from a gtfs-eta trained model registry entry.

    Args:
        model_type: One of ``historical_mean``, ``ewma``, ``polyreg_distance``,
            ``polyreg_time``, ``xgboost``. Selects which trained model family
            ``estimate_stop_times`` looks up in the registry.
        model_key: Optional exact registry key. When ``None`` (the default),
            ``estimate_stop_times`` picks the best model for ``model_type`` —
            preferring a route-specific model over the global one.
    """

    def __init__(self, model_type: str, model_key: str | None = None) -> None:
        self.model_type = model_type
        self.model_key = model_key
        self.predictor_id: str = f"gtfs_eta:{model_type}"

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
    ) -> list[StopPrediction]:
        """Emit one StopPrediction per upcoming stop the trained model can score.

        Args:
            trip: The active TripInstance — supplies ``route_id``/``trip_id`` for
                model selection and ``vehicle_id`` for the served feature payload.
            upcoming_stops: Ordered stops ahead of the vehicle.
            vehicle_state: Latest vehicle position sample.
            trip_updates: Not used by this predictor; accepted for Protocol
                compatibility.

        Returns:
            One :class:`StopPrediction` per stop ``estimate_stop_times`` returned
            a non-``None`` ``eta_seconds`` for. Empty if no upcoming stops, no
            registered model matches ``model_type``, or every per-stop prediction
            errored (e.g. missing trained feature columns).
        """
        if not upcoming_stops:
            return []

        made_at = vehicle_state.epoch
        vp_iso = datetime.fromtimestamp(made_at, tz=timezone.utc).isoformat()

        vehicle_position = {
            "vehicle_id": trip.vehicle_id,
            "lat": vehicle_state.lat,
            "lon": vehicle_state.lon,
            "timestamp": vp_iso,
            "speed": vehicle_state.speed,
            "current_status": vehicle_state.current_status,
            "route": trip.route_id,
        }
        stops_payload = [
            {
                "stop_id": s.stop_id,
                "stop_sequence": s.stop_sequence,
                "lat": s.lat,
                "lon": s.lon,
            }
            for s in upcoming_stops
        ]

        result = estimate_stop_times(
            vehicle_position=vehicle_position,
            upcoming_stops=stops_payload,
            route_id=trip.route_id,
            trip_id=trip.trip_id,
            model_key=self.model_key,
            model_type=self.model_type,
            max_stops=len(stops_payload),
        )

        n_upcoming = len(upcoming_stops)
        predictions: list[StopPrediction] = []
        for idx, p in enumerate(result.get("predictions", [])):
            eta_seconds = p.get("eta_seconds")
            if eta_seconds is None:
                continue
            predictions.append(
                StopPrediction(
                    stop_id=p["stop_id"],
                    stop_sequence=p["stop_sequence"],
                    predicted_arrival_epoch=made_at + eta_seconds,
                    made_at_epoch=made_at,
                    dist_remaining_m=p.get("distance_to_stop_m"),
                    stops_remaining=n_upcoming - idx,
                )
            )

        return predictions
