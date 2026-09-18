"""GtfsEtaModelPredictor — wraps stoptimes' trained ETA models for live validation.

Bridges val's ``Predictor`` protocol to ``stoptimes.serve.estimator.estimate_stop_times``
(``packages/serve``, a sibling workspace member of this package, not a separate
repo) so any model trained with ``stoptimes-train``/``stoptimes-models``
(historical_mean, polyreg_distance, polyreg_time, xgboost — ``ewma`` was
retired and is not in ``PREDICTOR_REGISTRY``) can be scored against real
arrivals the same way ``gtfsrt`` and ``baseline:constant_speed`` are.

New code should prefer the ``stoptimes:{model_type}`` predictor id (an alias
of this same class, registered in ``engine.assembly.PREDICTOR_REGISTRY``);
``gtfs_eta:{model_type}`` is kept only for backward compatibility with
existing run data and scripts and emits a ``DeprecationWarning`` when used
through the registry (constructing this class directly, as below, does not
warn — only the registry factory does).

Locating the trained-model registry (``_resolve_model_registry_dir``) predates
this codebase's merge into one ``stoptimes`` monorepo, back when ``etaval``
and ``gtfs-eta`` were separate repos linked by a ``uv`` path dependency and a
model registry directory had to be found across that boundary (a sibling
source checkout, an installed dist-info's ``direct_url.json``, or an
installed ``models`` package — see the resolver functions below). Those
fallbacks are harmless no-ops in this workspace (everything lives under
``packages/`` now, found the ordinary way via
``stoptimes.core.config.get_config().model_registry_dir``) and are kept only
because they still correctly resolve a real registry in the one case they
were built for: this file installed as a standalone package (e.g.
``pip install stoptimes-val[models]``) against a separately-installed
``stoptimes-models`` distribution, where ``__file__``-relative guesses are
meaningless and an explicit ``MODEL_REGISTRY_DIR`` may not be set. See the
README for the current recommended setup.

``predictor_id`` is ``f"gtfs_eta:{model_type}"`` (e.g. ``"gtfs_eta:xgboost"``,
``"gtfs_eta:polyreg_time"``) so runs can distinguish which trained model produced
each prediction. Smart model selection (route-specific model preferred over global,
best by ``test_mae_seconds``) is delegated entirely to ``estimate_stop_times``.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.predictors.base import StopPrediction

# Valid only when this file is still in-source (editable install, or pytest's
# rootdir-based import) -- see point 2 in the module docstring for why a
# non-editable install can't use this at all.
_SOURCE_TREE_GTFS_ETA_ROOT = Path(__file__).resolve().parents[3] / "gtfs-eta"


def _has_registry(candidate: Path) -> bool:
    """True if *candidate* looks like a real gtfs-eta models/trained/ dir."""
    try:
        return (candidate / "registry.json").is_file()
    except OSError:
        return False


def _from_env() -> Path | None:
    """Step 1: ``MODEL_REGISTRY_DIR`` env var, if set. Always wins when present.

    Trusted as-is (not verified against registry.json) -- an explicit
    operator override always wins, even if it points somewhere unexpected.
    """
    value = os.environ.get("STOPTIMES_MODEL_REGISTRY_DIR") or os.environ.get("MODEL_REGISTRY_DIR")
    if not value:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _from_source_tree() -> Path | None:
    """Step 2: sibling source checkout of gtfs-eta (…/gtfs-eta/models/trained).

    Only trusted after confirming it actually contains a real
    ``registry.json`` -- never returns a plausible-looking guess that turns
    out to point at an empty directory.
    """
    try:
        candidate = _SOURCE_TREE_GTFS_ETA_ROOT / "models" / "trained"
    except OSError:
        return None
    return candidate if _has_registry(candidate) else None


def _from_installed_dist_info() -> Path | None:
    """Step 3: gtfs-eta's installed dist-info ``direct_url.json``.

    Under a non-editable install, ``etaval`` is a uv-rebuilt wheel so its own
    ``__file__`` says nothing about the source tree -- but the ``gtfs-eta``
    distribution's dist-info still records its source path (editable or not)
    in ``direct_url.json`` as a ``file://`` URL. Parse that instead of
    guessing from etaval's own installed location. Defensive throughout:
    any failure (package not installed, no direct_url.json, malformed JSON,
    non-file:// URL) degrades to None rather than raising.
    """
    try:
        dist = importlib.metadata.distribution("gtfs-eta")
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None

    try:
        raw = dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None

    try:
        info = json.loads(raw)
        url = info.get("url", "")
    except (json.JSONDecodeError, AttributeError):
        return None

    if not url.startswith("file://"):
        return None

    try:
        from urllib.parse import unquote, urlparse

        source_root = Path(unquote(urlparse(url).path))
    except Exception:  # noqa: BLE001
        return None

    candidate = source_root / "models" / "trained"
    return candidate if _has_registry(candidate) else None


def _from_installed_models_package() -> Path | None:
    """Step 4: the installed top-level ``models`` package directory.

    gtfs-eta's ``models/`` (containing ``trained/``) is installed as its own
    top-level package alongside ``eta_service`` and ``gtfs_eta`` when
    packaged. Locate it via ``importlib.metadata`` rather than relative
    ``__file__`` walks, which are meaningless in a non-editable install.
    """
    try:
        dist = importlib.metadata.distribution("gtfs-eta")
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None

    try:
        files = dist.files
    except Exception:  # noqa: BLE001
        return None
    if not files:
        return None

    try:
        locate = dist.locate_file
        for f in files:
            parts = Path(str(f)).parts
            if parts and parts[0] == "models" and "trained" in parts:
                trained_idx = parts.index("trained")
                rel = Path(*parts[: trained_idx + 1])
                candidate = Path(str(locate(rel)))
                if _has_registry(candidate):
                    return candidate
    except Exception:  # noqa: BLE001
        return None
    return None


class GtfsEtaModelPredictor:
    """Serves predictions from a trained ``stoptimes-models`` registry entry.

    Args:
        model_type: One of ``historical_mean``, ``polyreg_distance``,
            ``polyreg_time``, ``xgboost`` (``ewma`` was retired upstream and
            is not a valid value here). Selects which trained model family
            ``estimate_stop_times`` looks up in the registry.
        model_key: Optional exact registry key. When ``None`` (the default),
            ``estimate_stop_times`` picks the best model for ``model_type`` —
            preferring a route-specific model over the global one.
    """

    @staticmethod
    def _resolve_model_registry_dir() -> Path | None:
        """Return a verified gtfs-eta models/trained/ dir, or ``None`` if unresolvable.

        Tries, in order, returning the first that exists (verified via a real
        ``registry.json`` — except the explicit env-var override, which
        always wins as-is):

        1. ``MODEL_REGISTRY_DIR`` env var.
        2. A sibling source checkout of gtfs-eta (``…/gtfs-eta/models/trained``).
        3. gtfs-eta's installed dist-info ``direct_url.json`` ``file://`` path.
        4. The installed top-level ``models`` package directory.

        Every step is defensive — it never raises, only degrades to the next
        option, finally returning ``None`` if nothing resolves.
        """
        for resolver in (
            _from_env,
            _from_source_tree,
            _from_installed_dist_info,
            _from_installed_models_package,
        ):
            try:
                result = resolver()
            except Exception:  # noqa: BLE001
                result = None
            if result is not None:
                return result
        return None

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
        vp_iso = datetime.fromtimestamp(made_at, tz=UTC).isoformat()

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


if not (os.environ.get("STOPTIMES_MODEL_REGISTRY_DIR") or os.environ.get("MODEL_REGISTRY_DIR")):
    _resolved = GtfsEtaModelPredictor._resolve_model_registry_dir()
    if _resolved is not None:
        os.environ["MODEL_REGISTRY_DIR"] = str(_resolved)
    else:
        warnings.warn(
            "MODEL_REGISTRY_DIR is not set and gtfs-eta's models/trained/ "
            "could not be located automatically (checked a sibling source "
            "checkout, gtfs-eta's installed dist-info, and the installed "
            "models/ package). gtfs_eta:* predictors will report 'No "
            "trained models found' until you export MODEL_REGISTRY_DIR "
            "pointing at gtfs-eta's models/trained/ directory -- see README. "
            "In a packaged install this is normally auto-resolved; this "
            "warning means none of the fallbacks found a real registry.json.",
            RuntimeWarning,
            stacklevel=2,
        )

try:
    from stoptimes.serve.estimator import estimate_stop_times
except ModuleNotFoundError:
    if str(_SOURCE_TREE_GTFS_ETA_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_TREE_GTFS_ETA_ROOT))
    from stoptimes.serve.estimator import estimate_stop_times
