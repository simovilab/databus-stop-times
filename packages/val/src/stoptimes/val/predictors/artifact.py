"""StoptimesArtifactPredictor -- scores a trained artifact straight from a
`core.contracts.artifact` registry, by artifact_id or by scope.

`stoptimes:<ref>` predictors resolve *what* to serve two ways:

- *ref* is a literal ``artifact_id`` present in the registry
  (``<registry>/<artifact_id>/artifact.json``) -- that one artifact serves
  every trip.
- otherwise *ref* names a **scope** -- today, a source id (agency), e.g.
  ``stoptimes:mbta`` -- and ``core.contracts.selection.select_artifact``
  picks the best-scoped candidate per trip (global vs. per-route vs.
  per-trip), exactly mirroring what ``stoptimes serve`` would choose for
  that trip (docs/CONTRACTS.md #7b) rather than delegating to the older,
  agency-unaware ``models.common.registry.get_best_model`` ranking.

Once an artifact is resolved for a trip, *how* it gets scored depends on
whether it carries pre-Phase-2 ``legacy`` migration metadata:

- **Migrated** (``ArtifactMetadata.legacy`` is set, from ``stoptimes models
  migrate``): delegated to ``stoptimes.serve.estimator.estimate_stop_times``
  via :class:`~stoptimes.val.predictors.gtfs_eta_models.GtfsEtaModelPredictor`,
  using the original pre-migration ``model_key`` verbatim -- exactly what
  ``estimate_stop_times`` needs, bypassing its own ranking since the
  artifact was already chosen here via ``core.contracts.selection``.
- **Native** (produced directly by ``stoptimes train run``, no ``legacy``
  block): loaded with ``stoptimes.serve.model_loading.load_artifact_model``
  and scored through the same feature/predict seam ``stoptimes serve`` uses
  for every native artifact -- ``stoptimes.serve.pipeline``'s
  ``default_features``/``default_predict`` (T3's ``core.features.registry``
  landed in Phase 2; the module-level feature building lives in
  ``serve.pipeline.default_features``, not here, so this predictor
  and ``stoptimes serve`` never compute a trip's features two different
  ways). ``trip``/``upcoming_stops``/``vehicle_state`` are already
  map-matched by val's own engine (``VehicleState.progress_m`` /
  ``Stop.progress_m``, from the same ``core.spatial.polyline`` primitives
  ``serve.pipeline.default_match`` uses), so this predictor builds
  ``MatchedObservation``s directly from them instead of re-matching --
  see ``_build_matched_observations`` below. Only the feature/predict
  stages are shared; ``match`` itself is a no-op here because val already
  did it. See ``docs/notes/f2.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stoptimes.core.contracts.artifact import ArtifactMetadata, read_metadata
from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import PipelineContext
from stoptimes.core.contracts.selection import SelectionPolicy, select_artifact
from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.serve.pipeline import MatchedObservation
from stoptimes.val.predictors.base import StopPrediction


def default_registry_dir() -> Path:
    """Default artifact registry root: ``core.config``'s ``model_registry_dir``."""
    from stoptimes.core.config import get_config

    return Path(get_config().model_registry_dir)


def load_registry_candidates(registry_dir: Path) -> list[ArtifactMetadata]:
    """Read every ``<registry_dir>/*/artifact.json`` into an ``ArtifactMetadata`` list.

    A subdirectory without an ``artifact.json``, or with one that fails to
    parse, is skipped rather than raising -- a registry is expected to be
    scanned repeatedly (once per predictor construction) and a single
    malformed entry must not take every other artifact down with it.
    """
    candidates: list[ArtifactMetadata] = []
    registry_dir = Path(registry_dir)
    if not registry_dir.is_dir():
        return candidates
    for child in sorted(registry_dir.iterdir()):
        meta_path = child / "artifact.json"
        if not meta_path.is_file():
            continue
        try:
            candidates.append(read_metadata(child))
        except Exception:  # noqa: BLE001 - malformed artifact.json, skip, not fatal
            continue
    return candidates


class _ConfigLocale:
    """Minimal ``ctx.source``-shaped stand-in for ``default_features``.

    ``serve.pipeline.default_features`` reads ``ctx.source.timezone`` /
    ``ctx.source.holiday_region`` to localize temporal features -- normally
    a real ``contracts.source.Source``, which val's ``Predictor.predict()``
    doesn't receive (only ``trip``/``upcoming_stops``/``vehicle_state``,
    identified by a bare ``source_id`` string). Sourcing the same
    ``core.config`` defaults ``stoptimes.serve.estimator.estimate_stop_times``
    uses for the legacy bridge keeps the two paths' temporal features
    computed the same way rather than silently falling back to UTC/no
    holiday region.
    """

    def __init__(self, source_id: str, timezone: str, holiday_region: str | None) -> None:
        self.id = source_id
        self.timezone = timezone
        self.holiday_region = holiday_region


def _build_matched_observations(
    trip: TripInstance,
    upcoming_stops: list[Stop],
    vehicle_state: VehicleState,
) -> list[MatchedObservation]:
    """Build one ``MatchedObservation`` per upcoming stop directly from val's
    already-matched inputs, mirroring what ``serve.pipeline``'s
    ``_match_vehicle_position`` computes -- ``distance_to_stop`` from the
    difference of two ``progress_m`` values along the same route polyline,
    and 0-indexed ``stops_ahead`` -- without re-running map-matching, since
    val's engine (``val/engine/runtime.py``) already projected both the
    vehicle and every upcoming stop onto the trip's shape via
    ``core.spatial.polyline`` before calling ``predict()``.
    """
    observation = Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id=trip.source,
        trip_id=trip.trip_id,
        route_id=trip.route_id,
        direction_id=trip.direction_id,
        start_date=trip.start_date,
        vehicle_id=trip.vehicle_id,
        ts=pd.Timestamp(vehicle_state.epoch, unit="s", tz="UTC"),
        lat=vehicle_state.lat,
        lon=vehicle_state.lon,
    )

    vehicle_progress_m = vehicle_state.progress_m
    matched: list[MatchedObservation] = []
    for idx, stop in enumerate(upcoming_stops):
        extra: dict[str, Any] = {"stops_ahead": float(idx)}
        if stop.progress_m is not None and vehicle_progress_m is not None:
            extra["distance_to_stop"] = float(stop.progress_m - vehicle_progress_m)
        matched.append(
            MatchedObservation(
                trip_id=trip.trip_id,
                route_id=trip.route_id,
                direction_id=trip.direction_id,
                start_date=trip.start_date,
                vehicle_id=trip.vehicle_id,
                stop_id=stop.stop_id,
                stop_sequence=stop.stop_sequence,
                observation=observation,
                extra=extra,
            )
        )
    return matched


def _score_native_artifact(
    artifact: ArtifactMetadata,
    registry_dir: Path,
    trip: TripInstance,
    upcoming_stops: list[Stop],
    vehicle_state: VehicleState,
    model_cache: dict[str, Any],
) -> list[StopPrediction]:
    """Score a native (non-``legacy``) artifact through the same
    ``serve.pipeline`` feature/predict seam ``stoptimes serve`` uses.

    Loads the model once per ``artifact_id`` (cached on *model_cache*, keyed
    by ``StoptimesArtifactPredictor`` instance) via
    ``serve.model_loading.load_artifact_model`` -- the same loader
    ``stoptimes serve`` calls -- then runs ``serve.pipeline.default_features``
    / ``default_predict`` over ``MatchedObservation``s built from val's
    already-matched inputs (see ``_build_matched_observations``).
    """
    from stoptimes.core.config import get_config
    from stoptimes.serve.model_loading import load_artifact_model
    from stoptimes.serve.pipeline import default_features, default_predict

    model = model_cache.get(artifact.artifact_id)
    if model is None:
        model = load_artifact_model(artifact, registry_dir)
        model_cache[artifact.artifact_id] = model

    matched = _build_matched_observations(trip, upcoming_stops, vehicle_state)
    if not matched:
        return []

    config = get_config()
    now = datetime.fromtimestamp(vehicle_state.epoch, tz=UTC)
    ctx = PipelineContext(
        source=_ConfigLocale(trip.source, config.default_timezone, config.default_region),
        static_feed=None,
        model=model,
        now=now,
    )

    features = default_features(matched, ctx)
    stop_time_predictions = default_predict(features, ctx)

    distance_by_key = {(item.stop_id, item.stop_sequence): item.extra.get("distance_to_stop") for item in matched}
    idx_by_key = {(item.stop_id, item.stop_sequence): idx for idx, item in enumerate(matched)}
    n_upcoming = len(upcoming_stops)

    predictions: list[StopPrediction] = []
    for pred in stop_time_predictions:
        if pred.predicted_arrival_time is None:
            continue
        key = (pred.stop_id, pred.stop_sequence)
        idx = idx_by_key.get(key, 0)
        predictions.append(
            StopPrediction(
                stop_id=pred.stop_id,
                stop_sequence=pred.stop_sequence,
                predicted_arrival_epoch=pred.predicted_arrival_time.timestamp(),
                made_at_epoch=vehicle_state.epoch,
                dist_remaining_m=distance_by_key.get(key),
                stops_remaining=n_upcoming - idx,
            )
        )
    return predictions


class StoptimesArtifactPredictor:
    """Serves predictions from a ``core.contracts.artifact`` registry entry, by id or scope.

    Args:
        ref: An exact ``artifact_id``, or a scope name (today: a
            ``source_id``) resolved per-trip via
            ``core.contracts.selection.select_artifact``.
        registry_dir: Registry root to scan; defaults to
            ``core.config.get_config().model_registry_dir``.
        source_id: Agency id passed to ``select_artifact`` when *ref* names a
            scope rather than one artifact. Defaults to *ref* itself -- the
            common case, where ``stoptimes:mbta`` selects for source
            ``"mbta"``.
        policy: Optional ``SelectionPolicy`` override; defaults to
            ``SelectionPolicy()``.
    """

    def __init__(
        self,
        ref: str,
        registry_dir: Path | None = None,
        source_id: str | None = None,
        policy: SelectionPolicy | None = None,
    ) -> None:
        self.ref = ref
        self.predictor_id: str = f"stoptimes:{ref}"
        self._registry_dir = Path(registry_dir) if registry_dir is not None else default_registry_dir()
        self._source_id = source_id if source_id is not None else ref
        self._policy = policy or SelectionPolicy()

        self._candidates = load_registry_candidates(self._registry_dir)
        by_id = {c.artifact_id: c for c in self._candidates}
        self._static_artifact: ArtifactMetadata | None = by_id.get(ref)
        self._native_model_cache: dict[str, Any] = {}
        """Loaded native models, keyed by artifact_id -- `load_artifact_model`
        deserializes a pickle per call, so a live/replay loop scoring the
        same artifact every tick would otherwise reload it every tick."""

    def _resolve(self, trip: TripInstance) -> ArtifactMetadata | None:
        """The artifact to serve *trip* with -- the static match if *ref* was
        an exact artifact_id, otherwise a fresh per-trip selection."""
        if self._static_artifact is not None:
            return self._static_artifact
        return select_artifact(
            self._candidates,
            self._policy,
            source_id=self._source_id,
            route_id=trip.route_id,
            direction_id=trip.direction_id,
            trip_id=trip.trip_id,
        )

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
    ) -> list[StopPrediction]:
        """Resolve an artifact for *trip* and delegate scoring to it.

        Migrated artifacts (``legacy.model_key`` set) go through
        ``GtfsEtaModelPredictor`` / ``estimate_stop_times``; native artifacts
        (``stoptimes train run`` output, no ``legacy`` block) go through
        ``serve.model_loading.load_artifact_model`` +
        ``serve.pipeline.default_features``/``default_predict`` -- see the
        module docstring.

        Returns:
            Empty list if there are no upcoming stops or no artifact
            resolves for this trip (e.g. an empty/mismatched registry --
            never invents predictions).
        """
        if not upcoming_stops:
            return []

        artifact = self._resolve(trip)
        if artifact is None:
            return []

        legacy = artifact.legacy or {}
        model_key = legacy.get("model_key")
        if model_key is not None:
            from stoptimes.val.predictors.gtfs_eta_models import GtfsEtaModelPredictor

            delegate = GtfsEtaModelPredictor(model_type=artifact.family, model_key=model_key)
            return delegate.predict(trip, upcoming_stops, vehicle_state, trip_updates)

        return _score_native_artifact(
            artifact,
            self._registry_dir,
            trip,
            upcoming_stops,
            vehicle_state,
            self._native_model_cache,
        )
