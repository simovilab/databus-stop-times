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

Scoring is delegated to ``stoptimes.serve.estimator.estimate_stop_times`` --
the same bridge :class:`~stoptimes.val.predictors.gtfs_eta_models.GtfsEtaModelPredictor`
uses -- because building model features generically from an
``ArtifactMetadata.required_features``-style list is ``core.features.registry``'s
job (T3's Phase 2 track; still a placeholder module as of this writing). For
an artifact produced by ``stoptimes models migrate``, ``ArtifactMetadata.legacy``
carries the original pre-migration ``model_key`` verbatim -- exactly what
``estimate_stop_times`` needs to serve that specific artifact, bypassing its
own ranking since the artifact was already chosen here via
``core.contracts.selection``. A "native" (non-migrated) artifact has no such
bridge yet: ``predict()`` raises a clear ``RuntimeError`` rather than
guessing at feature engineering -- this seam is documented in
``docs/notes/p2-t6.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stoptimes.core.contracts.artifact import ArtifactMetadata, read_metadata
from stoptimes.core.contracts.selection import SelectionPolicy, select_artifact
from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
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

        Returns:
            Empty list if there are no upcoming stops or no artifact
            resolves for this trip (e.g. an empty/mismatched registry --
            never invents predictions).

        Raises:
            RuntimeError: the resolved artifact has no ``legacy`` migration
                metadata to bridge to ``estimate_stop_times`` (a "native"
                Phase 2 artifact -- see module docstring).
        """
        if not upcoming_stops:
            return []

        artifact = self._resolve(trip)
        if artifact is None:
            return []

        legacy = artifact.legacy or {}
        model_key = legacy.get("model_key")
        if model_key is None:
            raise RuntimeError(
                f"stoptimes:{self.ref} resolved artifact {artifact.artifact_id!r} "
                f"(family={artifact.family!r}), but it carries no 'legacy' "
                "migration metadata to bridge to "
                "stoptimes.serve.estimator.estimate_stop_times. Native "
                "(non-migrated) artifacts need core.features.registry (T3) "
                "before they can be scored generically -- see docs/notes/p2-t6.md."
            )

        from stoptimes.val.predictors.gtfs_eta_models import GtfsEtaModelPredictor

        delegate = GtfsEtaModelPredictor(model_type=artifact.family, model_key=model_key)
        return delegate.predict(trip, upcoming_stops, vehicle_state, trip_updates)
