"""Contract: choosing which artifact serves a given trip.

A registry holds many artifacts for one agency — a global model, per-route
models, sometimes per-trip ones — and serving has to pick one per prediction.
The rule is **most specific scope that has a candidate, best metric within
it**, with explicit pins overriding everything. That is what
`stoptimes.models.common.registry.get_best_model` plus the per-agency pins in
`stoptimes.serve.estimator` do today, expressed as one pure function so
serving (T4), reporting (T3) and validation (T6) can't drift apart.

Selection reads `artifact.json` only: no pickle is opened to rank candidates.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from stoptimes.core.contracts.artifact import ArtifactMetadata, ScopeLevel

DEFAULT_SCOPE_ORDER: tuple[ScopeLevel, ...] = (
    ScopeLevel.TRIP,
    ScopeLevel.DIRECTION,
    ScopeLevel.ROUTE,
    ScopeLevel.GLOBAL,
)


class SelectionPolicy(BaseModel):
    """How an agency picks a model. Lives on `Source.serving` (per agency), overridable per call."""

    model_config = ConfigDict(extra="forbid")

    metric: str = "test_mae_seconds"
    """Key in `ArtifactMetadata.metrics`. Artifacts missing it are never selected."""

    lower_is_better: bool = True
    scope_order: list[ScopeLevel] = Field(default_factory=lambda: list(DEFAULT_SCOPE_ORDER))
    """Tried most specific first; the first level with a candidate wins."""

    families: list[str] = Field(default_factory=list)
    """Restrict to these model families; empty = any."""

    pins: dict[str, str] = Field(default_factory=dict)
    """Scope key -> artifact_id, bypassing the ranking: `global`, `route:22`,
    `route:22/1` (direction), `trip:t1`. A pin naming an artifact that isn't in
    the registry is ignored, so a deleted model degrades to ranking instead of
    to no predictions at all."""

    def pinned_for(self, *, route_id: str | None, direction_id: int | None, trip_id: str | None) -> list[str]:
        """Pin keys that apply here, most specific first."""
        keys = []
        if trip_id is not None:
            keys.append(f"trip:{trip_id}")
        if route_id is not None and direction_id is not None:
            keys.append(f"route:{route_id}/{direction_id}")
        if route_id is not None:
            keys.append(f"route:{route_id}")
        keys.append("global")
        return [self.pins[key] for key in keys if key in self.pins]


def select_artifact(
    candidates: Iterable[ArtifactMetadata],
    policy: SelectionPolicy | None = None,
    *,
    source_id: str,
    route_id: str | None = None,
    direction_id: int | None = None,
    trip_id: str | None = None,
) -> ArtifactMetadata | None:
    """The artifact that should serve this trip, or None if nothing qualifies.

    Callers decide what to do with None (skip the trip, emit a baseline,
    or raise) — this function never invents a model.
    """
    policy = policy or SelectionPolicy()
    usable = [
        artifact
        for artifact in candidates
        if (not policy.families or artifact.family in policy.families)
        and isinstance(artifact.metrics.get(policy.metric), (int, float))
        and artifact.scope.matches(
            source_id=source_id, route_id=route_id, direction_id=direction_id, trip_id=trip_id
        )
    ]
    if not usable:
        return None

    by_id = {artifact.artifact_id: artifact for artifact in usable}
    for pinned in policy.pinned_for(route_id=route_id, direction_id=direction_id, trip_id=trip_id):
        if pinned in by_id:
            return by_id[pinned]

    for level in policy.scope_order:
        at_level = [artifact for artifact in usable if artifact.scope.level == level]
        if at_level:
            return min(
                at_level,
                key=lambda artifact: (
                    artifact.metrics[policy.metric] if policy.lower_is_better else -artifact.metrics[policy.metric],
                    artifact.artifact_id,
                ),
            )
    return None
