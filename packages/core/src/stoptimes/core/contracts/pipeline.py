"""Contract: the serving pipeline stages.

Serving today (`stoptimes.serve.estimator.estimate_stop_times`) is one
function that reads a vehicle position, computes distance/temporal features,
picks a model by route, and returns a prediction. T4 rebuilds this as six
named, overridable stages so a user can, say, swap `match` for their own
map-matcher without touching `predict` or `publish`:

    ingest -> match -> features -> predict -> postprocess -> publish

This module defines the stage names, the per-stage hook signature, the
shared `PipelineContext`, and a `Pipeline` protocol describing the
`@pipeline.stage("predict")` registration mechanism. It ships one reference
implementation, `SequentialPipeline`, proving hooks compose in order; T4
owns the real one (retries, async IO, sinks) built against this contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

PipelineStageName = str
"""One of the six names in `STAGE_ORDER`, kept as `str` (not an Enum) so a
custom pipeline can insert stages T4 didn't anticipate, as long as it still
runs the six standard ones somewhere."""

STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "match",
    "features",
    "predict",
    "postprocess",
    "publish",
)
"""ingest      pull raw realtime records (Observations) for the window being served.
match       associate each Observation with a scheduled trip/stop (map-matching).
features    compute the model's `required_features` for each matched row.
predict     call the model; produces StopTimePrediction rows.
postprocess apply cross-cutting adjustments (clamping, smoothing, baseline blending).
publish     hand finished predictions to a sink (file, HTTP, MQTT — T4)."""


@dataclass
class PipelineContext:
    """State threaded through every stage of one pipeline run.

    Mutable by design (`state` is a scratch dict stages read/write to pass
    data forward) — pipelines are a single-threaded per-run object, not a
    shared/concurrent one, so this doesn't fight the rest of the codebase's
    immutability rule.
    """

    source: Any
    """A `contracts.source.Source`."""
    static_feed: Any
    """Whatever T1's reader returns for the source's GTFS static feed."""
    model: Any
    """Something satisfying `contracts.model.Model`."""
    now: datetime
    """The pipeline's logical clock — always UTC. Stages must read time
    through `ctx.now`, never `datetime.now()`, so replay (T6) is deterministic."""
    config: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("stoptimes.serve.pipeline"))
    state: dict[str, Any] = field(default_factory=dict)
    """Scratch space for inter-stage data that isn't the stage's own
    input/output (e.g. a match stage caching a spatial index for reuse)."""


StageHook = Callable[[Any, PipelineContext], Any]
"""A stage hook takes (payload_from_previous_stage, ctx) and returns the
payload for the next stage. The concrete payload type is per-stage:

    ingest:      (ctx)             -> list[Observation]
    match:       list[Observation] -> list[MatchedObservation]   (T2 shape)
    features:    list[MatchedObservation] -> pandas.DataFrame
    predict:     pandas.DataFrame  -> list[StopTimePrediction]
    postprocess: list[StopTimePrediction] -> list[StopTimePrediction]
    publish:     list[StopTimePrediction] -> None

`ingest`'s hook still matches the `Callable[[Any, PipelineContext], Any]`
shape by ignoring its first argument (conventionally `None`)."""


class Pipeline(Protocol):
    """The registration + execution contract every pipeline implementation follows."""

    def stage(self, name: PipelineStageName) -> Callable[[StageHook], StageHook]:
        """Decorator factory: `@pipeline.stage("predict")` registers the
        decorated function as that stage's hook and returns it unchanged, so
        it stays independently callable/testable.

        Registering a name twice replaces the previous hook (last write
        wins) — pipelines are configured once at startup, not accumulated.
        """
        ...

    def run(self, ctx: PipelineContext) -> Any:
        """Run every registered stage in `STAGE_ORDER`, threading each
        stage's return value into the next as its payload, and return the
        final stage's result. Raises `KeyError` if a stage has no hook."""
        ...


class SequentialPipeline:
    """Reference `Pipeline`: runs registered hooks in `STAGE_ORDER`, nothing else.

    No retries, no async, no sinks — those are T4's job. This exists so
    Phase 1 tests can prove the hook/context contract actually composes
    before any real stage exists.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, StageHook] = {}

    def stage(self, name: PipelineStageName) -> Callable[[StageHook], StageHook]:
        def register(hook: StageHook) -> StageHook:
            self._hooks[name] = hook
            return hook

        return register

    def run(self, ctx: PipelineContext) -> Any:
        payload: Any = None
        for name in STAGE_ORDER:
            if name not in self._hooks:
                raise KeyError(f"no hook registered for stage {name!r}")
            payload = self._hooks[name](payload, ctx)
        return payload
