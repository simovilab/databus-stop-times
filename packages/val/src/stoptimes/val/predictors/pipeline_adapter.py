"""PipelineTripUpdatesPredictor -- scores a user-authored serving pipeline (`pipeline:<file.py>`).

Bridges val's ``Predictor`` protocol to ``stoptimes.serve.pipeline.load_pipeline``,
T4's Phase 2 loader for a bring-your-own ``core.contracts.pipeline.Pipeline``.
This is a *small adapter val owns*, not a piece of the pipeline contract
itself (`docs/PLAN.md` rule 1: nothing depends on `val`, but `val`'s
`pipeline:*` predictor is allowed to depend on `serve`, optionally).

The seam this module leans on, and its current state (see
``docs/notes/p2-t6.md`` for the up-to-date status): as of this writing,
``stoptimes.serve.pipeline`` is still T4's Phase 2 placeholder module -- it
defines no ``load_pipeline`` yet. Every import of ``stoptimes.serve`` here is
therefore deferred to first use (construction succeeds; only ``predict()``
needs it), and every failure mode -- ``stoptimes-serve`` not installed at
all, or installed but without ``load_pipeline`` yet -- raises a specific,
actionable error instead of an opaque traceback. This keeps ``stoptimes.val``
importable (and its own test suite green) whether or not `serve` is
installed or finished, per the deliverable ("guarded so val still imports
when stoptimes-serve isn't installed").

Once a real pipeline is loaded, this adapter runs it once per ``predict()``
call and reads ``StopTimePrediction``-shaped results either from the
pipeline's own return value (``pipeline.run(ctx)``, for a pipeline whose
``publish`` stage simply returns its input rather than handing off to a
sink) or, if that's empty/``None``, from ``ctx.state["predictions"]`` -- a
scratch-space convention (`contracts.pipeline.PipelineContext.state` is
documented as exactly this: "scratch space for inter-stage data") a
`postprocess`/`publish` hook can use to make its output visible to a caller
that isn't a real sink, such as this predictor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.predictors.base import StopPrediction


def _import_pipeline_module() -> Any:
    """Import and return the ``stoptimes.serve.pipeline`` module.

    Split out from ``_load_pipeline_loader`` so tests can monkeypatch this
    one function directly instead of fighting namespace-package import
    semantics via ``sys.modules``.
    """
    import stoptimes.serve.pipeline as pipeline_module

    return pipeline_module


def _load_pipeline_loader() -> Any:
    """Return ``stoptimes.serve.pipeline.load_pipeline``, or raise a clear, actionable error."""
    try:
        pipeline_module = _import_pipeline_module()
    except ImportError as exc:
        raise ImportError(
            "The pipeline:<file.py> predictor requires stoptimes-serve installed "
            "(`uv sync --extra serve` in this workspace, or `pip install "
            "stoptimes-serve` standalone)."
        ) from exc

    load_pipeline = getattr(pipeline_module, "load_pipeline", None)
    if load_pipeline is None:
        raise AttributeError(
            "stoptimes.serve.pipeline has no load_pipeline() yet -- it is still "
            "T4's Phase 2 placeholder module as of this writing. See "
            "docs/notes/p2-t6.md for the current state of this seam."
        )
    return load_pipeline


class PipelineTripUpdatesPredictor:
    """Serves predictions from a bring-your-own pipeline file (``pipeline:<file.py>``).

    Args:
        path: Path to the user's pipeline module (passed to
            ``stoptimes.serve.pipeline.load_pipeline``).

    The pipeline is loaded lazily (on first ``predict()`` call, not at
    construction) so building the predictor -- e.g. listing it in
    ``PREDICTOR_REGISTRY`` -- never requires ``stoptimes-serve`` to be
    installed; only actually scoring with it does.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.predictor_id: str = f"pipeline:{self.path.name}"
        self._pipeline: Any = None
        self._static_feed: Any = None

    def attach_static_loader(self, static_loader: Any) -> None:
        """Opt-in hook (mirrors ``ScheduleBaselinePredictor``): supplies
        ``PipelineContext.static_feed`` for the run's GTFS static data."""
        self._static_feed = static_loader

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            load_pipeline = _load_pipeline_loader()
            self._pipeline = load_pipeline(self.path)
        return self._pipeline

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
    ) -> list[StopPrediction]:
        """Run the loaded pipeline for one tick and extract this trip's stop predictions.

        Raises:
            ImportError: ``stoptimes-serve`` isn't installed.
            AttributeError: ``stoptimes.serve.pipeline.load_pipeline`` doesn't exist yet.
        """
        if not upcoming_stops:
            return []

        from stoptimes.core.contracts.pipeline import PipelineContext

        pipeline = self._ensure_pipeline()
        ctx = PipelineContext(
            source=trip.source,
            static_feed=self._static_feed,
            model=None,
            now=datetime.fromtimestamp(vehicle_state.epoch, tz=UTC),
            state={
                "trip": trip,
                "upcoming_stops": upcoming_stops,
                "vehicle_state": vehicle_state,
                "trip_updates": trip_updates,
            },
        )
        result = pipeline.run(ctx)
        if not result:
            result = ctx.state.get("predictions")

        return self._to_stop_predictions(result, upcoming_stops, vehicle_state)

    @staticmethod
    def _to_stop_predictions(
        result: Any, upcoming_stops: list[Stop], vehicle_state: VehicleState
    ) -> list[StopPrediction]:
        """Convert whatever `StopTimePrediction`-shaped rows the pipeline produced
        into `StopPrediction`s, matching each row to its stop by `stop_id`.

        Rows that don't match an upcoming stop, or carry no arrival time, are
        skipped rather than raising -- a pipeline's own postprocess stage is
        free to drop/add stops val doesn't know about.
        """
        if not result:
            return []

        stops_by_id = {s.stop_id: s for s in upcoming_stops}
        n_upcoming = len(upcoming_stops)
        made_at = vehicle_state.epoch

        predictions: list[StopPrediction] = []
        for row in result:
            stop_id = row.get("stop_id") if isinstance(row, dict) else getattr(row, "stop_id", None)
            stop = stops_by_id.get(stop_id)
            if stop is None:
                continue

            arrival = (
                row.get("predicted_arrival_time")
                if isinstance(row, dict)
                else getattr(row, "predicted_arrival_time", None)
            )
            if arrival is None:
                continue
            arrival_epoch = arrival.timestamp() if isinstance(arrival, datetime) else float(arrival)

            predictions.append(
                StopPrediction(
                    stop_id=stop.stop_id,
                    stop_sequence=stop.stop_sequence,
                    predicted_arrival_epoch=arrival_epoch,
                    made_at_epoch=made_at,
                    stops_remaining=n_upcoming - upcoming_stops.index(stop),
                )
            )
        return predictions
