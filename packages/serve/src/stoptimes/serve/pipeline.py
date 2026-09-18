"""Customizable serving pipeline (stage hooks users override).

Implements `stoptimes.core.contracts.pipeline`: six named, independently
overridable stages -- `ingest -> match -> features -> predict -> postprocess
-> publish` -- threaded through one `PipelineContext`. This is the
"customizable inference function" the product promises: a user overrides one
stage (say, `predict`, to plug in a custom model call) via
`@pipeline.stage("predict")` without reimplementing the other five.

Seams (T1/T2/T3 not yet landed at the time this was written):

- `default_ingest` polls a source's JSON GTFS-RT feeds directly with httpx
  and `stoptimes.core.realtime.gtfsrt_json` (protobuf feeds are read via
  `google.protobuf.json_format.MessageToDict` into the same shape) instead of
  going through T1's `core.readers` (still a placeholder). Swap it out once
  that lands -- see docs/notes/p2-t4.md.
- `default_match` is a pass-through for `trip_update` Observations (they
  already carry `stop_id`/`stop_sequence`) and records a per-item error for
  `vehicle_position` Observations, since real map-matching is T2's
  `core.arrivals`/`core.segments` (also placeholders). A caller with a real
  matcher overrides just this stage.
- `default_features` is one small adapter this track owns, bridging to T3's
  not-yet-existing `core.features.registry`: it calls the temporal feature
  helper that already exists (`core.features.temporal`) and leaves spatial
  features to whatever `match` attached to `MatchedObservation.extra`.

Errors are contained per matched row, not per pipeline run: `default_predict`
and `default_postprocess` catch exceptions per item, append a structured
record to `ctx.state["errors"]`, and continue with the remaining rows.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import (
    STAGE_ORDER,
    Pipeline,
    PipelineContext,
    PipelineStageName,
    StageHook,
)
from stoptimes.core.contracts.prediction import StopTimePrediction, TripUpdatePrediction

__all__ = [
    "MatchedObservation",
    "StoptimesPipeline",
    "PipelineContext",
    "record_error",
    "default_ingest",
    "default_match",
    "default_features",
    "default_predict",
    "default_postprocess",
    "default_publish",
    "build_default_pipeline",
    "load_pipeline",
]


@dataclass
class MatchedObservation:
    """Stand-in for T2's real `MatchedObservation` shape (docs/CONTRACTS.md §5).

    Carries everything `default_features`/`default_predict` need: trip
    identity (for grouping into `TripUpdatePrediction`s) plus the source
    `Observation` and whatever extra fields a `match` stage computed
    (e.g. `distance_to_stop_m`, `stop_lat`/`stop_lon`).
    """

    trip_id: str
    route_id: str | None
    direction_id: int | None
    start_date: str | None
    vehicle_id: str | None
    stop_id: str
    stop_sequence: int
    observation: Observation
    extra: dict[str, Any] = field(default_factory=dict)


def record_error(ctx: PipelineContext, *, stage: str, trip_id: str | None, error: Exception) -> None:
    """Append a structured error to `ctx.state["errors"]` without raising.

    Convention every stage in this module (and any override) should follow
    so one bad trip never aborts the whole run.
    """
    errors: list[dict[str, Any]] = ctx.state.setdefault("errors", [])
    errors.append({"stage": stage, "trip_id": trip_id, "error": f"{type(error).__name__}: {error}"})
    ctx.logger.warning("pipeline stage %s failed for trip %s: %s", stage, trip_id, error)


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------


class StoptimesPipeline:
    """Concrete `Pipeline`: decorator-based stage registration + sequential run.

    Distinct from `contracts.pipeline.SequentialPipeline` (the Phase 1
    reference implementation proving the contract composes) by actually
    shipping default stages, `ctx.state["errors"]` accounting, and a
    `describe()` helper the CLI uses to report which stages are default vs.
    overridden.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, StageHook] = {}
        self._is_default: dict[str, bool] = {}

    def stage(self, name: PipelineStageName, *, _default: bool = False) -> Callable[[StageHook], StageHook]:
        def register(hook: StageHook) -> StageHook:
            self._hooks[name] = hook
            self._is_default[name] = _default
            return hook

        return register

    def is_default(self, name: str) -> bool:
        return self._is_default.get(name, False)

    def run(self, ctx: PipelineContext) -> Any:
        payload: Any = None
        for name in STAGE_ORDER:
            if name not in self._hooks:
                raise KeyError(f"no hook registered for stage {name!r}")
            payload = self._hooks[name](payload, ctx)
        return payload


# ---------------------------------------------------------------------------
# Default stages
# ---------------------------------------------------------------------------


def default_ingest(_payload: Any, ctx: PipelineContext) -> list[Observation]:
    """Produce `Observation`s for this run.

    Test/offline entry point: `ctx.config["observations"]` is used verbatim
    when present. Otherwise polls the source's JSON trip_updates feed (the
    only kind default `predict` below can use, since it needs a stop
    already identified) with httpx. Protobuf feeds are decoded to the same
    dict shape via `google.protobuf.json_format.MessageToDict` first.
    """
    injected = ctx.config.get("observations")
    if injected is not None:
        return list(injected)

    from stoptimes.core.contracts.source import FeedFormat, FeedKind

    source = ctx.source
    tu_feed = next((f for f in getattr(source, "realtime_feeds", []) if f.kind == FeedKind.TRIP_UPDATES), None)
    if tu_feed is None:
        return []

    import httpx

    from stoptimes.core.realtime.gtfsrt_json import parse_tu_feed

    response = httpx.get(tu_feed.url, headers=dict(tu_feed.headers), timeout=10.0)
    response.raise_for_status()
    if tu_feed.format == FeedFormat.PROTOBUF:
        from google.protobuf import json_format
        from google.transit import gtfs_realtime_pb2

        message = gtfs_realtime_pb2.FeedMessage()
        message.ParseFromString(response.content)
        feed = json_format.MessageToDict(message, preserving_proto_field_name=True)
    else:
        feed = response.json()

    _header_ts, records = parse_tu_feed(feed)
    observations: list[Observation] = []
    for record in records:
        for stu in record.stop_time_updates:
            if stu.stop_id is None or stu.stop_sequence is None:
                continue
            observations.append(
                Observation(
                    kind=ObservationKind.TRIP_UPDATE,
                    source_id=source.id,
                    trip_id=record.trip_id,
                    route_id=record.route_id,
                    direction_id=record.direction_id,
                    start_date=record.start_date,
                    stop_id=stu.stop_id,
                    stop_sequence=stu.stop_sequence,
                    arrival_time=(
                        pd.Timestamp(stu.arrival_time, unit="s", tz="UTC") if stu.arrival_time is not None else None
                    ),
                    departure_time=(
                        pd.Timestamp(stu.departure_time, unit="s", tz="UTC")
                        if stu.departure_time is not None
                        else None
                    ),
                )
            )
    return observations


def default_match(observations: list[Observation], ctx: PipelineContext) -> list[MatchedObservation]:
    """Pass-through match for `trip_update` observations (already stop-identified).

    `vehicle_position` observations need real map-matching (T2's job); this
    default records a per-item error and drops them rather than crashing the
    run, which is the "collect and report" behavior the contract asks for.
    """
    matched: list[MatchedObservation] = []
    for obs in observations:
        if obs.kind != ObservationKind.TRIP_UPDATE:
            record_error(
                ctx,
                stage="match",
                trip_id=obs.trip_id,
                error=NotImplementedError(
                    "default_match only handles trip_update observations; "
                    "vehicle_position map-matching is T2's core.arrivals/core.segments"
                ),
            )
            continue
        matched.append(
            MatchedObservation(
                trip_id=obs.trip_id,
                route_id=obs.route_id,
                direction_id=obs.direction_id,
                start_date=obs.start_date,
                vehicle_id=obs.vehicle_id,
                stop_id=obs.stop_id,  # type: ignore[arg-type]
                stop_sequence=obs.stop_sequence,  # type: ignore[arg-type]
                observation=obs,
            )
        )
    return matched


def default_features(matched: list[MatchedObservation], ctx: PipelineContext) -> pd.DataFrame:
    """Compute a features DataFrame, one row per `MatchedObservation`.

    Small adapter this track owns in place of T3's not-yet-existing
    `core.features.registry`: uses `core.features.temporal` (already
    landed) for time-of-day/holiday features, and reads any spatial fields
    a `match` stage already attached to `.extra`. Missing model features are
    filled with NaN so `model.predict` sees every declared column.
    """
    from stoptimes.core.features.temporal import extract_temporal_features

    holiday_region = getattr(ctx.source, "holiday_region", None)
    timezone = getattr(ctx.source, "timezone", "UTC")

    rows: list[dict[str, Any]] = []
    for item in matched:
        ts = item.observation.ts or item.observation.arrival_time or ctx.now
        temporal = extract_temporal_features(pd.Timestamp(ts).to_pydatetime(), tz=timezone, region=holiday_region)
        row: dict[str, Any] = {
            "trip_id": item.trip_id,
            "route_id": item.route_id,
            "direction_id": item.direction_id,
            "start_date": item.start_date,
            "vehicle_id": item.vehicle_id,
            "stop_id": item.stop_id,
            "stop_sequence": item.stop_sequence,
            **temporal,
            **item.extra,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def default_predict(features: pd.DataFrame, ctx: PipelineContext) -> list[StopTimePrediction]:
    """Call `ctx.model.predict` one row at a time so one bad row is dropped,
    not fatal, and produce `StopTimePrediction`s.

    The model's numeric output is treated as `horizon_s` (seconds from
    `ctx.now` to the predicted arrival) -- there is no scheduled arrival to
    diff against without a static-feed join (T1's `core.readers`/T3's
    dataset code), so `delay_s` is left `None` by this default and is a
    natural thing for a `postprocess` override to fill in once that join
    exists.
    """
    predictions: list[StopTimePrediction] = []
    if features.empty:
        return predictions

    model = ctx.model
    required = list(getattr(model, "required_features", features.columns))
    model_id = getattr(model, "family", getattr(model, "artifact_id", "unknown"))
    artifact_id = getattr(model, "artifact_id", None)

    for _, row in features.iterrows():
        trip_id = row.get("trip_id")
        try:
            X = pd.DataFrame([{name: row.get(name) for name in required}])
            horizon_s = float(model.predict(X)[0])
            predictions.append(
                StopTimePrediction(
                    trip_id=trip_id,
                    route_id=row.get("route_id"),
                    direction_id=(int(row["direction_id"]) if pd.notna(row.get("direction_id")) else None),
                    start_date=row.get("start_date") or ctx.now.strftime("%Y%m%d"),
                    stop_id=row.get("stop_id"),
                    stop_sequence=int(row.get("stop_sequence")),
                    vehicle_id=row.get("vehicle_id"),
                    predicted_arrival_time=ctx.now + timedelta(seconds=horizon_s),
                    horizon_s=horizon_s,
                    model_id=str(model_id),
                    artifact_id=artifact_id,
                    generated_at=ctx.now,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad trip must not kill the run
            record_error(ctx, stage="predict", trip_id=trip_id, error=exc)
    return predictions


def default_postprocess(predictions: list[StopTimePrediction], ctx: PipelineContext) -> list[StopTimePrediction]:
    """Identity by default, aside from dropping rows that fail re-validation."""
    clean: list[StopTimePrediction] = []
    for pred in predictions:
        try:
            if pred.predicted_arrival_time is not None and pred.predicted_arrival_time < ctx.now:
                # A prediction can't be for the past; clamp to "now".
                pred = pred.model_copy(update={"predicted_arrival_time": ctx.now})
            clean.append(pred)
        except Exception as exc:  # noqa: BLE001
            record_error(ctx, stage="postprocess", trip_id=pred.trip_id, error=exc)
    return clean


def group_by_trip(predictions: list[StopTimePrediction], ctx: PipelineContext) -> list[TripUpdatePrediction]:
    """Group flat `StopTimePrediction`s into one `TripUpdatePrediction` per trip."""
    by_trip: dict[tuple[str, str], list[StopTimePrediction]] = {}
    for pred in predictions:
        key = (pred.trip_id, pred.start_date)
        by_trip.setdefault(key, []).append(pred)

    trip_updates: list[TripUpdatePrediction] = []
    for (trip_id, start_date), stops in by_trip.items():
        stops = sorted(stops, key=lambda p: p.stop_sequence)
        try:
            trip_updates.append(
                TripUpdatePrediction(
                    trip_id=trip_id,
                    route_id=stops[0].route_id,
                    direction_id=stops[0].direction_id,
                    start_date=start_date,
                    vehicle_id=stops[0].vehicle_id,
                    stop_time_updates=stops,
                    generated_at=ctx.now,
                )
            )
        except Exception as exc:  # noqa: BLE001
            record_error(ctx, stage="publish", trip_id=trip_id, error=exc)
    return trip_updates


def default_publish(predictions: list[StopTimePrediction], ctx: PipelineContext) -> None:
    """Group predictions into `TripUpdatePrediction`s, stash them on
    `ctx.state["trip_updates"]`, and hand them to `ctx.config["sink"]` if
    one was provided. Returns `None` per the contract's `publish` stage;
    callers (the CLI) read `ctx.state` for what happened.
    """
    trip_updates = group_by_trip(predictions, ctx)
    ctx.state["trip_updates"] = trip_updates

    sink = ctx.config.get("sink")
    if sink is not None and trip_updates:
        from stoptimes.serve.tripupdates import encode_json, encode_protobuf

        fmt = ctx.config.get("format", "json")
        encoder = encode_protobuf if fmt == "pb" else encode_json
        payload = encoder(trip_updates, generated_at=ctx.now)
        sink.publish(payload, meta={"format": fmt, "source_id": ctx.source.id, "count": len(trip_updates)})
        ctx.state["published"] = True
    return None


def build_default_pipeline() -> StoptimesPipeline:
    """A `StoptimesPipeline` with every stage set to this module's default."""
    pipeline = StoptimesPipeline()
    pipeline.stage("ingest", _default=True)(default_ingest)
    pipeline.stage("match", _default=True)(default_match)
    pipeline.stage("features", _default=True)(default_features)
    pipeline.stage("predict", _default=True)(default_predict)
    pipeline.stage("postprocess", _default=True)(default_postprocess)
    pipeline.stage("publish", _default=True)(default_publish)
    return pipeline


# ---------------------------------------------------------------------------
# Loading a user-supplied pipeline
# ---------------------------------------------------------------------------


def load_pipeline(spec: str) -> Pipeline:
    """Resolve `"module.sub:attr"` or `"/path/to/file.py:attr"` to a `Pipeline`.

    `attr` may name a `Pipeline` instance directly, or a zero-argument
    factory returning one (checked by calling it when it isn't already a
    `Pipeline`-shaped object, i.e. has no `.run`).
    """
    if ":" not in spec:
        raise ValueError(f"pipeline spec must be 'module:attr' or 'path/to/file.py:attr', got {spec!r}")
    module_part, _, attr = spec.rpartition(":")

    looks_like_path = module_part.endswith(".py") or "/" in module_part or Path(module_part).exists()
    if looks_like_path:
        path = Path(module_part).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"pipeline file not found: {path}")
        module_name = f"_stoptimes_user_pipeline_{path.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"could not load pipeline module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    obj = getattr(module, attr)
    if not hasattr(obj, "run") and callable(obj):
        obj = obj()
    return obj
