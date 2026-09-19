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
- `default_match` matches both `trip_update` and `vehicle_position`
  Observations against the source's static feed, using the same spatial
  primitives `core.arrivals`/`core.segments` are built on
  (`core.spatial.polyline`'s `build_polyline`/`assign_stops_monotonic`/
  `project_point_to_polyline`/`upcoming_stops`) to attach real
  distance-along-shape features (`distance_to_stop`, `stops_ahead`) to
  `MatchedObservation.extra`. Static geometry comes from `ctx.static_feed`
  when the caller supplied one (T1's reader); otherwise this stage lazily
  loads it itself from `ctx.source.gtfs_static_url` via `core.readers.read_static`
  -- CLI wiring for `ctx.static_feed` doesn't exist yet (docs/notes/p3-u6.md),
  so this local fallback is what makes real matching reachable today. When no
  geometry can be resolved at all, behavior falls back to the historical
  default: `trip_update` observations still pass through (stop already
  identified, just without spatial extras) and `vehicle_position`
  observations still record a per-item error, since there's nothing to
  project them onto.
- `default_features` computes temporal features through T3's
  `core.features.registry.build` (landed in Phase 2) and leaves spatial
  features to whatever `match` attached to `MatchedObservation.extra`.
  `producible_features()`/`unproducible_registered_features()` below name
  exactly which registry entries this pipeline can and cannot compute --
  `serve.model_loading.check_feature_producibility` uses the latter to
  refuse an artifact before serving starts, instead of returning `None` for
  every row of a feature it can never produce (docs/PLAN.md §7.4).

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
from typing import Any, Callable, Sequence

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
    "producible_features",
    "unproducible_registered_features",
]

# ---------------------------------------------------------------------------
# What `default_features`/`default_match` can actually compute -- the
# contract `serve.model_loading.check_feature_producibility` refuses against.
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {"trip_id", "route_id", "direction_id", "start_date", "vehicle_id", "stop_id", "stop_sequence"}
)
"""Always present on a `MatchedObservation`-derived row, not from the registry."""

MATCH_SPATIAL_COLUMNS: frozenset[str] = frozenset({"distance_to_stop", "stops_ahead", "cross_track_error"})
"""The only `core.features.registry` "spatial"-tagged columns `default_match`
(`_match_trip_update`/`_match_vehicle_position`) actually attaches to
`MatchedObservation.extra` today. Every other registered spatial column
(`vp_lat`, `shape_progress`, `current_speed_kmh`, ...) describes geometry
only `train.datasets.dataset_builder` computes from an archived, densely
sampled trip -- this pipeline has no live source for it."""


def producible_features() -> frozenset[str]:
    """Feature names `default_features` can compute for a live serving row
    today: every `core.features.registry` "temporal" feature (derived live
    from a timestamp), the handful of "spatial" columns `default_match`
    attaches (`MATCH_SPATIAL_COLUMNS`), and the trip/stop identity columns
    always present on a `MatchedObservation` (`IDENTITY_COLUMNS`).

    Every other feature the registry knows about -- the rest of "spatial",
    all of "segment", all of "rolling" -- describes a column only a
    `stoptimes.train.datasets` builder materializes; nothing in this module
    computes it. See `unproducible_registered_features`.
    """
    from stoptimes.core.features.registry import list_features

    temporal_names = {f.name for f in list_features(tag="temporal")}
    return frozenset(temporal_names) | MATCH_SPATIAL_COLUMNS | IDENTITY_COLUMNS


def unproducible_registered_features(required: Sequence[str]) -> list[str]:
    """Which of `required` are `core.features.registry` entries this
    pipeline has no way to compute.

    A name the registry has never heard of (a BYO/legacy feature column,
    e.g. a pre-registry artifact's raw column name) is left alone -- there's
    no positive evidence it's unproducible, the same principle
    `model_loading.check_version_drift` uses for an artifact with no
    recorded library versions: absence of information is not evidence of a
    problem. Only a name the registry *does* know, and tags as something
    `default_features`/`default_match` don't compute, is reported.
    """
    from stoptimes.core.features.registry import list_features

    registered = {f.name for f in list_features()}
    producible = producible_features()
    return [name for name in required if name in registered and name not in producible]


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


def _load_static_geometry(ctx: PipelineContext) -> Any | None:
    """Resolve something satisfying `core.arrivals.StaticGeometry` (a
    `trip_geometry(trip_id) -> (shape_points, ordered_stops)` duck type) for
    this run, or `None` if none is available.

    Prefers `ctx.static_feed` (T1's reader, once a caller wires it up).
    Otherwise lazily loads it from `ctx.source.gtfs_static_url` via
    `core.readers.read_static` -- the CLI doesn't populate `ctx.static_feed`
    today (docs/notes/p3-u6.md), so without this fallback real map-matching
    would never be reachable from `stoptimes serve`. The result (including a
    failure) is cached on `ctx.state` so a bad/missing URL is only tried once
    per run, not once per observation.
    """
    if hasattr(ctx.static_feed, "trip_geometry"):
        return ctx.static_feed

    cache_key = "_p3_u6_static_geometry"
    if cache_key in ctx.state:
        return ctx.state[cache_key]

    geometry = None
    url = getattr(ctx.source, "gtfs_static_url", None)
    source_id = getattr(ctx.source, "id", "unknown")
    if url:
        try:
            from stoptimes.core.readers import read_static

            loader, _feed = read_static(url, source_id)
            geometry = loader
        except Exception as exc:  # noqa: BLE001 - a bad/missing static feed must not crash serving
            ctx.logger.debug("default_match: could not load static feed for %s: %s", source_id, exc)
            geometry = None
    ctx.state[cache_key] = geometry
    return geometry


def _trip_geometry_cached(ctx: PipelineContext, geometry: Any, trip_id: str) -> Any | None:
    """`geometry.trip_geometry(trip_id)` -> monotonic-progress stops + polyline,
    cached per trip on `ctx.state` (a run may match many observations for the
    same trip). Returns `None` when the trip has no usable geometry.
    """
    from stoptimes.core.spatial.polyline import assign_stops_monotonic, build_polyline

    cache = ctx.state.setdefault("_p3_u6_trip_geometry", {})
    if trip_id in cache:
        return cache[trip_id]

    resolved = None
    try:
        shape_points, stops = geometry.trip_geometry(trip_id)
    except (ValueError, KeyError):
        shape_points, stops = None, None

    if stops:
        polyline = build_polyline(shape_points) if shape_points else None
        ordered_stops = assign_stops_monotonic(stops, polyline) if polyline and len(polyline) >= 2 else stops
        resolved = (polyline, ordered_stops)
    cache[trip_id] = resolved
    return resolved


def _match_trip_update(obs: Observation, ctx: PipelineContext, geometry: Any | None) -> MatchedObservation:
    """`trip_update` match: the stop is already identified by the
    observation; when static geometry is available, attach the
    distance-along-shape approaching this stop (mirrors
    `core.segments`'s `distance_to_stop` -- the from-stop-to-this-stop
    segment length) and `stops_ahead` (stops remaining after this one,
    the same 0-from-the-end training convention `core.segments` uses).
    """
    extra: dict[str, Any] = {}
    resolved = _trip_geometry_cached(ctx, geometry, obs.trip_id) if geometry is not None else None
    if resolved is not None:
        _polyline, ordered_stops = resolved
        with_progress = [s for s in ordered_stops if s.progress_m is not None]
        by_sequence = {s.stop_sequence: s for s in with_progress}
        this_stop = by_sequence.get(obs.stop_sequence)
        if this_stop is not None:
            ordered = sorted(with_progress, key=lambda s: s.stop_sequence)
            idx = ordered.index(this_stop)
            prev_stop = ordered[idx - 1] if idx > 0 else None
            if prev_stop is not None:
                extra["distance_to_stop"] = float(this_stop.progress_m - prev_stop.progress_m)
            extra["stops_ahead"] = float(len(ordered) - 1 - idx)

    return MatchedObservation(
        trip_id=obs.trip_id,
        route_id=obs.route_id,
        direction_id=obs.direction_id,
        start_date=obs.start_date,
        vehicle_id=obs.vehicle_id,
        stop_id=obs.stop_id,  # type: ignore[arg-type]
        stop_sequence=obs.stop_sequence,  # type: ignore[arg-type]
        observation=obs,
        extra=extra,
    )


def _match_vehicle_position(
    obs: Observation, ctx: PipelineContext, geometry: Any | None
) -> list[MatchedObservation]:
    """`vehicle_position` match: project the vehicle onto its trip's shape
    (`core.spatial.polyline.project_point_to_polyline`) and emit one
    `MatchedObservation` per not-yet-reached stop (`upcoming_stops`), each
    carrying its real along-shape `distance_to_stop` and 0-indexed
    `stops_ahead` -- the live-serving analogue of `core.arrivals`'s
    map-matched arrival detection, projecting forward instead of back.

    Records a per-item error (as the old placeholder did) when there's no
    geometry, no shape for this trip, or the vehicle has already passed every
    scheduled stop.
    """
    if geometry is None:
        record_error(
            ctx,
            stage="match",
            trip_id=obs.trip_id,
            error=RuntimeError(
                "no static feed available to map-match this vehicle_position "
                "(ctx.static_feed is unset and ctx.source.gtfs_static_url could not be loaded)"
            ),
        )
        return []

    resolved = _trip_geometry_cached(ctx, geometry, obs.trip_id)
    if resolved is None or obs.lat is None or obs.lon is None:
        record_error(
            ctx,
            stage="match",
            trip_id=obs.trip_id,
            error=RuntimeError("no shape/stop geometry available for this trip"),
        )
        return []

    polyline, ordered_stops = resolved
    if polyline is None or len(polyline) < 2:
        record_error(
            ctx,
            stage="match",
            trip_id=obs.trip_id,
            error=RuntimeError("trip has no usable shape polyline to project onto"),
        )
        return []

    from stoptimes.core.spatial.polyline import project_point_to_polyline
    from stoptimes.core.spatial.polyline import upcoming_stops as _upcoming_stops

    projection = project_point_to_polyline(obs.lat, obs.lon, polyline)
    ahead = _upcoming_stops(projection.progress_m, ordered_stops)
    if not ahead:
        record_error(
            ctx,
            stage="match",
            trip_id=obs.trip_id,
            error=RuntimeError("vehicle has already passed every scheduled stop"),
        )
        return []

    matched: list[MatchedObservation] = []
    for idx, stop in enumerate(ahead):
        matched.append(
            MatchedObservation(
                trip_id=obs.trip_id,
                route_id=obs.route_id,
                direction_id=obs.direction_id,
                start_date=obs.start_date,
                vehicle_id=obs.vehicle_id,
                stop_id=stop.stop_id,
                stop_sequence=stop.stop_sequence,
                observation=obs,
                extra={
                    "distance_to_stop": float(stop.progress_m - projection.progress_m),
                    "cross_track_error": float(projection.cross_track_m),
                    "stops_ahead": float(idx),
                },
            )
        )
    return matched


def default_match(observations: list[Observation], ctx: PipelineContext) -> list[MatchedObservation]:
    """Match every observation against the source's static feed.

    `trip_update` observations already know their stop; a resolved static
    feed just adds real spatial features to it. `vehicle_position`
    observations are actually map-matched: projected onto their trip's shape
    to discover the upcoming stops and their along-shape distances. Either
    kind degrades to the historical placeholder behavior (pass-through /
    per-item error) when no static geometry can be resolved at all -- see
    `_load_static_geometry`.
    """
    geometry = _load_static_geometry(ctx)
    matched: list[MatchedObservation] = []
    for obs in observations:
        if obs.kind == ObservationKind.TRIP_UPDATE:
            matched.append(_match_trip_update(obs, ctx, geometry))
        else:
            matched.extend(_match_vehicle_position(obs, ctx, geometry))
    return matched


def default_features(matched: list[MatchedObservation], ctx: PipelineContext) -> pd.DataFrame:
    """Compute a features DataFrame, one row per `MatchedObservation`.

    Temporal features (time-of-day/holiday) go through T3's
    `core.features.registry.build`, not a direct call to
    `core.features.temporal` -- so this and any future caller that lists
    features by name (training, `val`'s `StoptimesArtifactPredictor`) agree
    on what a "temporal" feature is. Spatial features are read from whatever
    a `match` stage already attached to `.extra` -- `default_match` is the
    only thing in this pipeline that can compute them live; the registry
    only *describes* the rest (see `producible_features`). Missing model
    features are left absent (not filled with NaN) so `default_predict`'s
    `row.get(name)` behavior is unchanged; refusing an artifact that
    requires a column this stage can never produce is
    `model_loading.check_feature_producibility`'s job, at load time, not
    this stage's.
    """
    from stoptimes.core.features.registry import build as build_registry_features
    from stoptimes.core.features.registry import list_features as list_registry_features

    holiday_region = getattr(ctx.source, "holiday_region", None)
    timezone = getattr(ctx.source, "timezone", "UTC")

    rows: list[dict[str, Any]] = []
    for item in matched:
        ts = item.observation.ts or item.observation.arrival_time or ctx.now
        row: dict[str, Any] = {
            "trip_id": item.trip_id,
            "route_id": item.route_id,
            "direction_id": item.direction_id,
            "start_date": item.start_date,
            "vehicle_id": item.vehicle_id,
            "stop_id": item.stop_id,
            "stop_sequence": item.stop_sequence,
            "_ts": pd.Timestamp(ts).to_pydatetime(),
            **item.extra,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    temporal_names = [f.name for f in list_registry_features(tag="temporal")]
    frame = build_registry_features(
        temporal_names,
        frame,
        {"timestamp_column": "_ts", "tz": timezone, "holiday_region": holiday_region},
    )
    return frame.drop(columns=["_ts"])


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
