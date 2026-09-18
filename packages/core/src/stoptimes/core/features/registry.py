"""Feature registry: named, documented features with declared inputs, usable in training and serving.

Phase 2, track T3. Lives in core because serving (T4) will eventually need to
compute the same features from a live vehicle position that training computed
from an archived one (docs/PLAN.md).

Every entry here **wraps existing feature-computation code** — it does not
invent new features. One entry = one output column, matching how
`Experiment.features` lists them (see `examples/experiments/xgb-baseline.yaml`:
`distance_to_stop`, `progress_ratio`, `hour_sin`, ...):

- The `temporal` group (`hour`, `day_of_week`, `is_weekend`, `is_holiday`,
  `time_of_day_bin`, `is_peak_hour`, plus the cyclical encodings `hour_sin`/
  `hour_cos`/`dow_sin`/`dow_cos`) is computed live from a timestamp via
  `stoptimes.core.features.temporal.extract_temporal_features` (falls back to
  reusing the columns if a caller already has them, e.g. a dataset built by
  `stoptimes.train.datasets.dataset_builder`, which computes the same fields
  at build time).
- Every other feature describes one column already materialized by
  `stoptimes.train.datasets.dataset_builder.build_vp_training_dataset` (VP
  geometry), `stoptimes.train.datasets.segment_dataset_builder` (segment
  identity), or `stoptimes.train.datasets.rolling_conditions` (causal
  rolling traversal-time/congestion stats). ``core`` must stay upstream of
  ``train`` (docs/PLAN.md rule 1: ``core ← models ← serve ← train``), so
  this module cannot *call* those train-owned builder functions — instead
  each entry validates that its column is present and documents where it
  comes from. Whatever actually builds the frame (a ``train.datasets``
  builder today, a live feature computation in ``serve`` later) is
  responsible for having already computed it; ``build()`` raises a clear
  error naming the missing column and its source when it isn't there yet.

Public API: ``list_features()``, ``get(name)``, ``build(names, frame, ctx)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from stoptimes.core.features.temporal import extract_temporal_features

FeatureFn = Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]


class FeatureNotFoundError(LookupError):
    """Raised by `get`/`build` when a requested feature name isn't registered."""


class MissingFeatureInputsError(ValueError):
    """Raised by `build` when a feature's required columns aren't in the frame."""


@dataclass(frozen=True)
class Feature:
    """One named, documented feature — one output column.

    Attributes:
        name: registry key and output column name, used in
            `Experiment.features` and CLI/TUI listings.
        description: user-facing text — the Phase 3 TUI lists this verbatim.
        requires: input columns that must already be present in the frame
            (checked before `fn` runs; empty for features, like the
            temporal ones, that can derive their own inputs).
        tags: one or more of "temporal", "spatial", "segment", "rolling".
        fn: `(frame, ctx) -> frame` — returns a frame with `name`'s column
            present (a copy; `build()` always works off a fresh copy).
        source: where the computation actually lives, for the description
            and for error messages when the column is missing upstream.
    """

    name: str
    description: str
    requires: tuple[str, ...]
    tags: tuple[str, ...]
    fn: FeatureFn
    source: str = ""


_REGISTRY: dict[str, Feature] = {}


def _register(feature: Feature) -> Feature:
    _REGISTRY[feature.name] = feature
    return feature


def list_features(tag: str | None = None) -> list[Feature]:
    """All registered features, sorted by name. `tag` filters (e.g. "spatial")."""
    feats = sorted(_REGISTRY.values(), key=lambda f: f.name)
    if tag is not None:
        feats = [f for f in feats if tag in f.tags]
    return feats


def get(name: str) -> Feature:
    """Look up one feature by name. Raises `FeatureNotFoundError` if unknown."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise FeatureNotFoundError(f"unknown feature {name!r}; available: {available}") from exc


def build(names: Sequence[str], frame: pd.DataFrame, ctx: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Resolve `names` against the registry and return a frame with all of
    their columns present.

    `ctx` carries feature-specific settings (e.g. the temporal features' `tz`
    / `holiday_region` / `timestamp_column`). Raises `FeatureNotFoundError`
    for an unknown name and `MissingFeatureInputsError` naming exactly which
    input column a feature needed but didn't find.
    """
    ctx = ctx or {}
    out = frame.copy()
    for name in names:
        feature = get(name)
        missing = [c for c in feature.requires if c not in out.columns]
        if missing:
            raise MissingFeatureInputsError(
                f"feature {name!r} needs columns {missing} not present in the input frame "
                f"(source: {feature.source or 'unknown'}; have: {sorted(out.columns)})"
            )
        out = feature.fn(out, ctx)
        if name not in out.columns:
            raise MissingFeatureInputsError(
                f"feature {name!r} did not produce its column (source: {feature.source or 'unknown'})"
            )
    return out


# --------------------------------------------------------------------------
# temporal — computed live, or reused if the frame already has it.
# --------------------------------------------------------------------------

_TEMPORAL_BASE_COLUMNS: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "time_of_day_bin",
    "is_peak_hour",
)
_TEMPORAL_CYCLICAL_COLUMNS: tuple[str, ...] = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")
_TEMPORAL_COLUMNS: tuple[str, ...] = _TEMPORAL_BASE_COLUMNS + _TEMPORAL_CYCLICAL_COLUMNS


def _temporal_fn(frame: pd.DataFrame, ctx: Mapping[str, Any]) -> pd.DataFrame:
    out = frame
    if not all(c in out.columns for c in _TEMPORAL_BASE_COLUMNS):
        ts_col = ctx.get("timestamp_column", "vp_ts")
        if ts_col not in out.columns:
            raise MissingFeatureInputsError(
                f"temporal features need either precomputed columns {list(_TEMPORAL_BASE_COLUMNS)} "
                f"or a {ts_col!r} timestamp column (set ctx['timestamp_column'] to override) "
                f"to derive them (source: stoptimes.core.features.temporal.extract_temporal_features)"
            )
        tz = ctx.get("tz", "America/New_York")
        region = ctx.get("holiday_region", "US_MA")
        rows = [extract_temporal_features(ts, tz=tz, region=region) for ts in out[ts_col]]
        computed = pd.DataFrame(rows, index=out.index)
        out = out.copy()
        for col in _TEMPORAL_BASE_COLUMNS:
            out[col] = computed[col]

    if not all(c in out.columns for c in _TEMPORAL_CYCLICAL_COLUMNS):
        out = out.copy()
        if "hour_sin" not in out.columns:
            out["hour_sin"] = out["hour"].apply(lambda h: math.sin(2 * math.pi * h / 24))
        if "hour_cos" not in out.columns:
            out["hour_cos"] = out["hour"].apply(lambda h: math.cos(2 * math.pi * h / 24))
        if "dow_sin" not in out.columns:
            out["dow_sin"] = out["day_of_week"].apply(lambda d: math.sin(2 * math.pi * d / 7))
        if "dow_cos" not in out.columns:
            out["dow_cos"] = out["day_of_week"].apply(lambda d: math.cos(2 * math.pi * d / 7))

    return out


_TEMPORAL_DESCRIPTIONS: dict[str, str] = {
    "hour": "Hour of day (0-23) in the agency's local timezone.",
    "day_of_week": "Day of week (0=Monday..6=Sunday) in the agency's local timezone.",
    "is_weekend": "Whether the local day is Saturday or Sunday.",
    "is_holiday": "Whether the local date is a holiday in the agency's holiday region.",
    "time_of_day_bin": "Time-of-day bucket: morning/midday/afternoon/evening.",
    "is_peak_hour": "Weekday morning (7-9) or evening (16-19) commuter peak indicator.",
    "hour_sin": "Cyclical (sine) encoding of hour of day, so 23:59 is numerically close to 00:00.",
    "hour_cos": "Cyclical (cosine) encoding of hour of day.",
    "dow_sin": "Cyclical (sine) encoding of day of week.",
    "dow_cos": "Cyclical (cosine) encoding of day of week.",
}

for _name in _TEMPORAL_COLUMNS:
    _register(
        Feature(
            name=_name,
            description=_TEMPORAL_DESCRIPTIONS[_name],
            requires=(),
            tags=("temporal",),
            fn=_temporal_fn,
            source="stoptimes.core.features.temporal.extract_temporal_features",
        )
    )


# --------------------------------------------------------------------------
# spatial — vehicle-position geometry already computed by the VP dataset
# builder (train/datasets/dataset_builder.py). Wrapped as validated
# passthroughs: core can describe and check these columns without importing
# stoptimes.train (which depends on core, not the other way around).
# --------------------------------------------------------------------------


def _passthrough(frame: pd.DataFrame, ctx: Mapping[str, Any]) -> pd.DataFrame:
    return frame


_DATASET_BUILDER_SOURCE = "stoptimes.train.datasets.dataset_builder.build_vp_training_dataset"

_SPATIAL_DESCRIPTIONS: dict[str, str] = {
    "vp_lat": "Vehicle-position fix latitude.",
    "vp_lon": "Vehicle-position fix longitude.",
    "vp_bearing": "Vehicle-position fix heading (degrees).",
    "distance_to_stop": "Straight-line distance from the vehicle to the target stop, in meters.",
    "distance_to_next_stop": "Straight-line distance from the vehicle to the next stop ahead, in meters.",
    "shape_distance_to_stop": "Distance to the target stop measured along the route shape, in meters.",
    "shape_progress": "How far along the route shape the vehicle has traveled (0-1).",
    "cross_track_error": "Perpendicular distance of the vehicle's fix from the route shape, in meters.",
    "progress_ratio": "Fraction of the current stop-to-stop segment already traveled (0-1).",
    "stops_ahead": "Number of scheduled stops remaining before the target stop.",
    "current_speed_kmh": "Vehicle's current speed, in km/h.",
    "bearing_to_stop": "Compass bearing from the vehicle to the target stop, in degrees.",
    "bearing_diff": "Angular difference between the vehicle's heading and its bearing to the target stop.",
    "is_at_stop": "Whether the vehicle is currently stopped/dwelling at a stop.",
}

for _name in _SPATIAL_DESCRIPTIONS:
    _register(
        Feature(
            name=_name,
            description=_SPATIAL_DESCRIPTIONS[_name],
            requires=(_name,),
            tags=("spatial",),
            fn=_passthrough,
            source=_DATASET_BUILDER_SOURCE,
        )
    )

_SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "from_stop_id": "The stop the vehicle is currently traveling away from (the start of its current segment).",
    "segment_index": "Index of the current stop-to-stop segment along the trip.",
    "route_id": "The route this row's trip belongs to -- a grouping key for route-level models and features.",
    "stop_id": "The target stop for this row's prediction -- a grouping key for stop-level models and features.",
    "stop_sequence": "The target stop's position (1-based) along its trip -- a grouping key for historical-mean/EWMA-style models.",
    "trip_id": "The scheduled trip this row belongs to.",
}

for _name in _SEGMENT_DESCRIPTIONS:
    _register(
        Feature(
            name=_name,
            description=_SEGMENT_DESCRIPTIONS[_name],
            requires=(_name,),
            tags=("segment",),
            fn=_passthrough,
            source="stoptimes.train.datasets.segment_dataset_builder.build_bucr_segment_dataset",
        )
    )

_ROLLING_DESCRIPTIONS: dict[str, str] = {
    "recent_seg_traversal_s": "EWMA of how long the last few vehicles took to traverse this segment, in seconds.",
    "recent_seg_ratio": "Ratio of the recent per-segment traversal EWMA to its running baseline.",
    "seg_n_obs_before": "Number of prior observations feeding this segment's rolling baseline.",
    "headway_since_last_traversal_s": "Seconds since this segment was last traversed by any vehicle.",
    "leader_last_duration_s": "Traversal duration of the most recent vehicle through this segment, in seconds.",
    "route_congestion_index": "Causal EWMA of recent per-segment traversal-ratio-to-baseline across the whole route.",
    "vehicle_recent_delay_ratio": "This vehicle's own recent delay ratio, from its last few segments.",
}

for _name in _ROLLING_DESCRIPTIONS:
    _register(
        Feature(
            name=_name,
            description=(
                _ROLLING_DESCRIPTIONS[_name]
                + " NaN where there isn't enough history yet — never silently filled."
            ),
            requires=(_name,),
            tags=("rolling",),
            fn=_passthrough,
            source="stoptimes.train.datasets.rolling_conditions.add_rolling_conditions_features",
        )
    )
