"""
ETA Service - Enhanced Implementation
Estimates stop arrival times from vehicle positions with route-specific model support.
"""

from datetime import datetime, timezone
import math

# Use centralized configuration (eliminates sys.path manipulation)
from stoptimes.core.config import get_config
from stoptimes.core.logging import get_logger
from stoptimes.core.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    PredictionError,
)
from stoptimes.core.validation import (
    validate_vehicle_position,
    validate_stops_list,
)
from stoptimes.core.exceptions import InvalidVehiclePositionError, InvalidStopError

# Initialize config to ensure paths are set up
_config = get_config()
_logger = get_logger("estimator")

from stoptimes.core.features.temporal import extract_temporal_features
from stoptimes.core.spatial.geo import calculate_distance_features_with_shape
from stoptimes.models.common.registry import get_registry

# Phase 3 U6: model *choice* (not model *loading*, which stays the legacy
# registry's job -- see model_loading.py's docstring, owned by U7) is routed
# through the same pure selection function the pipeline (serve.pipeline) and
# a future artifact-registry migration will share, instead of a bespoke
# route/global if-else duplicated here. See docs/notes/p3-u6.md for the
# bridge design and its one deliberate, documented exception.
from stoptimes.core.contracts.artifact import ArtifactMetadata, ModelScope, Provenance, ScopeLevel
from stoptimes.core.contracts.pipeline import PipelineContext
from stoptimes.core.contracts.selection import SelectionPolicy, select_artifact


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def _initial_bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees 0-360."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angle_diff(a, b):
    """Smallest absolute angular difference (0-180), None-safe."""
    if a is None or b is None:
        return None
    d = abs(float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _progress_features(vehicle_position, stop, next_stop, total_segments_hint):
    """Distance/progress metrics computable online without route shapes.

    Returns ``(distance_to_stop, distance_to_next_stop, progress_ratio)``. The
    internal ``progress_on_segment`` only feeds ``progress_ratio`` and is not a
    served feature anymore.
    """
    vp_lat = vehicle_position["lat"]
    vp_lon = vehicle_position["lon"]
    stop_lat = stop["lat"]
    stop_lon = stop["lon"]

    distance_to_stop = haversine_distance(vp_lat, vp_lon, stop_lat, stop_lon)

    progress_on_segment = 0.0
    if next_stop:
        next_lat = next_stop["lat"]
        next_lon = next_stop["lon"]
        segment_length = haversine_distance(stop_lat, stop_lon, next_lat, next_lon)
        if segment_length == 0.0:
            distance_to_next = 0.0
        else:
            distance_to_next = haversine_distance(vp_lat, vp_lon, next_lat, next_lon)
            progress_on_segment = max(0.0, min(1.0, 1.0 - (distance_to_next / segment_length)))
    else:
        distance_to_next = None

    stop_seq = (
        stop.get("stop_sequence")
        or stop.get("sequence")
        or stop.get("stop_order")
        or 1
    )
    total_segments = (
        stop.get("total_stop_sequence")
        or total_segments_hint
        or stop_seq
    )
    completed = max(float(stop_seq) - 1.0, 0.0)
    denom = max(float(total_segments), 1.0)
    progress_ratio = max(0.0, min(1.0, (completed + progress_on_segment) / denom))

    return distance_to_stop, distance_to_next, progress_ratio


def _predict_with_model(model_key, model_type, features, distance_m):
    """
    Call the appropriate predict_eta function based on model type.
    """

    if model_type == 'historical_mean':
        from stoptimes.models.historical_mean.predict import predict_eta
        return predict_eta(
            model_key=model_key,
            route_id=features.get('route_id', 'unknown'),
            stop_sequence=features.get('stop_sequence', 0),
            hour=features.get('hour', 0),
            day_of_week=features.get('day_of_week', 0),
            is_peak_hour=features.get('is_peak_hour', False)
        )

    elif model_type == 'ewma':
        # EWMA is retired as a serving candidate (2026-08-30): negative R^2 at
        # 3 of 4 MBTA dataset sizes and strictly dominated by historical_mean
        # everywhere. The module is kept for offline ablation
        # (train_all_models.py --models ewma) but is no longer served, so a
        # pinned or auto-selected ewma key fails loudly rather than silently
        # serving a dominated model.
        raise ValueError(
            "ewma is retired as a serving candidate and is kept only for "
            "offline ablation. Serve xgboost (primary) or "
            "polyreg_distance/historical_mean (fallbacks) instead."
        )

    elif model_type == 'polyreg_distance':
        from stoptimes.models.polyreg_distance.predict import predict_eta
        return predict_eta(
            model_key=model_key,
            distance_to_stop=distance_m
        )

    elif model_type in ('polyreg_time', 'xgboost'):
        if model_type == 'polyreg_time':
            from stoptimes.models.polyreg_time.predict import predict_eta
        else:
            from stoptimes.models.xgb.predict import predict_eta

        return predict_eta(
            model_key=model_key,
            distance_to_stop=distance_m,
            distance_to_next_stop=features.get('distance_to_next_stop'),
            shape_distance_to_stop=features.get('shape_distance_to_stop'),
            shape_progress=features.get('shape_progress'),
            cross_track_error=features.get('cross_track_error'),
            progress_ratio=features.get('progress_ratio'),
            stops_ahead=features.get('stops_ahead'),
            current_speed_kmh=features.get('current_speed_kmh'),
            bearing_to_stop=features.get('bearing_to_stop'),
            bearing_diff=features.get('bearing_diff'),
            is_at_stop=features.get('is_at_stop'),
            hour=features.get('hour', 0),
            day_of_week=features.get('day_of_week', 0),
            is_peak_hour=features.get('is_peak_hour', False),
            is_weekend=features.get('is_weekend', False),
            is_holiday=features.get('is_holiday', False),
            hour_sin=features.get('hour_sin'),
            hour_cos=features.get('hour_cos'),
            dow_sin=features.get('dow_sin'),
            dow_cos=features.get('dow_cos'),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _legacy_registry_candidates(registry, model_type: str | None) -> list[ArtifactMetadata]:
    """View the legacy `ModelRegistry`'s entries as `ArtifactMetadata`, so
    ranking can go through `select_artifact` instead of a bespoke loop.

    Mirrors exactly `ModelRegistry.get_best_model`'s own candidate-gathering
    (same `model_type` filter, same requirement that `metrics.test_mae_seconds`
    be present and numeric) so the two never rank a different candidate set --
    faithfulness, not just interface tidiness. Doesn't touch model *loading*:
    `artifact_id` here is still a legacy `model_key` string, resolved back
    into a prediction the same way it always was (`_predict_with_model`).
    """
    candidates: list[ArtifactMetadata] = []
    for key, info in registry.registry.items():
        if model_type and info.get("model_type") != model_type:
            continue
        route_id = info.get("route_id")
        try:
            meta = registry.load_metadata(key)
        except Exception:
            continue
        metric_value = meta.get("metrics", {}).get("test_mae_seconds")
        if not isinstance(metric_value, (int, float)):
            continue
        scope = (
            ModelScope(level=ScopeLevel.GLOBAL)
            if route_id in (None, "global")
            else ModelScope(level=ScopeLevel.ROUTE, route_ids=[route_id])
        )
        try:
            candidates.append(
                ArtifactMetadata(
                    artifact_id=key,
                    family=info.get("model_type", "unknown"),
                    scope=scope,
                    metrics={"test_mae_seconds": float(metric_value)},
                    provenance=Provenance(created_at=datetime.now(timezone.utc)),
                )
            )
        except Exception:  # pragma: no cover - a malformed legacy key never ranks
            continue
    return candidates


def _select_ranked_model_key(
    registry,
    *,
    model_type: str | None,
    route_id: str | None,
    prefer_route_model: bool,
) -> tuple[str | None, str | None]:
    """Replace the old `prefer_route_model` if/else with one call into
    `core.contracts.selection.select_artifact`: try the route-specific scope
    first (when a real route is known and preferred), then fall back to
    global -- the exact "most specific scope with a candidate wins" rule
    every other track's serving goes through.

    Returns `(model_key, model_scope)`, `(None, None)` if nothing qualifies.

    Not covered by this: the served-model *pin* (`_config.served_model_key`),
    which stays a direct registry-membership check in `estimate_stop_times`
    -- it's presence-only and route/model_type-agnostic by design (an agency
    pin must win even over a "better" model the ranking would pick), which
    doesn't fit `SelectionPolicy`'s scope+metric ranking model. See
    docs/notes/p3-u6.md.
    """
    query_route_id = route_id if (prefer_route_model and route_id and route_id != "unknown") else None
    scope_order = [ScopeLevel.ROUTE, ScopeLevel.GLOBAL] if query_route_id else [ScopeLevel.GLOBAL]
    candidates = _legacy_registry_candidates(registry, model_type)
    policy = SelectionPolicy(
        metric="test_mae_seconds",
        scope_order=scope_order,
        families=[model_type] if model_type else [],
    )
    selected = select_artifact(
        candidates, policy, source_id=query_route_id or "unknown", route_id=query_route_id
    )
    if selected is None:
        return None, None
    scope_label = "route" if selected.scope.level is ScopeLevel.ROUTE else "global"
    return selected.artifact_id, scope_label


def _stage_match_and_features(
    vehicle_position: dict,
    stops_to_predict: list[dict],
    route_id: str,
    shape: object,
    temporal_features: dict,
) -> list[dict]:
    """The adapter's `match` + `features` stages: attach spatial and
    temporal features to each already-identified upcoming stop.

    Unlike `serve.pipeline.default_match` (which discovers upcoming stops
    itself by projecting a live position onto a static feed's shape), this
    adapter's stops are already matched -- supplied by the caller, per the
    `estimate_stop_times` contract databus relies on -- so "match" here means
    attaching distance/shape/kinematic features to a known stop, not
    discovering one. Logic is unchanged from the pre-pipeline implementation
    (see docs/notes/p3-u6.md); only its home moved.
    """
    approx_total_segments = (
        max(
            (
                stop.get('total_stop_sequence')
                or stop.get('stop_sequence')
                or stop.get('sequence')
                or 0
            )
            for stop in stops_to_predict
        )
        if stops_to_predict
        else 0
    )
    if approx_total_segments <= 0:
        approx_total_segments = max(len(stops_to_predict), 1)

    matched: list[dict] = []
    for idx, stop in enumerate(stops_to_predict):
        next_stop = stops_to_predict[idx + 1] if idx + 1 < len(stops_to_predict) else None

        stop_sequence_value = (
            stop.get('stop_sequence')
            or stop.get('sequence')
            or stop.get('stop_order')
            or idx + 1
        )

        distance_m, distance_to_next, progress_ratio = _progress_features(
            vehicle_position,
            stop,
            next_stop,
            approx_total_segments,
        )

        # ------------------------------------------------------------------
        # SHAPE HOOK: precomputed distance (from the stop dict) beats
        # ShapePolyline projection, which beats the haversine proxy above.
        # Defaults below preserve today's proxy behavior when neither a
        # per-stop precomputed distance nor a shape object is supplied.
        # ------------------------------------------------------------------
        shape_distance_to_stop = distance_m
        shape_progress = progress_ratio
        cross_track_error = 0.0

        precomputed_dist = stop.get('shape_distance_to_stop')
        if precomputed_dist is not None:
            # Databus pre-computes a loop-back-safe monotonic distance along
            # the shape for each upcoming stop; use it directly as the
            # authoritative distance_m, bypassing both ShapePolyline
            # projection and the haversine fallback for this stop.
            distance_m = float(precomputed_dist)
            shape_distance_to_stop = distance_m
        elif shape is not None:
            shape_features = calculate_distance_features_with_shape(
                vehicle_position,
                stop,
                next_stop,
                shape=shape,
                vehicle_stop_order=stop_sequence_value,
                total_segments=approx_total_segments,
            )
            if shape_features.get('shape_distance_to_stop') is not None:
                distance_m = shape_features['shape_distance_to_stop']
                shape_distance_to_stop = distance_m
            if shape_features.get('cross_track_error') is not None:
                cross_track_error = shape_features['cross_track_error']
            if shape_features.get('shape_progress') is not None:
                shape_progress = shape_features['shape_progress']

        # Kinematic / status features straight off the live VehiclePosition.
        bearing_to_stop = _initial_bearing(
            vehicle_position['lat'], vehicle_position['lon'],
            stop['lat'], stop['lon'],
        )
        vp_bearing = vehicle_position.get('bearing', vehicle_position.get('heading'))
        vp_speed = vehicle_position.get('speed')  # m/s, matching the builder
        # Fallbacks keep every trained feature non-null online: a None feature is
        # skipped by predict_eta and the model then rejects the missing column.
        current_speed_kmh = vp_speed * 3.6 if vp_speed is not None else 0.0
        bearing_diff = _angle_diff(vp_bearing, bearing_to_stop)
        if bearing_diff is None:
            bearing_diff = 0.0
        if distance_to_next is None:  # last stop in the window has no successor
            distance_to_next = distance_m
        is_at_stop = vehicle_position.get('current_status') == 'STOPPED_AT'

        # Cyclical temporal encodings (must match the builder).
        hour = temporal_features['hour']
        dow = temporal_features['day_of_week']

        # Build features. Shape-derived features use real shape projections when
        # the SHAPE HOOK above resolved one (precomputed distance or a
        # ShapePolyline); otherwise they fall back to the geometric proxies
        # they've always used so the served input matches the trained schema.
        features = {
            'route_id': route_id,
            'stop_sequence': stop_sequence_value,
            'distance_to_stop': distance_m,
            'distance_to_next_stop': distance_to_next,
            'shape_distance_to_stop': shape_distance_to_stop,
            'shape_progress': shape_progress,
            'cross_track_error': cross_track_error,
            'progress_ratio': progress_ratio,
            # 0-indexed to match the training-side convention: the dataset
            # builder assigns stops_ahead=0 to the closest upcoming stop
            # (now stoptimes.train.datasets.dataset_builder, `stops_ahead =
            # int(stop_order_arr[j] - closest_stop_order)` starting at
            # j=closest_idx). upcoming_stops[0] is that same closest stop,
            # so it must be served as stops_ahead=0, not 1 (roadmap 1.2:
            # this off-by-one was the primary driver of the measured
            # +71s served-model bias).
            'stops_ahead': idx,
            'current_speed_kmh': current_speed_kmh,
            'bearing_to_stop': bearing_to_stop,
            'bearing_diff': bearing_diff,
            'is_at_stop': is_at_stop,
            'hour': hour,
            'day_of_week': dow,
            'is_weekend': temporal_features['is_weekend'],
            'is_holiday': temporal_features['is_holiday'],
            'is_peak_hour': temporal_features['is_peak_hour'],
            'hour_sin': math.sin(2 * math.pi * hour / 24.0),
            'hour_cos': math.cos(2 * math.pi * hour / 24.0),
            'dow_sin': math.sin(2 * math.pi * dow / 7.0),
            'dow_cos': math.cos(2 * math.pi * dow / 7.0),
        }

        matched.append(
            {
                'stop': stop,
                'stop_sequence_value': stop_sequence_value,
                'distance_m': distance_m,
                'features': features,
            }
        )

    return matched


def _stage_predict(ctx: PipelineContext, matched: list[dict]) -> list[dict]:
    """The adapter's `predict` stage: dispatch each matched, featurized stop
    through `_predict_with_model` (unchanged), containing errors per stop
    exactly like `serve.pipeline.default_predict` contains them per trip --
    one bad stop never drops the rest of the run.
    """
    model_key, actual_model_type = ctx.model
    predictions: list[dict] = []
    for item in matched:
        stop = item['stop']
        stop_sequence_value = item['stop_sequence_value']
        distance_m = item['distance_m']
        features = item['features']
        try:
            result = _predict_with_model(model_key, actual_model_type, features, distance_m)

            eta_seconds = result.get('eta_seconds', 0.0)
            eta_minutes = eta_seconds / 60.0
            eta_formatted = result.get(
                'eta_formatted',
                f"{int(eta_minutes)}m {int(eta_seconds % 60)}s"
            )

            eta_ts = datetime.fromtimestamp(ctx.now.timestamp() + eta_seconds, tz=timezone.utc)

            predictions.append({
                'stop_id': stop['stop_id'],
                'stop_sequence': stop_sequence_value,
                'distance_to_stop_m': round(distance_m, 1),
                'eta_seconds': round(eta_seconds, 1),
                'eta_minutes': round(eta_minutes, 2),
                'eta_formatted': eta_formatted,
                'eta_timestamp': eta_ts.isoformat(),
            })

        except Exception as e:
            predictions.append({
                'stop_id': stop['stop_id'],
                'stop_sequence': stop_sequence_value,
                'distance_to_stop_m': round(distance_m, 1),
                'eta_seconds': None,
                'eta_minutes': None,
                'eta_formatted': None,
                'eta_timestamp': None,
                'error': str(e),
            })
    return predictions


def estimate_stop_times(
    vehicle_position: dict,
    upcoming_stops: list[dict],
    route_id: str = None,
    trip_id: str = None,
    model_key: str = None,
    model_type: str = None,
    prefer_route_model: bool = True,
    max_stops: int = 3,
    shape: object = None,
) -> dict:
    """Estimate arrival times for upcoming stops from a live vehicle position.

    ``shape`` is an optional pre-loaded ``ShapePolyline`` (see
    ``stoptimes.core.spatial.geo``) for the vehicle's current trip. When
    provided, it's used to compute shape-projected distance/progress
    features instead of the haversine proxy. Per-stop, an even more
    authoritative signal wins over both: if an ``upcoming_stops`` entry
    carries a non-None ``shape_distance_to_stop`` key, that precomputed
    distance is used directly (see the SHAPE HOOK block below) — this is
    the path the databus caller exercises, since it precomputes a
    loop-back-safe monotonic distance along the shape itself.
    """

    # Validate inputs
    if not vehicle_position or not upcoming_stops:
        return {
            'vehicle_id': vehicle_position.get('vehicle_id', 'unknown') if vehicle_position else 'unknown',
            'route_id': route_id,
            'trip_id': trip_id,
            'computed_at': datetime.now(timezone.utc).isoformat(),
            'model_key': None,
            'predictions': [],
            'error': 'Missing vehicle position or stops'
        }

    stops_to_predict = upcoming_stops[:max_stops]

    # Parse timestamp
    vp_timestamp_str = vehicle_position['timestamp']
    if vp_timestamp_str.endswith('Z'):
        vp_timestamp_str = vp_timestamp_str.replace('Z', '+00:00')
    vp_timestamp = datetime.fromisoformat(vp_timestamp_str)

    # Extract temporal features (using config for timezone)
    temporal_features = extract_temporal_features(
        vp_timestamp,
        tz=_config.default_timezone,
        region=_config.default_region
    )

    # Determine route
    if route_id is None:
        route_id = vehicle_position.get('route', 'unknown')

    # Load registry
    registry = get_registry()
    model_scope = 'unknown'

    # ============================================================
    #    SMART MODEL SELECTION
    # ============================================================
    if model_key is None:
        # Served-model pin (the active agency's profile file --
        # examples/agencies/<agency>.yaml's `serving.primary_model`,
        # resolved via stoptimes.core.config's ProjectConfig.served_model_key
        # / STOPTIMES_MODEL_KEY): when the active agency has one, it wins
        # outright. This is what
        # guarantees the databus path serves the BUCR corpus model rather
        # than registry.get_best_model()'s metric-only ranking, which has no
        # dataset/agency awareness and would happily pick an MBTA model
        # (e.g. lower test_mae_seconds) over a BUCR one. An agency with no
        # pin configured (e.g. mbta) falls straight through to that ranking,
        # unchanged from prior behavior.
        pinned_key = _config.served_model_key
        if pinned_key is not None:
            pinned_fallback_key = _config.served_model_fallback_key
            if pinned_key in registry.registry:
                model_key = pinned_key
                model_scope = 'pinned'
            elif pinned_fallback_key is not None and pinned_fallback_key in registry.registry:
                model_key = pinned_fallback_key
                model_scope = 'pinned_fallback'
                _logger.warning(
                    "Pinned served model missing from registry, using fallback",
                    pinned_key=pinned_key,
                    fallback_key=pinned_fallback_key,
                )
            else:
                return {
                    'vehicle_id': vehicle_position['vehicle_id'],
                    'route_id': route_id,
                    'trip_id': trip_id,
                    'computed_at': datetime.now(timezone.utc).isoformat(),
                    'model_key': None,
                    'predictions': [],
                    'error': (
                        f'Pinned served model {pinned_key!r} and fallback '
                        f'{pinned_fallback_key!r} are both missing from the registry'
                    ),
                }
        else:
            model_key, model_scope = _select_ranked_model_key(
                registry,
                model_type=model_type,
                route_id=route_id,
                prefer_route_model=prefer_route_model,
            )
            if model_scope is None:
                model_scope = 'global'

        # Last fallback: best model of any scope, route or global, ignoring
        # the requested route entirely. `select_artifact`'s scope contract
        # requires naming a specific route to consider ROUTE-scoped
        # candidates, so it can't express "any route, don't care which" --
        # kept as the one direct legacy call (see docs/notes/p3-u6.md).
        if model_key is None:
            model_key = registry.get_best_model(model_type=model_type)

        if model_key is None:
            return {
                'vehicle_id': vehicle_position['vehicle_id'],
                'route_id': route_id,
                'trip_id': trip_id,
                'computed_at': datetime.now(timezone.utc).isoformat(),
                'model_key': None,
                'predictions': [],
                'error': 'No trained models found for model_type'
            }

    # Load metadata
    try:
        model_metadata = registry.load_metadata(model_key)
        actual_model_type = model_metadata.get('model_type', 'unknown')
        model_route_id = model_metadata.get('route_id')

        if model_route_id not in (None, 'global'):
            model_scope = 'route'
        elif model_scope == 'unknown':
            model_scope = 'global'

    except Exception as e:
        return {
            'vehicle_id': vehicle_position['vehicle_id'],
            'route_id': route_id,
            'trip_id': trip_id,
            'computed_at': datetime.now(timezone.utc).isoformat(),
            'model_key': model_key,
            'predictions': [],
            'error': f'Failed to load model metadata: {str(e)}'
        }

    # ============================================================
    #    RUN THE ADAPTER PIPELINE: match+features, then predict
    # ============================================================
    # `estimate_stop_times` is a thin adapter over the same six-stage shape
    # `serve.pipeline` runs (docs/PLAN.md §7 U6): a `PipelineContext` threads
    # `now`/`model`/`static_feed` through, and `_stage_match_and_features` /
    # `_stage_predict` do the same jobs `default_match`+`default_features` /
    # `default_predict` do there. It isn't literally `StoptimesPipeline.run`,
    # because this function's contract -- stops the caller already matched,
    # not a static feed to match against -- doesn't fit `default_match`'s
    # payload shape (see docs/notes/p3-u6.md); the goal here is one feature
    # implementation, not a second one, which this keeps.
    ctx = PipelineContext(
        source=None,
        static_feed=shape,
        model=(model_key, actual_model_type),
        now=vp_timestamp,
        config={'route_id': route_id, 'max_stops': max_stops},
    )
    matched = _stage_match_and_features(vehicle_position, stops_to_predict, route_id, shape, temporal_features)
    predictions = _stage_predict(ctx, matched)

    return {
        'vehicle_id': vehicle_position['vehicle_id'],
        'route_id': route_id,
        'trip_id': trip_id,
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'model_key': model_key,
        'model_type': actual_model_type,
        'model_scope': model_scope,
        'predictions': predictions
    }
