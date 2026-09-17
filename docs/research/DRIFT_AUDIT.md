# ETA Library Divergence Audit — gtfs-django (canonical) vs databus/gtfs-eta (vendored)

Read-only audit. No files modified in either repo.

- **Canonical**: `/Users/dotj/Desktop/SIMOVI/git.no_sync/gtfs-django/eta_prediction/`
- **Vendored**: `/Users/dotj/Desktop/SIMOVI/git.no_sync/databus/backend/gtfs-eta/gtfs_eta/`

## Executive summary

- The `core/*` slimming (config/logging/exceptions/validation) is **inert, not just intentional slimming**: canonical's `eta_service/estimator.py` imports `validate_vehicle_position`, `validate_stops_list`, `ModelNotFoundError`, `ModelLoadError`, `PredictionError`, `InvalidVehiclePositionError`, `InvalidStopError` but never calls/raises any of them (dead imports). Vendored's `estimator.py` doesn't import `core.*` at all — it uses stdlib `logging` directly and hardcodes `"America/Costa_Rica"/"CR"` instead of calling `get_config()`. Reconciling `core/` is safe and mechanical either way — nothing behavioral rides on it.
- **`eta_service/estimator.py` is the critical divergence.** Vendored is *not* a slimmed-down port — it has a real feature: shape-aware distance projection (`_progress_features_with_shape`, via `feature_engineering/spatial.py`) plus a "precomputed-distance hook" that databus depends on directly. Canonical never developed this; it still uses haversine-only geometric proxies with a different feature set (bearing, speed, cyclical hour/dow encodings) that vendored dropped. **These are two genuinely different, incompatible feature schemas feeding the model** — see the estimator deep-dive below.
- **The databus seam is exactly one function call plus one class import**: `gtfs_eta.eta_service.estimator.estimate_stop_times(...)` and `gtfs_eta.feature_engineering.spatial.ShapePolyline`. Any consolidation must preserve `estimate_stop_times`'s signature (specifically the `shape=` kwarg and the `shape_distance_to_stop` stop-dict key) and `ShapePolyline`'s constructor contract, or databus's `stop_times.py` breaks.
- **Model registry contract is compatible today by luck, not by design.** Canonical writes absolute paths into `registry.json`; vendored writes basenames only and always resolves via basename-stripping (`_resolve_in_registry`) regardless of what's stored. The live `registry.json` on databus (`eta_models/registry.json`) has one entry (`polyreg_distance_global_baseline_v0`, basename paths) — currently loadable by both sides — but the two `save_model()` implementations produce structurally different JSON, and vendored's `_predict_with_model` has dead branches for `historical_mean`/`ewma`/`polyreg_time`/`xgboost` that would `ModuleNotFoundError` if the registry ever pointed at one of those model types (those model packages were never vendored).
- Only `polyreg_distance` is fully vendored and live in production; `models/{historical_mean,ewma,polyreg_time,xgb}` and all `feature_engineering/{dataset_builder,progress,rt_source,weather}.py` and every `models/*/train.py` exist only in canonical (expected — training-only), but the **prediction-side** siblings (`historical_mean/predict.py`, `ewma/predict.py`, `polyreg_time/predict.py`, `xgb/predict.py`) are also missing from vendored, which is inference-relevant, not just training-relevant, and is a latent break if the registry is ever seeded with a non-`polyreg_distance` model.

## The databus seam (exact API surface)

File: `/Users/dotj/Desktop/SIMOVI/git.no_sync/databus/backend/runs/domain/progression/stop_times.py`

Two lazy imports inside `compute_stop_time_updates`:

```python
from gtfs_eta.feature_engineering.spatial import ShapePolyline   # noqa: PLC0415
...
import gtfs_eta.eta_service.estimator as _estimator_mod          # noqa: PLC0415
```

Usage:

```python
shape = ShapePolyline([(pt[0], pt[1]) for pt in geom.polyline])

result = _estimator_mod.estimate_stop_times(
    vehicle_position,
    upcoming_stops,
    route_id=run_hash.get("route_id"),
    trip_id=run_hash.get("trip_id"),
    prefer_route_model=True,
    max_stops=max_stops,
    shape=shape,
)
```

Required contract:
1. `ShapePolyline(list[tuple[float, float]])` — constructor takes a plain list of `(lat, lon)` tuples. **Canonical's `ShapePolyline` also exists but the class only lives in `feature_engineering/spatial.py`, which the databus seam imports as `gtfs_eta.feature_engineering.spatial` — this module path must survive consolidation.**
2. `estimate_stop_times(vehicle_position, upcoming_stops, route_id=, trip_id=, prefer_route_model=, max_stops=, shape=)` — the `shape=` kwarg is **vendored-only**; canonical's signature has no `shape` parameter at all (see deep-dive).
3. Each `upcoming_stops[i]` dict is built by databus with keys: `stop_id`, `stop_sequence`, `lat`, `lon`, `total_stop_sequence`, **`shape_distance_to_stop`** (a pre-computed loop-back-safe monotonic distance along the route shape, in metres). `shape_distance_to_stop` is read by vendored's estimator as the "precomputed-distance hook" (see below) — canonical's estimator has no code path that reads this key at all, so if canonical's estimator were substituted as-is, this field would silently be ignored and every prediction would fall back to raw haversine, changing model input distribution.
4. Return value fields consumed by databus: `result.get("error")`, `result["predictions"]` (list), and per-prediction: `pred.get("error")`, `pred["stop_sequence"]`, `pred["stop_id"]`, `pred.get("eta_timestamp")`. Vendored's estimator emits `shape_used` at top level and optional `cross_track_error_m` / `shape_progress` / `shape_distance_to_stop_m` per-prediction that databus does **not** currently read — safe to keep or drop.
5. `vehicle_position` dict built by databus: `vehicle_id`, `lat`, `lon`, `speed` (m/s), `timestamp` (ISO string), `route`. Matches both estimators' expectations.

**Environment contract** (from `stop_times.py` module docstring): `MODEL_REGISTRY_DIR` (read by `gtfs_eta`'s registry singleton from the environment, per the vendored `_resolve_registry_dir()`), `ETA_MAX_STOPS`, `ETA_DEFAULT_UNCERTAINTY_S`. The last two are consumed entirely inside `stop_times.py`, not inside `gtfs_eta` — no seam impact.

## Vendored-only work at risk (highest priority)

These exist **only** on the vendored side and would be silently lost if canonical's copy were declared authoritative and vendored discarded.

### 1. Shape-aware distance/progress features — `feature_engineering/spatial.py` `calculate_distance_features_with_shape` consumption in `estimator.py`

Vendored `estimator.py` adds two helper functions canonical does not have:

```python
def _progress_features_with_shape(
    vehicle_position, stop, next_stop, shape, vehicle_stop_order, total_segments
):
    """
    Shape-aware distance / progress metrics using a pre-loaded ShapePolyline.
    Returns enhanced spatial features including cross-track error and
    shape-based distances.
    """
    features = calculate_distance_features_with_shape(...)
    return {
        "distance_to_stop_m": features.get("distance_to_stop", 0.0),
        "progress_on_segment": features.get("progress_on_segment", 0.0),
        "progress_ratio": features.get("progress_ratio", 0.0),
        "cross_track_error": features.get("cross_track_error"),
        "shape_progress": features.get("shape_progress"),
        "shape_distance_to_stop": features.get("shape_distance_to_stop"),
    }
```

Canonical has no `shape` parameter on `estimate_stop_times` and no `_progress_features_with_shape` at all — it only ever computes haversine-based `_progress_features`. If canonical is used as-is, cross-track error and shape-projected distance are permanently lost, and predictions degrade to straight-line-distance accuracy on curved/looping routes.

### 2. The "precomputed-distance hook" — the actual thing keeping databus's GTFS-RT feed accurate today

```python
# -------------------------------------------------------------------
# PRECOMPUTED-DISTANCE HOOK
# Databus pre-computes a loop-back-safe monotonic distance along the
# shape for each upcoming stop and stores it as shape_distance_to_stop.
# When present, we use it directly as the authoritative distance_m,
# skipping both ShapePolyline projection and haversine fallback.
# This avoids projection artifacts on looping routes and ensures the
# distance is always non-decreasing across the stop list.
# -------------------------------------------------------------------
precomputed_dist = stop.get("shape_distance_to_stop")
if precomputed_dist is not None:
    distance_m = float(precomputed_dist)
    spatial_features = { ... "shape_distance_to_stop": distance_m }
elif shape_available:
    spatial_features = _progress_features_with_shape(...)
    distance_m = spatial_features["distance_to_stop_m"]
else:
    spatial_features = _progress_features_fallback(...)
    distance_m = spatial_features["distance_to_stop_m"]
```

This is exactly what `stop_times.py` relies on (it always sets `shape_distance_to_stop` on every upcoming stop — see seam section, point 3). **This branch is exercised on every single production prediction today.** It has zero counterpart in canonical. Losing it means every prediction reverts to raw haversine and loses the "loop-back-safe monotonic" guarantee, which will actively break ETAs on loop routes (haversine distance can *decrease* near a loop-back even though the vehicle is moving further along the route).

### 3. Registry relocatability — `_find_existing_registry_dir()` / `_resolve_registry_dir()` / `_resolve_in_registry()` in `models/common/registry.py`

```python
def _resolve_registry_dir() -> Path:
    """
    Determine the registry directory with this priority:
      1. MODEL_REGISTRY_DIR env var (authoritative).
      2. Walk CWD upward for an existing registry (dev convenience).
      3. Raise at runtime if neither is available.
    """
```
```python
def _resolve_in_registry(self, stored_path: str) -> Path:
    """Resolve a registry-stored path against the registry directory.
    Only the basename of ``stored_path`` is used, so entries written with
    an absolute path on another host ... still load wherever the registry
    directory currently lives.
    """
    return self.base_dir / Path(stored_path).name
```
Canonical's `ModelRegistry.__init__` instead does `base_dir or os.getenv("MODEL_REGISTRY_DIR") or "models/trained"` (silently falls back to a relative-to-CWD default rather than raising) and `save_model` writes `str(model_path)` (an **absolute** path baked into `registry.json`) rather than a basename. This is real portability work done on the vendored side — it is what makes `MODEL_REGISTRY_DIR` bind-mountable per the vendored README's stated design ("Paths are resolved relative to the registry directory... relocatable"). Canonical does not have this design goal executed; it only has a partial fallback (`load_model`/`load_metadata` check `is_absolute()` and fall back to `self.base_dir / model_path.name`, but `save_model` still writes absolute paths, so a registry.json written by canonical training and then moved to another host is not guaranteed to resolve — canonical's own "relocatability" is incomplete).

### 4. `models/polyreg_distance/model.py` — inference-only class extraction

Vendored-only file. Docstring states intent precisely:
```
Extracted verbatim from eta_prediction/models/polyreg_distance/train.py ...
Only the class and its sklearn/numpy/pandas imports are present here —
training functions, dataset loaders, metrics, and ModelKey helpers are
deliberately excluded so pickles can be loaded without any training-time
dependency.
```
Canonical has no equivalent standalone module — `PolyRegDistanceModel` presumably still lives inside canonical's `models/polyreg_distance/train.py` bundled with training code (not independently confirmed by reading that file in this pass — see "could not determine" below). This split-out-for-pickle-compat pattern is real vendored engineering: if canonical's `train.py` changes the class in an incompatible way without this seam, pickled models become unloadable by the inference-only side.

### 5. `models/polyreg_distance/predict.py` — extra `route_id` parameter

Vendored's `predict_eta` takes `route_id: Optional[str] = None` and raises `ValueError("route_id is required for a route-specific model")` when `model.route_specific` is true and `route_id` is `None`. Confirm whether canonical's `models/polyreg_distance/predict.py` has the same guard — **not read in this pass; flagged as unverified** (see gaps section).

### 6. `seed_baseline_model.py` — vendored-only deployment/bootstrap script

No canonical equivalent found. This is the only way to produce a working `registry.json` + `.pkl`/`_meta.json` triplet from scratch without the full training pipeline; it is what seeded the live `eta_models/registry.json` (`dataset: "synthetic_constant_speed"`, `saved_at: "2026-06-25..."` matches this script's synthetic-data approach). Losing this script removes the only lightweight way to bootstrap a registry for a fresh environment/CI.

## Per-module divergence table

| Path | Canonical lines | Vendored lines | Classification | Reconciliation difficulty |
|---|---:|---:|---|---|
| `core/__init__.py` | 52 | 1 | Intentional slimming (vendored is empty stub) | Easy |
| `core/config.py` | 228 | 28 | Intentional slimming, **but vendored estimator doesn't call `get_config()` at all** — dead code on both sides re: estimator | Easy |
| `core/exceptions.py` | 300 (19 exception classes) | 13 (3 classes: `GTFSEtaError`, `ModelNotFoundError`, `PredictionError`) | Intentional slimming; canonical's richer hierarchy (`ModelLoadError`, `InvalidVehiclePositionError`, `InvalidStopError`, `RedisError`, etc.) is imported by canonical's estimator but never raised — dead either way | Easy |
| `core/logging.py` | 266 (structured `ETALogger`, JSON formatter) | 15 (thin `get_logger` wrapper) | Intentional slimming; vendored `estimator.py` doesn't even call vendored `core.logging.get_logger` — it uses stdlib `logging.getLogger(__name__)` directly | Easy |
| `core/validation.py` | 449 (`VehiclePosition`/`Stop` dataclasses, full validators) | 15 (`require_keys`, `require_positive` — generic, unused by estimator) | Intentional slimming; canonical's `validate_vehicle_position`/`validate_stops_list` are imported by canonical's own estimator but never called — so this is genuinely dead validation logic on both sides today | Easy |
| `eta_service/estimator.py` | 410 | 495 | **Genuine conflict** — see deep-dive. Vendored added shape-awareness + precomputed-distance hook (real, load-bearing); canonical has bearing/speed/cyclical-time features vendored dropped. Different feature schemas passed to `_predict_with_model` for polyreg_time/xgboost. | **Hard** |
| `feature_engineering/temporal.py` | 94 | 98 | Intentional port — functionally identical (`diff` shows only comment/docstring/arrow-notation changes, no logic delta) | Easy |
| `feature_engineering/spatial.py` | 372 | 250 | Partial slimming: vendored drops `load_shape_from_gtfs`, `load_shape_for_trip` (DB-loading helpers, correctly training/DB-side only) and `example_usage()`; keeps `ShapePolyline`, `calculate_distance_features_with_shape` — **need line-level diff of the kept functions**, not verified in this pass whether their bodies also diverged | Moderate (verify kept-function bodies match before assuming pure subset) |
| `feature_engineering/{dataset_builder,progress,rt_source,weather}.py` | present | absent | Canonical-only, training/dataset-build only — expected | N/A (don't reconcile, keep training-only) |
| `models/common/registry.py` | 390 | 316 | **Vendored-only work** — relocatable path resolution (`_resolve_in_registry`, `_resolve_registry_dir`, `_find_existing_registry_dir`); canonical drops `compare_routes()` counterpart (training/reporting tool, expected). `get_best_model` logic is byte-for-byte equivalent between the two. | Moderate |
| `models/common/utils.py` | 287 | 126 | Intentional slimming — vendored drops `_ui_on`, `_metric`, `print_metrics_table`, `train_test_summary`, `create_feature_importance_df` (all reporting/CLI, training-side only); keeps `safe_divide`, `clip_predictions`, `calculate_speed_kmh`, `haversine_distance`, `format_seconds`, `add_lag_features`, `smooth_predictions` | Easy |
| `models/polyreg_distance/{predict.py,model.py}` | present (bundled differently — not fully verified) | present, `model.py` is vendored-only extraction | Vendored-only work (pickle-compat class split) — see item 4/5 above | Moderate |
| `models/polyreg_distance/train.py` | present | absent | Canonical-only, training only — expected | N/A |
| `models/{historical_mean,ewma,polyreg_time,xgb}/predict.py` | present | **absent** | Canonical-only but **inference-relevant** — vendored's `_predict_with_model` has live `elif` branches that `import` these exact modules by vendored path (`gtfs_eta.models.historical_mean.predict`, etc.) and will raise `ModuleNotFoundError` if ever invoked | Moderate–Hard (must port predict.py siblings before any non-polyreg_distance model is registered) |
| `models/{historical_mean,ewma,polyreg_time,xgb}/train.py`, `models/train_all_models.py`, `models/evaluation/*`, `models/check_registry.py` | present | absent | Canonical-only, training/ops tooling — expected | N/A |
| `bytewax/*`, `prefect/*` | present | absent | Canonical-only, pipeline/orchestration — expected, no inference relevance found | N/A |
| `__init__.py` (package root), `models/polyreg_distance/__init__.py` | absent/N/A | present | Vendored-only — packaging plumbing for `gtfs_eta` as an installable namespace; expected given "candidate for extraction" intent | Easy |
| `seed_baseline_model.py` | absent | present | Vendored-only work — bootstrap script, no canonical counterpart found | Easy to port back, but **should be ported**, not discarded |

## `eta_service/estimator.py` deep dive

### Imports

| | Canonical | Vendored |
|---|---|---|
| Config | `from core.config import get_config` (called, but only `.default_timezone`/`.default_region` attrs used) | none — hardcodes `tz="America/Costa_Rica", region="CR"` literals inline |
| Logging | `from core.logging import get_logger` → `_logger = get_logger("estimator")` | `import logging` → `_log = logging.getLogger(__name__)` |
| Exceptions | imports `ModelNotFoundError`, `ModelLoadError`, `PredictionError`, `InvalidVehiclePositionError`, `InvalidStopError` — **none referenced in the function body** | none imported |
| Validation | imports `validate_vehicle_position`, `validate_stops_list` — **never called** | none imported |
| Temporal | `from feature_engineering.temporal import extract_temporal_features` | `from gtfs_eta.feature_engineering.temporal import extract_temporal_features` |
| Registry | `from models.common.registry import get_registry` | `from gtfs_eta.models.common.registry import get_registry` |
| Spatial | not imported | `try: from gtfs_eta.feature_engineering.spatial import ShapePolyline, calculate_distance_features_with_shape; SHAPE_SUPPORT = True; except ImportError: SHAPE_SUPPORT = False` |

### Function/class inventory

| Name | Canonical | Vendored | Notes |
|---|---|---|---|
| `haversine_distance(lat1, lon1, lat2, lon2)` | yes | yes | Identical formula, cosmetic-only diff (`R = 6371000` vs `R = 6_371_000`) |
| `_initial_bearing(...)` | **yes** | **no** | Canonical-only. Computes vehicle→stop bearing for `bearing_to_stop`/`bearing_diff` features. Dropped in vendored along with the features it feeds. |
| `_angle_diff(a, b)` | **yes** | **no** | Canonical-only, feeds `bearing_diff`. |
| `_progress_features(vehicle_position, stop, next_stop, total_segments_hint)` | yes (single haversine-only version, returns 3-tuple) | renamed/split into `_progress_features_fallback` (same haversine logic, returns a dict instead of a tuple) **plus** `_progress_features_with_shape` (new) | Vendored's fallback is functionally the haversine algorithm ported almost verbatim, but the **return shape changed from tuple to dict** and gained `cross_track_error`/`shape_progress`/`shape_distance_to_stop` keys (all `None` in the fallback path). |
| `_progress_features_with_shape(...)` | **no** | **yes** | Vendored-only, see "vendored-only work" section. |
| `_predict_with_model(model_key, model_type, features, distance_m)` | yes | yes | **Different branches/feature keys — see below.** |
| `estimate_stop_times(...)` | yes, no `shape` param | yes, **adds `shape: object = None` param** | Public entry point — signature diverges (additive on vendored side, backward compatible for positional/keyword calls that omit `shape`). |

### `_predict_with_model` — feature schema conflict (the actual conflict, not just size)

Canonical's `polyreg_time`/`xgboost` branch passes:
```python
distance_to_stop, distance_to_next_stop, shape_distance_to_stop, shape_progress,
cross_track_error, progress_ratio, stops_ahead, current_speed_kmh,
bearing_to_stop, bearing_diff, is_at_stop, hour, day_of_week, is_peak_hour,
is_weekend, is_holiday, hour_sin, hour_cos, dow_sin, dow_cos
```
Vendored's `polyreg_time`/`xgboost` branches (kept separate `elif`s, not combined like canonical) pass:
```python
distance_to_stop, progress_on_segment, progress_ratio, hour, day_of_week,
is_peak_hour, is_weekend, is_holiday, temperature_c, precipitation_mm,
wind_speed_kmh
```
These are **disjoint feature sets** — canonical is kinematic/bearing-based with cyclical time encodings; vendored is weather-augmented (`temperature_c`, `precipitation_mm`, `wind_speed_kmh`, sourced from `core/config.py`'s `DEFAULT_TEMPERATURE_C`/etc. defaults, hardcoded as literals `25.0`/`0.0`/`None` directly in `estimate_stop_times` rather than via `get_config()`). Neither side's `models/{polyreg_time,xgb}/predict.py` is present in the vendored tree to check which schema the actual trained pickle expects — **this can only be resolved by reading canonical's `models/polyreg_time/predict.py` and `models/xgb/predict.py` signatures**, not done in this pass (out of the requested `core/eta_service/feature_engineering/models` scope check but flagged as a hard blocker for reconciling this function). Since only `polyreg_distance` is live in production, this conflict is currently dormant but **will surface the moment a `polyreg_time` or `xgboost` model is registered** — whichever `predict_eta` module design a consolidated package keeps, the other side's trained pickles/feature-column assumptions are incompatible.

### `historical_mean` / `ewma` branches
Byte-for-byte parameter parity between canonical and vendored (`route_id`, `stop_sequence`, `hour`, `day_of_week`, `is_peak_hour` for `historical_mean`; `route_id`, `stop_sequence`, `hour` for `ewma`). Only the import path differs (`models.X.predict` vs `gtfs_eta.models.X.predict`). Low risk — but **neither `historical_mean/predict.py` nor `ewma/predict.py` exists in the vendored tree**, so these branches are unreachable dead code there today (would `ModuleNotFoundError` if a matching model_type were ever registered).

### `estimate_stop_times` body differences (beyond the shape/precomputed-distance hook already covered)

- **`stop_sequence_value` resolution**: canonical uses `stop.get('stop_sequence') or stop.get('sequence') or stop.get('stop_order') or idx + 1` (falsy-coalescing — **bug**: a real `stop_sequence == 0` is treated as missing and silently relabeled). Vendored explicitly fixed this with sequential `is not None` checks and a code comment:
  ```python
  # Use explicit None checks, not `or`: stop_sequence == 0 is a valid
  # GTFS 0-based sequence and must not be treated as falsy (that would
  # relabel stop 0 as 1 and collide with a real stop 1).
  ```
  **This is vendored-only bug-fix work; porting canonical's estimator wholesale would reintroduce a real bug for any agency using 0-based GTFS `stop_sequence`.**
- Vendored's success-path prediction dict conditionally adds `cross_track_error_m`, `shape_progress`, `shape_distance_to_stop_m` when present; canonical's dict never has these keys at all.
- Vendored's top-level return adds `"shape_used": shape_available`; canonical's does not.
- Canonical computes `bearing_to_stop`, `bearing_diff`, `current_speed_kmh`, `is_at_stop`, `hour_sin/cos`, `dow_sin/cos` per-stop inline in the loop; vendored has none of this — it was fully replaced by the shape/precomputed-distance path and weather placeholders.
- Docstring at the top of vendored's file is an explicit, self-documenting changelog of every change made when it was ported (see the file header) — useful as a divergence log in itself; recommend preserving it or migrating its content into a CHANGELOG when consolidating.

### What the extra ~85 lines actually are (495 − 410)

Net line delta breaks down roughly as:
- **+37 lines**: expanded module/function docstrings (top-of-file changelog comment, `estimate_stop_times` args doc, precomputed-distance-hook comment block)
- **+~30 lines**: `_progress_features_with_shape` + shape-aware branch in the per-stop loop + optional-shape-metrics output block
- **+~15 lines**: precomputed-distance hook branch and its comment
- **−~45 lines**: removal of `_initial_bearing`, `_angle_diff`, and the bearing/speed/cyclical-time feature computation inline in the loop
- **−~10 lines**: `_predict_with_model`'s `historical_mean`/`ewma`/`polyreg_distance` branches are near-identical length; `polyreg_time`/`xgboost` vendored branches are each ~4 lines shorter (fewer params) but are two separate `elif`s instead of one combined branch (+a few lines from the split)

Net: real added functionality (shape-awareness, precomputed-distance hook, stop_sequence 0-bug fix, richer docs) outweighs the removed bearing/speed feature code, landing at +85 lines.

## Registry contract

- **Format**: both sides use the same on-disk shape — `registry.json` is `{model_key: {model_path, meta_path, saved_at, model_type, route_id, dataset}}`, plus per-model `{model_key}.pkl` and `{model_key}_meta.json` sitting next to it in the same directory. Confirmed against the live file:
  ```json
  {
    "polyreg_distance_global_baseline_v0": {
      "model_path": "polyreg_distance_global_baseline_v0.pkl",
      "meta_path": "polyreg_distance_global_baseline_v0_meta.json",
      "saved_at": "2026-06-25T18:06:35.423137",
      "model_type": "polyreg_distance",
      "route_id": null,
      "dataset": "synthetic_constant_speed"
    }
  }
  ```
  Path: `/Users/dotj/Desktop/SIMOVI/git.no_sync/databus/backend/eta_models/registry.json`
- **Path storage differs**: canonical's `save_model()` writes `metadata['model_path'] = str(model_path)` — an **absolute path** baked in at write time. Vendored's `save_model()` writes `metadata["model_path"] = model_path.name` — **basename only**.
- **Path resolution differs but happens to be compatible today**: canonical's `load_model`/`load_metadata` check `if not model_path.is_absolute(): model_path = self.base_dir / model_path.name` (handles relative but trusts absolute paths verbatim — **not portable across hosts if a registry.json with absolute paths is copied elsewhere**). Vendored's `_resolve_in_registry()` always does `self.base_dir / Path(stored_path).name`, discarding any directory component unconditionally — this is why vendored can load the file above (which happens to already be basenames) and would *also* tolerate a canonical-written absolute-path entry (it strips to basename regardless). The reverse is not guaranteed: if vendored moves the registry directory and a canonical process still has the old absolute path cached, canonical will fail to find the file.
- **`get_best_model()`, `get_routes()`, `list_models()`**: verified logic is essentially identical (line-by-line equivalent filtering/sorting) between both sides — this part of the contract is not at risk.
- **Model-type coverage gap**: the live registry only has a `polyreg_distance` entry, and vendored only ships `polyreg_distance`'s `predict.py`/`model.py`. If canonical's training pipeline (`models/train_all_models.py`, `models/{historical_mean,ewma,polyreg_time,xgb}/train.py`) ever writes an entry with a different `model_type` into this same `registry.json`, vendored's `estimate_stop_times` → `_predict_with_model` will hit a live `ModuleNotFoundError` in production (the corresponding `gtfs_eta.models.<type>.predict` module doesn't exist).
- **Recommendation**: standardize on vendored's basename-storage + basename-stripped-resolution scheme (it's strictly more portable and is already documented as the intended design in the vendored README), and port `_resolve_in_registry`/`_resolve_registry_dir`/`_find_existing_registry_dir` back into canonical so both sides write/read the identical, portable format.

## Reconciliation difficulty and recommended order

1. **`core/{config,logging,exceptions,validation}.py` — Easy.** Neither side's estimator actually exercises the richer canonical logic (dead imports on canonical's side) nor the vendored stubs (vendored's estimator bypasses them entirely). Safe to either (a) keep vendored's minimal versions and delete the unused imports from canonical's estimator, or (b) wire canonical's richer validation/exceptions into a consolidated estimator going forward — pure greenfield decision, no drift to reconcile.
2. **`feature_engineering/temporal.py` — Easy.** Verified functionally identical; take either copy verbatim (recommend canonical's, it has denser inline comments) and drop vendored's copy.
3. **`models/common/utils.py` — Easy.** Vendored is a strict functional subset of canonical (verified via function name diff — no renamed/altered kept functions found). Keep canonical, vendored's copy can be deleted post-merge.
4. **`models/common/registry.py` — Moderate.** Logic-compatible (`get_best_model`, `get_routes`, `list_models` match), but the **path-storage scheme must be unified** before merging (see registry contract section) — pick vendored's basename scheme, port it into canonical, then treat vendored's copy as redundant. Do this **before** step 6, since step 6 (estimator) depends on a working registry either way and you don't want a save/load mismatch discovered mid-estimator-merge.
5. **`feature_engineering/spatial.py` — Moderate.** Vendored drops `load_shape_from_gtfs`/`load_shape_for_trip`/`example_usage()` (fine, DB/demo-only) but **the bodies of the kept `ShapePolyline` and `calculate_distance_features_with_shape` were not diffed line-by-line in this audit** — do that diff before assuming it's a pure subset, since the estimator's shape-aware path (item 1 in "vendored-only work") depends entirely on this module's correctness.
6. **`eta_service/estimator.py` — Hard.** This is the real merge:
   - Keep vendored's shape-awareness, precomputed-distance hook, and the `stop_sequence` `is not None`-check bugfix — these are load-bearing for the live databus seam and fix a real bug.
   - Keep canonical's `_initial_bearing`/`_angle_diff`/bearing+speed+cyclical-time feature computation as an **available but currently-unused** feature set — don't discard it, since it represents real modeling work, but it must not be silently dropped from whichever `polyreg_time`/`xgboost` `predict.py` ends up canonical.
   - **Before merging, resolve the `_predict_with_model` feature-schema conflict for `polyreg_time`/`xgboost`** by reading canonical's `models/polyreg_time/predict.py` and `models/xgb/predict.py` to determine which feature schema (bearing/speed/cyclical vs weather-augmented) their pickled models actually expect. This determines which branch is "correct" and which was speculative/never-deployed. **Not resolved in this audit — do this first**, since it gates whether vendored's weather features or canonical's kinematic features get kept in the merged `_predict_with_model`.
   - Port `historical_mean/predict.py`, `ewma/predict.py`, `polyreg_time/predict.py`, `xgb/predict.py` into the inference package (whatever it's namespaced as) before considering `_predict_with_model`'s non-`polyreg_distance` branches reconciled — right now they're live landmines.
7. **`models/polyreg_distance/{model.py,predict.py}` — Moderate.** Verify whether canonical's `models/polyreg_distance/train.py` still contains an inline, non-extracted version of `PolyRegDistanceModel` (not confirmed in this audit) — if so, port vendored's `model.py` extraction pattern back so canonical's training code imports the same class vendored uses for inference, guaranteeing pickle compatibility going forward.
8. **`seed_baseline_model.py` — Easy, but don't skip it.** Port this into the consolidated package; it's the only bootstrap path that doesn't require the full training pipeline and is presumably how CI/fresh-environment registries get seeded.

**Overall recommended order**: registry (4) → spatial.py line-level verification (5) → resolve the polyreg_time/xgboost feature-schema question by reading the missing predict.py files (prerequisite for 6) → estimator.py merge (6) → port remaining predict.py siblings (part of 6) → polyreg_distance model/predict split (7) → seed script (8) → core/utils/temporal cleanup (1–3, can happen anytime, lowest risk).

## What could not be determined in this pass

- **`models/polyreg_time/predict.py` and `models/xgb/predict.py` (canonical) were not read.** This blocks a definitive resolution of the `_predict_with_model` feature-schema conflict — I inferred the conflict from what each side's estimator *passes into* `_predict_with_model`, but did not confirm which schema the actual trained-model code on the canonical side consumes.
- **`feature_engineering/spatial.py`'s `ShapePolyline` and `calculate_distance_features_with_shape` bodies were not diffed line-by-line** between canonical (372 lines) and vendored (250 lines) — only the function/class name list was compared. The 122-line gap is attributed to the two dropped DB-loading functions plus `example_usage()`, but the retained functions' internals could still have diverged.
- **Whether canonical's `models/polyreg_distance/train.py` contains an inline `PolyRegDistanceModel` class or already imports from a shared location** was not verified — needed to confirm whether vendored's `model.py` extraction is a net-new split or already mirrors a canonical refactor.
- **`models/polyreg_distance/predict.py` on canonical was not read** — could not confirm whether the vendored `route_id` parameter and its `ValueError` guard (item 5 in vendored-only work) also exist on canonical, or is itself vendored-only work.
- **No git history / blame was consulted** (task is read-only and time-boxed to a tree comparison) — the audit is a snapshot diff; it does not establish *which side's change happened first* or *why*, only what currently differs.
- **`models/common/data.py`, `models/common/keys.py`, `models/common/metrics.py`** (canonical-only) were inventoried but not opened — assumed training-only based on naming and absence of any reference from vendored's `estimator.py`/`registry.py`, but not exhaustively confirmed.
