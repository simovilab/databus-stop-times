# stoptimes contracts (`stoptimes.core.contracts`)

`CONTRACTS_VERSION = "1.0.0-draft"`

This is Phase 1's deliverable: the frozen interfaces Phase 2's six tracks
(T1 data, T2 arrivals, T3 train, T4 serve, T5 record, T6 val) build against
instead of against each other. Everything here is pydantic v2 models,
`typing.Protocol`s, enums, and small pure functions — no agency names, no
pyarrow/protobuf/xgboost/torch imports. Source lives in
`packages/core/src/stoptimes/core/contracts/`; tests in
`packages/core/tests/test_contract_*.py`.

Read this if you are about to write T1–T6 code, or if you are the user
deciding whether Phase 2 can start.

---

## 1. `source` — feed and agency descriptions

**File:** `contracts/source.py`. **Consumers:** T1 (readers connect to
`realtime_feeds`), T5 (recorder polls them, writes to `archive`), T2/T6
(read `detector` params), T4 (serve reads `timezone`, `gtfs_static_url`).

A `Source` is one agency/feed, described generally enough that bUCR and MBTA
are two values, not two code paths (docs/PLAN.md rule 4).

```
Source
  id: str                 # "mbta" — used as a key everywhere (datasets, artifacts, archive paths)
  name: str
  timezone: str = "UTC"
  holiday_region: str | None   # "CR", "US_MA" (COUNTRY[_SUBDIVISION]); None = no holidays
  gtfs_static_url: str    # URL or local path to the GTFS .zip
  realtime_feeds: list[RealtimeFeed]
  archive: Archive | None
  detector: GroundTruthDetector
  serving: ServingDefaults     # primary_model / fallback_model registry keys serve uses by default
  tags: list[str]

RealtimeFeed
  kind: FeedKind           # vehicle_positions | trip_updates | alerts
  url: str
  format: FeedFormat = json  # pb | json
  poll_interval_s: float = 15.0
  headers: dict[str, str]    # static headers, non-secret
  auth_env: dict[str, str]   # header name -> ENV VAR NAME (never a secret value)

Archive
  kind: ArchiveKind = local  # local | s3
  uri: str                   # local path or s3://bucket/prefix

GroundTruthDetector
  arrival_radius_m: float = 40.0
  dwell_s: float = 0.0
  shape_threshold_m: float = 30.0
```

**Invariants:** at most one `RealtimeFeed` per `FeedKind`; `id` is
`[a-z0-9][a-z0-9_-]*`; extra fields are rejected (`extra="forbid"`) so a typo
in a YAML fails loudly, not silently.

**Agency files today.** `examples/agencies/{mbta,bucr}.yaml` (moved from val by C2) are
flat (`name`, `gtfs_static_url`, `vehicle_positions_url`,
`trip_updates_url`, `poll_interval_s`, `headers`, `arrival_radius_m`,
`dwell_s`, `shape_threshold_m` — today parsed into
`stoptimes.core.domain.models.SourceConfig`). Rather than widen `Source`
with legacy flat fields, `contracts.adapters.source_from_val_yaml(dict) ->
Source` adapts it. The same files carry the agency profile (`timezone`,
`holiday_region`, `serving`) read by `stoptimes.core.config`; the adapter maps those too.
Looked up via `--sources-dir` → `STOPTIMES_SOURCES` → `./examples/agencies`.

**Example** (`examples/agencies/mbta.yaml`, loaded through `source_from_val_yaml`):

```yaml
name: mbta
gtfs_static_url: https://cdn.mbta.com/MBTA_GTFS.zip
vehicle_positions_url: https://cdn.mbta.com/realtime/VehiclePositions.json
trip_updates_url: https://cdn.mbta.com/realtime/TripUpdates.json
poll_interval_s: 15
headers: {}
arrival_radius_m: 40.0
dwell_s: 0.0
shape_threshold_m: 30.0
```

---

## 2. `data` — canonical data tables

**File:** `contracts/data.py`. **Consumers:** T1 (readers produce
`Observation`/`StaticFeed`, validate with `validate_frame`), T2 (produces
`Arrival`), T3 (trains against `Arrival` + `Observation`-derived features),
T5 (writes `Observation` rows using `OBSERVATION_*_PARTITION`), T6 (reads
archived `Observation`s for replay).

### Timezone rule (non-negotiable)

Every timestamp column is `datetime64[ns, UTC]` (`UTC_DTYPE`). Pandas 2.x/3.x
infers coarser units (`us`) from some inputs — `coerce_utc_ns(series)` is the
one place that pins it back to `ns` and treats a naive input as already-UTC
(matching `record.storage.schema.add_partition_columns`'s existing rule).
`validate_frame` fails a column that is timezone-aware UTC but not `ns`
resolution — "close enough" datetime dtypes are treated as a defect, not a
pass, because silent unit mismatches are exactly the bug Phase 0 hit.

### `StaticFeed`

Not a data container — `stoptimes.core.gtfs.static.GtfsStaticLoader` (owned
by T1) holds the actual DataFrames. `StaticFeed` records which tables loaded
so callers can check completeness:

```
StaticFeed(source_id: str, tables: list[str])
  .has(table) -> bool
  .missing_required() -> list[str]

REQUIRED_STATIC_TABLES = ("stops", "trips", "stop_times")
OPTIONAL_STATIC_TABLES = ("shapes", "routes", "calendar", "calendar_dates")
STATIC_TABLE_SCHEMAS: dict[str, TableSchema]   # per-table required columns
```

### `Observation`

One realtime record — a vehicle-position sample, or one stop-time prediction
from a TripUpdate — `kind`-discriminated, source-agnostic:

```
Observation
  kind: vehicle_position | trip_update
  source_id, feed_name, trip_id, route_id, direction_id, start_date, vehicle_id
  ts: UTC timestamp            # VP: the fix; TU: prediction's own timestamp
  ingested_at: UTC timestamp | None
  # vehicle_position: lat, lon, bearing, speed, current_status, current_stop_sequence
  # trip_update: stop_id, stop_sequence, arrival_time, arrival_delay, arrival_uncertainty,
  #              departure_time, departure_delay, departure_uncertainty,
  #              schedule_relationship, trip_schedule_relationship, revenue
```

Column names deliberately mirror `stoptimes.record.storage.schema.DATA_COLUMNS`
(VP) and `.tu_schema.DATA_COLUMNS` (TU) **by name, not by import** — `core`
must not depend on `record` (docs/PLAN.md rule 1: `core ← record`). See
`OBSERVATION_VP_SCHEMA` / `OBSERVATION_TU_SCHEMA` for the exact column list,
and `contracts.adapters.observation_from_vp_row` /
`observation_from_tu_row` for row-level conversion (T5's writer or T1's
reader calls these, not the reverse).

Partition layouts (Hive-style, agency/bucket agnostic — no "mbta"/"transit"
baked in, unlike today's `record.storage.schema.BASE_PREFIX`):

```
OBSERVATION_VP_PARTITION: year={year}/month={month}/day={day}/route_id={route_id}
OBSERVATION_TU_PARTITION: year={year}/month={month}/day={day}
```

### `Arrival`

One actual (derived or reported) stop arrival. Keyed by (trip, service day,
`stop_sequence`) — **not** by `stop_id`: a loop route visits the same stop
twice in one trip and both visits are real, independent evidence.

```
Arrival
  source_id, trip_id, start_date, route_id, vehicle_id, stop_id, stop_sequence
  arrival_time: UTC timestamp
  method: reported | map_matched | interpolated
  confidence: float | None      # [0, 1], derived methods only
```

Backed today by `stoptimes.core.domain.models.ActualArrival` (epoch-seconds
float, `trip_key` instead of separate identifiers); adapt with
`contracts.adapters.arrival_from_actual_arrival`.

### `validate_frame(df, schema) -> list[str]`

Returns human-readable problems: missing required columns, wrong dtype
(string/int64/float64/bool), or a datetime column that isn't exactly
`datetime64[ns, UTC]`. Empty list = passes.

---

### Does training require GTFS?

No — training consumes an `Arrival` table plus features, so
`data.datasets` can point at a prepared dataset from any source (an AVL CSV,
an existing arrivals export) with no GTFS zip present. GTFS static is required
for the parts that genuinely need a schedule: deriving arrivals by
map-matching (stop locations, shapes, stop sequences), schedule-based features
and baselines, and emitting GTFS-RT TripUpdates, which are addressed by
`trip_id`/`stop_id` and so cannot exist without a feed the consumer also has.

## 3. `experiment` — experiment.yaml

**File:** `contracts/experiment.py`. **Consumers:** T3 (`train run
experiment.yaml`, feature registry lookups), U1 (TUI writes this file and
nothing else — docs/PLAN.md rule 2).

```
Experiment
  name: str
  data: DataSelection        # sources/datasets, date range, routes
  target: str = "arrival_delay_s"
  model: ModelSpec           # family + params (scalar or search space)
  features: list[str]        # names resolved against core.features.registry (T3)
  split: SplitStrategy       # time | trip | kfold
  seed: int = 42
  outputs: OutputSpec        # registry_dir, tags
  baselines: list[str] = ["schedule", "constant_speed"]   # always included (rule 5)

DataSelection: sources, datasets, start_date, end_date, routes  (>= 1 of sources/datasets)
ModelSpec: family: str (free-form, not an enum), params: dict[str, ParamValue]
ParamValue = float | int | str | bool | ParamRange | ParamChoices
SplitStrategy: kind=time (needs train_fraction+test_fraction) | trip (same) | kfold (needs n_folds)
```

`load_experiment(path)` / `dump_experiment(exp, path)` round-trip losslessly
through YAML (verified in `test_contract_experiment.py::test_round_trip_yaml`).

**Design choice:** one `Experiment` trains one model spec. Today's
`train_all_models.py --models a b c` sweep becomes a loop over several
`Experiment`s (or several `model` values in a wrapper the TUI drives) rather
than a `models: list[ModelSpec]` field — keeps "what did this specific
artifact come from" unambiguous, one experiment file per artifact.

**Examples:** `examples/experiments/xgb-baseline.yaml`,
`examples/experiments/historical-mean-kfold.yaml` (both validated by
`test_contract_experiment.py::test_shipped_examples_validate`).

---

## 4. `model` — the model interface

**File:** `contracts/model.py`. **Consumers:** T3 (families implement
`Model`), T4 (serve calls `predict`, loads via `Model.load` or
`ModelLoader`).

```python
class Model(Protocol):          # runtime_checkable
    family: str
    required_features: list[str]
    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, directory: Path) -> ModelFormat: ...
    @classmethod
    def load(cls, directory: Path) -> "Model": ...
```

Existing families (`stoptimes.models.{historical_mean,polyreg_distance,
polyreg_time,xgb}.predict.predict_eta`) are **functions**, not objects, and
take flat keyword args rather than a DataFrame. They satisfy `Model` through
a thin per-family adapter (T3's job — not written here; a fake demonstrating
the shape passes in `test_contract_model.py`), not a rewrite.
`stoptimes.serve.estimator` picks a family by string (`_predict_with_model`)
today; T4's pipeline `predict` stage becomes "resolve `family` to a `Model`,
call `.predict`" instead of the current if/elif chain.

Bring-your-own models don't implement `Model` directly:

```
ModelRef
  kind: pickle | onnx | callable | http
  location: str        # path, "module:function", or URL
  family: str = "byo"
  required_features: list[str]
  headers, auth_env: dict[str, str]   # http only; auth_env mirrors Source.RealtimeFeed
  timeout_s: float = 5.0

ModelLoader.load(ref: ModelRef) -> Model     # T4 implements per-kind loaders
```

`stoptimes.core` never imports `pickle`-unsafe code, `onnxruntime`, or an
HTTP client for a `ModelRef` — only the matching loader (in `serve`) does.

---

## 5. `pipeline` — serving stages

**File:** `contracts/pipeline.py`. **Consumers:** T4 (rebuilds `serve` on
this), T6 (`pipeline:<file.py>` predictor loads a user pipeline).

```
STAGE_ORDER = ("ingest", "match", "features", "predict", "postprocess", "publish")

PipelineContext(source, static_feed, model, now: datetime, config: dict, logger, state: dict)
StageHook = Callable[[payload, PipelineContext], payload]

class Pipeline(Protocol):
    def stage(self, name: str) -> Callable[[StageHook], StageHook]: ...   # @pipeline.stage("predict")
    def run(self, ctx: PipelineContext) -> Any: ...
```

Each stage's payload type (documented in the module, not enforced by the
type system — stages are user-substitutable):

| stage | in | out |
|---|---|---|
| ingest | — | `list[Observation]` |
| match | `list[Observation]` | `list[MatchedObservation]` (T2 shape) |
| features | matched observations | `pandas.DataFrame` |
| predict | features DataFrame | `list[StopTimePrediction]` |
| postprocess | predictions | `list[StopTimePrediction]` |
| publish | predictions | `None` |

`ctx.now` is the pipeline's logical clock — stages read time through it, not
`datetime.now()`, so T6's replay is deterministic.

`SequentialPipeline` is a reference implementation (no retries, no async, no
sinks — T4 owns those) proving the contract composes:
`test_contract_pipeline.py::test_hooks_compose_in_order_and_share_context_state`
registers all six stages and confirms `run()` threads payloads through them
in order and `ctx.state` is shared and mutable across stages.

---

## 6. `prediction` — serving output

**File:** `contracts/prediction.py`. **Consumers:** T4 (`predict`/
`postprocess`/`publish` stages produce and consume these), T6 (scores any
source's predictions against these fields via the `Predictor` interface —
docs/PLAN.md rule 3).

```
StopTimePrediction
  trip_id, route_id, direction_id, start_date, stop_id, stop_sequence, vehicle_id
  predicted_arrival_time, predicted_departure_time: datetime | None   # UTC required
  delay_s, uncertainty_s, horizon_s: float | None
  schedule_relationship: SCHEDULED | SKIPPED | NO_DATA | UNSCHEDULED
  model_id: str
  artifact_id: str | None
  generated_at: datetime   # UTC required — ctx.now when the pipeline ran

TripUpdatePrediction
  trip_id, route_id, direction_id, start_date, start_time, vehicle_id
  schedule_relationship: SCHEDULED | ADDED | UNSCHEDULED | CANCELED | DUPLICATED | DELETED
  stop_time_updates: list[StopTimePrediction]   # >= 1, all matching this trip_id/start_date
  generated_at: datetime
```

Both render themselves as **plain dicts shaped exactly like the GTFS-RT
TripUpdate/StopTimeUpdate JSON message** — the same snake_case field names
`stoptimes.core.realtime.gtfsrt_json.parse_tu_feed` already reads, just
written instead of parsed:

```python
StopTimePrediction(...).to_stop_time_update()
# {"stop_sequence": 3, "stop_id": "s1",
#  "arrival": {"time": 1789646400, "delay": 45, "uncertainty": 30}}

TripUpdatePrediction(...).to_trip_update_dict()
# {"trip": {"trip_id": "t1", "start_date": "20260917", "route_id": "r1", ...},
#  "vehicle": {"id": "v1"},
#  "stop_time_update": [...],
#  "timestamp": 1789646400}
```

No protobuf dependency — a caller that needs protobuf wraps this dict
(`google.protobuf.json_format.ParseDict` or hand assembly), which is out of
scope for `core`. `uncertainty_s` follows the GTFS-RT spec loosely: it's
"expected error in seconds," and the spec leaves the exact statistic to the
producer — document your model's choice (e.g. MAE vs. a quantile width) in
its model card.

`stoptimes.core.domain.models.PredictionSample` (epoch-seconds float,
`trip_key`-addressed) adapts via
`contracts.adapters.stop_time_prediction_from_sample`.

---

## 7. `artifact` — model artifacts on disk

**File:** `contracts/artifact.py`. Fixed by the orchestrator before Phase 1
fan-out so C1 and C3 (`stoptimes models migrate`) build on the same layout;
unchanged by C1 except this documentation pass. See the module docstring for
the full shape (`ArtifactMetadata`, `ModelFormat`, `DataWindow`,
`Provenance`, `HorizonScore`) and `artifact_dir`/`read_metadata`/
`write_metadata`/`version_mismatches`.

```
<registry>/<artifact_id>/
  artifact.json   ArtifactMetadata — schema_version, artifact_id, family, target,
                  format, features, params, data (DataWindow), provenance,
                  scope (ModelScope — see §7b), metrics, scores_by_horizon,
                  tags, legacy (C3 migration only)
  model.<ext>     the fitted model (ext from ModelFormat)
  card.md         human-readable model card (optional)
```

C1 added no fields — every field a Phase 2 track needs was already present.

### Loading refuses on load-critical version drift (U7)

**File:** `serve/model_loading.py` (`load_artifact_model`, `check_version_drift`).
`version_mismatches(meta, installed)` (above) has existed since C1 but nothing
called it until U7 — a stale artifact was loaded silently, only ever warned
about. This is the real failure `version_mismatches`' own docstring names: a
model pickled under different library versions than the deploy target runs
gets deserialized with those library's classes anyway, silently.

`load_artifact_model` now calls `check_version_drift(meta)` before opening
anything:

- **Which libraries are load-critical is per `ModelFormat`**
  (`LOAD_CRITICAL_LIBS`): `pickle` → `{sklearn, xgboost}` (either may own the
  pickled class depending on family); `xgboost-json` → `{xgboost}`; `onnx` →
  `{onnxruntime}`; `torchscript` → `{torch}`. A drift in anything else
  currently tracked in `provenance.lib_versions` (`numpy`, `pandas`) — or
  anything not tracked at all — is **informational**: logged as a warning,
  never a refusal. This is what keeps a matplotlib or numpy point-release
  from blocking a model that doesn't care.
- **No recorded `lib_versions` → always loads.** Absence of information is
  not evidence of drift; every pre-Phase-1 artifact keeps working.
- **A load-critical drift raises `ArtifactVersionDriftError`**, naming the
  library, the built vs. installed versions, and what to do: re-export the
  artifact under the versions installed on the target, or override.
- **The override** is `STOPTIMES_ALLOW_VERSION_DRIFT` — an environment
  variable, not a config default, so it's opt-in per-deploy: `all`/`1`/
  `true`/`*` overrides every artifact, or a comma-separated list of artifact
  ids overrides only those. `load_artifact_model(..., allow_version_drift=True)`
  is the equivalent explicit code-level override for a caller (e.g. an
  offline re-scoring tool) that wants it hardcoded. Every override is logged
  at `WARNING`, naming the artifact and the exact drift — it is never silent
  and never the default.

This directly targets the situation SIMOVI has today, generically: a serving
host running different `scikit-learn`/`xgboost` builds than the host a model
was trained on. Before U7 that produced a warning and kept going; now it
refuses unless the operator says otherwise.

---

## 7b. `selection` — which artifact serves a trip

**File:** `contracts/selection.py`. **Consumers:** T4 (serving picks a model per
trip), T3 (reports say which artifact would be chosen), T6 (validation scores
the same choice serving would make).

A registry holds many artifacts for one agency: a global model, per-route
models, sometimes per-trip ones. `ArtifactMetadata.scope` says what an artifact
may serve, and `select_artifact` picks one:

```
ModelScope
  level: global | route | direction | trip
  source_id: str | None          # None = serves any agency
  route_ids: list[str]; direction_id: int | None; trip_ids: list[str]
  .matches(source_id=, route_id=, direction_id=, trip_id=) -> bool

SelectionPolicy                   # on Source.serving, overridable per call
  metric: str = "test_mae_seconds"    # key in ArtifactMetadata.metrics
  lower_is_better: bool = True
  scope_order: [trip, direction, route, global]
  families: list[str]                 # restrict to model families; empty = any
  pins: dict[str, str]                # "global" | "route:22" | "route:22/1" | "trip:t1" -> artifact_id

select_artifact(candidates, policy, *, source_id, route_id=, direction_id=, trip_id=) -> ArtifactMetadata | None
```

**Rule:** explicit pins first (a pin naming a missing artifact is ignored, so a
deleted model degrades to ranking rather than to no predictions); then the most
specific scope in `scope_order` that has a candidate; within that level, the
best `metric`, ties broken by `artifact_id` so the choice is deterministic.
Artifacts lacking the metric are never selected, and `None` is a real answer —
the caller decides between skipping the trip and falling back to a baseline.

Ranking reads `artifact.json` only; no model file is opened to choose.
This replaces `registry.get_best_model`'s metric-only ranking (which had no
agency awareness — the reason bUCR needed a hard pin) and the route/global
if-else chain in `serve.estimator`. `ServingDefaults.primary_model` /
`fallback_model` stay as shorthand pins for the existing deployments.

`stoptimes models migrate` maps a legacy `route_id` onto a route scope, so
migrated per-route models keep serving only their route.

### `models list --json` shape

**File:** `cli/commands/models.py::models_list`. Reads every
`<registry>/<artifact_id>/artifact.json` (skipping directories without one)
and, with `--json`, prints a JSON array — one object per artifact, filtered
by any of `--source`/`--route`/`--family`:

```json
[
  {
    "artifact_id": "xgboost_mbta_20260818_120020",
    "family": "xgboost",
    "scope": "route",                  // ModelScope.level.value
    "routes": ["22"],                  // ModelScope.route_ids ([] for global/trip scope)
    "metric": 31.5,                    // metrics["test_mae_seconds"], or null if absent
    "created": "2026-08-18T12:00:20"   // provenance.created_at, ISO 8601
  }
]
```

One bad `artifact.json` (fails to parse) is skipped with a warning on
stderr, not a hard failure of the whole listing — an operator auditing a
registry with one corrupt artifact still sees the rest.

## 7c. Promote / rollback (U7)

**Files:** `cli/commands/models.py` (`promote`, `rollback`, `pins`),
`models/common/promotion.py` (the history sidecar).

**The problem this solves:** `models promote` existed before U7 but its
behavior — what it writes, where, and how to undo it — was unspecified and
untested. "Production must always have a way back" (docs/PLAN.md §8) needs
promote and rollback to both exist and be exercised by tests, not just the
promote half.

**The design — smallest honest thing, not a deployment system:**

- **The pointer** an operator promotes into is unchanged: `SelectionPolicy.pins`
  (§7b) inside `serving.selection.pins` of the agency's profile YAML
  (`<sources-dir>/<source>.yaml`, default `./examples/agencies`). This is
  what `resolve_artifact`/`select_artifact` already read at serve time — no
  new file for serving to consult, no new schema.
- **The history** is a sidecar JSON file next to the profile, named by
  suffix: `<source>.yaml` → `<source>.promotions.json`. It is *not* folded
  into the profile YAML's `serving`/`selection` keys, because those are
  parsed into `ServingDefaults`/`SelectionPolicy` (`contracts/source.py`),
  both `extra="forbid"` — adding an unrecognized key there would break every
  other reader of that file (`source_from_val_yaml`), not just this track's
  own commands. Shape:

  ```json
  {
    "pins": {
      "global": [
        {"artifact_id": "m1", "at": "2026-09-10T12:00:00+00:00"},
        {"artifact_id": "m2", "at": "2026-09-18T09:00:00+00:00"}
      ],
      "route:22": [ ... ]
    }
  }
  ```

  Each pin key (same key space as `SelectionPolicy.pins`: `global`,
  `route:<id>`, `route:<id>/<direction>`, `trip:<id>`) has its own
  chronological list; the last entry is what's currently live for that pin.

- **`models promote <artifact_id> --as <pin> --source <id>`**: refuses if
  the artifact isn't in `--registry` (as before); on success, writes the pin
  into the profile *and* appends `{artifact_id, at}` to that pin's history —
  this is what makes rollback possible. `--json` reports
  `promotions_for_pin` (the new history depth for that pin) alongside the
  existing fields.
- **`models rollback --as <pin> --source <id>`**: pops the current history
  entry for `pin` and restores the pointer to whatever is now on top —
  i.e. what was live immediately before the promotion being undone. Refuses
  with a clear "nothing to roll back to" message (exit 1) if fewer than two
  promotions are on record for that pin; it never guesses or clears the pin.
  If the restored artifact id no longer exists in `--registry` (deleted
  since), it still restores the pointer and warns on stderr rather than
  refusing — a pin naming a missing artifact degrades to metric-ranked
  selection at serve time (§7b's own rule), so this is a safe, honest
  outcome, not a broken one. Rolling back and then promoting again is how
  you move forward again; there is no separate "redo".
- **`models pins --source <id>`**: shows what is live — every pin currently
  set in the profile, its artifact id, and how many promotions are on record
  (so an operator can tell at a glance whether `rollback` has somewhere to
  go). `--json`: `[{"pin": str, "artifact_id": str | null, "promotions": int}, ...]`.

Tests: `packages/models/tests/test_promotion.py` (history bookkeeping in
isolation) and `packages/cli/tests/test_cli_models_promote.py` (the CLI
commands end to end, including the missing-artifact warning and the
nothing-to-roll-back-to refusal).

## 8. `adapters` — bridging existing code to these contracts

**File:** `contracts/adapters.py`. Pure functions, no side effects, no
behavior change to the code they read from. All currently implemented:

| function | from | to |
|---|---|---|
| `source_from_val_yaml` | val source YAML dict | `Source` |
| `source_from_archive_uri` | `Source` + a uri string | `Source` (copy, `archive` set) |
| `observation_from_vp_row` | record VP archive row (mapping) | `Observation` |
| `observation_from_tu_row` | record TU archive row (mapping) | `Observation` |
| `arrival_from_actual_arrival` | `domain.models.ActualArrival` | `Arrival` |
| `stop_time_prediction_from_sample` | `domain.models.PredictionSample` | `StopTimePrediction` |

Not here (owned elsewhere): legacy `_meta.json` dict → `ArtifactMetadata`
(C3, `stoptimes models migrate`).

**Dependency direction check:** `adapters.py` imports
`stoptimes.core.domain` (same package, fine) and `stoptimes.core.contracts.*`
only. It does **not** import `stoptimes.record`, `stoptimes.val`,
`stoptimes.serve`, `stoptimes.models`, or `stoptimes.train` — `core` is
upstream of all of them (docs/PLAN.md rule 1). Row-shaped adapters
(`observation_from_{vp,tu}_row`) take a plain `Mapping`, documented to match
`record`'s column names, rather than importing `record.storage.schema`.

---

## Open questions for sign-off

Each has a recommendation; approving this document means accepting the recommendations
unless you say otherwise. Resolved at integration: the static table set now includes
`calendar`/`calendar_dates`; agency files live in `examples/agencies/` and carry the profile.

1. **What `uncertainty_s` means.** GTFS-RT leaves the statistic open.
   *Recommendation:* standardize it — the model's held-out mean absolute error, in seconds,
   for the prediction's horizon bucket (taken from `scores_by_horizon`). Val can then check it
   is calibrated, and every family reports the same thing.

2. **One experiment = one model.** Alternative: an experiment lists several models.
   *Recommendation:* keep one model per experiment file (clear provenance per artifact).
   Sweeps and "compare A, B, C" become a separate `study` file in Phase 3 that expands into experiments.

3. **`Observation.schedule_relationship` as free text.**
   *Recommendation:* keep the raw string in the archive (never lose upstream data), and have
   readers expose a normalized enum column next to it.

4. **Fourth arrival method.** *Recommendation:* add `schedule_fallback` now, but never use it as
   ground truth (val and training filter it out). Cheap now, a contract change later.

5. **BYO model family defaults to `"byo"`.** *Recommendation:* require a caller-chosen `name`
   instead, so two custom models don't merge in reports.

6. **Timestamps are `datetime64[ns, UTC]` everywhere; local time only inside feature code.**
   *Recommendation:* confirm (already implemented).
