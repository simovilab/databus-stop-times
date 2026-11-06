# MLflow integration path

Status: **revised 2026-11-06, see §0, which overrides the rest where they differ** · first written 2026-09-18.
For how databus would consume MLflow-backed predictions, see [DATABUS_INTEGRATION.md](DATABUS_INTEGRATION.md) (O3, model source).

---

## 0. Revision 2026-11-06: ml.simovi.org is SIMOVI's model registry

### 0.1 What changed since the first version

- **There is a server.** MLflow runs at `https://ml.simovi.org` behind MLflow's built-in basic auth
  (`/health` returns 200; every `/api/2.0/mlflow/*` call and `/version` return 401 without credentials).
  So MLflow is no longer an optional backend kept only for comparison. **For SIMOVI it is the registry.**
  The filesystem registry stays the package default, so a user without an MLflow server (the
  agency-agnostic goal in PLAN §8.1) loses nothing. It is also the local cache for the MLflow backend.
- **An artifact is now several files, and serving keeps state.** Since F4 (PLAN §8.1), a servable model is
  `artifact.json` (schema v2, `timezone`, `holiday_region`, input tags), plus the model file, plus asset
  sidecars (segment statistics, route vocabulary) with a sha256 each. Its live features need per-trip state
  built from successive pings. A pyfunc that takes a feature frame cannot reproduce our numbers on its own,
  because the features are computed by the live session, not by the caller.

### 0.2 Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **The unit stored in MLflow is a whole stoptimes artifact directory.** It is logged as a pyfunc model whose `artifacts` are the directory, so nothing is lost and MLflow-native tools still see a model. There is one registered model per agency, `stoptimes-<source_id>`, and one model version per artifact. | MLflow handles versions, aliases, UI and audit. The directory keeps our contract intact. |
| D2 | **stoptimes never runs the pyfunc.** Loading from MLflow downloads the version's directory into a local cache (`~/.cache/stoptimes/mlflow/<name>/<version>/`, immutable per version) and then takes the existing filesystem path: `read_metadata` → asset sha256 check → U7 drift refusal → live session. | All the guarantees we already test still apply unchanged. MLflow's own dependency handling only warns on a mismatch, so it can't replace the U7 refusal. |
| D3 | **Version tags carry what selection needs**: `stoptimes.artifact_id`, `family`, `target`, `scope.level`, `scope.keys`, `inputs` (live/clock/…), `schema_version`, `timezone`, `metric.val_mae_s`, `data.test_window`. `list()` reads tags only and downloads nothing. | `SelectionPolicy` ranks candidates without fetching 35 MB per version. |
| D4 | **Aliases are the promotion mechanism.** `@production` is what the databus prediction service serves; `@staging` is what runs in shadow. Pins such as `@route-22` follow the reversible encoding in I2. `stoptimes models promote` moves an alias. It can optionally require a passing `val` result logged for that version (D6) first. Rollback moves the alias back. | MLflow records every alias change, so promotion is auditable without a databus deploy. |
| D5 | **The backend is chosen by URI.** `STOPTIMES_REGISTRY=mlflow+https://ml.simovi.org`, or a plain path for the filesystem registry. `STOPTIMES_MODEL_REGISTRY_DIR` keeps working. | One switch, used the same way by train, serve, val and the UI. |
| D6 | **Tracking.** `train run` logs a run in experiment `stoptimes/<source_id>` with params, metrics, per-horizon scores, the card and baseline child runs, and the model version links to that run. `val` runs (CLI and UI) log to `stoptimes-val/<source_id>`, tagged with the model version, and include the baseline MAEs (for example MBTA `gtfsrt`). | The MLflow UI then answers "does v7 beat MBTA live, and since when?" |
| D7 | **Credentials** come only from MLflow's standard `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` environment variables. They never go in YAML and are never logged. Use two accounts: a **trainer** that can write experiments and register models, and a **server** with READ on `stoptimes-*` registered models only. | MLflow basic auth supports per-experiment and per-registered-model permissions, so a leaked serving credential cannot change models. |
| D8 | **Serve hot-swaps on alias change.** The service re-resolves `@production` every few minutes. When the version changes it downloads, checks and loads the new one, then swaps model and assets atomically. Trip state is kept, because it is pings and arrivals, which don't depend on the model. If loading fails, it keeps serving the old version and logs an event. | Promote and rollback reach production with no restart and no databus deploy. |
| D9 | **The validator lists MLflow models.** `/api/predictors` adds `mlflow:<name>@<alias>` and `mlflow:<name>/<version>` next to `stoptimes:<artifact_id>`, built through the same stateful artifact predictor. | This is the UI picker change; it is the same mechanism for both backends. |

What stays from §§1–9: ground rules (§2: optional extra, lazy imports, `mlflow-skinny` for serve), I2's
`RegistryBackend` refactor, I4 and I6 (bring-your-own MLflow models), §7 (don't replace serve, val or
`experiment.yaml`). §6's comparison becomes secondary, because the decision has already been made for SIMOVI.

### 0.3 How databus fits

```
                 ┌──────────────────────── ml.simovi.org (MLflow) ───────────────────────┐
                 │ experiments stoptimes/bucr, stoptimes-val/bucr                         │
                 │ registered model stoptimes-bucr: v1 … vN, aliases @production @staging │
                 └──────▲ log + register ───────▲ log val runs ────────│ resolve alias, ──┘
                        │                       │                     │ download version
  stoptimes train ──────┘          stoptimes val (UI/CLI) ──┐          ▼
        ▲                                                   │   prediction service (stoptimes serve
        │ archive                                           │   + databus pipeline file, O3)
  stoptimes record ◀── VehiclePositions + GTFS ──┐          │          │ read            │ write EX 60
                                                 │          │          ▼                 ▼
  databus: realtime-engine ──▶ Redis (runs, positions, stop status)   run:<id>:stop_time_updates
           schedule-engine ──▶ /feed/{vehicle_positions,trip_updates}.pb ◀── reads ───────┘
```

- **databus never talks to MLflow.** O4 stays rejected. MLflow is a model source for the prediction service
  (O3) only, and databus can't see it. Because the server is external, databus's compose file needs no
  `mlflow` service (this supersedes DATABUS_INTEGRATION §5.3). The prediction service needs outbound HTTPS to
  `ml.simovi.org` and the read-only credential.
- **databus is also the data source.** `stoptimes record` archives databus's own VehiclePositions and
  GTFS feed, and that archive is what `train` learns from. Every trained version therefore comes from the
  feed it will serve.
- **databus's published TripUpdates become a baseline**, just as MBTA's are. `val` can score
  `mlflow:stoptimes-bucr@staging` against databus's own `trip_updates.pb`, which comes from `@production`,
  over the same trips. That is the shadow comparison of G2b, logged in MLflow.
- **The live model can't use G2a.** G2a swaps databus's in-process, stateless `estimate_stop_times` call.
  The 19-feature model needs per-trip state across pings, which only the stateful live session (O3) keeps.
  G2a can still carry the older stateless models as a reversible first step, but the model that beats the
  baseline only reaches databus through O3.

**Server facts (read-only probe, 2026-11-06):** MLflow **3.17.0**. Artifacts are **proxied** by the server
(`mlflow-artifacts:/<id>`), so clients need no bucket credentials. The probing account is not an admin. It
can see one experiment (`test-jae`), gets 403 on the Default experiment, and sees no registered models.
Calls are **slow, 6–21 s each** (even an empty registered-model search takes about 20 s). So `list()` and
`/api/predictors` need a TTL cache, alias polling must not block serving, and a cold download of a version
has to happen before the swap, never on the predict path.

### 0.4 Order of work (M-tracks)

| Track | Scope | Gate |
|---|---|---|
| **M0** (user) | Server facts: credentials for the two accounts, MLflow server version, artifact store (proxied `mlflow-artifacts:` or direct S3, which decides whether clients also need bucket credentials), the naming in D1 | — |
| **M1** | `RegistryBackend` protocol and `FilesystemRegistry`. Move every caller behind it: `serve.model_loading`, `val.predictors.artifact`, `train` artifact write, `models list/card/promote`, `/api/predictors`. Filesystem only. | All existing tests pass unchanged |
| **M2** | `MlflowRegistry` (D1–D5): write, tag-only list, cached download, aliases, promote and rollback. A shared contract test suite runs against both backends, and the MLflow side uses a local `sqlite:///` store in `tmp_path`, with no network. | Contract suite green on both |
| | **Status 2026-11-06:** M1 done (`stoptimes.core.registry`: `RegistryBackend`, `FilesystemRegistry`, `open_registry`). M2 done (`MlflowRegistry` in `core/registry/mlflow_backend.py`, extra `stoptimes-core[mlflow]`). The real xgb-live round-tripped through a local store (published in 5 s, metadata identical, model and assets load). A read-only listing of ml.simovi.org works with `~/.mlflow/credentials` (about 21 s uncached). Finding: xgb-live has no `scope.source_id`, so it would publish to `stoptimes-global`, not `stoptimes-mbta`; the trainer should set it from the experiment's source. Implementation notes: `search_model_versions` returns no aliases (they're read from the registered models); resolving `models:/…` partly uses mlflow's process-global URIs, which `_bound_uris` sets under a lock while publishing or downloading. | |
| **M3** | Tracking (D6): train run and val run logging, with best-effort failure handling (a warning, never a failed run) | Tested offline against a `file:` store |
| **M4** | `mlflow:` predictor IDs in val and the API/UI (D9) | xgb-live picked from the UI by alias |
| **M5** | Serve alias polling and hot swap (D8), plus the `serve` Docker image (G2 prerequisite) | A swap during replay loses no trip state |
| **First real use** | Register xgb-live as `stoptimes-mbta` v1 on `ml.simovi.org` with `@staging`, then run the validator from the UI on `mlflow:stoptimes-mbta@staging` vs `gtfsrt` | Needs the user's go: it is the first write to the server |

This document says where MLflow can stand in for something stoptimes already does, where it
can't, and how to wire each option in as an **optional** extra. The filesystem registry,
`experiment.yaml`, the serving pipeline and `val` stay the defaults. MLflow is a second
implementation behind the same seams, which also gives us a baseline to compare our own
implementation against (§6).

---

## 1. Stance

stoptimes is mostly domain work MLflow doesn't do: GTFS/GTFS-RT readers, recording,
map-matching raw vehicle positions into ground-truth arrivals, feature parity between
training and serving, spec-valid TripUpdates, and horizon-based validation against
published feeds. MLflow assumes you already have a feature table with a label column.

It overlaps with the **model lifecycle**: tracking runs, versioning artifacts, promoting and
rolling back, and packaging a model so something else can call it. That's where it fits.

## 2. Ground rules

1. **Optional only.** A `mlflow` extra; nothing in `core` imports MLflow; every `import mlflow`
   is lazy, inside the function that needs it, with an install hint when it's missing (same
   pattern U2 uses for the other extras).
2. **Serving stays light.** `pip install 'stoptimes[serve]'` must not pull MLflow. Where
   serving needs it (I4), depend on `mlflow-skinny`, behind its own extra.
3. **Contracts first.** `stoptimes.core.contracts` stays the source of truth. MLflow objects
   are mapped to and from `ArtifactMetadata`, `ModelRef` and `Model`. They never replace
   them. Any contract change (only I4 needs one) goes through an orchestrator-approved note,
   as PLAN.md §5 requires after sign-off.
4. **Design rule 3 still holds.** `val` scores through the `Predictor` interface only. MLflow
   is never a source of ground truth.
5. **Our guarantees don't weaken.** U7's refusal to load a drifted artifact applies to
   MLflow-backed artifacts too. MLflow on its own only *warns* on dependency mismatch.

## 3. What maps to what

| stoptimes today | MLflow counterpart | Replaceable? | Option |
|---|---|---|---|
| `train run` writes metrics, horizon scores and a card into `artifact.json` / `card.md` | Tracking: runs, params, metrics, artifacts, comparison UI | **Complement.** Log alongside the artifact | I1 |
| Filesystem registry `<registry>/<artifact_id>/{artifact.json, model.<ext>, card.md}` | Model Registry: registered models, versions, tags, descriptions | **Yes, as an alternative backend** | I2 |
| U7 `models promote` / rollback, `SelectionPolicy.pins` | Model aliases (`models:/<name>@<alias>`), mutable and auditable | **Yes, for pins and promotion** | I2 |
| U7 drift refusal (`version_mismatches`) | pyfunc warns on mismatch; `mlflow.pyfunc.get_model_dependencies()` | **No.** MLflow warns and we refuse; keep ours and feed it MLflow's recorded deps | I2 |
| `Model` families saved as pickle/ONNX | pyfunc / `PythonModel` packaging with a signature and environment | **Complement.** Export to it and import from it | I3, I4 |
| BYO loaders (`ModelRefKind` pickle/onnx/callable/http) | `mlflow.pyfunc.load_model("models:/…")` | **Adds a loader kind** | I4 |
| `HttpModel` (`{"rows": …}` → `{"predictions": …}`) | `mlflow models serve` `/invocations` (`{"dataframe_records": …}` → `{"predictions": …}`) | **Small change.** Different request shape | I6 |
| `val` predictors (`stoptimes:`, `pipeline:` prefixes) | none | **Adds a predictor prefix** | I5 |
| `val` reports | Tracking runs | **Complement.** Log val results | I5 |
| `experiment.yaml` + TUI | MLflow Projects | **No.** See §7 | — |
| `serve` pipeline and publisher | `mlflow models serve` | **No.** See §7 | — |
| `val` scoring | `mlflow.models.evaluate` | **No.** See §7 | — |
| Record, Data, Arrivals, Diagnose | none | **No** | — |

---

## 4. Integration options

Each option lists what it replaces or adds, where it plugs in, the design, and what stays ours.
They're independent except I5 (needs I4) and I2's promote path (needs U7 done).

### I1 · Tracking sink for `train run`

**Adds:** an MLflow run per experiment, so modelers get MLflow's run comparison UI next to
(not instead of) the artifact and card.

**Plugs in:** `stoptimes.train.experiment.run_experiment`, after `_write_artifact`. Configured
by an optional block in `experiment.yaml`, which is an optional field on `OutputSpec`. Adding an
optional field is allowed after sign-off, but still file the note.

```yaml
outputs:
  registry_dir: artifacts/models
  tracking:                      # optional; absent = no MLflow
    backend: mlflow
    tracking_uri: file:./mlruns  # or http://…; env STOPTIMES_MLFLOW_TRACKING_URI overrides
    experiment: bucr-route-models
```

**Design:** a small `stoptimes.train.tracking` module with a `TrackingSink` protocol
(`log_run(experiment, meta, card_path, baseline_metrics)`) and one `MlflowTrackingSink`.
The mapping from `ArtifactMetadata`:

| ArtifactMetadata | MLflow |
|---|---|
| `experiment.name`, `artifact_id` | run name; tag `stoptimes.artifact_id` |
| `family`, `target`, `scope`, `features`, `data.*`, `provenance.git_sha`, `seed` | tags / params |
| `params` | params (flattened; ranges from `ParamRange`/`ParamChoices` as child runs if tuning lands) |
| `metrics` | metrics |
| `scores_by_horizon` | metrics named `mae_s.h<min>_<max>` (plus rmse/p90) and a CSV artifact |
| baselines (rule 5) | child runs under the model's run, so "better than the schedule?" is one UI view |
| `card.md`, `artifact.json` | run artifacts |

**Stays ours:** the artifact on disk is still written first and is still what `serve` and
`val` load. The run points to it through a tag, not the other way round. If MLflow is
unreachable, the run is logged as a warning and training still succeeds.

**Effort:** small (~200 lines + tests with a `file:` tracking URI in `tmp_path`).

### I2 · MLflow Model Registry as an alternative registry backend

**Replaces:** the filesystem registry, including U7's promote/rollback, when the user picks it.

**Plugs in:** after Phase 3 there are several callers of the filesystem layout:
`serve.model_loading.iter_registry_artifacts` / `resolve_artifact` / `load_artifact_model`,
`val.predictors.artifact.load_registry_candidates`, `train.experiment._write_artifact`, and the
`models list/card/promote/migrate` commands. Step one is to put them behind one interface. It's
worth doing even without MLflow, because it's the refactor that makes a second backend possible:

```python
class RegistryBackend(Protocol):
    def list(self) -> list[ArtifactMetadata]: ...
    def get(self, artifact_id: str) -> ArtifactMetadata: ...
    def load_model(self, meta: ArtifactMetadata) -> Model: ...        # U7 drift check happens here
    def write(self, meta: ArtifactMetadata, model_dir: Path, card: Path) -> None: ...
    def promote(self, pin_key: str, artifact_id: str) -> str | None: ...  # returns previous, for rollback
    def pins(self) -> dict[str, str]: ...
```

`FilesystemRegistry` is today's behaviour; `MlflowRegistry` is the new one. Choosing:
`STOPTIMES_REGISTRY=mlflow+http://host:5000` (or `mlflow+file:./mlruns`), plain paths meaning
filesystem as today. The `registry_dir` field and `STOPTIMES_MODEL_REGISTRY_DIR` keep working.

**Mapping:**

- **One registered model per agency** (`stoptimes.<source_id>`), one **model version per
  artifact**. Scope, family, target and features go in version tags so `SelectionPolicy` can rank
  candidates without downloading models. `artifact.json` is stored with the version and is what
  `list()` returns. The ranking itself (`select_artifact`) is unchanged.
- **Pins become aliases.** `global` → `@global`, `route:22` → `@route-22`,
  `route:22/1` → `@route-22-1`, `trip:t1` → `@trip-t1`. Check MLflow's allowed alias
  characters and reserved names (`latest`, `v<N>`) and make the encoding reversible. Store the
  original key in a tag so `pins()` doesn't have to parse the alias. `promote` = `set_registered_model_alias`.
  Rollback = set the alias back to the version `promote` returned. MLflow keeps the audit trail.
- **Drift refusal.** `load_model` reads `provenance.lib_versions` from `artifact.json`, not
  from MLflow's environment files, and runs the same `version_mismatches` check and override
  flag as U7. MLflow's `get_model_dependencies()` is just a second place to see them.
- **Model bytes.** The model version's artifacts hold the same `model.<ext>` directory the
  family's `save()` writes, so `Model.load` and `LegacyModuleUnpickler` work unchanged.
  Wrapping as pyfunc is optional (I3). The registry doesn't need it.

**Stays ours:** the artifact contract, selection policy, drift policy, legacy-pickle loading.
MLflow provides storage, versioning, aliases, UI and audit.

**Caveats:** needs a tracking server, or a file/SQLite store, which small agencies may not
want. So filesystem stays the default. Serving from this backend needs `mlflow-skinny` in the
serve environment, behind a `serve[mlflow]` extra. Models are fetched over the network on cold
start, so cache them locally, keyed by version.

**Effort:** medium. The interface refactor is the bigger half and should come first as its own
commit, with the filesystem tests unchanged. Then `MlflowRegistry` with a contract test suite
that runs against **both** backends.

### I3 · Export a stoptimes artifact as an MLflow pyfunc

**Adds:** `stoptimes models export <artifact_id> --mlflow [--register <name>]`, so a stoptimes
model can be used by MLflow-native tooling (`mlflow models serve`, Docker images, Spark UDFs,
managed platforms).

**Design:** a `StoptimesPyfunc(mlflow.pyfunc.PythonModel)` whose `load_context` calls the
family's `Model.load` on the bundled artifact directory, and whose `predict` calls
`Model.predict`. Log it with a **signature** built from `ArtifactMetadata.features` and an
input example from the test split, pinning requirements from `provenance.lib_versions`.
Attach `artifact.json` and `card.md` as artifacts.

**Be explicit in the docs:** the pyfunc takes the **feature frame**, not raw GTFS-RT. Feature
computation lives in the serve pipeline (`match` → `features`). An MLflow-served model only
agrees with ours if the caller computes features the same way. That's the parity problem U6
fixed for our own serving. Bundling the feature stage into the pyfunc is possible but pulls
`core` into the model environment. Leave it as a later decision.

**Effort:** small.

### I4 · Load an MLflow model as a bring-your-own model

**Adds:** serving and validating a model that lives in MLflow, whether it was trained with
stoptimes or not.

**Plugs in:** a new `ModelRefKind.MLFLOW` in `core/contracts/model.py` (contract change, needs
a note) and an `MlflowModelLoader` in `serve/model_loading.py`, next to the pickle/onnx/callable/http
loaders and dispatched by `load_model_ref`.

```yaml
model:
  kind: mlflow
  location: models:/stoptimes.bucr@route-22   # or runs:/<run_id>/model
```

**Design:** `mlflow.pyfunc.load_model(location)` wrapped to satisfy `Model`:
`required_features` from the signature's input names when `ModelRef.required_features` is
empty; `fit` raises (inference-only, like `HttpModel`); `predict` returns
`np.asarray(..., dtype=float)`. Load failures and signature mismatches become the same
`RuntimeError`s the other loaders raise, so the pipeline's per-stage error recording
(`record_error`) handles them.

**Effort:** small.

### I5 · MLflow models in `val`, and val results in MLflow

**Adds:** (a) an `mlflow:<uri>` predictor, so an MLflow-hosted model can be replayed against
ours over identical trips; (b) optionally, logging a val report as an MLflow run.

**Plugs in:** (a) `val/engine/assembly.py`: add `"mlflow:"` to `_DYNAMIC_PREDICTOR_PREFIXES`
and a branch in `resolve_predictor_factory` that builds the predictor from I4's loader
through the default pipeline (the same route `pipeline:` takes), so features are computed by
our stages. (b) a `--log-mlflow <experiment>` flag on `val replay` / `val live` that logs overall
and per-horizon/route/time-of-day errors as metrics and the report as an artifact.

**Stays ours:** ground truth, matching, scoring, statistics. MLflow only receives the numbers.

**Effort:** small once I4 exists.

### I6 · Call `mlflow models serve` over HTTP

**Adds:** using a model served by MLflow's scoring server (or anything with the same API)
without installing MLflow in the stoptimes process.

**Plugs in:** `HttpModel` already exists, but it POSTs `{"rows": [...]}`. MLflow's `/invocations`
expects `{"dataframe_records": [...]}` (or `dataframe_split`) and replies `{"predictions": [...]}`.
Add an optional `request_format: rows | mlflow` field to `ModelRef` (HTTP only, default `rows`)
and switch the body shape in `HttpModel.predict`. The response side already accepts
`{"predictions": …}`.

**Effort:** tiny. This is the cheapest way to try MLflow serving, and it keeps MLflow out of
the serving environment entirely.

---

## 5. Packaging

| Extra | Pulls | Used by |
|---|---|---|
| `stoptimes-train[mlflow]` | `mlflow` | I1, I3, I2 writes |
| `stoptimes-models[mlflow]` | `mlflow-skinny` | I2 registry backend |
| `stoptimes-serve[mlflow]` | `mlflow-skinny` | I2 reads, I4 |
| `stoptimes-val[mlflow]` | `stoptimes-serve[mlflow]` | I5 |
| `stoptimes[mlflow]` (cli) | all of the above | convenience |

I6 needs no extra. Confirm with U2's startup check that `stoptimes --help` stays under 300 ms
with the extra installed (no import at module level). Add a CI matrix job with the extra
installed; the default jobs run without it, which also proves it stays optional.

## 6. Comparison against our implementation

The point of having both is a fair comparison. After I1 + I2 land, run the same experiments
through both backends and record:

| Question | How to measure |
|---|---|
| Same behaviour? | The shared registry contract test suite passes on both; `select_artifact` picks the same artifact for the same pins/metrics; `val replay` of one recorded day gives identical scores through both backends |
| Promote / rollback | Time and steps from "new model trained" to "served", and back; what the audit trail shows |
| Drift safety | A deliberately drifted artifact is refused by both |
| Cost to run | Services needed (none vs tracking server + store); install size of `serve` with and without the extra; cold-start load time |
| Modeler experience | Comparing N runs by horizon: `models list`/cards vs the MLflow UI |
| Newcomer experience | Quickstart length with each; whether MLflow adds steps a small agency won't take |

Write the results in `docs/notes/mlflow-comparison.md`. The likely outcome is filesystem as
the default for small agencies and MLflow for teams that already run it. The measurements
should decide it, not this document.

## 7. What not to replace, and why

- **`serve` pipeline / publisher → `mlflow models serve`.** MLflow serves a function over
  HTTP. It doesn't poll GTFS-RT, map-match, compute features, choose models per route/trip, or
  emit TripUpdates, and databus calls a Python function in-process (PLAN.md §7.2, G2).
  Use I3/I6 to put MLflow *behind* the `predict` stage, never in place of the pipeline.
- **`val` → `mlflow.models.evaluate`.** It scores a logged model on a static labelled
  dataset. `val` scores any TripUpdates feed, including ones we didn't produce, against
  published-feed ground truth over time, by horizon. Different question (design rule 3).
- **`experiment.yaml` + TUI → MLflow Projects.** Projects describe *how to run code*.
  `experiment.yaml` describes *what to train on which data*, validated by the `experiment`
  contract. You could run `stoptimes train run` as a Project entry point, but it adds nothing.

## 8. Order of work

1. **Registry interface refactor** (part of I2), filesystem only, no MLflow, tests unchanged.
   Can start once U7 is merged.
2. **I6** (tiny) and **I1** (small): first working integration, low risk.
3. **I4 → I5**: MLflow models become first-class in serve and val.
4. **I2 `MlflowRegistry`** + dual-backend contract tests.
5. **I3** export.
6. **§6 comparison**, then decide defaults and write the U5 docs page ("stoptimes and MLflow").

## 9. Open questions

- Registered-model naming: one per agency (proposed) or one per scope level? It affects how
  many aliases a registered model carries and how the MLflow UI reads.
- Should `models migrate` (C3) be able to target an MLflow registry directly?
- Bundle the feature stage into the exported pyfunc (I3), or keep pyfunc as model-only?
- Does I1 log when training fails partway (a failed run), or only on success?
