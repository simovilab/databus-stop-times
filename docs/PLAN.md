# stoptimes — implementation plan

Status: **Phase 3 tracks all merged (2026-09-18)** — 1497 passed, 20 skipped; the clean-machine quickstart gate has
**not** been run yet · next: run that gate, then Phase 4 (note G4 now precedes G2a — see §8) ·
branch `feat/stoptimes-scaffold`, not pushed (the account has read-only access to the remote) · last updated 2026-09-18

This is the working plan for turning SIMOVI's ETA research code (`gtfs-eta`) and its
validation tool (`etaval`) into **stoptimes**: a general toolkit that lets anyone with
GTFS data record feeds, train arrival-time models, serve GTFS-RT TripUpdates with them,
and validate any prediction source. SIMOVI (bUCR, MBTA) is the first user, not the design
target.

The plan is written to be executed by several Sonnet subagents in parallel, coordinated
by one orchestrator (Opus). Each track owns a set of paths, has explicit inputs, and has a
checkable definition of done.

---

## 1. Product in one page

| Part | User gets | Package |
|---|---|---|
| Data | Load GTFS static + realtime in any common form (GTFS-RT protobuf/JSON, archive parquet, AVL CSV); inspect coverage | `stoptimes-core` |
| Record | Archive a live GTFS-RT feed, compacted, local or S3 | `stoptimes-record` |
| Arrivals | Actual stop arrivals derived from vehicle positions (map-matching) | `stoptimes-core` |
| Train | Pick target, model, features, hyperparameters (TUI or YAML); get a versioned artifact + model card | `stoptimes-train`, `stoptimes-models` |
| Serve | A prediction library (load a model, compute its inputs, produce stop time updates) with overridable pipeline stages, plus a standalone publisher that emits spec-valid TripUpdates. Anyone who already owns a feed builder — databus does — calls the library and skips the publisher; see §7.2 | `stoptimes-serve` |
| Validate | Horizon-based scoring against published-feed ground truth, live or replayed, for any TripUpdates source | `stoptimes-val` |
| Diagnose | Feed quality report before training | `stoptimes-core` |

One command: `stoptimes {data,record,train,models,serve,val,diagnose}`. Installed in parts:
`pip install 'stoptimes[serve]'` never pulls training dependencies.

### Design rules (non-negotiable)

1. **Dependency direction is acyclic:** `core ← models ← serve ← train`, `core ← record ← train`,
   `core, serve(optional) ← val`. **Nothing depends on `val`.** `cli` depends on everything optionally.
2. **UIs write config; the CLI runs config.** The TUI/web never hold state the YAML can't express.
3. **Ground truth comes only from published feeds.** Validation never reads training internals;
   it scores models through the `Predictor` interface.
4. **No agency-specific code in `packages/`.** bUCR and MBTA live in `examples/` as configs.
   (Phase 0 still carries migrated `bucr_*` modules; Phase 2 T2 removes the special cases.)
5. **Baselines always ship** (schedule-only, constant-speed) and appear in every report.
6. **Faithful before better.** A move keeps behavior and tests; improvements are separate commits.

---

## 2. Repository layout

```
databus-stop-times/
├── packages/
│   ├── core/     stoptimes.core     config, domain models, gtfs/, realtime/, spatial/, features/,
│   │                                segments, contracts/, readers*, arrivals*, diagnose*, api*
│   ├── models/   stoptimes.models   common/ (registry, metrics, splits, provenance), families
│   ├── serve/    stoptimes.serve    estimator, pipeline*, tripupdates*, sinks*
│   ├── record/   stoptimes.record   storage/ (spool, S3 writer, schemas), compaction/, recorder*
│   ├── train/    stoptimes.train    datasets/, sources/, evaluation/, experiment*, tui*
│   ├── val/      stoptimes.val      etaval: engine, feeds, groundtruth, predictors, scoring, stats, api, tui
│   └── cli/      stoptimes.cli      the `stoptimes` command
├── web/          Nuxt UI (etaval console today; experiments later)
├── examples/     agencies/ and experiments/ configs (Phase 1)
├── docs/         PLAN.md, MIGRATION_MAP.md, CONTRACTS.md (Phase 1), research/, history/, val/
└── legacy/       not-yet-migrated code, kept with history for reference; deleted in Phase 4
```
`*` = placeholder module today.

Namespace: every member ships `src/stoptimes/<part>`; there is **no** `stoptimes/__init__.py`
(PEP 420 namespace). Name is provisional; renaming is one mechanical pass.

---

## 3. Phases at a glance

| Phase | Goal | Shape | Gate |
|---|---|---|---|
| **0 Scaffold** | Whole project skeleton; all existing code in its final home, tests at baseline | 4 parallel fix-up tracks | Test parity table (§4.4) |
| **1 Contracts** | Freeze the interfaces every later track builds on | 1 main track + 2 small parallel | User signs off `docs/CONTRACTS.md` |
| **2 Capabilities** | Data, arrivals, train, serve, record, val working through contracts | 6 parallel tracks | End-to-end smoke on MBTA sample |
| **3 Experience** | TUI, diagnostics, serving on the pipeline, registry lifecycle, Python API, docs | 7 parallel tracks | Quickstart works from a clean machine |
| **4 Generality + cutover** | bUCR + MBTA with zero special cases; SIMOVI production on stoptimes | Sequential, user-approved steps | databus + VPS on stoptimes; old repos frozen |
| **5 Ecosystem** | Model zoo, benchmark + leaderboard, PyPI | Parallel, optional | — |

Why this order: Phase 0 makes every later agent start from working code instead of a blank
repo. Phase 1 is deliberately narrow so it finishes fast and unblocks the widest fan-out
(Phase 2). Cutover waits until the generality test passes, so SIMOVI production never runs
on half-generalized code.

---

## 4. Phase 0 — Scaffold

### 4.1 Done (orchestrator)

- [x] **0.1 History import.** `gtfs-eta@feat/lstm-segment-model (c3e3dff)` and
      `etaval@feat/ops-console-eval-repro (4156710)` rewritten with `git filter-repo`
      so each file's history lives at its new path; merged into this branch. `git log --follow` works.
- [x] **0.2 Import rewrite.** Mechanical rewrite of all legacy module paths to `stoptimes.*`
      (152 files). Map in `docs/MIGRATION_MAP.md`.
- [x] **0.3 Workspace.** uv workspace, 7 members, Python 3.12 dev pin, `stoptimes` CLI
      (planned commands exit 2 and name their phase; `val` mounts etaval; `record compact`,
      `train legacy-all`, `train seed-baseline`, `models check` pass through to migrated code),
      contract + feature placeholders, root `.gitignore`.

### 4.2 Baseline (source repos, before migration)

| Suite | Result |
|---|---|
| gtfs-eta `core eta_service feature_engineering models` | 365 passed, 15 skipped; 3 known collection errors (`test_dataset_builder` needs `sch_pipeline`, `test_rt_source` needs `rt_pipeline`, `test_lstm` needs torch) |
| gtfs-rt-pipeline `rt_pipeline/storage rt_pipeline/compaction` | 88 passed |
| etaval `tests` | 478 passed, 1 skipped |

### 4.3 Parallel fix-up tracks (Sonnet)

State after 0.3: `cli` 3 passed · `core` 105 passed · `val` 430 passed 1 skipped ·
`serve` 4 failed 17 passed · `models` collection error (`test_lstm`) ·
`record` collection error (`storage/test_spool_compaction`) ·
`train` collection errors (`test_dataset_builder`, `test_weather`).

**P0-A · models + serve** — owns `packages/models/**`, `packages/serve/**`, `packages/core/src/stoptimes/core/config.py`
- `test_lstm.py`: skip cleanly when torch is absent (`pytest.importorskip`); passes with `uv sync --group dl`.
- `serve/tests/test_eta_wiring.py`: fixture moved to `packages/serve/tests/fixtures/sample.parquet`.
- Registry location: no repo-root discovery. `MODEL_REGISTRY_DIR` keeps working; add `STOPTIMES_HOME`;
  default `./artifacts/models` (P0-A owns `core/config.py` for this; no other Phase 0 track touches core).
- **Legacy pickles:** loading an artifact pickled under `models.*` / `core.*` / `eta_service.*` /
  `feature_engineering.*` must work, via a `find_class` remapping unpickler in the registry
  (not a global import hook). Test with a pickle built from a class re-homed under a legacy module name.
- Remove the `models → train` lazy `train_all_models` attribute (dependency inversion); update
  `test_lazy_xgboost_import.py` accordingly.
- Fix stale path references in docstrings/READMEs inside owned packages.
- Done: models + serve tests ≥ baseline share; `import stoptimes.serve.estimator` does not import xgboost or torch.

**P0-B · train** — owns `packages/train/**`
- `test_dataset_builder.py` / `test_weather.py`: Django-backed legacy builders. Skip cleanly when
  Django / `sch_pipeline` are absent; with `--extra django` they run as far as they did before.
- `train_all_models.py`: replace bare `from ewma…`/`from common…`/`sys.path` hacks with package imports;
  `stoptimes train legacy-all --help` works.
- `test_rt_source.py`: now importable (`stoptimes.record.storage`) — make it pass.
- `seed_baseline_model.py`: drop the gtfs_eta alias-package narrative; `stoptimes train seed-baseline --help` works.
- Done: train tests ≥ baseline share, plus `test_rt_source` passing (was a known error).

**P0-C · record** — owns `packages/record/**`, `legacy/collector/README.md`
- `storage/test_spool_compaction.py` collection error; all 88 baseline tests pass.
- Check `status.py` / `tu_records.py` have no Django/Celery imports left; tests pass.
- `legacy/collector/README.md`: state that storage/compaction/tu_records/status moved to
  `stoptimes.record`, that this Django app is reference-only here, and that production still runs
  from the pre-extraction tree (see §8 Phase 4 G3).
- Done: record tests = 88 passed (+ `test_tu_records`).

**P0-D · CLI, val docs, CI, web** — owns `packages/cli/**`, `packages/val/README.md`, `web/**`, `.github/**`
- Split `cli/main.py` into `cli/commands/{data,record,train,models,serve,val,diagnose}.py` so each
  Phase 2 track owns one file. Behavior and `packages/cli/tests` unchanged (extend tests per command).
- `packages/val/README.md`: `etaval …` → `stoptimes val …`, `etaval-api`/`etaval-tui` equivalents.
  Add `stoptimes val api` / `stoptimes val tui` commands if val exposes `main()` for them.
- `web/`: confirm the API base URL/env still matches `stoptimes.val.api`; document how to run it.
- `.github/workflows/ci.yml`: uv, Python 3.12, `uv sync --group dev`, pytest per package (matrix), ruff (non-blocking for now).
- Done: `stoptimes val --help`, `stoptimes val api --help` work; CI file validates (`actionlint` if available).

### 4.4 Integration and exit (orchestrator)

1. Merge track branches in order A → B → C → D; `uv lock` once; full suite.
2. Fill the parity table below; any shortfall is explained or fixed.
3. Update root `README.md` status and this file.

| Package | Expected (from baseline) | Actual |
|---|---|---|
| core | 105 passed | 105 passed ✅ |
| models | all non-torch pass; lstm skipped w/o torch | 105 passed, 1 skipped (lstm, no torch) ✅ |
| serve | 21 passed | 21 passed ✅ |
| record | 88 + tu_records passed | 106 passed ✅ |
| train | all migrated pass; Django tests skip w/o extra; rt_source passes | 217 passed, 15 skipped (bUCR snapshot absent, as before) ✅ — Django tests and rt_source now pass |
| val | 430 passed, 1 skipped | 430 passed, 1 skipped ✅ |
| cli | all passed | 24 passed ✅ |

Total: **1008 passed, 17 skipped, 0 failed** (baseline across the three source suites: 931 passed, 16 skipped, 3 collection errors).
Also verified: nothing in `packages/` imports legacy modules; `import stoptimes.serve.estimator` loads neither xgboost nor torch;
6 real pre-migration artifacts load through `LegacyModuleUnpickler` (they use bare module names like `xgb.train`).

### 4.5 Follow-ups carried into Phase 1

- `stoptimes.models.common.data.load_dataset` falls back to `packages/serve/tests/fixtures/` — production code must not
  know test paths. Replace with an explicit dataset path/catalog (C1 `data` contract, T1 catalog).
- pandas now infers `datetime64[us]` where code assumed `ns` (fixed in `navsat_adapter`); C1 `data` contract pins timestamp dtype.
- `val/predictors/gtfs_eta_models.py` docstrings still describe a sibling gtfs-eta repo (T6 renames to `stoptimes:*`).
- Track notes with details: `docs/notes/p0-{a,b,d}.md`.

**Phase 0 exit:** ✅ parity table complete, `stoptimes --help` shows every area, the branch is pushed
(with the user's OK), and nothing in `packages/` imports from `legacy/`.

---

## 5. Phase 1 — Contracts

Narrow on purpose. Output is code (pydantic models / Protocols + tests) and `docs/CONTRACTS.md`.

**Orchestrator, before fan-out (done):** `contracts/artifact.py` fixed (layout `<registry>/<artifact_id>/{artifact.json, model.<ext>, card.md}`,
`ArtifactMetadata`, atomic write, `version_mismatches`) so C1 and C3 share it. C1 may add optional fields only.

**C1 · contracts** (1 agent, Opus review) — owns `packages/core/src/stoptimes/core/contracts/**` (except artifact field changes),
`packages/core/tests/test_contract_*.py`, `docs/CONTRACTS.md`, `examples/experiments/**`
- `source`: feed/agency description (static URL, realtime endpoints + format, timezone, auth, archive, ground-truth detector).
  Must accept today's val source YAMLs unchanged (new fields optional); test validates them wherever they live.
- `data`: StaticFeed, Observation (VP/TU, source-agnostic), Arrival (with derivation method); column schemas + parquet layout;
  timestamps pinned (UTC, `datetime64[ns, UTC]`). Reconcile `core.domain.models`, `record.storage.schema/tu_schema`.
- `experiment`: experiment.yaml schema (data selection, target, model + params, features, split, seed, outputs).
- `model`: fit/predict Protocol, declared feature needs, serialization; how BYO models (pickle, ONNX, callable, HTTP) satisfy it.
- `pipeline`: stages `ingest → match → features → predict → postprocess → publish`, hook signatures, context object.
- `prediction`: per-stop prediction + exact GTFS-RT TripUpdate/StopTimeUpdate mapping.
- Adapters from existing structures to contracts in `contracts/adapters.py` (pure functions, no behavior change elsewhere).
- `examples/experiments/xgb-baseline.yaml` validates against the schema.

**C2 · de-SIMOVI config** (parallel) — owns `packages/core/src/stoptimes/core/config.py`, its call sites
(`serve/estimator.py`, `train/datasets/dataset_builder.py` imports only) and their tests, `packages/val/src/stoptimes/val/config/**`,
`packages/val/src/stoptimes/val/feeds/source_config.py`, val source tests, `examples/agencies/**`
- Agency profiles (timezone, holiday region, served-model pins) move to `examples/agencies/<agency>.yaml`; settings become
  generic with a `STOPTIMES_` env prefix; old `ETA_*` names still honored with a `DeprecationWarning`.
- `git mv` val `config/sources/*.yaml` to `examples/agencies/`; search path `--sources-dir` → `STOPTIMES_SOURCES` → `./examples/agencies`.

**C3 · artifact migration** (parallel) — owns `packages/models/src/stoptimes/models/compat/**`, its tests, `models migrate` in `cli/commands/models.py`
- `stoptimes models migrate <legacy_dir> <registry>`: `{key}.pkl` + `{key}_meta.json` → artifact layout, original meta kept in `legacy`,
  library versions recorded, idempotent, `--dry-run`; migrated artifacts load and predict identically.

**Result (2026-09-17):** C1, C2, C3 merged. Suite: core 189, models 116+1s, serve 21, record 106, train 217+15s,
val 430+1s, cli 29 — 1108 passed, 17 skipped, 0 failed. Integration fixes: `Source` gained `holiday_region`/`serving`;
`calendar`/`calendar_dates` optional static tables; `STOPTIMES_MODEL_REGISTRY_DIR` honored by registry + val predictor;
holiday calendar generic (`COUNTRY[_SUBDIV]`, None = no holidays).
Carried into Phase 2: val API `/api/sources` tests assume cwd = repo root (T6); `_LEGACY_BUILTIN_PROFILES` in core config
keeps production bUCR pins without a profile file, delete in G1; migrated `params` include some non-hyperparameter
fields (T3 may curate per family); `models/common/data.py` test-fixture fallback (T1). Notes: `docs/notes/p1-{c1,c2,c3}.md`.

**Gate (passed):** user signed off `docs/CONTRACTS.md` with the six recommendations, plus the added
`selection` contract (per-route/per-trip model choice) answering their serving question. After sign-off, contract changes need an orchestrator-approved note.

---

## 6. Phase 2 — Capabilities (6 parallel tracks)

Tracks interact only through `stoptimes.core.contracts`. Each owns its modules and its CLI command file.

| Track | Owns | Delivers | Depends on |
|---|---|---|---|
| **T1 Data** | `core/readers.py`, `core/catalog/**`, `cli/commands/data.py` | Readers: GTFS-RT pb + JSON (live + files), archive parquet, AVL CSV with column mapping; local data catalog; `data add`, `data inspect` (coverage: trips matched, arrivals present, gaps) | C1 |
| **T2 Arrivals** | `core/arrivals.py`, `core/segments.py`, `train/sources/**` | Agency-agnostic trip inference + map-matching → Arrival table; bUCR navsat becomes an AVL-CSV reader config, not code; val ground-truth detectors reuse it; `data arrivals` | C1, T1 reader interface |
| **T3 Train** | `train/experiment.py`, `core/features/registry.py`, `models/**` adapters, `cli/commands/{train,models}.py` | Feature registry wrapping existing temporal/spatial/segment/rolling features; families on the model contract; `train run experiment.yaml`; artifacts + cards; `models list/card/promote` | C1, C3 |
| **T4 Serve** | `serve/**`, `cli/commands/serve.py` | Pipeline with overridable stages (decorators); estimator rebuilt on it; TripUpdates encoder (pb + JSON, spec-valid); sinks: file, HTTP, MQTT (optional extra); BYO model loaders; `serve` | C1, T3 feature registry |
| **T5 Record** | `record/**`, `cli/commands/record.py` | Standalone recorder (poll loop, spool, flush, compaction) without Django/Celery; local FS + S3; `record start/compact/status` (replaces `simovi-status`); Dockerfile | C1 |
| **T6 Val** | `val/**`, `cli/commands/val.py`, `web/**` | Protobuf feeds; replay from T5 archive; `stoptimes:<artifact>` and `pipeline:<file.py>` predictors (keep `gtfs_eta:*` aliases); score any TripUpdates feed; drop `ewma` from predictor registry; rename etaval branding | C1, T4 for pipeline predictor |

Soft dependencies are handled by coding against the contract with a fake, not by waiting.

**Result (2026-09-18):** all six tracks merged, plus **T7 · Training frame** (added by the orchestrator: nothing
turned observations + arrivals + static into the frame the families train on — the only builder was Django/Postgres/MBTA-shaped)
and the orchestrator's wiring of `data arrivals`. Suite: core 289, models 125+2s, serve 57+2s, record 130, train 239+15s,
val 468+1s, cli 70, e2e 1 — **1379 passed, 20 skipped, 0 failed**. Gate test: `tests/e2e/test_end_to_end.py`, in CI.

Bugs the gate caught (all fixed with tests): val couldn't open an archive whose partition key is also a column
(pandas 3 `large_string` vs hive `string`); val replay coerced `ts` with `to_numeric`, so archives written by the new
recorder looked empty; `get_config()` snapshotted the environment at first use, writing a test's catalog into the
working directory.

Carried into Phase 3 (the first two are now **U6**, the fifth is now **U7**):
- `serve.estimator` was left on its pre-Phase-2 code rather than rebuilt on the pipeline (it is what databus calls
  in production; unifying needs T1's readers + T2's matcher — do it before G2b).
- The default `match` stage handles trip_update observations only; vehicle-position map-matching in serving should
  call `core.arrivals`/`core.segments` (the gate's pipeline supplies distance features by hand).
- bUCR trip inference (~1240 lines, ~20 tuned thresholds) is still agency-specific — Phase 4 G1, deserves its own track.
- `record status` reports local archive size only; the Dockerfile has never been built.
- `models list --json` shape and `models promote`'s profile write are unspecified in CONTRACTS.md (now **U7**).
- Contract wording: an `Arrival` is keyed by (trip, service day, **stop_sequence**) — a loop route visits a stop twice
  and both visits are real. CONTRACTS.md §2 says "trip+stop+day"; T2 implemented the sequence key, which is correct.
- s3:// reads need `fsspec`/`s3fs`, not yet a declared extra; the s3 archive path is untested.

**Gate — end-to-end smoke** (shipped as `tests/e2e/test_end_to_end.py`, orchestrator-owned; a synthetic agency built
in `tmp_path` rather than a committed MBTA sample, so it needs no fixture data and no network): `data add` →
`data arrivals` → `data frame` → `train run` → `serve --once` through a user pipeline → `val replay`. In CI.

---

## 7. Phase 3 — Experience (7 parallel tracks)

Scope reviewed with the user 2026-09-18 after walking through the user stories (§7.1) and the
databus integration question (§7.2). U6 and U7 are new; they were "carried into Phase 3" debt
and are now prerequisites, because G2 step two depends on them and because discovering either
during the cutover is the worst possible time.

| Track | Delivers |
|---|---|
| **U1 Experiment TUI** | Textual builder: target, model, features (from registry, with descriptions), params (+ ranges for tuning), split; writes experiment.yaml; `train ui` |
| **U2 CLI polish** | Startup < 300 ms for `--help`; consistent tables; `--json` output everywhere; install hints for every optional part |
| **U3 Diagnose** | `diagnose` report: unmatched trip_ids, missing `shape_dist_traveled`, VP gaps, stale timestamps, schedule/realtime mismatch; reused by `data inspect` |
| **U4 Python API** | `stoptimes.core.api` (re-exported as `stoptimes`) with the same verbs as the CLI; example notebooks |
| **U5 Docs** | mkdocs site: quickstart ("I have a GTFS zip and a realtime URL"), concepts, contracts, CLI reference, SIMOVI case study; **serving positioned per §7.2** |
| **U6 Serving on the pipeline** | Rebuild `serve.estimator` on `serve.pipeline` (it is what databus calls today); real vehicle-position map-matching in the default `match` stage via `core.arrivals`/`core.segments`; the legacy estimator becomes a thin compatibility shim over the pipeline, not a second implementation |
| **U7 Registry lifecycle** | Refuse (or require an explicit override) to load an artifact whose library versions have drifted — `version_mismatches` exists and nothing acts on it; a tested `models promote` / rollback workflow, documented in CONTRACTS.md along with the `models list --json` shape |

**Gate:** quickstart executed from a clean machine by an agent that has not seen the repo. **Not yet run.**
U6 additionally must keep the end-to-end gate green **without** the hand-supplied distance
features in the test's pipeline — that removal is the proof map-matching actually landed.

### 7.3 Phase 3 status — all seven tracks merged 2026-09-18

Suite: core 321, models 134+2s, serve 79+2s, record 130, train 267+15s, val 473+1s, cli 92, e2e 1 —
**1497 passed, 20 skipped, 0 failed**. `mkdocs build --strict` clean.

**The U6 gate was passing vacuously and U6 said so.** Deleting the hand-supplied features proved
nothing: `serve` hard-coded `static_feed=None`, so matching never fired through the CLI, and
`HistoricalMeanModel` imputes missing features and returns numbers anyway. A silently broken
`match` stage would have looked identical. Fixed by the orchestrator (`--static`, threaded into
the context) and the gate now asserts on what the *stages produced* — spatial features present
and finite for every stop except each trip's first, where the distance is genuinely unknown.
Lesson for later phases: assert on the intermediate, not only on the output, whenever the model
tolerates missing inputs.

Defects found and fixed this phase, each by a track following its own instructions rather than by
review: a silent date-range drop when editing an experiment in the TUI (U1); `--help` importing
sklearn and matplotlib, 3-5 s → 0.13 s (U2); an install hint naming an extra that could not fix
the error it was printed for (U2, predates the phase); an unbounded archive read in `diagnose`
(U3); the async entry point breaking scoring inside a notebook (U4); a hard-coded deployment
path in `record` violating rule 4 (U5); a refused model reaching the operator as a traceback
instead of its own actionable message (U5→orchestrator).

Carried forward: `read_archive_parquet` has no column projection, so a bounded read still loads
every column (U3, U5); `stoptimes-models` exposes no public family registry, so the TUI reads a
private mapping (U1); no target registry exists (U1); ONNX/TorchScript drift is undetectable
because provenance records only python/sklearn/xgboost/numpy/pandas (U7); `record compact` is
still the legacy agency-specific CLI (U5, folds into G1); a `val` leaf subcommand still takes
~4-6 s to start, down from 20-60 s (U5); `import stoptimes` is not a usable entry point — there
is no root namespace module, so verbs import from their packages (U4, **user decision, ties to
the naming question**).

### 7.1 User stories (the acceptance list)

Written 2026-09-18 to describe the whole toolkit from outside. Phase 3 is judged by whether a
stranger can do these without asking us.

**Newcomer** — diagnose whether a feed can support ETA modelling at all before investing weeks
in it; record a live feed unattended for weeks with self-compaction and an honest answer to "how
much data do I have"; get from archive to trained model through a guided flow rather than a
hand-written config; be told plainly when the model is *not* better than the printed schedule.

**Modeler** — describe an experiment in one file and reproduce it later on new data; compare
families over one dataset with errors broken out by prediction horizon; plug in a custom model
through a small interface without forking; bring a prepared dataset that never came from GTFS-RT;
have every artifact carry enough provenance to identify it six months later.

**Operator** — publish spec-valid TripUpdates continuously so existing consumers need no change;
override one pipeline stage without rewriting the server; get per-route/per-trip model choice
automatically with a manual pin available; have a stale model refused rather than loaded quietly
(**U7**); install serving without training dependencies; promote and roll back quickly (**U7**).

**Skeptic** — score *any* TripUpdates feed against what happened without access to how it was
produced; replay one recorded day across several prediction sources over identical trips; break
error down by horizon, route and time of day; watch a live feed's accuracy degrade.

**SIMOVI** — bUCR and MBTA as configuration only (Phase 4 G1); databus production moved with the
old repos frozen and a way back (G2); bUCR's trip-less raw AVL handled by a configurable
inference step rather than 1,240 hand-tuned lines.

### 7.2 What "serving" means, and how databus uses it

Two different things get called serving, and the plan must not blur them.

The **prediction library** — load a model, compute its inputs, produce stop time updates — is
load-bearing regardless: databus calls exactly that, and `val` scores through the same seam.

The **standalone publisher** (`stoptimes serve`: poll on a timer, predict, publish a feed) is
what databus does *not* need, since databus owns its own feed. It stays, because an agency
without a databus would otherwise be left with a model file and nowhere to put it, and because
the loop around the prediction code is thin. But the docs must not imply it solves the
operations problem of running a public realtime feed: it ships as the straightforward path for
a small agency and the way to watch your own model work end to end, while anyone with real
infrastructure calls the library from their own publisher. **U5 writes it that way.**

databus is the first proof that seam is in the right place: it takes the prediction and skips
the loop cleanly. See G2 for the two-step cutover that follows from this.

---

## 8. Phase 4 — Generality proof and SIMOVI cutover (sequential)

Each step touching production needs the user's explicit go.

- **G1 Generality:** bUCR and MBTA both run end to end with configs only; `grep -ri bucr packages/` is empty (tests/fixtures excepted).
- **G2 databus (two deploys, not one):** databus keeps building its own feed; only the prediction call moves.
  - **G2a — drop-in.** `runs/domain/progression/stop_times.py` imports `estimate_stop_times` from `stoptimes.serve`
    instead of `gtfs_eta.eta_service.estimator` (same signature, including the `shape_distance_to_stop` hook databus
    relies on); the `backend/gtfs-eta` symlink becomes a pinned `stoptimes` dependency; `ETA_*` env vars become
    `STOPTIMES_*` (old names still read, deprecated). Nothing else changes — progression, the Redis
    `stop_time_updates` projection and `schedule_engine`'s `build_trip_updates` are untouched. Deliberately a
    one-line-import cutover so rollback is a revert.
  - **G2b — pipeline.** Once G2a is proven live, databus supplies a pipeline overriding `ingest` (its own vehicles,
    no polling) and `publish` (write the Redis projection), inheriting `match`/`features`/`predict`/`postprocess`.
    `build_trip_updates` still untouched. This is what kills the duplicated feature computation (today training and
    serving agree only by inspection), makes bUCR's loop-back-safe shape distance a declared `match` override rather
    than logic buried in progression, and lets a newly trained route-level model be picked up by the selection policy
    with no databus deploy. Requires **U6**.
  - Before either: a thin databus-facing adapter tested against a recorded bUCR day, so the cutover is verified
    off-VPS first.
- **G3 Collector:** VPS collector moves to `stoptimes record`. Production still runs from
  `~/git/gtfs-django/eta_prediction/gtfs-rt-pipeline` with uncommitted changes; confirm with
  `docker inspect … working_dir`, back up `.env`, run side by side before switching.
- **G4 Served models — must now happen BEFORE G2a, not after.** Re-export served artifacts under the deploy
  target's library versions (the deferred RUNBOOK task). Phase 3's U7 makes a load-critical version drift a
  **refusal** rather than a warning, and the currently served artifact is exactly that case: pickled under the
  training host's scikit-learn/xgboost, loaded under different minor versions in the deployment. Before U7 that was
  hygiene; after U7 the cutover would refuse to load the live model and serve nothing. Two acceptable orders:
  re-export first (preferred — it removes the drift), or set `STOPTIMES_ALLOW_VERSION_DRIFT` deliberately in the
  deploy environment as a temporary, logged exception. Choosing neither is an outage. Verify which applies by
  loading the artifact under the target's versions before scheduling G2a.
- **G5 Freeze:** tag `gtfs-eta` and `etaval` (citation), update `CITATION.cff`, archive both repos with a pointer here.
- **G6 Main:** replace `main` of `simovilab/databus-stop-times` with this branch (user decision).
- Then delete `legacy/`.

## 9. Phase 5 — Ecosystem (optional, parallel)

Model cards + pretrained zoo · public MBTA benchmark with fixed splits + leaderboard · simulator for data-free testing
(databus-sim) · PyPI release · positioning vs TheTransitClock and a prior-art survey · further targets (dwell, headway, bunching).

---

## 10. Agent protocol

**Setup.** The orchestrator creates one git worktree per track next to the repo:
`git worktree add ../dst-<track> -b p<phase>/<track> <base>`. Agents work only there.

**Environment.**
- `uv sync --group dev` in the worktree (add `--group dl` only when the task needs torch).
- Run tests with `uv run --no-sync python -m pytest packages/<pkg> -q -p no:cacheprovider`.
  Never plain `pytest` (a global pytest can shadow the project's).
- Repo lives under iCloud (`~/Desktop`): ignore `* 2.py` style duplicates, never commit them.
  A venv inside iCloud gets its `.pth` files flagged hidden and Python skips them (imports of
  `stoptimes` fail at random). Worktree setup: `uv venv --python 3.12 ~/.venvs/<worktree>`,
  `ln -s ~/.venvs/<worktree> .venv`, then `uv sync --group dev`. See README.

**Ownership.**
- Edit only paths your track owns. Need a change elsewhere? Don't make it: add a note to
  `docs/notes/<track>.md` (what, where, why) and work around it with a local shim if blocking.
- Never edit `uv.lock`, the root `pyproject.toml`, or another track's `pyproject.toml`. Dependency
  needs go in the notes file; the orchestrator relocks.

**Quality.**
- Faithful moves first, improvements in separate commits. Don't delete tests to get green; a skip needs a reason string.
- Conventional commits (`fix:`, `refactor:`, `test:`, `docs:`, `feat:`), one logical change each, on your track branch.
- Don't push, don't merge, don't touch `main`.

**Report back** (final message, ≤ 30 lines): branch + commits; test results per owned package (passed/failed/skipped vs expected);
notes filed; anything you were unsure about.

**Orchestrator.** Merges tracks in the listed order, relocks, runs the full suite, resolves notes, updates this file's
checkboxes and parity tables, and asks the user before any push or production step.
