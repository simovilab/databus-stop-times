# stoptimes — implementation plan

Status: **Phase 1 in progress (C1, C2, C3)** · Phase 0 complete (not pushed) · branch `feat/stoptimes-scaffold` · last updated 2026-09-16

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
| Serve | Spec-valid TripUpdates from a pretrained, trained-here or bring-your-own model; overridable pipeline stages | `stoptimes-serve` |
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
| **3 Experience** | TUI, diagnostics, Python API, docs | 5 parallel tracks | Quickstart works from a clean machine |
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

**Gate:** user reviews `docs/CONTRACTS.md`. After sign-off, contract changes need an orchestrator-approved note.

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

**Gate — end-to-end smoke** (`tests/e2e/test_mbta_smoke.py`, orchestrator-owned): committed MBTA sample →
`data add` → `data arrivals` → `train run` → `serve --sink file` → `val replay` scores the served feed with
baselines present. Runs in CI in < 5 min.

---

## 7. Phase 3 — Experience (5 parallel tracks)

| Track | Delivers |
|---|---|
| **U1 Experiment TUI** | Textual builder: target, model, features (from registry, with descriptions), params (+ ranges for tuning), split; writes experiment.yaml; `train ui` |
| **U2 CLI polish** | Startup < 300 ms for `--help`; consistent tables; `--json` output everywhere; install hints for every optional part |
| **U3 Diagnose** | `diagnose` report: unmatched trip_ids, missing `shape_dist_traveled`, VP gaps, stale timestamps, schedule/realtime mismatch; reused by `data inspect` |
| **U4 Python API** | `stoptimes.core.api` (re-exported as `stoptimes`) with the same verbs as the CLI; example notebooks |
| **U5 Docs** | mkdocs site: quickstart ("I have a GTFS zip and a realtime URL"), concepts, contracts, CLI reference, SIMOVI case study |

**Gate:** quickstart executed from a clean machine by an agent that has not seen the repo.

---

## 8. Phase 4 — Generality proof and SIMOVI cutover (sequential)

Each step touching production needs the user's explicit go.

- **G1 Generality:** bUCR and MBTA both run end to end with configs only; `grep -ri bucr packages/` is empty (tests/fixtures excepted).
- **G2 databus:** databus consumes `stoptimes-serve` (pinned) instead of the `backend/gtfs-eta` symlink; PR to `simovilab/databus`
  (currently imports `gtfs_eta.eta_service.estimator` from `runs/domain/progression/stop_times.py`).
- **G3 Collector:** VPS collector moves to `stoptimes record`. Production still runs from
  `~/git/gtfs-django/eta_prediction/gtfs-rt-pipeline` with uncommitted changes; confirm with
  `docker inspect … working_dir`, back up `.env`, run side by side before switching.
- **G4 Served models:** re-export served artifacts under the deploy target's library versions (the deferred RUNBOOK task).
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
