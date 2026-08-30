# ETA Prediction — Research & System Roadmap

> Status: draft v3.3, 2026-08-30. Supersedes nothing; complements `REFACTOR_PLAN.md`
> (which covered the completed schema-alignment + S3 migration, Phases A–E).
>
> **v3.3 (2026-08-30):** The bUCR vertical slice built on 2026-08-24 (segment ETA models
> beating the offset schedule 35–39%, deployed as the databus served model — full writeup in
> `SESSION_2026-08-24_BUCR_ETA_MODELS.md`) is now **merged to `main`** (was on
> `feat/bucr-eta-models`, merged and pushed this session). That work delivered a
> *bUCR-specific* path through **Phases 2 and 3** — the navsat→canonical adapter (2.2), trip/
> route inference via etaval map-matching (2.3), the trip-quality report (2.4), stale-fix
> dropping (2.5), and the segment dataset builder (3.1) — plus a position-derived schedule
> baseline (3.4, bUCR). **Status columns are now on the Phase 2/3 tables** to record this
> precisely: it is a bUCR slice, *not* the agency-agnostic generalization. Still open: 2.1's
> unified `AgencyConfig`, the MBTA segment reformulation (Phase 3 for MBTA), and the formal
> purge/embargo split + provenance of 3.2/3.3. Also this session: **ewma retired** as a
> serving/default candidate (negative R² at 3 of 4 MBTA sizes, strictly dominated by
> `historical_mean` — kept only for offline ablation via `--models ewma`; the estimator now
> rejects a pinned/selected ewma key rather than silently serving a dominated model).
> **Hygiene item surfaced but not yet done:** the served XGBoost artifact bundles a sklearn
> `ColumnTransformer` + xgboost regressor pickled under sklearn 1.7.2/xgboost 3.1.2, while the
> databus container runs 1.9.0/3.4.1 → version warnings on load (predicts sanely). A clean
> re-export must run under container-matching versions and only lands via a databus redeploy,
> so it is **deferred with the databus work**. 1.7 and 1.8 remain untouched.
>
> **v3 (2026-08-17):** The ETA suite now lives in its own repository,
> **`github.com/dotjae/gtfs-eta`**, extracted from `gtfs-django/eta_prediction` with full
> history preserved across the three branches that touched it (89 / 52 / 77 commits).
> This repo is intended to be the artifact the paper(s) cite. 1.6 done (Bytewax retired).
> Adds **1.7 — reconcile the `gtfs_eta` fork in databus**, which is new scope and the most
> consequential open correctness item: the inference half of this repo was vendored into
> databus and has since diverged *in both directions*, so this repo is no longer canonical
> despite that package's README saying so. Full analysis in `DRIFT_AUDIT.md`. Adds 1.8
> (package restructure), deliberately sequenced after 1.7.
>
> **v3.1 (2026-08-18):** First real end-to-end run of the pipeline this repo was
> extracted with — `docker compose up --build` had never actually succeeded post-extraction,
> and `build_eta_sample` had never been run to completion against the real S3 corpus. Both
> now work. Found and fixed two severe, previously-invisible performance bugs in
> `feature_engineering/` (see *Known defects*) — one made the dataset builder
> non-terminating in practice at any real scale, the other became the new bottleneck once
> the first was fixed. Also fixed the Docker build's stale pre-extraction paths (see
> *Docker / cross-repo build*, below). None of this changes 1.1–1.8's status or the
> databus reconciliation, which is untouched — see *What's next* at the bottom of this
> file for where a fresh agent should pick up.
>
> **v3.2 (2026-08-18, same day):** First complete pipeline pass — four real datasets built
> (1/7/15/28-day windows, routes 222/15/37) and all five model families trained on each
> (20 models total, in the registry now). xgboost is the clear winner throughout (28-day:
> 1.83m MAE, R² 0.803); `ewma` looks cuttable (negative R² at 3 of 4 sizes). Wired `etaval`
> (separate repo, `main` branch) to serve predictions from these trained models via a new
> `GtfsEtaModelPredictor`, verified against real MBTA traffic — found and fixed two real
> bugs along the way (one here, one in etaval) plus documented an environment-specific
> `--no-editable`/`MODEL_REGISTRY_DIR` workaround. See *First real dataset builds + model
> training + etaval wiring*, below, for the full writeup.
>
> **v2 (2026-08-14):** Phase 0.1, 0.2, 0.3, 0.4 and 0.4b done — both collectors rebuilt on
> a spool → hourly staging → daily compaction architecture, weekly static-GTFS snapshots
> for both agencies, and the 28 historical MBTA days re-compacted with dedup, all running
> on / verified against the Hetzner VPS. See *Collector rebuild* below. Also corrected the
> archive's daily-volume estimate, which turned out to be theoretical and wrong (real
> measured volume is several times denser). 0.5 (TripUpdates) remains open.
>
> Adds *Collection strategy*: a 90-day replication window (**2026-08-14 → 2026-11-12**)
> collected in the background while the whole study is rehearsed end-to-end on the
> existing 28 days. Every phase below runs twice.

## Context

The ETA suite currently works as an engineering artifact: a Celery collector writes MBTA
VehiclePositions to Hive-partitioned Parquet on MinIO, a dataset builder joins them
against static GTFS, five model families train against a shared feature contract, and a
serving path publishes predictions to Redis. Separately, `etaval` (standalone repo)
independently scores arrival predictions against detected ground truth.

Two goals now drive the work, and they are deliberately coupled:

1. **Research.** A publishable cross-agency study: *how much does an ETA model trained in
   a data-rich regime (MBTA) transfer to a data-poor one (bUCR), and what is the minimum
   local data needed to beat the local baseline?* Dataset-size ablation, feature ablation,
   and training/inference cost become sections of this paper rather than separate papers.
2. **System.** Ship a first version of bUCR ETA predictions. Models need only be
   reasonable, not optimal — but the pipeline must be genuinely agency-agnostic.

Coupling them is intentional: the paper's cross-agency claim cannot be made without
fixing bUCR, and bUCR cannot ship without the same normalization work. One effort, two
deliverables.

The gap between "works" and "defensible" is large, and this document is the plan to close
it. It is phased by dependency, not by calendar, so scope can be cut at any phase
boundary once submission dates are known.

---

## Current state (verified 2026-08-14)

### Data actually in S3 (`s3://transit/feeds/`)

| Feed | Coverage | Volume | Status |
|---|---|---|---|
| `mbta/vehicle_positions` | 2026-07-01 → 07-29, then 08-14 → | 500 MiB, 4 601 objects | **Live again** (5 s poll, restored 08-14) |
| `bucr/navsat` | 2026-06-30 → present | ~7 000 objects, 44 days compacted | Live, hourly staging |
| `incofer` | — | `incofer.duckdb` only | No RT pipeline |

**16 days are missing from the MBTA archive (2026-07-30 → 08-13)** — the collector was
dead for that window and no feed replay exists. The gap is permanent and must be treated
as a hard discontinuity when constructing splits: it straddles a month boundary, so any
"last N days" window crossing it silently mixes two collection regimes.

Sampled row counts: ~45 M raw rows across the MBTA archive, ~32 M unique
`(vehicle_id, ts)` observations after dedup. Training volume is not the constraint.

### Collector rebuild (2026-08-14) — Phase 0.1, 0.3, 0.4

Both collectors previously wrote to S3 **on every poll**. For MBTA that meant
`COPY ... PARTITION_BY (year, month, day, route_id)` fanning out into ~160 separate PUTs
per poll across the Atlantic:

```
poll_vehicle_positions_s3 succeeded in 331.92s: {'s3_rows': 690}
```

331 s per poll against a 5 s beat schedule. The queue grew unbounded until the worker was
OOM-killed (`Exited 137`) on 07-29, while MinIO accumulated ~500 k objects/day at 2–3
inodes each — the inode exhaustion that took the server down. bUCR had the same shape at
smaller scale: ~1 262 objects/day, each ~4 KiB of mostly Parquet footer.

Both now run one architecture:

```
poll ──▶ local DuckDB spool ──hourly──▶ staging prefix ──daily──▶ curated layout
         (durable, dedup'd)             (1–2 objects/hr)          (compacted, dedup'd)
```

Dedup is applied at all three stages, keyed `(feed_name, vehicle_id, ts)` for MBTA and
`(plate_number, cr_datetime)` for bUCR, keeping the most recently ingested row. The
curated layout is **unchanged**, so `read_vehicle_positions` and the dataset builder need
no modification.

Measured after deployment: polls 0.09–0.55 s (from 331 s); one 322 KiB staging object per
flush in place of ~160 tiny ones; **0 duplicate keys** in the written Parquet. Object
budget over 90 days drops from ~500 k/day to ~164/day.

Observability: `ssh jae@hetzner simovi-status` prints both feeds. Status files at
`/var/lib/simovi/status/<feed>.{json,txt,events.log}` — snapshot for `cat`, event log for
`tail -f`.

Code: `gtfs-django` branch `fix/collector-spool-and-compaction`; `navsat-bridge` now under
version control (it never was before).

### Schema asymmetry — the core research asset

| Field | MBTA (GTFS-RT) | bUCR (navsat AVL) |
|---|---|---|
| vehicle identity | `vehicle_id` | `plate_number` (6 vehicles) |
| `trip_id` | given | **absent** |
| `route_id` | given (partition key) | **absent** |
| stop context | `stop_id`, `current_stop_sequence`, `current_status` | **absent** (`estado` ∈ movimiento/detenido) |
| kinematics | `bearing`, `speed` (m/s) | `speed_kmh`, `odometer_km` (cumulative) |
| time | UTC `datetime64` | strings; `cr_datetime` local CR, `ingested_at_utc` |
| other | — | `lugar` (reverse-geocoded, Spanish) |

bUCR requires inferring route, trip, and stop sequence from raw GPS traces. This is the
data-poor regime the paper is about.

### Known defects (all verified in code)

**Silent correctness bugs**

- **Train/serve timezone skew.** Builder uses `tz_for_temporal="America/New_York"`
  (`dataset_builder.py:142`); estimator uses `_config.default_timezone` →
  `"America/Costa_Rica"` (`core/config.py:126`, set in both compose files). Every temporal
  feature is shifted 1–2 h at inference only. Invisible to offline metrics.
- **Train/serve feature skew.** `estimator.py` substitutes proxies because route geometry
  is unavailable online: `shape_distance_to_stop` ← haversine `distance_to_stop`,
  `shape_progress` ← `progress_ratio`, `cross_track_error` ← `0.0`. The builder computes
  all three from real shapes.
- **Bytewax serving path is dead.** `pred2redis.py:424` passes `shape=` to
  `estimate_stop_times()`, which has no such parameter → `TypeError` on every vehicle,
  swallowed by a broad `except`. Zero predictions, logged as errors. Prefect path is fine.

**Research-blocking methodology defects**

- **Split leakage.** `ETADataset.temporal_split` (`common/data.py:109`) cuts by *row-count
  quantile*. One VehiclePosition fans out to `max_stops_ahead=5` rows sharing an identical
  `vp_ts`, and consecutive VPs of a trip share the same arrival event as their label. No
  grouping by `trip_id`, no purge, no embargo. `sort_values('vp_ts')` uses pandas' default
  unstable quicksort against massive ties, so split boundaries are not reproducible.
- **Comparison is not apples-to-apples.** EWMA calls `model.update(val_df, y_val)` before
  test evaluation (trains on train+val). `polyreg_time` and `xgb` default to
  `handle_nan='drop'` *and* discard features with >30 % NaN, training on a smaller
  non-random subset while scored on the full test set. `historical_mean` does not clip
  predictions; the other four do.
- **Label pooling.** `find_actual_arrival_time` fires within 50 m, but silently falls back
  to closest approach up to **200 m**. No column records which branch fired.
  `arrival_source` (`computed` vs `stopped_at`) is stamped nowhere.
- **Validation split computed and never used.** No early stopping, no model selection, no
  HPO. All hyperparameters are hardcoded constants.
- **No statistics anywhere.** No CIs, no significance tests, no bootstrap. `scipy.stats` is
  never imported. `metrics.py::prediction_intervals` and `error_analysis` are dead code.
- **No schedule baseline.** `scheduled_arrival`/`scheduled_travel_time` were dropped in
  `653e54f`. "Better than the published timetable?" is a mandatory comparator for a transit
  ETA paper.
- ~~**Data duplication — still present in the historical archive.**~~ **Fixed 08-14
  (0.4b).** 5 s polling against a slower feed produced duplicate `(vehicle_id, ts)` rows
  (Green-E, 07-09: 21 090 → 11 376 unique, ~1.85×), and the Aug 1 compaction had merged
  with a bare `SELECT *` and no dedup. All 28 historical days have now been re-compacted
  with dedup: 4,601/4,601 leaves rewritten, 0 errors. A full 28-day scan (not just a
  spot-check) confirms the backfill worked as intended: **32,192,234 rows total, 1,027
  residual duplicate `(feed_name, vehicle_id, ts)` keys (~0.0032%)**, present in every one
  of the 28 days at a consistent ~7–55 keys/day rate. These residuals are not a defect in
  0.4b — they're a structural blind spot, documented below. The archive is otherwise
  internally consistent — old and new days both dedup'd to the same standard.
- **Known residual dedup gap: same `(vehicle_id, ts)` under two different routes.** The
  natural dedup key `(feed_name, vehicle_id, ts)` deliberately excludes `route_id`, but the
  curated layout partitions by `route_id` and compaction dedups per-partition — so a vehicle
  reassigned mid-trip (two polls ~20 s apart, same vehicle+ts, different route_id/trip_id)
  produces one surviving row per route, invisible to per-partition dedup. Confirmed via a
  concrete example (vehicle y3121) to be a live MBTA source-feed artifact, not a pipeline
  bug — and it applies to the ongoing 90-day collection too, not just the historical corpus.
  1,027 / 32,192,234 rows (~0.0032%) is small enough to leave as a known limitation rather
  than re-architecting dedup to be cross-partition; note it in the paper's data-quality
  section rather than fixing it.
- **The historical archive was collected at coarser-than-5s cadence, not 5 s.** The
  original ~563 k rows/day figure was a theoretical estimate (4 fork workers ÷ 331 s/poll,
  the rate right before the OOM crash) and turned out to be wrong — **0.4b's backfill
  measured the real post-dedup total directly**: day 1 = 1,334,616 rows, day 8 = 1,574,494,
  day 15 = 820,659. Real daily volume is ~0.8–1.6 M rows, several times denser than the old
  estimate implied (the 331 s/poll rate clearly didn't hold for the whole month — most of
  it ran faster, with the fan-out only reaching its worst point near the 07-29 crash). Still
  coarser than a true 5 s poll's ~12 M/day, but not by the ~16× the old figure implied. This
  is a *research* fact, not an ops one: matters most for the segment-based reformulation
  (Phase 3) where traversal times are derived from consecutive fixes.
  Either restrict the study window to one regime or downsample the new data to match, and
  say which in the paper.
- **DuckDB silently reinterprets tz-aware timestamps in the host's zone.** Inserting a
  UTC-aware pandas timestamp into a DuckDB `TIMESTAMP` column applies the *host's* offset
  unless `SET TimeZone='UTC'` is issued on the connection — verified empirically: on a
  UTC−6 host, `08:00Z` landed as `02:00`. The collector runs on a German VPS, so this
  would have corrupted every spooled `ts` at collection time, where no downstream fix
  could recover it. Pinned in `spool.py`; the pre-existing Parquet path was unaffected.
  Same failure class as the train/serve timezone skew above — see 1.1.

**Missing archival**

- **Static GTFS is never snapshotted.** Feeds are republished weekly; without dated
  snapshots, historical dataset rebuilds are irreproducible. **This one is genuinely
  unrecoverable** — start snapshotting immediately.
- **TripUpdates are archived nowhere.** `tasks.py` schedules only
  `poll_vehicle_positions_s3`; TU tasks are defined but unscheduled. This does *not* block
  the agency-baseline comparison: `etaval` polls VP and TU concurrently in its live tick
  loop, so a live validation run captures the GTFS-RT predictions in-flight and scores them
  against VP-derived ground truth with no archive at all. What the archive buys is
  **replayability** — scoring many model variants against the *same* TU baseline on the
  *same* trips. Without it, every ablation arm needs its own live run on different days and
  the arms aren't comparable. Cheap to add; do it before the ablation grid (Phase 6.2),
  not urgently.

**Performance defects — found and fixed 2026-08-18**

Discovered while running `build_eta_sample` end-to-end against real S3 data for the first
time since extraction (never previously exercised past unit-test scale). Both were silent —
correct output, just unboundedly slow — so offline metrics or code review wouldn't have
caught them; only running at real volume did.

- **`feature_engineering/dataset_builder.py` STEP 4 was O(V²) per trip, not O(V).** For
  every VP in a trip, for each of up to 5 upcoming stops, it re-sliced and re-scanned the
  *entire remaining VP history of that trip* from scratch (`future_vps = trip_vps[trip_vps['ts']
  > vp_ts]` recomputed on every (VP, stop) pair, each requiring a fresh row-wise
  `.apply(haversine)` pass). A Green Line trip at 5 s cadence over a 30–45 min run is
  ~300–500 VPs; the old code needed ~5×V² row-wise Python calls per trip — for the original
  28-day/Green-D+E rehearsal build (1,794 trips) this never completed even 1 trip in 25+
  minutes before being killed. Fixed by sorting each trip's VPs by `ts` once, then
  precomputing (a) a vectorized numpy VP×stop haversine distance matrix, and (b) an O(n)
  "index of the next match after position i" helper (`_next_true_after`, backward-fill based)
  used once per stop instead of once per (VP, stop) pair. `find_actual_arrival_time` /
  `find_stopped_at_arrival` themselves are untouched (still unit-tested, still usable
  standalone) — only the STEP 4 caller changed. Validated against 400+ randomized
  brute-force-reference trials plus the existing test suite (all pass) before re-running.
- **`feature_engineering/spatial.py`'s `ShapePolyline.project_point` was O(shape_points) per
  call, called ~2–3× per (VP, stop) pair.** Invisible before because the STEP 4 bug above was
  orders of magnitude worse; once that was fixed this became the dominant cost. Fixed by
  precomputing the shape's per-segment planar-projection arrays once in `__init__` and
  vectorizing the "closest segment" search across all segments with numpy instead of a
  Python loop calling `_project_onto_segment` per segment (now-dead code, removed). Validated
  against 600 randomized brute-force-reference trials (shapes up to 400 points, including
  degenerate/duplicate-point segments) plus the existing test suite.
- **Combined effect, measured on real S3 data** (bus routes 15/37/222, chosen for low VP
  volume relative to the Green Line): 1-day build **5m10s → 1m21s** total (3.8×), byte-identical
  output (65,889 rows, same stats) confirming pure perf, no behavior change. 7-day build:
  previously never finished (still 0/1,130 trips after 21+ min); after both fixes,
  **10m9s total**, 641,459 rows, 1,130 trips — the "Match VPs" step itself was 5m51s for
  757,915 (VP, stop) pairs. The original failing case (28-day, Green-D/E, subway-density VPs)
  was not re-attempted this session — extrapolating from the bus-route numbers, expect it to
  be substantially slower than the bus case (denser VPs per trip) but now a *finite, tractable*
  run rather than the previous effectively-unbounded blowup.
- **Not fixed / out of scope this pass:** `models/evaluation/roll_validate.py` and other
  per-row Python loops elsewhere in the codebase were not audited for the same O(n²) pattern;
  worth a targeted look before the ablation grid (Phase 6.2) runs at full 90-day scale.

**Docker / cross-repo build — fixed 2026-08-18**

`docker compose up --build` in `gtfs-rt-pipeline/` had never actually succeeded since the
extraction — it still assumed the pre-extraction nesting
(`gtfs-django/eta_prediction/gtfs-rt-pipeline/`), which no longer exists now that this repo
stands alone as `gtfs-eta/`. Two distinct problems, both fixed without touching the read-only
`gtfs-django` repo itself:
- `gtfs-rt-pipeline/sch_pipeline/models.py` does `from gtfs.models import (...)` — a real,
  load-bearing runtime dependency on `gtfs-django`'s `gtfs` app (the abstract `Base*` GTFS
  models), not just a training-time one. `gtfs-rt-pipeline/pyproject.toml`'s editable-path
  source for the `gtfs-django` dependency (`{ path = "../.." }`) assumed the old two-levels-up
  nesting; it's a true sibling now, so the path must be `../../gtfs-django`.
- `docker-compose.yml` / `Dockerfile` hardcoded `eta_prediction/...` paths throughout (build
  context, `COPY` sources, container working dirs). Fixed by mirroring the two repos as true
  siblings inside the image (`/app/gtfs-django`, `/app/gtfs-eta/...`), matching how they sit
  on the host, so the same relative-path convention (`../../gtfs-django`) resolves identically
  in both places. `feature_engineering/dataset_builder.py:12`'s `from core.config import ...`
  also needed `core/` added to the image's `COPY` list — it wasn't there before either
  (pre-existing gap, not a regression).
- This is a stopgap, not a structural fix — 1.8's package restructure is still the right place
  to actually remove the `gtfs-django` runtime dependency (vendor or reimplement the `Base*`
  models this repo actually needs) so `gtfs-eta` is genuinely standalone. Until then, anyone
  building this image needs a local checkout of `gtfs-django` as a sibling directory.
- **Also found: no `.gitignore` existed anywhere in this repo's history.** `.env` (real AWS
  S3 credentials) had zero protection against an accidental `git add -A`. Added one
  (root `.gitignore`), covering `.env*`, `__pycache__/`, `.venv/`, `*.egg-info/`,
  `celerybeat-schedule`, trained-model artifacts (`*.pkl`, `models/trained/`), and generated
  datasets (`datasets/*.parquet`, excluding the tracked `datasets/sample.parquet`).

**First real dataset builds + model training + etaval wiring — 2026-08-18**

With the Docker build and both perf fixes above validated, ran the first real,
complete pipeline pass: four datasets at 1/7/15/28-day windows (all anchored at
2026-07-01, the start of the continuous 28-day corpus), routes 222/15/37 (MBTA
buses — deliberately not Green-D/E, to keep iteration cheap), built via
`build_eta_sample --no-weather` and saved as
`datasets/mbta_bus_222_15_37_{1,7,15,28}d.parquet` (128K / 558K / 1.08M / 2.03M
rows). All five model families trained globally (not per-route) on each dataset
size via `models/train_all_models.py`; all 20 resulting models are in
`models/trained/registry.json`.

| Days | xgboost MAE | xgboost R² | polyreg_time MAE | polyreg_time R² |
|---|---|---|---|---|
| 1  | 3.00m | 0.542 | 6.35m | 0.257 |
| 7  | 2.08m | 0.801 | 2.71m | 0.689 |
| 15 | 1.94m | 0.863 | 2.85m | 0.730 |
| 28 | 1.83m | 0.803 | 2.46m | 0.698 |

xgboost wins outright at every size; both peak in R² at 15 days then dip
slightly at 28 (MAE still improves) — read as ordinary week-to-week variance in
real data, not a regression. `historical_mean` is weak-but-positive (R² 0.206–
0.324, needs no live kinematic features — the one model still useful as a
cold-start fallback). `ewma` has negative R² at 3 of 4 sizes and is strictly
worse than `historical_mean` everywhere — no scenario found where it wins;
candidate for cutting entirely rather than carrying forward.

**Data-quality finding:** `current_speed_kmh` is uniformly `0.0` for all three
bus routes, confirmed at the raw MBTA VehiclePosition feed level (`speed=0.0`
even while `current_status=IN_TRANSIT_TO`) — not a builder bug. The "kinematic"
feature group these models actually learn from is position/bearing/distance
only; speed contributes nothing for MBTA buses specifically. Unknown whether
this holds for MBTA subway (Green-D/E) or is bus-specific — worth checking
before trusting `current_speed_kmh` on any other route class.

**etaval wiring (on `etaval`'s `main` branch — not the older, unmerged
`feat/model-validation` branch mentioned below):** added
`etaval/predictors/gtfs_eta_models.py`, a `GtfsEtaModelPredictor` wrapping this
repo's `eta_service.estimator.estimate_stop_times`, registered in
`etaval/engine/assembly.py`'s `PREDICTOR_REGISTRY` as
`gtfs_eta:{historical_mean,ewma,polyreg_distance,polyreg_time,xgboost}`.
Verified end-to-end against real MBTA traffic (`etaval run --predictor
gtfs_eta:xgboost --predictor gtfs_eta:polyreg_time ...`; `etaval report` shows
both scored alongside `gtfsrt`/`baseline:constant_speed` in the same table).
Two bugs found and fixed along the way, both in `etaval` (not this repo):
- `models/common/registry.py:39` here (this repo) — default model path was
  cwd-relative (`"models/trained"`), silently wrong for any out-of-tree
  consumer. Fixed to fall back through `core.config.get_config().model_registry_dir`
  instead. Still not sufficient on its own for etaval specifically — see below.
- `etaval/spatial/polyline.py`'s `build_polyline` crashed on real MBTA shapes
  where `shape_dist_traveled` is blank for a subset of points within an
  otherwise-populated shape (checked only the first point, assumed the rest
  matched). Pre-existing, blocked *all* live etaval runs against MBTA, not just
  this work. Fixed to fall back to cumulative haversine when any point lacks it.
- **Environment quirk, not a code bug, still requires a documented workaround:**
  on at least one dev machine (Python 3.14 + uv-managed venv), `.pth`-based
  editable-install activation silently never fires at interpreter startup —
  reproduced independent of `uv` by invoking the venv's `python` binary
  directly with a stripped env. This breaks etaval's own console-script
  entrypoint unless installed with `uv sync --no-editable`, which in turn
  means this repo's packages get physically copied into etaval's venv, which
  breaks `core.config._find_project_root()`'s walk-up-for-pyproject.toml
  heuristic (finds etaval's own pyproject.toml first, not this repo's). Net
  effect: etaval requires both `--no-editable` *and* an explicit
  `export MODEL_REGISTRY_DIR=.../gtfs-eta/models/trained` — see etaval's
  `README.md` for the full writeup and exact commands. Nothing here to fix on
  this repo's side; `models/common/registry.py`'s fix above is still correct
  and worth keeping for the in-tree/editable case.

### Assets already built and unused

- **`etaval` branch `origin/feat/model-validation`** (8 commits, +3 914 lines, unmerged):
  `MLModelPredictor` wrapping `estimate_stop_times`, feeding etaval's authoritative
  along-shape distances into the estimator so models and baselines share identical
  geometry; stops-ahead bucketing; ground-truth detector as a run parameter. **Confirmed
  2026-08-17: this branch's vendored `gtfs_eta` copy (`vendor/gtfs-eta/`) matches databus's
  design, not this repo's** — same precomputed-distance hook, same `progress_on_segment`/
  weather-augmented schema, same basename-only registry paths. It also ships a `_compat.py`
  pickle shim for loading models trained on a third gtfs-django lineage
  (`feature/eta-prediction/core`), which genuinely has its own `weather` `FEATURE_GROUPS`
  entry — but the one real trained pickle found from that lineage (in a separate,
  previously-untracked repo, `github.com/simovilab/gtfs-rt-db`) has `include_weather: false`
  in its own metadata, so weather was never actually used by any known trained model on any
  lineage. See 1.7's entry above for the full chain of evidence. Relevant to 1.7's merge:
  etaval is a *second* thing that will need to adapt to this repo's schema, not just databus.
- **`models/evaluation/roll_validate.py`**: a correct calendar-windowed walk-forward
  backtester reporting `mean ± std` across windows — the only dispersion estimate in the
  repo. Orphaned (no `__init__.py`, no caller, no test).
- **`etaval/spatial/polyline.py`**: `project_point_to_polyline`, `assign_stops_monotonic`
  (loop-back-safe DP) — exactly the map-matching needed for bUCR trip inference.
- **`gtfs/fixtures/example.json`**: bUCR static GTFS (2 routes, 22 stops, 896 shape
  points, 130 trips). Referenced by nothing.

---

## Collection strategy — rehearse on 28 days, replicate on 90

**Decided 2026-08-14.** The 90-day collection window runs **2026-08-14 → 2026-11-12**.
Rather than wait for it, the existing 28 July days become a **rehearsal corpus**: every
phase below — dataset build, splits, models, evaluation, figures, and a complete paper
draft — is executed end-to-end against those 28 days first. When the 90 days land, the
entire study is re-run against them.

Why this is the right shape:

- **The critical path stops being data collection.** Everything except the final numbers
  can be finished while the collector runs. On 2026-11-12 the remaining work is a re-run,
  not a research project.
- **Every pipeline defect surfaces on cheap data.** A split-leakage bug or a broken
  ablation arm found in September costs a re-run; found in November it costs the schedule.
- **The final run becomes a reproducibility artifact.** Executing the identical pipeline
  over a second, larger, independently collected window is exactly the robustness check
  reviewers ask for — obtained for free as a by-product of the plan.

**What does *not* transfer from rehearsal to replication.** The two corpora are not the
same kind of data, and treating rehearsal numbers as preliminary results would be wrong:

| | Rehearsal (28 days, 2026-07) | Replication (90 days, 08-14 →) |
|---|---|---|
| effective cadence | ~80 s between fixes | 5 s |
| duplication | 1.85×, fixed 08-14 (0.4b) | deduplicated at write |
| rows/day | ~563 k | ~3–4 M unique |
| static GTFS | no dated snapshot | snapshotted from 0.2 onward |

So the rehearsal validates **pipeline correctness, experimental design, and code paths**.
It does *not* produce transferable accuracy numbers, and anything sensitive to fix density
— above all the segment-based reformulation in Phase 3, where traversal times come from
consecutive fixes — will behave differently on the real corpus. Report only replication
numbers; use the rehearsal to fix the *method*.

**Consequences for the phases below:**

- **0.2 (static GTFS)** is done — see Phase 0 below.
- **0.4b (re-compact the 28 days with dedup)** is done — see Phase 0 below. Also
  corrected the archive's known daily-volume estimate along the way: real measured
  volume is several times denser than the old theoretical figure (~563k/day) implied.
- **Phase 6.1 freezes *two* test periods**, one per corpus, using identical selection
  logic.
- Down-sampling the replication corpus to ~80 s is worth running as a **deliberate
  ablation arm** rather than a correction — it directly measures what fix density buys,
  which is a paper-worthy result in its own right and turns the July/August cadence
  discontinuity from a liability into a finding.

---

## Research thesis

> **Transit ETA prediction under data-quality asymmetry.** Learned models trained on a
> data-rich agency feed (MBTA: standardized GTFS-RT with trip/stop assignment) are compared
> against the agency's own published predictions and against the same model families
> applied to a data-poor feed (bUCR: bespoke AVL with no trip, route, or stop context).
> We quantify what the inference pipeline costs in accuracy, how much history a new agency
> needs before learning beats its local baseline, and the accuracy/latency trade-off for
> real-time deployment.

**Baselines (non-negotiable, in priority order)**

| # | Baseline | Availability |
|---|---|---|
| 1 | Agency's own GTFS-RT TripUpdates ETA | Live via `etaval`; archive TU to replay across ablation arms |
| 2 | Published timetable / schedule-derived ETA | Removed in `653e54f` — restore in 3.4 |
| 3 | Constant-speed white-box | Built into `etaval` |
| 4 | Historical mean, EWMA | Existing models |

**Paper sections that fall out of the main study**

- Dataset-size ablation: 1 / 7 / 15 / 30 / 60+ days → "how much history does a new agency need?"
- Feature ablation across `FEATURE_GROUPS`
- Formulation ablation: direct stop-level regression vs. segment-decomposed
- Training and inference cost (accuracy per millisecond, accuracy per dollar)

**Venue targets.** IEEE ITSC, TRB Annual Meeting, or *Transportation Research Part C*. For
application purposes an arXiv preprint captures most of the value at a fraction of the
latency — target the preprint first, submit after.

---

## Phase 0 — Stop the bleeding

**Date-independent. The static-GTFS gap is now the only item still losing unrecoverable
data.**

| # | Task | Size | Status |
|---|---|---|---|
| 0.1 | Restart the MBTA collector; add liveness monitoring | S | **Done 08-14.** Rebuilt rather than restarted — the original design was the cause of death. 5 s poll, `expires` so a backlog can never re-accumulate, `simovi-status` + status files for liveness |
| 0.2 | Weekly static-GTFS snapshot task → `feeds/<agency>/gtfs_static/<ISO date>.zip`, for both agencies | S | **Done 08-14.** `rt_pipeline.storage.static_gtfs` + `snapshot_static_gtfs` task (maint queue, Mondays 04:00 UTC). Deployed and manually triggered on the VPS: MBTA snapshot 18.4 MB, bUCR 31 KB, both verified present in MinIO via `mc stat`. Caught and fixed one bug in verification — the bUCR snapshot's status was first written under status name `bucr`, which `simovi-status` never reads (the live collector and monitor both key on `navsat`); fixed and re-verified. `simovi-status` now shows a `static gtfs` row per feed |
| 0.3 | Fix the bUCR writer: batch to hourly objects instead of one row per file | M | **Done 08-14.** Durable DuckDB spool, hour-boundary flush, staging prefix. Window widened to 06:00–23:00 CR |
| 0.3b | Re-partition bUCR by *event* date (`cr_datetime`), not ingestion date | M | **Open, deliberately deferred.** Stale device fixes (the 07-01 file holds `cr_datetime` back to 06-05) mean event-date partitioning has to merge into arbitrary past days. A `cr_datetime_utc` column was added instead, so this can be resolved in the builder rather than the collector |
| 0.4 | Backfill-compact existing bUCR objects | S | **Done** — 44 days compacted, 1/day |
| 0.4b | Re-compact the 28 existing MBTA days **with dedup** | S | **Done 08-14.** Added a `--force` flag to compaction (the routine skip-guard treats any leaf holding a compacted `<date>.parquet` as already done, which blocked re-running dedup on it). Ran live: 4,601/4,601 leaves rewritten, 0 errors. Full 28-day verification scan: 32,192,234 rows, 1,027 residual duplicate keys (~0.0032%) — all attributable to the cross-route reassignment gap noted above, not a backfill defect. Also corrected the record — the old "15.7 M → ~8.5 M" estimate was theoretical and wrong; real measured per-day post-dedup volume is 0.8–1.6 M rows (day 1/8/15 sampled directly), several times denser than assumed |
| 0.5 | Schedule `poll_trip_updates_s3` → `feeds/<agency>/trip_updates/`, mirroring the VP storage layer. Not urgent for the head-to-head, which `etaval` does live — this exists so the ablation grid (6.2) can replay one fixed window | M | Open |

**Verification (0.1/0.3/0.4, confirmed on the VPS 2026-08-14):** MBTA poll age 0.3 s and
0 failures; bUCR 0 failures in 88 polls; one hourly staging object per feed; a forced
flush wrote 59 776 rows in 3.9 s; a 23 157-row staging object held 0 duplicate keys with
`ts` in correct UTC. Docker log rotation added (there was none anywhere on the host).

**Still unproven:** the first *automatic* daily compaction. It has been dry-run in-container
(43 days already compacted, 0 errors) but has not yet run on its 03:15 UTC schedule.

## Phase 1 — Correctness

*Blocks all modeling work. Nothing measured before this is trustworthy.*

| # | Task | Size |
|---|---|---|
| 1.1 | ~~Make timezone and holiday region per-agency config, sourced from one place, consumed identically by builder and estimator.~~ **Done 08-16.** Added `AGENCY_TEMPORAL_DEFAULTS` to `core/config.py` (single table, keyed by agency) and an `ETA_AGENCY` env var (default `mbta`); `ProjectConfig.default_timezone`/`default_region` and `dataset_builder.build_vp_training_dataset`'s new `agency` param both resolve through it. Found and fixed a **live train/serve skew**: `docker-compose.yml`/`docker-compose.full.yml`/`.env.example` pinned `eta`, `eta-dev`, `eta-cli`, `prefect-flow`, and `bytewax` to `ETA_TIMEZONE=America/Costa_Rica`, so the deployed estimator was computing `hour`/`is_weekend`/`is_holiday`/`is_peak_hour` in Costa Rica time+holidays for every live MBTA prediction, while `dataset_builder` trained on America/New_York + US_MA — corrected to `ETA_AGENCY=mbta` everywhere. Added parity tests in `core/tests/test_core.py::TestAgencyTemporalParity` asserting the same timestamp produces identical temporal features via both resolution paths | S |
| 1.2 | Resolve the train/serve geometry skew: either ship shapes to the serving path (preferred — `etaval`'s `MLModelPredictor` already does this) or drop the three shape features from the contract. Do not keep silent proxies | M |
| 1.3 | Record the arrival-detection branch as a dataset column (`arrival_method` ∈ `within_50m`/`closest_approach_200m`/`stopped_at`); stamp `arrival_source` into dataset metadata and `ModelKey` | S |
| 1.4 | ~~Deduplicate `(feed_name, vehicle_id, ts)` at write time~~ — **done 08-14** (spool `ON CONFLICT`, flush, and compaction), and the 28 historical days are now deduplicated at rest too (0.4b). Remaining, lower-stakes: make `dedup=True` the non-optional default on *read* rather than an opt-out, as defense in depth | S |
| 1.5 | Make `backfill_s3` idempotent (delete-then-write per partition, or content-hash filenames) | S |
| 1.6 | ~~Fix or retire the Bytewax path. Retiring is defensible — Prefect works and two serving paths is one too many for a solo project.~~ **Done 08-17 — retired.** Removed `bytewax/`, `docker/Dockerfile.bytewax`, and its compose services. It was already dead in production: `pred2redis.py` called `estimate_stop_times(..., shape=...)`, but no `shape` parameter exists on this repo's estimator, so every prediction raised `TypeError` into a broad `except` and was silently swallowed. That is independent confirmation of the divergence in 1.7 — the `shape=` kwarg exists *only* in databus's vendored copy. The MQTT→Redis bridge (`subscriber/mqtt2redis.py`) is not Bytewax-specific and survives as `mqtt-subscriber/`; `cache-seeder` now runs `prefect/mock_stops_and_shapes.py` | S |
| 1.7 | **Reconcile the `gtfs_eta` fork in databus.** The inference half of this repo (`core/`, `eta_service/`, `feature_engineering/`) was vendored into databus as `backend/gtfs-eta/gtfs_eta/` and has since diverged **in both directions** — this repo is *not* canonical, despite that package's README saying so. Vendored-only and load-bearing: **(a)** shape-aware distance/progress (`_progress_features_with_shape`, cross-track error, shape-projected distance); **(b)** a **precomputed-distance hook** — databus passes a loop-back-safe monotonic `shape_distance_to_stop` on every upcoming stop and the vendored estimator consumes it as authoritative. That branch runs on *every* production prediction and has no counterpart here, so substituting this repo's estimator would not error — it would silently ignore the field and fall back to haversine, which can *decrease* near a loop-back while the vehicle is still progressing; **(c)** a real bugfix — this repo resolves stop sequence via falsy `or`-coalescing, so a valid `stop_sequence == 0` is relabelled `1` and collides with a real stop 1. Held only here: bearing/speed/cyclical-time features that the vendored copy dropped. **Seam that must not break:** `estimate_stop_times(..., shape=)` plus the `gtfs_eta.feature_engineering.spatial.ShapePolyline` import path, called from databus `runs/domain/progression/stop_times.py`. ~~**Blocking open question:** the `polyreg_time`/`xgb` branches pass *disjoint* feature schemas on the two sides (kinematic here, weather-augmented there) — dormant while only `polyreg_distance` is live, but not resolvable without reading `models/polyreg_time/predict.py` and `models/xgb/predict.py` to establish which schema the trained pickles actually expect.~~ **Settled 08-17: this repo's kinematic/bearing/cyclical schema is correct; databus's weather-augmented branch was speculative and never matched a trained pickle.** `models/polyreg_time/predict.py` and `models/xgb/predict.py` have byte-identical signatures — `distance_to_stop`, `distance_to_next_stop`, `shape_distance_to_stop`, `shape_progress`, `cross_track_error`, `progress_ratio`, `stops_ahead`, `current_speed_kmh`, `bearing_to_stop`, `bearing_diff`, `is_at_stop`, `hour`, `day_of_week`, `is_peak_hour`, `is_weekend`, `is_holiday`, `hour_sin/cos`, `dow_sin/cos` — with no `temperature_c`/`precipitation_mm`/`wind_speed_kmh` anywhere. `models/common/data.py::ETADataset.FEATURE_GROUPS` (what `train.py` on both model families actually fits against, via `_get_feature_groups`) has no `weather` group at all. `feature_engineering/dataset_builder.py:542–548` confirms why: weather columns are a dated, explicit stub — "Weather features — STUB (intentionally disabled)... needs rethinking (source, caching, leakage) and is out of scope. Re-add ... when ready, then register them in `FEATURE_GROUPS`." `fetch_weather()` in `weather.py` exists and is unit-tested, but its output has never reached a trained model. So databus's weather branch would both feed a model columns it never saw in training and omit several it was trained on (`distance_to_next_stop`, `bearing_to_stop`, `bearing_diff`, `is_at_stop`, `stops_ahead`, `hour_sin/cos`, `dow_sin/cos`, `cross_track_error`, `shape_progress`) — confirming it was written against a schema that was never deployed. **Consequence for the estimator merge:** when `_predict_with_model`'s `polyreg_time`/`xgboost` branches are merged, keep this repo's kinematic schema; drop vendored's hardcoded weather placeholders (`25.0`/`0.0`/`None` literals in `estimator.py`) rather than reconciling them — they don't correspond to anything a model was ever fit on. **Corroborated 08-17 via `etaval` and a third, previously-untracked repo:** `etaval`'s unmerged `origin/feat/model-validation` branch vendors its own `gtfs_eta` copy (`vendor/gtfs-eta/`) that matches *databus's* design exactly — same precomputed-distance hook (nearly verbatim comment), same `progress_on_segment`/weather-augmented `_predict_with_model` schema, same basename-only registry paths — not this repo's. Its `models/_compat.py` documents *why*: it's a pickle-compat shim for models trained on gtfs-django branch `feature/eta-prediction/core` (a third lineage, distinct from the branches this repo was extracted from), which really does define a `weather` `FEATURE_GROUPS` entry. That could have overturned the "never matched a trained pickle" claim above — but checking the one real artifact that exists, `github.com/simovilab/gtfs-rt-db` (a separate, previously-unmentioned repo with actual trained `.pkl`s from 2025-10/11 on MBTA Green Line data), the real `polyreg_time` model's own metadata records `"include_weather": false` and a 6-feature set (`distance_to_stop`, `hour`, `day_of_week`, `is_weekend`, `is_peak_hour`, `current_speed_kmh`) — a strict subset of this repo's schema, and weather-free. No `xgb` pickle exists there either. So: three schemas now on record (this repo's 19-feature kinematic/shape/cyclical, databus/etaval's 11-param weather/`progress_on_segment`, and `gtfs-rt-db`'s 6-feature bare-bones one), but every artifact anyone has actually trained uses a weather-free schema, and this repo's superset schema is the only one of the three that could directly load and serve the one real `polyreg_time` pickle found so far (its 6 features are all valid optional kwargs in this repo's `predict_eta`) — *modulo* pickle module-path compatibility, not verified. Also latent: `historical_mean`/`ewma`/`polyreg_time`/`xgb` have no `predict.py` in the vendored tree, so any non-`polyreg_distance` registry entry is a live `ModuleNotFoundError` in production. **Direction (decided 08-17): harvest databus's implementation, but `gtfs-eta` holds design authority — databus adapts to it, not the reverse.** Two separable questions. *Whose code is the better starting point:* databus's, for the serving surface — adopt its `eta_service/estimator.py`, its `models/common/registry.py` path scheme, and the `polyreg_distance` `model.py`/`predict.py` split. The estimator's three-tier distance design (precomputed override → `ShapePolyline` projection → haversine fallback) is genuinely general, not a databus quirk, and the middle tier is what keeps the library usable standalone by a replicator. *Who governs the API going forward:* this repo, because it is the published library and the artifact the paper cites, and because databus already consumes it as a workspace dependency whose own README keeps the seam narrow so extraction "stays mechanical". Two things a wholesale adopt would get wrong: **`core/*` stays this repo's** — databus's `config.py` hardcodes `DEFAULT_TIMEZONE = "America/Costa_Rica"` / `DEFAULT_REGION = "CR"` at module level, which is right *for databus* (it serves bUCR) and wrong for a library; the 1.1 agency abstraction is the correct design and databus resolves it by setting `ETA_AGENCY=bucr`. Adopting it verbatim reverts `f074949`. **`feature_engineering/spatial.py` stays this repo's superset** — databus dropped `load_shape_from_gtfs`/`load_shape_for_trip`, which `dataset_builder.py:15,267` imports; adopt the vendored *bodies* of the retained functions inside this file. Note the precomputed-distance hook is *not* databus internals leaking into the library: databus owns run progression (`project_point_to_polyline` plus per-stop `progress_m`, `stop_times.py:122–144`) as a core responsibility, so passing the derived distance in is a clean split — the library only has to keep working without it. **First action, before any module-level merge: delete databus's vendored `backend/gtfs-eta/` and replace it with a real dependency on this repo.** The copy is the drift generator; every day it survives, the reconciliation grows. The adaptation surface in databus is genuinely small — `runs/domain/progression/stop_times.py` (two lazy imports, one call), three patch targets in `tests/test_stop_times_producer.py`, and two `pyproject.toml` entries. **Order:** unify the registry path scheme (adopt the vendored basename storage — it is the portable one) → line-diff `spatial.py`'s retained bodies → settle the feature-schema question → merge `estimator.py` → port the missing `predict.py` siblings. Full analysis: `docs/DRIFT_AUDIT.md` | **L** |
| 1.8 | **Package restructure** — split into installable packages under a `uv` workspace, namespaced `gtfs_eta.*` to match what databus already imports, and rename the distribution from `eta_prediction` to `gtfs-eta` (confirmed available on PyPI 08-17, unclaimed). Dependency *metadata* was already split into `train`/`collect`/`viz` extras on 08-17 — base install verified at 12 packages with no Django, celery, xgboost, matplotlib or psycopg — so what remains is moving modules. **The dependency direction is already acyclic and correct**: collector imports nothing from training/inference, and the only cross-package edges are 5 sites in `feature_engineering/` reaching into the collector. Work, in order: **(a)** `dataset_builder.py:11` `from sch_pipeline.models import StopTime, Stop, Trip` (+ `:10` `django.db.connection`) — the substantive one. Training reaches into the collector's Django ORM for static GTFS, so no dataset can be built without Django + Postgres standing up, which is exactly what makes a paper artifact unreproducible. Fix by reading the weekly S3 static-GTFS snapshots the collector already writes (0.2, done 08-14); overlaps Phase 2. **(b)** Promote `rt_pipeline/storage/` (and `compaction/`) out of the collector's Django app into a shared data-access package — it is already 100% framework-free (~2,100 lines, zero Django references), so `rt_source.py:17`'s import of it is a mislabelled seam rather than a leak; this likely makes the split four packages (`storage` / `collect` / `train` / `infer`) rather than three. **(c)** `weather.py:3` `django.core.cache` → `functools.lru_cache` or a small protocol; trivial. **(d)** The split cuts *through* `models/`: `*/predict.py` + `common/registry.py` + `common/utils.py` are inference, while `*/train.py` + `train_all_models.py` + `common/{data,keys,metrics}.py` + `evaluation/` are training — mechanical, but touches all five model families. **(e)** Adopt a `src/` layout; nothing is namespaced `gtfs_eta.*` yet. **Deliberately sequenced after 1.7** — the restructure moves exactly the files 1.7 has to merge, and doing both at once turns a two-way merge into a two-way merge across renamed paths | M |

**Verification:** the temporal-parity test passes; a rebuilt dataset has an
`arrival_method` column with a sane distribution; re-running `backfill_s3` over a date
range leaves row counts unchanged.

## Phase 2 — Agency-agnostic pipeline

*Unblocks bUCR, and is the system deliverable in its own right.*

| # | Task | Size | Status |
|---|---|---|---|
| 2.1 | Introduce an `AgencyConfig` (feed name, S3 prefix, timezone, holiday region, feed protocol, static GTFS source). Remove hardcoded `feeds/mbta/...` from `storage/schema.py` and `region="US_MA"` from `dataset_builder.py:498` | M | **Partial.** Per-agency *temporal* and *served-model* defaults live in `core/config.py` (`AGENCY_TEMPORAL_DEFAULTS`, `AGENCY_MODEL_DEFAULTS`, resolved via `ETA_AGENCY`); the unified `AgencyConfig` (feed name, S3 prefix, feed protocol, static-GTFS source) and removing the hardcoded `feeds/mbta/...`/`region="US_MA"` are **not** done — bUCR currently runs through dedicated `navsat_*`/`bucr_*` modules rather than a shared config |
| 2.2 | `navsat` → canonical VP adapter: parse `cr_datetime` with explicit CR tz, map `plate_number` → `vehicle_id`, `speed_kmh` → m/s, `estado` → `current_status`, retain `odometer_km`. Emit the canonical 12-column frame `write_vehicle_positions()` expects | M | **Done (bUCR), 08-24.** `feature_engineering/navsat_adapter.py` + `navsat_normalize.py` (+ tests) |
| 2.3 | **bUCR trip/route inference.** Load bUCR static GTFS; map-match each plate's trace onto route shapes using `assign_stops_monotonic`; segment traces into trip instances; derive `route_id`, `trip_id`, `current_stop_sequence`, `stop_id`. Port or import `etaval/spatial/polyline.py` rather than reimplementing | **L** | **Done (bUCR), 08-24.** `bucr_gtfs.py` + `bucr_trip_inference.py` map-match via etaval polyline; 48.5% of points assigned → 3,094 trips (+ tests) |
| 2.4 | Quality report for inferred bUCR trips: match rate, ambiguous assignments, dropped traces. This becomes a paper table | S | **Done (bUCR), 08-24.** `bucr_quality_report.py` + `bucr_trip_scoring.py` (+ tests) |
| 2.5 | Drop stale device fixes (`cr_datetime` far from `ingested_at_utc`) with a recorded threshold and drop-rate | S | **Done (bUCR), 08-24.** `navsat_cleaning.py`; ~0.1% of rows dropped |

**Verification:** `build_eta_sample --agency bucr` produces a dataset with the same 37-column
schema as MBTA; trip-inference match rate is reported and defensible; a spot-check of
inferred trips against the timetable looks plausible.

**Risk:** 2.3 is the single largest and least certain item. With 6 vehicles on 2 routes the
matching problem is tractable, but validating inference quality without ground-truth trip
labels is genuinely hard. Budget generously; consider hand-labelling a day of traces as a
validation set.

## Phase 3 — Segment-based reformulation

*The core modeling change. Chosen as primary formulation.*

Replace the stop-level target (`time_to_arrival_seconds` per VP × up-to-5-stops-ahead)
with **stop-to-stop segment traversal time**. Stop-level ETA is then derived by summing
predicted segment times along the remaining path.

This is not just a modeling preference — it structurally fixes the leakage in §Current
state: one observation per segment traversal instead of five correlated rows per VP, and a
natural sequence for a recurrent model to consume.

| # | Task | Size | Status |
|---|---|---|---|
| 3.1 | Segment dataset builder: one row per (trip instance × segment), target = observed traversal seconds. Keep the stop-level builder intact for the formulation ablation | **L** | **Done (bUCR), 08-24.** `feature_engineering/segment_dataset_builder.py` (+ 12 tests); one row per (trip × segment), **position-derived** target so the offset timetable can't poison labels. MBTA stop-level builder kept for the formulation ablation |
| 3.2 | Correct splitting: calendar-boundary temporal splits with `trip_id` grouping and an explicit purge/embargo gap. Stable sort with a deterministic secondary key. Replace `temporal_split` and the duplicated `_temporal_split_df` | M | **Partial (leakage probe done 08-30).** The bUCR baseline uses a clean **day-grouped** holdout (train 33 d / test 5 d), which keeps whole trips inside one split, and the shuffle-label **leakage probe now passes** (`models/evaluation/leakage_probe.py`: control 33.4 s/R² 0.694 vs shuffled 72.3 s/R² −0.046 collapsing to the 71.2 s naive baseline → no target leakage). Still open: the formal calendar-boundary split with explicit purge/embargo, the stable secondary sort, and replacing `temporal_split`/`_temporal_split_df` (the original MBTA stop-level leakage) |
| 3.3 | Determinism and provenance: global seeding, seed recorded in metadata, plus git SHA, dataset content hash, library versions, split boundaries, `arrival_source`, and the exact feature list | S | **Done 08-30.** `models/common/provenance.py` stamps git SHA, sklearn/xgboost/numpy/pandas/python versions, dataset sha256, seed, and feature list into every saved model's metadata (wired centrally in `registry.save_model`); `models/tests/test_provenance.py` includes a fixed-seed reproducibility check (identical metrics across runs). Remaining minor: `arrival_source` stamping is tied to 1.3 (dataset-level), not this item |
| 3.4 | Restore a schedule-derived baseline (as a *comparator*, and optionally as features) | M | **Partial.** bUCR has an offset-invariant, position-derived schedule baseline (`scratchpad/baseline_compare_corpus.py`); restoring the schedule-derived baseline in the **MBTA** builder (removed in `653e54f`) is not done |

**Verification:** a leakage probe — train on a shuffled-label variant and confirm metrics
collapse to baseline; confirm no `trip_id` appears in more than one split; re-running
training twice with a fixed seed produces byte-identical metrics.

## Phase 4 — Evaluation authority

*`etaval` becomes the single source of measured truth for every model and baseline.*

| # | Task | Size |
|---|---|---|
| 4.1 | Merge / rebase `etaval` `origin/feat/model-validation` onto `main`; re-import real trained models (registry `.pkl`s are gitignored; the only tracked entry is a synthetic-data baseline) | M |
| 4.2 | **S3 replay source for `etaval`**: a `FeedSource` that replays the archived VP + TU Parquet as `FeedSnapshot`s, so a month of history can be scored offline through the same engine as live runs | M |
| 4.3 | Statistical layer: bootstrap CIs on MAE, paired tests across rolling windows (Diebold-Mariano or paired Wilcoxon) for model-vs-model claims, multiple-comparison correction. Revive `prediction_intervals` and `error_analysis` | M |
| 4.4 | Wire `roll_validate.py` in: shared windows across all models in one run, seeded, results persisted to a file. Reconcile the leaderboard's 70/15 split with training's 70/10/20 | M |
| 4.5 | Ground-truth sensitivity: report headline metrics under all three detectors, not just `shape_distance` | S |

**Verification:** one command scores every model plus the GTFS-RT and schedule baselines on
the same held-out period, emitting a table with CIs and horizon stratification; the
timezone bug from §1.1 would have been caught by it (regression-test the fix this way).

## Phase 5 — Models

| # | Task | Size |
|---|---|---|
| 5.1 | Hyperparameter tuning on the *validation* split (Optuna with pruning; CPU-friendly budgets locally, GPU rental for the heavier sweeps). Record every trial | M |
| 5.2 | **LSTM/GRU over segment sequences.** Small enough to train on a rented GPU in hours. This is the deep-learning entry the AI-masters framing wants | **L** |
| 5.3 | Fix the apples-to-apples breaks: EWMA must not see validation data; unify NaN handling and prediction clipping across all model families | S |
| 5.4 | Cost instrumentation: training wall-clock and cost, inference p50/p95/p99 latency per model, measured through the serving path rather than the stale `bytewax/profiling/*.csv` snapshots | M |

**Verification:** every model trained on identical splits with recorded seeds; the
leaderboard reproduces from a clean checkout; latency numbers come from a live run.

## Phase 6 — Paper

| # | Task | Size |
|---|---|---|
| 6.1 | Freeze a held-out final test period, untouched during all development | S |
| 6.2 | Run the full experiment grid: {models} × {agencies} × {history sizes} × {feature sets} × {formulations}. Needs the TU archive from 0.5 so every arm replays the same window | M |
| 6.3 | Figures: error-vs-horizon curves, convergence, per-route breakdowns, ablation tables, cost/accuracy Pareto. Reuse `etaval`'s `@unovis` dashboard for exploration; generate publication figures from the Parquet | M |
| 6.4 | Reproducibility bundle: pinned environment, dataset hashes, seeds, one-command rebuild | M |
| 6.5 | Draft → arXiv preprint → venue submission | **L** |

---

## Sequencing

```
Phase 0 ──┬─→ Phase 1 ──┬─→ Phase 2 ──┐
          │             │             ├─→ Phase 3 ──┬─→ Phase 5 ──→ Phase 6
          └─────────────┴─→ Phase 4 ──┘             │
                              ▲                      │
                              └──────────────────────┘
```

- Phase 0 is landed except 0.5 (0.1/0.2/0.3/0.4/0.4b done 2026-08-14); data now accrues
  while other work proceeds, and the historical archive is dedup'd and internally
  consistent. What remains — 0.5 TripUpdates and the deferred 0.3b sub-item — is
  independent of everything downstream except the ablation grid (6.2).
- Phase 1 blocks everything downstream.
- **1.7 (databus reconciliation) is the long pole in Phase 1** and partly overlaps 1.2 — the
  train/serve *geometry* skew 1.2 describes is the same skew the vendored copy already
  solved with shape-awareness and the precomputed-distance hook. Do 1.7 first and 1.2 may
  reduce to adopting what the merge brings back. 1.8 (restructure) is gated on 1.7.
- Phases 2 and 4 are independent of each other and can interleave.
- Phase 3 needs Phase 2 (bUCR data) only for the cross-agency arm; the MBTA arm can start as soon as Phase 1 lands.
- Every phase runs twice: once against the 28-day rehearsal corpus, then again against the
  90-day replication corpus once collection closes on **2026-11-12**. The second pass is a
  re-run, not new work — see *Collection strategy* above.
- Cut points, in order of least damage: 5.2 (LSTM), 6.3 (figure polish), 2.4/4.5 (secondary analyses).

## Open questions

1. **Does databus serve bUCR TripUpdates?** If yes, bUCR gets an agency baseline too and
   the cross-agency comparison becomes symmetric. If no, the bUCR arm compares against
   schedule and constant-speed only — still valid, but state it explicitly.
2. **Is 6 vehicles / 2 routes enough for the bUCR claims?** Plan the framing as a
   data-poor *case study* with honest power limitations rather than a large-sample result.
3. **MBTA retention.** 500 MiB for 28 days at the old ~80 s cadence; at a true 5 s poll,
   budget ~60 MB/day compressed, so ~5.4 GiB over 90 days against a 186 GiB bucket. Still
   no lifecycle policy needed for the paper horizon.
4. **Does `incofer` enter the study?** A third agency is available but has no RT pipeline.
   Recommend deferring — two agencies already carry the argument.
5. **How is the 07-30 → 08-13 gap handled?** 16 days missing mid-archive, either side
   collected at different cadences. Options: restrict the study to the post-08-14 regime
   (clean, but discards July), treat the two windows as separate folds, or downsample the
   new data to the old cadence. This decision shapes the dataset-size ablation and should
   be made before Phase 6.1 freezes a test period.
6. **Rotate the NavSat API token?** It sat in `.env.example` in plaintext (the endpoint
   path carries the token and account id). Redacted 08-14 and never pushed, but it lived
   on the VPS unprotected.

## What's next (2026-08-18, updated same day)

**Update 2026-08-30 (v3.3), supersedes the ordering below where they overlap.** Since v3.2:
the bUCR slice is **merged to `main`**, and **ewma is retired** (item 2 in the 08-18 list —
now decided: cut from defaults/serving, kept for ablation). Current priorities:

1. **1.7 (databus reconciliation)** — still the long pole, still untouched. Its "first
   action" (delete databus's vendored `backend/gtfs-eta/`, replace with a real dependency on
   this repo) touches a live-production repo — **do not start without the user's explicit
   go-ahead.**
2. **Re-export the served XGBoost artifact** under container-matching sklearn/xgboost
   versions. It bundles a sklearn `ColumnTransformer`, so xgboost's native version-portable
   format alone won't fix the sklearn half — the re-export must run under 1.9.0/3.4.1 (in the
   container or a matching env) and only lands via a databus redeploy, so **deferred with the
   databus work**.
3. **Generalize the bUCR slice** — 2.1's unified `AgencyConfig` and the **MBTA** segment
   reformulation (Phase 3 for MBTA), so the cross-agency claim rests on one pipeline rather
   than a bUCR-only path plus the older MBTA stop-level one.
4. **Rigor before the paper** — 3.2 formal split + leakage probe, 3.3 provenance/seed, plus
   the bUCR report's open items (under-prediction-bias calibration, min-traversal outlier
   filter, fixed-seed re-run check).
5. **90-day replication corpus** (closes **2026-11-12**) still not started.

The 08-18 list below remains valid for its lower-priority open items (0.5 TripUpdates, 1.3
`arrival_method`, the O(n²) audit, per-route vs global models, the `current_speed_kmh=0.0`
check).

For a fresh agent picking this up: the state above (through v3.2) is current as of this
commit. 1.1–1.8's status is still unchanged by today's work — datasets, training, and
etaval wiring were groundwork those items depend on, not progress on them directly. The
one exception is a small piece of new evidence for 1.7 (etaval's `main` branch now has an
independent, correctly-schema'd model predictor — see below). In priority order:

1. **1.7 (databus reconciliation) is still the long pole** — untouched this session, and
   now has one more piece of evidence: `etaval`'s unmerged `feat/model-validation` branch
   (weather-augmented, wrong schema — see *Assets already built and unused*) is no longer
   the only or best reference for "what should etaval's model predictor look like" — this
   session added a working one on `main` (`etaval/predictors/gtfs_eta_models.py`, kinematic
   schema, verified against real MBTA traffic) that could inform or replace it. The 1.7
   entry's own "first action" — delete databus's vendored `backend/gtfs-eta/` and replace
   with a real dependency on this repo — has not been started; **do not start it without
   the user's explicit go-ahead**, it touches a separate live-production repo.
2. **Decide on `ewma` and `historical_mean`.** `ewma` has negative R² at 3 of 4 dataset
   sizes trained this session and is strictly dominated by `historical_mean` everywhere —
   no dataset size or metric found where it wins. Recommend cutting it from
   `train_all_models.py`'s default model list and `eta_service/estimator.py`'s
   `_predict_with_model` dispatch. `historical_mean` is weak (R² 0.206–0.324) but the only
   model needing no live kinematic features, so worth keeping *only* as a cold-start/
   degraded-feature fallback — not as a serving candidate on its own. This is a judgment
   call for the user, not yet made.
3. **Per-route models, not just global.** Everything trained this session
   (`mbta_bus_222_15_37_{1,7,15,28}d`) was global-only (`train_all_models.py` without
   `--by-route`); route-specific models (one xgboost per route instead of one shared across
   222/15/37) were never compared. `train_all_models.py --by-route` already supports this
   and prints a trips/observations-vs-MAE correlation table for free — worth running once,
   cheap, before assuming global is good enough.
4. **Full 90-day / all-routes runs remain undone.** Everything so far is 3 bus routes,
   28 days max, global models. The subway stress case (Green-D/E, the original failing
   case pre-perf-fix) has still never been run end-to-end even after the fixes were
   validated on buses — worth doing once as a scale check, separate from the paper's actual
   90-day replication window (**2026-08-14 → 2026-11-12**, not yet started).
5. **`current_speed_kmh=0.0` for MBTA buses** (see *First real dataset builds...* above) —
   worth a quick check on whether this also holds for Green-D/E or is bus-specific, since it
   affects how much the "kinematic" feature group is actually contributing vs. just
   position/bearing/distance doing all the work.
6. **1.2 (train/serve geometry skew)** likely shrinks once 1.7 lands (see 1.7's note on the
   overlap) — don't start it first.
7. **1.8 (package restructure)** is the real fix for the Docker cross-repo coupling
   documented above under *Docker / cross-repo build* — the stopgap there works but a
   standalone `gtfs-eta` shouldn't need a sibling `gtfs-django` checkout to build at all.
   Still deliberately sequenced after 1.7 per the existing plan.
8. **Lower priority, pick up opportunistically:** 0.5 (TripUpdates archival), 1.3
   (`arrival_method` column), the `roll_validate.py`-and-elsewhere O(n²) audit noted above.

Housekeeping worth knowing about: 08-18 also created the repo's first `.gitignore` (none
existed before — `.env` was unprotected), confirmed the local Docker dev stack
(`gtfs-rt-pipeline/docker compose up`) builds and runs correctly, and — same day, second
pass — built the first four complete real datasets and trained the first 20 real models
(see above). If `docker compose` fails with a path-resolution error again, check whether
`.env`/`gtfs-rt-pipeline/.env` still hold valid `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
for `data.simovilab.org` before assuming it's a code regression — that exact failure mode
(silent empty results, not a raised error) cost real time this session. For anything
involving `etaval`, see its own `README.md` first — it now documents a required
`--no-editable` + `MODEL_REGISTRY_DIR` workaround that isn't obvious from this repo alone.
