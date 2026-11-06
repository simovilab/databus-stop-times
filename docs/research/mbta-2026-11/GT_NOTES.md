# Phase 1 NOTES - data foundation (gt/ + corpus/)

Repo @ fbd5ff5 (read-only). Seed 42. Provenance: `gt/MANIFEST.json` (sha256 of GT, comparator, splits, corpus), `logs/pip_freeze.txt`. Code: `scripts/` (run order in `scripts/README.md`). Library issues: `FINDINGS.md` (F1-F7).

## 1. Environment
`uv venv --python 3.12 $E2E/.venv` then, from the repo root (non-editable):
`uv pip install './packages/cli[core,models,record,train,serve,val]' './packages/models[xgboost,torch]' optuna scipy pyarrow duckdb s3fs fsspec matplotlib` - extras all existed, ~1 min, no resolver error.
Friction (F1): silently resolves numpy 2.5.4 + pandas 3.1.1 next to torch 2.2.2 (numpy-1 ABI) -> "Failed to initialize NumPy". Fixed with `uv pip install 'numpy<2' 'pandas>=2.3,<3' 'scipy<1.15' 'scikit-learn>=1.7.2' 'contourpy<1.3' 'matplotlib<3.10' tabulate`; `uv pip check` clean; `stoptimes --help` (0.2 s), `import torch, xgboost`, `torch.from_numpy` all OK. Final: numpy 1.26.4, pandas 2.3.3, torch 2.2.2, xgboost 3.4.1, sklearn 1.9.1, duckdb 2.0.0.

## 2. Mirror (`$E2E/data`, total 3.4 GB incl. outputs; raw 1.8 GB)
* VP routes 15/37/222 (hive layout, UTC-day partitions): 111/108/111 day-files = 103 MB (2026-07-01 .. 2026-11-04).
* TU staging: 1,505 files, 359 MB (41.4M rows, 2026-09-04 .. 2026-11-05).
* Static: 13 weekly zips (2026-08-14 .. 2026-11-02) 319 MB; MBTA public archive: 45 zips (every feed version overlapping 2026-06-29..11-05) in `data/gtfs_static_archive/` (~1.1 GB) + `archived_feeds.txt`. `data/gtfs_filtered/<tag>/` = routes 15/37/222 only (`w_` weekly, `a_` archive).
* Archive semantics: entry (feed_start_date..feed_end_date) = the feed version in effect for service dates in that range; used as the primary feed per day (`common.best_tag_for`), weekly = fallback.

## 3. Audit - see `gt/AUDIT.md` (per-day tables), `gt/audit_*.csv`
Key facts:
* 112 VP service days present (06-30..11-04); **104 VP-complete**. Missing: 07-27, 07-30..08-13. Partial (excluded): 06-30, 07-26, 07-28, 07-29, 08-14, 11-04; degraded (<50% of same-era same-weekday median rows): 07-15, 07-16.
* Ping cadence: median 21 s through 2026-07-29 vs 6 s from 08-14 (regime break); ~30-100 vehicles/day; route 37 has no Sunday service.
* TU: 63 service days (09-04..11-05); 62 complete (11-05 partial: ends 20:00 local, no VP). Cadence 30 s (p95 32 s). Schema: has route_id; arrival_time/departure_time are naive-UTC TIMESTAMP (absolute epoch); arrival_delay and revenue 100% NULL; arrival_uncertainty set (300/301) in 0.9% of rows only; **passed stops are dropped** (0.3% of rows at stops before the vehicle's current stop); RLE'd (see F6).
* Trip-id match, instance level (VP trip vs trips.txt of the feed in effect): summer **99.94% with archive** vs **87.3% without** (earliest weekly 08-14 zip; stop-seq identity 87.9%); fall **99.98%** with archive or weekly alike (stop_seq/stop_id identity of STOPPED_AT pings 100% both). Identity caveat = ghost instances (F4): archive "active service" match is 88.7% in July (ghosts), 99.93% in fall.

## 4. Splits (`gt/splits.json`, `gt/day_status.csv`; rule is in `scripts/08_splits.py` docstring and the JSON)
Service day = local minus 3 h. "VP-complete" = first ping <= 04:00 local and last >= 24:00 local, >=22 of 24 local hours with pings, and rows >= 50% of the median of same era (pre-gap / Aug14+) and weekday class. "TU-complete" = >=2400 snapshots, first <= 03:05 local, last >= 02:50 next day (DST-safe).
* **test** (14, VP+TU complete): 2026-10-21 .. 2026-11-03 (consecutive).
* **val** (7): 2026-10-14 .. 2026-10-20.
* **train pool** (83 VP-complete days before val): 2026-07-01 .. 2026-10-13 (23 July days at 21 s cadence + 60 days 2026-08-15..10-13; gap = 07-15/16, 07-26..08-14).
* Ablation (N most recent available pool days; N counts days, not calendar span): N=7: 10-07..10-13; 15: 09-29..10-13; 30: 09-14..10-13; 60: 08-15..10-13 (all pure 6-s era); 90 = all = 83 days from 07-01 (only 83 exist; N=90 == 'all'). Explicit lists in the JSON. **Confound to report**: N=83 adds the 23 sparse-ping July days (different cadence + feed version), so "all vs 60" mixes more data with a regime change.
* 2026-11-04 (TU-complete but VP stops 18:59 local) is excluded from test.

## 5. Ground truth (`gt/arrivals.parquet`, 1,201,090 rows, 33,174 trip instances, 112 days; incl. partial days, flagged by `gt/day_status.csv`/split)
Schema: trip_id, service_date (date), stop_sequence, stop_id, route_id, direction_id, shape_id, vehicle_id (most-pinged vehicle; `multi_vehicle` flag), stop_progress_m (package `assign_stops_monotonic`), sched_arr (epoch, GTFS noon-12h origin), **arr_interp**, **arr_stopped_at** (epoch s, NaN if none), interp_reason (ok / before_first_ping / never_reached / gap_gt_60s / too_few_valid_pings), interp_gap_s, n_pings, n_valid_pings, xt_med_m, first_ping, last_ping, feed_tag. One row for every scheduled stop of every observed instance (>=10 pings, calendar-active); ghost instances in `gt/ghost_instances.parquet`.
Method (`scripts/gt_core.py`): forward-only windowed map matching of pings onto the shape (package polyline + stop progress; off-route >100 m ignored; back-jitter 60 m; no loop-back snapping), arrival = linear interpolation of the time progress first reaches the stop, rejected if bracketing valid-ping gap >60 s or no bracket. stopped_at = first STOPPED_AT ping with `current_stop_sequence == stop_sequence`.
* **Coverage** (interp, % of scheduled stop visits of observed instances): overall 90.4%; **test 94.2%**, val 94.0%, train pool 91.2% (Aug+ 93.9%, **July 77.1%** because of the 21 s cadence/60 s gap rule); by route 15: 89.7%, 222: 92.8%, 37: 88.1% (37: ~10% of instances have <50% of pings). Interior stops (excl. first/last 5%): 94.4%; misses are concentrated at trip ends (on 2026-10-14, first 5% of stops: `before_first_ping` 54%/20%/60% for routes 15/222/37; last 5%: `never_reached` 49%/32%/53%). stopped_at coverage 88.5% test / 82.0% overall; either method 96.3% test.
* **GT-method sensitivity = arr_stopped_at - arr_interp** (rows with both; seconds): overall n=947,186: median **+1.6**, IQR [-6.1, +5.5], p5 **-42.6**, p95 **+11.4**; |diff| median 5.7 s, 57% within 10 s, 3.3% beyond 60 s. Test days (n=145,660): median +1.5, IQR [-6.6, +5.4], p5 -42.5, p95 +10.8. Route 222 is shifted positive (median +3.9 s, IQR [+0.3, +7.0]) vs route 15 (+0.3, [-10.2, +4.3]). Negative tail = STOPPED_AT status set before the bus reaches the projected stop point (stopped_at first); positive = dwell/queueing.
* **Agreement with package code** (80 sampled trips of 2026-10-14, `gt/gt_vs_package.json`): `StopStatusDetector` = my `arr_stopped_at` **100% exact**; `ShapeDistanceDetector(30 m)` (closest-approach in progress space, global nearest projection) vs my `arr_interp`: median +0.07 s, IQR [-1.6, +1.6], 88% within 10 s, 97.5% within 60 s; segment-builder 50 m-radius arrival (`core.segments._detect_arrival`): median **-4.2 s**, IQR [-14.6,-1.1] vs interp (F3). Package `core.arrivals.derive_arrivals` (CLI `data arrivals`, which also has reported + map_matched + interpolated modes) was not usable: one UTC day of three routes did not finish in 23 min / up to 7.3 GB RSS (F2), so no agreement number against it.
* Plausibility vs comparator (`gt/mbta_vs_gt_preview.json`, test days; MBTA latest prediction made h s before the GT arrival, no staleness filter): MAE 30 s (h=60), 40 s (120), 63 s (300), 95 s (600), 144 s (1200); median signed error -12 s (h=60) growing to -75 s at 20 min (MBTA predicts earlier than actual). This is a sanity preview, not the protocol evaluation.

## 6. MBTA comparator (`gt/mbta_predictions.parquet`, 41,402,018 rows, 260 MB, all 63 TU days; sorted by service_date, trip_id, stop_sequence, feed_ts)
Columns: trip_id, service_date (= TU start_date), stop_sequence, stop_id, **feed_ts** (header_ts epoch = when the poll saw it; use for no-leak joins), tu_ts, arrival_time, departure_time (epoch s), arrival_uncertainty, route_id, direction_id, vehicle_id, stop_rel, trip_rel (SKIPPED/CANCELED/ADDED), start_time. RLE: a row only when the (trip,stop) prediction changed.
* Identity: all 20,043 GT instances on TU-complete days have a TU instance with the **same (trip_id, service_date)**; none match only on another date. 20,277 TU instances, 20,043 (98.8%) have GT (the rest are TU-only: no VP pings/ghost/cancelled).
* **Coverage of GT arrivals on test days** (158,786 GT interp arrivals): 100% have >=1 MBTA row; 98.7% have an MBTA arrival_time; **98.7% have a prediction made >=60 s before the GT arrival**; 97.4% have one in the [arr-600 s, arr-60 s] window (val: 98.7%/97.4%). `gt/mbta_stats.json`.
* Staleness: use trip-level freshness (F6). Predictions already in the past at feed time: 1.2%. Horizon p50 1150 s, p99 3566 s (feed capped at 3600 s).

## 7. Training corpus (`corpus/segments/sd=YYYY-MM-DD/part.parquet`, one file per service day; `corpus/segments_meta.csv`, `corpus/CORPUS.json`)
Built through the public API `stoptimes.train.datasets.mbta_segments.build_mbta_segment_dataset` (the CLI chain is not viable: F2) on ghost-filtered VP, per day against the feed in effect, extra columns `service_date`, `feed_tag`. Dir key is `sd` (a hive key named `service_date` collides with the column and breaks pyarrow dataset reads - hit this once).
* Rows: **train pool 810,851 (83 d) / val 79,543 (7 d) / test 159,746 (14 d)**; total 1,050,140 over 104 days, 68 MB. Median 6.8k rows/day (July) vs 11.0k (Aug+).
* Ablation rows: N=7: 78,633; 15: 170,006; 30: 340,778; 60: 645,828; 90/all: 810,851.
* Build time: 745 s wall for 104 days with 3 workers (~14-28 s/day single, contended machine); no rewrite of package logic.
* Ghost contamination (07-05): 29% of segments from ghost trips in an unfiltered build (F4).
* Leakage probe (PROTOCOL 5): XGBoost (300 trees, depth 6, seed 42) on the last 30 train-pool days, scored on val: real target MAE 23.3 s / R2 0.674 (consistent with RESULTS.html ~24.6 s); **shuffled target MAE 48.2 s / R2 0.001** = mean baseline (48.2 s). Passes.
* Caveat for modelers: corpus target is segment-entry -> next-stop traversal (50 m-radius arrival, F3), not the GT definition; stop-level ETA at arbitrary made_at must be constructed by later phases (e.g. summing segments, or `data frame` path if F2 is solved).

## 8. PROTOCOL section 5 checklist - verified in phase 1
* Leakage shuffle-label probe: done (above). "Features at t use only data <= t": segment features come from the entry ping only (by construction of `core.segments`), not audited further.
* Static GTFS in effect per day with trip-match per day: done (`AUDIT.md`, `audit_day.csv`, `audit_consistency.csv`), with and without archive.
* Seeds 42; artifacts stamped via MANIFEST (git SHA, hashes), freeze in `logs/pip_freeze.txt`.
* Library bugs recorded in FINDINGS.md; repo untouched.
* NOT in scope of phase 1 (open): train/serve parity, shapes supplied to serving (GT/corpus both use real shapes for 100% of trips, no haversine fallback).

## 9. Blockers / open points for later phases
1. No documented-CLI path scales (F2); later phases must use Python API / own scripts, or accept long CLI runs. Whether `train run` can consume the segment corpus is untested.
2. Label definition mismatch (F3) between corpus target and GT.
3. July regime (21 s cadence, 77% GT coverage) vs Aug+; decide whether N=83/all is "more data" or "more data + regime shift".
4. MBTA predictions: use `feed_ts` and trip-level staleness (F6).
