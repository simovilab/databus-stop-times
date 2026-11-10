# bUCR inferred trips: navsat pings → timetable-matched live frame (2026-11-09)

bUCR's navsat AVL pings carry no trip, run or route. This session turns them into a
GTFS-RT-shaped VehiclePosition archive whose `trip_id`s come from the bUCR timetable, so
the same live-feature frame and models used for MBTA can be trained on bUCR. It is step 1
of the structured-vs-unstructured comparison in `RESEARCH_ROADMAP.md`.

## Pipeline

```bash
# 1. pings + GTFS -> matched archive (+ legs.parquet, report.json)
uv run --no-sync python -m stoptimes.train.sources.bucr_observations \
    --navsat-dir <dir of daily navsat parquet> --gtfs-zip <bucr gtfs.zip> --out <out>
# 2. archive -> live frame
stoptimes data frame --live --observations <out>/archive --arrival-radius-m 30 ...
# 3. frame -> model
stoptimes train run examples/experiments/xgb-live-bucr.yaml --dataset-dir ~/stoptimes-datasets
```

| Module | Job |
|---|---|
| `train/sources/corridor_legs.py` | Splits each vehicle-day into legs at the corridor's turning points. Terminals, detour stops and variants are read from the GTFS shapes. |
| `train/sources/schedule_match.py` | Stop passages per leg; one-to-one (Hungarian) assignment of legs to the day's scheduled trips per direction, with the inferred shape as a prior. |
| `train/sources/bucr_observations.py` | bUCR glue: calendar, local service day, archive writer, report. |
| `live_frame.with_radius_arrivals` | Optional `arrival_radius_m`: a stop is reached when a ping comes within the radius, not only when it is crossed. |

Inputs: the curated pings in the bucket (`feeds/bucr/navsat/year=/month=/day=/`) and the
schedule feed `http://feeds.simovi.org/bucr/schedule/gtfs.zip` (1 route, 122 weekday trips,
7 shapes). Outputs are regenerable and not in git.

## Results (semester, 10 Aug – 6 Nov 2026, 62 weekdays)

| | Old `bucr_trip_inference` | This pipeline |
|---|---|---|
| Clean pings in trips | 55% | 63.5% |
| Variant agrees with timetable | 44% | 88.4% |
| Legs matched to a scheduled trip | — | 6,110 of 6,305 (97%), 98.5 per day |
| Kept (uncontested, margin ≥ 0) | — | 5,844 |

Matched offset (observed − scheduled): median −48 s, 5th–95th percentile −261 to +319 s;
median spread along a trip 69 s.

Model `xgb-live-bucr`, frame of 413k rows: trained on 10 Aug – 23 Oct, validated on
26 Oct – 6 Nov.

| Predictor | MAE |
|---|---|
| Timetable (`sched_rem`) | 94.3 s |
| Segment history (`prior_rem`) | 54.4 s |
| **xgb-live-bucr** | **48.7 s** (median 32.9 s, bias +8 s) |

By horizon (model vs timetable): < 2 min 19 vs 44 s; 2–5 min 37 vs 78; 5–10 min 56 vs 116;
10–20 min 83 vs 130; 20+ min 201 vs 302. Inbound 53.4 s, outbound 43.5 s.

Caveat: the validation days also drove early stopping (`best_iteration` 100), so this is
not yet a held-out test. Next: rolling-origin by week + day-grouped CV.

## Pitfalls found

- **Quick terminal turnarounds.** Buses turn around at Educación/Artes in 30 s – 3 min.
  The old 90 s dwell rule glued the new trip's first leg onto the previous trip; with the
  origin lost, the Educación-vs-Artes variant was a coin flip. Educación (0_01) and Artes
  (0_02) are 349 m apart on a loop, each on the other variant's path.
- **Depot breaks.** The depot (Sección de Transportes) is on the inbound route, ~580 m
  before Odontología. A stationary wait of ≥ 5 min away from a terminal ends the leg
  (`out_of_service_wait_s`); without it the smoke model's MAE was 131 s.
- **Parking short of Odontología.** Buses lay over 6 m before stop 0_11's progress point,
  so crossing-based arrivals fired on departure, ~30 min late. `--arrival-radius-m 30`
  fixes it (smoke MAE 75.5 → 46.6 s).
- **Day files are cut at UTC midnight** (18:00 in Costa Rica). Concatenate all files and
  regroup by local service day.
- **Contested matches.** A negative `margin_s` means the assignment gave the leg's
  best-fitting trip to another leg; `--min-margin-s 0` (default) drops them.
- **The calendar ends 27 Nov 2026** (`entresemana`, holidays 31 Aug, 15 Sep, 12 Oct,
  1 Dec removed). Next term needs the new feed.
- `simovi-status` says bUCR static GTFS is "none yet", but weekly snapshots exist in
  `feeds/bucr/gtfs_static/`.

## Cross-validation, tuning and the frozen model (same day, later)

`python -m stoptimes.train.live_cv` (commits 5f46a25, 42651a5) refits the segment-history
prior and route vocabulary on each fold's fit days only, and early-stops on training days.

| Scheme | Model | Timetable | Segment history |
|---|---|---|---|
| Original split (early-stopped on its own test days) | 48.7 s | 94.3 s | 54.4 s |
| Rolling origin by ISO week, 9 test weeks (W37–W45) | 52.6 ± 4.1 s | 93.5 s | 58.1 s |
| 5-fold by day | 51.0 ± 2.1 s | 92.2 s | 56.6 s |

Sweep (12 configs, rolling): only depth matters. `max_depth 10, learning_rate 0.03` gives
pooled 52.1 s against 52.5 s, winning 9 of 9 folds by 0.42 ± 0.15 s. Learning rate,
`min_child_weight` and patience move it ≤ 0.2 s. The 20+ min horizon stays at ~240 s
whatever the settings, so it needs features, not tuning.

**Frozen model:** `examples/experiments/xgb-live-bucr-final.yaml`.
- Trained on all 62 days with `early_stopping_rounds: 0` and `n_estimators: 415`
  (CV median best iteration 360 × 1.15).
- Artifact: `~/stoptimes-registry/xgb-live-bucr-final`. In-sample train MAE is 45.6 s.
- An in-sample replay of 4 Nov, 06–12 h, through `val`: 34.9 s, against 87.1 s for the
  timetable. That checks the wiring only.
- Publish as `stoptimes-bucr@staging`, then run the live test in
  [the runbook](../guides/bucr-live-test.md).
