# MBTA study, 2026-11 (live features)

Study behind PLAN.md §8.1: 83 train days, 7 val, 14 held-out test days (2026-10-21..11-03), routes 15/222/37.
Ground truth = interpolated shape-distance arrivals; baseline = MBTA's own TripUpdates.
Test MAE: rs-ens-avg-3 92.0 s, rs-xgb-servable-plus (19 features) 95.5 s, MBTA 118.1 s, package e2e-stop-xgb 211.8 s.

`reference/` is the study's feature code, kept as the **behavioral spec** for the F-tracks. It is not a package,
is never imported from `packages/`, and depends on study-local helpers that are not copied here.

- `gt_core.py`: forward-only map matching and the interpolated arrival rule (gap ≤ 60 s).
- `rs_prep.py`: arrival events with detection time (the causal rule: usable only when detect ≤ t).
- `rs_stats.py`: segment statistics with hierarchical fallback and out-of-fold fitting.
- `rs_feat.py`: every feature; `SERVABLE_PLUS` is the 19-feature list.
- `GT_NOTES.md`: ground-truth construction and coverage.
