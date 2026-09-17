# BUCR trip-inference quality report

## Cleaning
- Days: 5, rows in: 37998, rows kept: 37983
- Overall drop rate: 0.04%
  - bad_timestamp: 0.00%
  - stale: 0.04%
  - null_coord: 0.00%
  - zero_coord: 0.00%
  - out_of_range: 0.00%
- Staleness threshold: 900s

## Inference match rate
- Points in: 37983
- Assigned: 21366 (56.25%)
- Parked (ambiguous): 2614 (6.88%)
- Anomaly-rejected: 2223 (5.85%)
- Dropped as noise (idle/too-short instances): 11780 (31.01%) across 1245 instances
- Trips inferred: 345

| route_id | direction_id | shape_id | trips |
|---|---|---|---|
| bUCR | 0 | desde_artes_a_odontologia_con_milla | 8 |
| bUCR | 0 | desde_artes_a_odontologia_sin_milla | 138 |
| bUCR | 0 | desde_educacion_a_odontologia_con_milla | 4 |
| bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 18 |
| bUCR | 1 | desde_edufi_a_educacion | 77 |
| bUCR | 1 | desde_odontologia_a_artes | 30 |
| bUCR | 1 | desde_odontologia_a_educacion | 70 |

## Trips/day
| service_day | vehicle_id | route_id | direction_id | shape_id | n_trips |
|---|---|---|---|---|---|
| 2026-08-18 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 1 |
| 2026-08-18 | 299-1014 | bUCR | 1 | desde_edufi_a_educacion | 1 |
| 2026-08-18 | 299-921 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 2 |
| 2026-08-18 | 299-921 | bUCR | 1 | desde_edufi_a_educacion | 1 |
| 2026-08-18 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 1 |
| 2026-08-18 | 299-922 | bUCR | 1 | desde_edufi_a_educacion | 1 |
| 2026-08-18 | 299-922 | bUCR | 1 | desde_odontologia_a_educacion | 1 |
| 2026-08-19 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_con_milla | 1 |
| 2026-08-19 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 10 |
| 2026-08-19 | 299-1014 | bUCR | 1 | desde_edufi_a_educacion | 4 |
| 2026-08-19 | 299-1014 | bUCR | 1 | desde_odontologia_a_artes | 2 |
| 2026-08-19 | 299-1014 | bUCR | 1 | desde_odontologia_a_educacion | 5 |
| 2026-08-19 | 299-604 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 7 |
| 2026-08-19 | 299-604 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 2 |
| 2026-08-19 | 299-604 | bUCR | 1 | desde_edufi_a_educacion | 5 |
| 2026-08-19 | 299-604 | bUCR | 1 | desde_odontologia_a_educacion | 5 |
| 2026-08-19 | 299-921 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 15 |
| 2026-08-19 | 299-921 | bUCR | 0 | desde_educacion_a_odontologia_con_milla | 1 |
| 2026-08-19 | 299-921 | bUCR | 1 | desde_edufi_a_educacion | 7 |
| 2026-08-19 | 299-921 | bUCR | 1 | desde_odontologia_a_artes | 4 |
| 2026-08-19 | 299-921 | bUCR | 1 | desde_odontologia_a_educacion | 7 |
| 2026-08-19 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_con_milla | 1 |
| 2026-08-19 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 10 |
| 2026-08-19 | 299-922 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 1 |
| 2026-08-19 | 299-922 | bUCR | 1 | desde_edufi_a_educacion | 8 |
| 2026-08-19 | 299-922 | bUCR | 1 | desde_odontologia_a_artes | 1 |
| 2026-08-19 | 299-922 | bUCR | 1 | desde_odontologia_a_educacion | 7 |
| 2026-08-20 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_con_milla | 1 |
| 2026-08-20 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 11 |
| 2026-08-20 | 299-1014 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 1 |
| 2026-08-20 | 299-1014 | bUCR | 1 | desde_edufi_a_educacion | 3 |
| 2026-08-20 | 299-1014 | bUCR | 1 | desde_odontologia_a_artes | 5 |
| 2026-08-20 | 299-1014 | bUCR | 1 | desde_odontologia_a_educacion | 5 |
| 2026-08-20 | 299-604 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 7 |
| 2026-08-20 | 299-604 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 1 |
| 2026-08-20 | 299-604 | bUCR | 1 | desde_edufi_a_educacion | 7 |
| 2026-08-20 | 299-604 | bUCR | 1 | desde_odontologia_a_artes | 1 |
| 2026-08-20 | 299-604 | bUCR | 1 | desde_odontologia_a_educacion | 4 |
| 2026-08-20 | 299-921 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 14 |
| 2026-08-20 | 299-921 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 1 |
| 2026-08-20 | 299-921 | bUCR | 1 | desde_edufi_a_educacion | 8 |
| 2026-08-20 | 299-921 | bUCR | 1 | desde_odontologia_a_artes | 4 |
| 2026-08-20 | 299-921 | bUCR | 1 | desde_odontologia_a_educacion | 4 |
| 2026-08-20 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 11 |
| 2026-08-20 | 299-922 | bUCR | 0 | desde_educacion_a_odontologia_con_milla | 1 |
| 2026-08-20 | 299-922 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 2 |
| 2026-08-20 | 299-922 | bUCR | 1 | desde_edufi_a_educacion | 5 |
| 2026-08-20 | 299-922 | bUCR | 1 | desde_odontologia_a_artes | 3 |
| 2026-08-20 | 299-922 | bUCR | 1 | desde_odontologia_a_educacion | 6 |
| 2026-08-21 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 12 |
| 2026-08-21 | 299-1014 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 2 |
| 2026-08-21 | 299-1014 | bUCR | 1 | desde_edufi_a_educacion | 8 |
| 2026-08-21 | 299-1014 | bUCR | 1 | desde_odontologia_a_artes | 3 |
| 2026-08-21 | 299-1014 | bUCR | 1 | desde_odontologia_a_educacion | 4 |
| 2026-08-21 | 299-604 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 9 |
| 2026-08-21 | 299-604 | bUCR | 1 | desde_edufi_a_educacion | 7 |
| 2026-08-21 | 299-604 | bUCR | 1 | desde_odontologia_a_artes | 2 |
| 2026-08-21 | 299-604 | bUCR | 1 | desde_odontologia_a_educacion | 4 |
| 2026-08-21 | 299-921 | bUCR | 0 | desde_artes_a_odontologia_con_milla | 1 |
| 2026-08-21 | 299-921 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 10 |
| 2026-08-21 | 299-921 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 3 |
| 2026-08-21 | 299-921 | bUCR | 1 | desde_edufi_a_educacion | 3 |
| 2026-08-21 | 299-921 | bUCR | 1 | desde_odontologia_a_artes | 3 |
| 2026-08-21 | 299-921 | bUCR | 1 | desde_odontologia_a_educacion | 7 |
| 2026-08-21 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 11 |
| 2026-08-21 | 299-922 | bUCR | 0 | desde_educacion_a_odontologia_con_milla | 1 |
| 2026-08-21 | 299-922 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 4 |
| 2026-08-21 | 299-922 | bUCR | 1 | desde_edufi_a_educacion | 4 |
| 2026-08-21 | 299-922 | bUCR | 1 | desde_odontologia_a_artes | 2 |
| 2026-08-21 | 299-922 | bUCR | 1 | desde_odontologia_a_educacion | 8 |
| 2026-08-22 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_con_milla | 1 |
| 2026-08-22 | 299-1014 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 1 |
| 2026-08-22 | 299-1014 | bUCR | 1 | desde_edufi_a_educacion | 2 |
| 2026-08-22 | 299-604 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 1 |
| 2026-08-22 | 299-921 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 2 |
| 2026-08-22 | 299-921 | bUCR | 0 | desde_educacion_a_odontologia_con_milla | 1 |
| 2026-08-22 | 299-921 | bUCR | 1 | desde_odontologia_a_educacion | 1 |
| 2026-08-22 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_con_milla | 3 |
| 2026-08-22 | 299-922 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 3 |
| 2026-08-22 | 299-922 | bUCR | 1 | desde_edufi_a_educacion | 2 |
| 2026-08-22 | 299-922 | bUCR | 1 | desde_odontologia_a_educacion | 1 |

## Self-loop artifact
Flagged shapes: desde_educacion_a_odontologia_sin_milla, desde_artes_a_odontologia_sin_milla

- Flagged shapes: 155 trips
  - duration_s: n=155, median=1441.0, mean=1528.3, p25=1101.0, p75=1721.0
  - n_points: n=155, median=70.0, mean=69.1, p25=58.0, p75=85.0
  - n_stops_covered: n=155, median=7.0, mean=6.3, p25=6.0, p75=7.0
- Con_milla siblings: 12 trips
  - duration_s: n=12, median=4895.5, mean=4919.5, p25=1398.0, p75=7438.2
- All other shapes: 175 trips
  - duration_s: n=175, median=1050.0, mean=988.1, p25=759.5, p75=1217.5
- Fragmentation-inflation estimate (flagged trips/vehicle-day ÷ sibling trips/vehicle-day):
  - desde_educacion_a_odontologia_sin_milla: 1.80x
  - desde_artes_a_odontologia_sin_milla: 5.71x

## Spot-check vs. timetable
| trip_id | vehicle_id | route_id | direction_id | shape_id | inferred_start_local | nearest_scheduled_start | delta_minutes | n_scheduled_trips_same_direction |
|---|---|---|---|---|---|---|---|---|
| bucr:299-921:bUCR:1:desde_edufi_a_educacion:20260818T233844Z | 299-921 | bUCR | 1 | desde_edufi_a_educacion | 2026-08-18 17:38:44 | 17:35:00 | 3.7333333333333334 | 60 |
| bucr:299-604:bUCR:1:desde_odontologia_a_educacion:20260819T134122Z | 299-604 | bUCR | 1 | desde_odontologia_a_educacion | 2026-08-19 07:41:22 | 07:40:00 | 1.3666666666666667 | 60 |
| bucr:299-604:bUCR:0:desde_artes_a_odontologia_sin_milla:20260819T155901Z | 299-604 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 2026-08-19 09:59:01 | 09:55:00 | 4.016666666666667 | 62 |
| bucr:299-922:bUCR:1:desde_odontologia_a_educacion:20260819T205538Z | 299-922 | bUCR | 1 | desde_odontologia_a_educacion | 2026-08-19 14:55:38 | 14:50:00 | 5.633333333333334 | 60 |
| bucr:299-921:bUCR:0:desde_artes_a_odontologia_sin_milla:20260820T123229Z | 299-921 | bUCR | 0 | desde_artes_a_odontologia_sin_milla | 2026-08-20 06:32:29 | 06:30:00 | 2.4833333333333334 | 62 |
| bucr:299-921:bUCR:1:desde_odontologia_a_educacion:20260821T021403Z | 299-921 | bUCR | 1 | desde_odontologia_a_educacion | 2026-08-20 20:14:03 | 20:10:00 | 4.05 | 60 |
| bucr:299-921:bUCR:1:desde_odontologia_a_artes:20260821T134929Z | 299-921 | bUCR | 1 | desde_odontologia_a_artes | 2026-08-21 07:49:29 | 07:50:00 | 0.5166666666666667 | 60 |
| bucr:299-921:bUCR:1:desde_odontologia_a_educacion:20260821T172508Z | 299-921 | bUCR | 1 | desde_odontologia_a_educacion | 2026-08-21 11:25:08 | 11:25:00 | 0.13333333333333333 | 60 |
| bucr:299-922:bUCR:1:desde_edufi_a_educacion:20260821T173651Z | 299-922 | bUCR | 1 | desde_edufi_a_educacion | 2026-08-21 11:36:51 | 11:35:00 | 1.85 | 60 |
| bucr:299-922:bUCR:0:desde_educacion_a_odontologia_sin_milla:20260821T202434Z | 299-922 | bUCR | 0 | desde_educacion_a_odontologia_sin_milla | 2026-08-21 14:24:34 | 14:20:00 | 4.566666666666666 | 62 |

## Sanity
- Timetable trip count: 122
- Vehicle-days observed: 19
- Max trips in a single (vehicle, day, shape) bucket: 15

## Verdict
AUTOMATED PLACEHOLDER -- replace with a reasoned call before treating this run as final. See the self-loop inflation estimate and match rate above.
