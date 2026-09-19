# Guide: diagnose a feed

For the newcomer user story: "diagnose whether a feed can support ETA modelling at all
before investing weeks in it."

`stoptimes diagnose` reports feed-quality problems — unmatched trips, missing shape
distances, vehicle-position gaps, stale records, schedule/realtime mismatches — as a report
you can read or feed to a script, before you spend any time on training.

## Point it at what you have

```bash
stoptimes diagnose \
  --static ./gtfs.zip \
  --vehicle-positions ./archive --vehicle-positions-format archive_parquet \
  --source demo
```

Run for real against the [quickstart](../quickstart.md)'s fixture (vehicle positions only,
no trip updates — the realistic "just started recording" case):

```text
⚠ reading all of './archive' with no --from/--to bound — this loads every matched parquet
file's full row set into memory. Pass --from/--to to scan only part of a large archive.
DEGRADED — modelling is possible, but 1 issue(s) will cost accuracy or coverage:
No trip updates provided.

                                    findings
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ severity ┃ id              ┃ title               ┃ count ┃ total ┃ share ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ degraded │ tu_missing      │ No trip updates ...  │ 0     │ —     │ —     │
│ info     │ vp_sampling_... │ VP sampling ...      │ 0     │ 96    │ 0%    │
│ info     │ vp_staleness... │ Staleness not check…│ 0     │ —     │ —     │
└──────────┴─────────────────┴─────────────────────┴───────┴───────┴───────┘

tu_missing: No trip_update observations were supplied.
  blocks: There is no independently-published prediction to validate against, and no
  TU-reported arrivals to cross-check derived (map-matched) ones. Training from vehicle
  positions alone is still possible.
  do: Record or point diagnose at a trip_updates GTFS-RT feed or archive if you plan to
  validate or serve TripUpdates.
```

Exit code is `0` here — a feed *with* quality problems is a successful diagnosis; the
verdict (`OK` / `DEGRADED` / `BLOCKED`) and the findings table are the point, not the process
exit status. A non-zero exit means diagnose itself couldn't read a given location (bad
path/URL/format), a usage error rather than a feed-quality finding.

## Severity, and what it actually means

Findings are `blocker` / `degraded` / `info` — deliberately three-valued rather than a
numeric score, because the underlying counts don't support more precision than that.
`blocker` means the specific thing it's about genuinely cannot happen without fixing this
(no static feed at all blocks `data arrivals` unconditionally; no vehicle positions blocks it
too, since VP is the only source of positions to map-match). `degraded` means modelling is
still possible but something downstream will cost accuracy or coverage — missing trip
updates, for instance, doesn't stop training but does mean nothing to validate or
cross-check derived arrivals against. `--majority-threshold` (default `0.5`) is the one
tunable line that decides when a share of affected rows tips a mismatch finding from
`degraded` to `blocker`.

## Only what your data can actually support

`diagnose` doesn't invent findings your data can't answer. Staleness (computed from
`ingested_at - ts`, not wall-clock-now, so it works the same on a historical archive as on a
live poll) degrades to an `info` finding explaining *why* it couldn't be checked when
`ingested_at` is absent — e.g. an AVL CSV import that never had a fetch timestamp — instead
of silently skipping it or guessing. Similarly, `shape_dist_traveled` coverage is checked
against what `stoptimes.core.spatial.polyline` actually falls back to (cumulative haversine
distance when it's missing), not against the GTFS spec's general recommendation — the
finding's `blocks` text describes a real consequence, not a style complaint.

## On a large archive

`--from`/`--to` (same spelling as `val replay`) bound an `archive_parquet` read to matching
partitions *before* any file is opened — not a post-hoc filter. Without them, `diagnose`
warns (as shown above) that it's about to load every matched file's full row set into memory.
Column projection isn't implemented yet (`readers.read_archive_parquet` has no `columns=`
parameter), so even a date-bounded read still loads every column of the matched files — see
`docs/notes/p3-u5.md` for this filed as a follow-up (owned by whoever owns `core/readers.py`
next, not this docs track).

## Reused by `data inspect`?

`docs/PLAN.md`'s plan for this feature says diagnose logic should be reused by `data
inspect`. As of this writing that wiring hasn't landed — `diagnose_feed` is a plain function
(`stoptimes.core.diagnose.diagnose_feed`) that whoever owns `cli/commands/data.py` next can
call directly; `data inspect` today has its own, narrower coverage report. Don't expect
`data inspect`'s output to include diagnose's findings yet.
