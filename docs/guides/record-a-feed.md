# Guide: record a feed

For the newcomer user story: "record a live feed unattended for weeks with self-compaction
and an honest answer to 'how much data do I have.'"

## Describe the agency once

`stoptimes record` reads a [`Source`](../concepts/observations.md) YAML — the same file
`data arrivals`, `data frame`, `serve`, and `val` all read:

```yaml
# agencies/myagency.yaml
name: myagency
gtfs_static_url: https://example.invalid/gtfs.zip
vehicle_positions_url: https://example.invalid/VehiclePositions.pb
trip_updates_url: https://example.invalid/TripUpdates.pb
poll_interval_s: 15
timezone: UTC
arrival_radius_m: 40.0
dwell_s: 0.0
shape_threshold_m: 30.0
```

Real feed URLs and, if the feed needs one, an auth header via `auth_env` (a header name
mapped to an *environment variable name* — never a literal secret in the file itself; see
[Contracts §1](../CONTRACTS.md#1-source-feed-and-agency-descriptions)).

## Poll it

```bash
stoptimes record start --source myagency --sources-dir ./agencies \
  --spool-dir ./spool --status-dir ./status --duration 3600
```

Runs until `--duration` seconds elapse, or indefinitely until `SIGINT`/`SIGTERM` if you omit
it — the shape for "unattended for weeks" under something like systemd or a supervisor.
`--feeds vp,tu` restricts which feed kinds to poll if you only want one. `--dry-run` polls
and parses without writing anything, useful for checking a new agency file's URLs are
reachable before committing to a real recording. `--out` overrides the source's configured
archive destination (local path or `s3://bucket/prefix`).

`stoptimes record start --source myagency --sources-dir ./agencies --duration 1 --dry-run`
was run against the [quickstart](../quickstart.md)'s synthetic agency file while writing
this page and printed:

```json
{"stopped_reason": "duration_elapsed", "stats": {"vehicle_positions": {"polls": 1, "written": 0, "skipped_unchanged": 0, "errors": 0}, "trip_updates": {"polls": 1, "written": 0, "skipped_unchanged": 0, "errors": 0}}}
```

(zero writes because the quickstart's feed URLs are deliberately unreachable placeholders —
against a real feed, `written` counts real polled rows.)

!!! warning "Always pass `--status-dir`"
    If you omit `--status-dir`, the recorder falls back to a hard-coded
    `/var/lib/simovi/status` — a path baked in from SIMOVI's own deployment, left over in
    `stoptimes.record.status.DEFAULT_STATUS_DIR`. On any machine that isn't that specific
    VPS this fails outright (`PermissionError`/`FileNotFoundError`), not gracefully. Always
    pass an explicit `--status-dir` (or `--spool-dir`) until this default is fixed — see
    `docs/notes/p3-u5.md` for the bug report filed against this.

## Ask "how much data do I have"

```bash
stoptimes record status --source myagency --sources-dir ./agencies \
  --spool-dir ./spool --status-dir ./status --json
```

```json
{
  "source": "myagency",
  "vehicle_positions_spool": {"rows": 0, "oldest_ts": null, "newest_ts": null, "bytes": 0},
  "trip_updates_spool": null,
  "last_status": {},
  "archive_bytes": null
}
```

(shape confirmed against the quickstart's empty spool; a running recorder fills in real row
counts, timestamps, and the compacted archive's size.) This is the generic replacement for
SIMOVI's old ops-only `simovi-status` script — no agency hostnames or paths baked into the
command itself, everything comes from the flags you pass.

## Compaction

Spooled data needs to be compacted into the curated archive periodically —
`stoptimes record compact` does this, but as shipped today it's still the legacy compaction
tool's own argument shape (`--alias`, `--bucket`, `--endpoint`, `--feed
{mbta_vp,bucr_navsat}`), which is visibly agency-specific (it doesn't take a generic
`--source`, and its help text shows MBTA/bUCR feed names and a default `alias`/`bucket`
pair). Treat this subcommand as a pass-through to the pre-migration compaction CLI rather
than as a generalized `stoptimes record` command yet — see
`docs/notes/p3-u5.md` for the specifics filed here, and `docs/PLAN.md` §8 G1 (bUCR/MBTA
generality) for where this is meant to be finished.

## Next

Before training on what you've recorded, run [Diagnose a feed](diagnose-a-feed.md) to check
it's actually usable.
