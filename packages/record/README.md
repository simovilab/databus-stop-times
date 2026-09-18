# stoptimes-record

Record live GTFS-RT feeds into a compacted, queryable archive (local or S3).

## What's here today (Phase 0)

Migrated, Django/Celery-free code from the `gtfs-rt-pipeline` collector
(`legacy/collector/` in this repo), moved faithfully with its tests:

- `storage/` — the local DuckDB spool (`spool.py`, `spool_base.py`,
  `tu_spool.py`), the S3 writer (`s3_writer.py`), on-disk/S3 schemas
  (`schema.py`, `tu_schema.py`), manifest tracking (`manifest.py`), static
  GTFS handling (`static_gtfs.py`), and config (`config.py`).
- `compaction/` — turns spooled/staged data into the curated, Hive-partitioned
  Parquet archive: planning (`planning.py`), execution (`execution.py`),
  DuckDB operations (`duckdb_ops.py`), per-feed layout/dedup config
  (`feeds.py`), storage backends (`storage.py`), S3 credentials
  (`credentials.py`), and the `compact` CLI (`cli.py`).
- `tu_records.py` — flattens GTFS-RT TripUpdates into archive rows (route
  scoping, horizon capping); unit-testable without a database.
- `status.py` — atomic, human- and machine-readable status snapshots for a
  running collector (`<name>.json` / `.txt` / `.events.log`).

None of this imports Django, Celery, or `legacy.collector` at runtime. Where a
docstring references the code that still supplies configuration at the call
site (the legacy collector's Celery tasks), it says so explicitly — that code
has not moved.

## The standalone recorder (Phase 2, track T5)

`recorder.py` is an async poll loop over a `Source`'s realtime feeds
(`stoptimes.core.contracts.source.Source`, JSON or protobuf, vehicle
positions and/or trip updates), with no Django/Celery dependency anywhere in
the loop:

- one task per configured feed, each on its own jittered interval
  (`http.py`: conditional GETs via ETag/If-Modified-Since, a body-hash
  fallback for servers that don't support conditional requests, retry with
  backoff on 5xx/timeouts);
- parsed rows (`parsing.py`, `vp_records.py`, `tu_records.py`) are appended to
  the existing local DuckDB spool (`storage/spool.py`, `storage/tu_spool.py`)
  — a failed poll never touches the spool, so a transient outage can't lose
  already-spooled data;
- a periodic + final flush drains the spool to the existing Parquet staging
  layout (`storage/spool_base.flush_spool_to_staging`), whose prefix is now
  parameterized by source id (`storage/schema.staging_prefix`,
  `storage/tu_schema.staging_prefix`) instead of hardcoded to `mbta` —
  `staging_prefix("mbta")` still equals the original literal, so existing
  archives are unaffected;
- SIGINT/SIGTERM stop the loop cleanly: the current poll finishes, every spool
  with pending rows gets a final flush, and the process exits non-zero only on
  a real failure.

### Running the recorder

```bash
uv run --no-sync stoptimes record start --source mbta --out ./data/archive
uv run --no-sync stoptimes record start --source mbta --out s3://bucket/prefix --feeds vp
uv run --no-sync stoptimes record start --source mbta --duration 60 --dry-run
uv run --no-sync stoptimes record status --source mbta --json
```

`--source <id>` loads `<sources-dir>/<id>.yaml` (default:
`$STOPTIMES_SOURCES` or `./examples/agencies`), tried first as the
`contracts.source` YAML shape and falling back to the flat val-source shape
(`contracts.adapters.source_from_val_yaml`) — see `docs/CONTRACTS.md` §1.
`--out` overrides the source's configured archive with a local path or an
`s3://bucket/prefix` URI. `record status` replaces SIMOVI's ops-only
`simovi-status` script: spool row counts/sizes, the last status snapshot
(`status.py`), and archive size for a local archive — no agency hostnames or
paths are baked in; everything comes from `--source`/`--spool-dir`/
`--status-dir`.

### Docker

```bash
docker build -f packages/record/Dockerfile -t stoptimes-record .
docker run --rm -v "$PWD/examples/agencies:/sources:ro" -v recorder-data:/data \
  -e STOPTIMES_SOURCES=/sources \
  stoptimes-record --source mbta --out /data/archive --spool-dir /data/spool
```

Non-root, uv-based, nothing agency-specific baked in — credentials and the
chosen source are always passed at `docker run` time.

## Running compaction

```bash
uv run --no-sync stoptimes record compact --help
```

This passes through to `stoptimes.record.compaction.cli`, e.g.:

```bash
uv run --no-sync stoptimes record compact --dry-run
uv run --no-sync stoptimes record compact --feed bucr_navsat
uv run --no-sync stoptimes record compact --since 2026-07-01 --until 2026-07-31
```

Tests that need `mc`, the `duckdb` CLI binary, or a real S3 endpoint skip
cleanly when those aren't available (see `packages/record/tests/compaction/conftest.py`,
which otherwise runs against a local-directory `Storage` + in-process DuckDB).

## Tests

```bash
uv run --no-sync python -m pytest packages/record -q -p no:cacheprovider
```
