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

## What's planned (Phase 2, track T5)

`recorder.py` is a placeholder for a standalone recorder: poll loop -> spool
-> hourly flush -> nightly compaction, with no Django/Celery dependency
anywhere in the loop, plus `record start` / `record compact` / `record
status` CLI commands (`record status` replacing the ops script
`simovi-status`) and a Dockerfile. See `docs/PLAN.md` §6 (T5).

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
