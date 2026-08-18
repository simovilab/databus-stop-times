# ETA Pipeline — Runbook

Useful commands for running, inspecting, and testing the GTFS-RT ingestion +
S3 ETA pipeline. Paths assume the ingest project at `gtfs-rt-pipeline/`,
relative to the repo root (`gtfs-eta/`), unless noted.

> Rebuilt 2026-08-14 (roadmap Phase 0). The pipeline described below —
> spool → hourly staging → daily compaction, both collectors — replaced a
> design that wrote to S3/Postgres per poll and OOM-killed itself on
> 2026-07-29. See `RESEARCH_ROADMAP.md` for the incident writeup and
> `S3_LAYOUT.md` for the storage contract this runbook operates against.
>
> **Updated 2026-08-18:** paths below corrected from `eta_prediction/gtfs-rt-pipeline/`
> to `gtfs-rt-pipeline/` — this doc predated the 2026-08-17 extraction into a standalone
> `gtfs-eta` repo and still assumed the old nesting under a `gtfs-django/eta_prediction/`
> checkout. The Docker build itself had the same stale-path bug (now fixed — see
> `RESEARCH_ROADMAP.md`'s *Docker / cross-repo build* note) and had never actually been
> run successfully post-extraction until this pass. `docker compose exec web` commands in
> this doc also gained a couple of undocumented prerequisite steps that were only
> discovered by actually running them — see the *Build a training dataset* section below.

## Prerequisite: a sibling `gtfs-django` checkout

The Docker build (`gtfs-rt-pipeline/docker-compose.yml` / `Dockerfile`) requires a local
checkout of `github.com/simovilab/gtfs-django` at `../gtfs-django` relative to this repo
(i.e. both repos as siblings under the same parent directory) — `gtfs-rt-pipeline`'s
`sch_pipeline` app imports `gtfs.models` (the abstract `Base*` GTFS models) from it at
runtime, a real dependency, not just a training-time one. Without it, `docker compose build`
fails resolving the `gtfs-django` editable-path dependency. This is a stopgap (see
`RESEARCH_ROADMAP.md`'s *Docker / cross-repo build* note under v3.1) — `gtfs-django` is
read-only from this repo's perspective; never modify it to unblock a build here. 1.8's
package restructure is the eventual real fix (removing this dependency entirely).

## Components
- **`gtfs-rt-pipeline/`** — Django + Celery. `poll_vehicle_positions_s3`
  polls MBTA GTFS-RT into a local DuckDB spool (no Postgres, no per-poll S3
  write); `flush_vp_spool_s3` moves it to S3 hourly; `compact_vp_day` folds
  closed days into the curated layout nightly; `snapshot_static_gtfs` takes
  a weekly dated snapshot of each agency's static GTFS.
- **`rt_pipeline/storage/`** — the spool (`spool.py`), curated read/write +
  partition index (`s3_writer.py`, `manifest.py`), static GTFS snapshots
  (`static_gtfs.py`).
- **`rt_pipeline/compaction/`** — staging→curated compaction, standalone
  package (own tests, own CLI: `python -m rt_pipeline.compaction.cli`).
- **`feature_engineering/`** — dataset builder (reads RT from S3).
- **`models/`, `eta_service/`** — training, registry, estimator.

## Environment (`gtfs-rt-pipeline/.env`)
Required: `REDIS_URL`, `FEED_NAME`, `GTFSRT_VEHICLE_POSITIONS_URL`,
`GTFSRT_TRIP_UPDATES_URL`, `POLL_SECONDS` (default 5). `DATABASE_URL` is
still required by Django but VP polling no longer writes to it.

S3 / MinIO (spool→staging→curated path, and static snapshots):
```
AWS_ENDPOINT_URL=https://data.simovilab.org
AWS_ACCESS_KEY_ID=...        # from SIMOVILAB-S3.md (gitignored — never commit)
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

SPOOL_PATH=/data/spool/vp_spool.duckdb          # DuckDB is single-writer:
                                                 # the poll queue MUST run at
                                                 # --concurrency=1
STATUS_DIR=/var/lib/simovi/status               # cat/tail-able health files
S3_VP_STAGING_BASE_URI=s3://transit/feeds/mbta/vehicle_positions_staging
SPOOL_FLUSH_MINUTE=2
COMPACT_HOUR_UTC=3
COMPACT_MINUTE=15
MBTA_GTFS_STATIC_URL=https://cdn.mbta.com/MBTA_GTFS.zip
BUCR_GTFS_STATIC_URL=...                        # SIMOVI-served, moves; set explicitly
STATIC_GTFS_SNAPSHOT_DOW=1                      # Monday
STATIC_GTFS_SNAPSHOT_HOUR_UTC=4
```
> ⚠️ A malformed `AWS_ENDPOINT_URL` makes the hourly flush / nightly
> compaction fail loudly now (they raise and log `last_error`, visible via
> `simovi-status`) — this is a deliberate change from the old dual-write
> sink, which failed silently.
> ⚠️ **`queue=` on a task's `@shared_task` decorator overrides
> `task_routes` in `celery.py`.** The flush task must stay on the `fetch`
> queue (same worker as polling — DuckDB is single-writer, and only that
> worker mounts the spool volume). Setting the route in `celery.py` alone
> is not enough; this caused a real incident where the flush silently ran
> against an empty database on the wrong worker and reported
> `{'flushed': 0}` as success.

Load creds into a shell for host-side tools without committing them. Pull
these from wherever you keep secrets (password manager, `pass`, a
`chmod 600` `.env` sourced separately, etc.) — never from a plaintext file
checked into or living alongside the repo:
```bash
export AWS_ENDPOINT_URL=https://data.simovilab.org AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:?set this from your secrets manager}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:?set this from your secrets manager}
```

## Run — full Docker stack
```bash
cd gtfs-rt-pipeline
docker compose up -d --build
docker compose ps
ssh jae@hetzner simovi-status   # one-screen health check, both collectors — see below
docker compose down             # stop
```
Four Celery services matter: `celery-worker` (`-Q fetch`, `--concurrency=1` —
polling + hourly flush, must stay serialized against the DuckDB spool),
`celery-maint` (`-Q maint` — daily compaction + weekly static-GTFS
snapshot, no spool volume mounted), `celery-beat` (schedules all of the
above), `redis`.

## Run — local app + Docker infra (use if container DNS is flaky)
```bash
cd gtfs-rt-pipeline
docker compose up -d postgres redis
set -a; source .env; set +a
export DATABASE_URL=postgresql://gtfs:gtfs@localhost:15432/gtfs   # host-mapped ports
export REDIS_URL=redis://localhost:16379/0
uv run python manage.py migrate
uv run celery -A ingestproj worker -Q fetch -c 1 -l INFO          # terminal 1 (poll + flush)
uv run celery -A ingestproj worker -Q maint -l INFO               # terminal 2 (compaction + snapshots)
uv run celery -A ingestproj beat -l INFO                          # terminal 3
```

## Monitor
```bash
ssh jae@hetzner simovi-status
```
One screen, both collectors: last poll age, spool size, next/last flush,
last static-GTFS snapshot, MinIO disk/inode/object counts with a runway
projection, staging build-up (a nightly-compaction health signal), VPS
resources. Exits 1 if a collector has stalled. Source:
`gtfs-rt-pipeline/ops/simovi-status` — install a change with
`sudo cp ops/simovi-status /usr/local/bin/simovi-status` on the VPS after
syncing the repo (it is not run from the repo checkout directly). **Unverified
2026-08-18:** this path was written pre-extraction as `eta_prediction/gtfs-rt-pipeline/...`;
corrected here to match this repo's current layout, but the actual VPS checkout's path
was not confirmed via SSH this pass — check `ssh jae@hetzner` before trusting it blindly.

Per-collector status files, atomic + `cat`/`tail -f`-able, at
`$STATUS_DIR` (`/var/lib/simovi/status` in prod): `mbta.json`/`.txt`,
`navsat.json`/`.txt` (bUCR — named `navsat`, the collector's package name,
**not** `bucr`; a status write under the wrong name is silently invisible
to `simovi-status` and to the collector itself), plus `*.events.log` for
flushes/compactions/errors (`tail -f`).

## Inspect the S3 bucket
```bash
# structure / sizes (mc alias 'simovilab')
~/.local/bin/mc tree simovilab/transit/feeds/mbta/
~/.local/bin/mc ls --recursive simovilab/transit/feeds/mbta/vehicle_positions/          # curated
~/.local/bin/mc ls simovilab/transit/feeds/mbta/vehicle_positions_staging/year=2026/month=8/day=14/  # today's staging, pre-compaction
~/.local/bin/mc du simovilab/transit/feeds/mbta/vehicle_positions/

# project helpers (partition index + routes); needs AWS_* exported
cd gtfs-rt-pipeline
PYTHONPATH="$(pwd)" uv run python -c "from rt_pipeline.storage import list_partitions, available_routes; print(available_routes()); print(list_partitions().to_string(index=False))"
```
DuckDB query (footer-only, no data scan — use `parquet_file_metadata`, not
`read_parquet`+`count(*)`, when just counting rows across many files; a
full-month `read_parquet` glob has taken 15+ minutes against this MinIO):
```sql
CREATE OR REPLACE SECRET simovi (TYPE s3, PROVIDER config,
  KEY_ID '...', SECRET '...', REGION 'us-east-1',
  ENDPOINT 'data.simovilab.org', USE_SSL true, URL_STYLE 'path');
SELECT sum(num_rows), count(*)
FROM parquet_file_metadata('s3://transit/feeds/mbta/vehicle_positions/**/*.parquet');
```

## Compaction — manual / backfill runs
```bash
cd gtfs-rt-pipeline
python -m rt_pipeline.compaction.cli --dry-run                      # what would happen, today
python -m rt_pipeline.compaction.cli --since 2026-07-01 --until 2026-07-31
python -m rt_pipeline.compaction.cli --feed bucr_navsat
# --force: re-process a leaf even if it already holds a compacted
# <date>.parquet (the routine guard otherwise skips it forever). Only for
# backfilling dedup onto a day a PRE-dedup compaction already merged —
# never needed for routine runs. This is how roadmap 0.4b was run:
python -m rt_pipeline.compaction.cli --force --feed mbta_vp --since 2026-07-01 --until 2026-07-31
```
For a run expected to take more than a couple of minutes (0.4b's took
~70), don't run it as a plain foreground `ssh ... docker compose exec`.
The SSH session has died mid-run at least twice in this project from what
looks like ordinary network flakiness, killing the process with it (each
already-completed leaf survives — the swap is atomic — but you lose the
final summary and have to verify completion against the data directly).
Prefer `nohup ... > /path/inside/the/bind-mounted/repo/log 2>&1 & disown`
(note: `/tmp` on the host is **not** mounted into the containers) or a
detached `tmux` session.

## Build a training dataset (reads RT from S3)

**Verified working end-to-end 2026-08-18** (first time since extraction — see
`RESEARCH_ROADMAP.md`'s v3.1 note). Two one-time setup steps below were previously
undocumented; without them `import_gtfs` fails immediately and `build_eta_sample` silently
returns 0 records with no error (looks identical to a real "no data" condition — see the
credentials gotcha further down).

```bash
# One-time: a GTFSProvider row must exist before import_gtfs can attach a Feed to it.
# provider_id is a BigAutoField — the first row created gets id=1.
docker compose exec web python manage.py shell -c "
from sch_pipeline.models import GTFSProvider
GTFSProvider.objects.create(code='MBTA', name='Massachusetts Bay Transportation Authority')
"

# import_gtfs needs --url explicitly (GTFS_SCHEDULE_ZIP_URL isn't set by default,
# and the command silently prints 'No GTFS URL provided' and returns — no exception).
docker compose exec web python manage.py import_gtfs --url https://cdn.mbta.com/MBTA_GTFS.zip --provider-id 1

# --end-date is a half-open [start, end) boundary on `ts` — it does NOT include
# the end date itself. For a genuine single day, end-date must be the day AFTER.
docker compose exec web python manage.py build_eta_sample --route-ids Green-D,Green-E \
  --start-date 2026-07-01 --end-date 2026-07-02
# success log: "Fetch VehiclePositions from S3 ... N records" then a step-by-step
# progress trace ending in a "Dataset ready" summary box.
```

**Credentials gotcha:** an invalid/expired `AWS_ACCESS_KEY_ID` does **not** raise — every S3
read (`list_partitions`, `fetch_vehicle_positions`, `build_eta_sample`) just silently
returns empty/0 rows, indistinguishable from "no data for this filter." Verify credentials
work *before* debugging route/date filters:
```bash
docker compose exec web python -c "
import duckdb, os
con = duckdb.connect(); con.execute('INSTALL httpfs; LOAD httpfs;')
con.execute(f\"SET s3_endpoint='{os.environ['AWS_ENDPOINT_URL'].replace('https://','').replace('http://','')}'\")
con.execute(f\"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}'\")
con.execute(f\"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}'\")
con.execute(\"SET s3_region='us-east-1'; SET s3_url_style='path';\")
print(len(con.execute(\"SELECT * FROM glob('s3://transit/feeds/mbta/vehicle_positions/year=2026/month=7/day=15/**')\").fetchall()), 'objects found')
"
```
A `403 InvalidAccessKeyId` here means the key itself is stale — fix `.env` /
`gtfs-rt-pipeline/.env` and `docker compose up -d --force-recreate web celery-worker
celery-maint celery-beat` (env vars are read at container start, not live-reloaded).

At real scale (multi-day, high-frequency routes), `build_eta_sample` can run long — see
*Performance defects* in `RESEARCH_ROADMAP.md` for two bugs found and fixed 2026-08-18
that made this previously non-terminating at 28-day/Green-line scale. Run it detached for
anything beyond a day or two, same caution as the compaction note above.

## Train models

**Verified working end-to-end 2026-08-18** — first real training run (20 models: 5 families
× 4 dataset sizes) against `datasets/mbta_bus_222_15_37_{1,7,15,28}d.parquet`. Runs on the
**host**, not in Docker — training needs `sklearn`/`xgboost`/`matplotlib`, which are behind
the `train`/`viz` extras, not the collector container's dependency set.

```bash
# One-time: sync the host venv with training deps (sklearn/xgboost/Django/etc. — the
# collector container doesn't have these, and `models/` doesn't need Django at all).
uv sync --all-extras

uv run python models/train_all_models.py --dataset mbta_bus_222_15_37_28d --models all
# global models only by default; add --by-route to also train one model per route
# (prints a trips/observations-vs-MAE correlation table — not yet done for any dataset
# built this session, see RESEARCH_ROADMAP.md's "What's next").
```

`--dataset <name>` looks up `datasets/<name>.parquet` — no path, no extension. Models save
to `models/trained/{model_key}.pkl` + `_meta.json`, and register in
`models/trained/registry.json`; `model_key` embeds the dataset name, so training the same
model type against multiple dataset sizes doesn't collide.

`uv run pytest` and `uv run python -m pytest` **silently fall back to a global interpreter**
if `pytest` isn't itself a project dependency — `uv run pytest` on this repo picks up
`/usr/local/bin/pytest` (wrong Python, missing `sklearn`) rather than erroring. Use
`uv run --with pytest python -m pytest models/tests/ core/tests/` instead, which forces the
project venv.

## Validate trained models with etaval

`etaval` (sibling repo, `github.com/dotjae/etaval`) can score any model trained above
against real live traffic — see its own `README.md` for the full picture, but in short:

```bash
cd ../etaval
export MODEL_REGISTRY_DIR=$(pwd)/../gtfs-eta/models/trained   # see etaval README for why
uv sync --no-editable                                          # see etaval README for why
uv run --no-editable etaval run --source mbta \
  --predictor gtfsrt --predictor gtfs_eta:xgboost --predictor gtfs_eta:polyreg_time \
  --route 222 --route 15 --route 37 --duration 10m --out r.parquet
uv run --no-editable etaval report r.parquet
```

`--predictor gtfs_eta:{historical_mean,ewma,polyreg_distance,polyreg_time,xgboost}` all work
— `estimate_stop_times`'s smart model selection (route-specific model preferred over
global, ranked by `test_mae_seconds`) picks the best registered model automatically; no
`model_key` needs specifying by hand.

## Backfill existing Postgres VPs -> S3
```bash
docker compose exec web python manage.py backfill_s3 --start 2026-06-01 --end 2026-06-29 [--route-ids Green-D,Green-E] [--dry-run]
```
(A different backfill from the compaction one above — this one is for VPs
still sitting in Postgres from before the S3 sink existed, not for
re-deduplicating already-curated S3 data.)

## Tests
```bash
# package (Base* models) — repo root
# BROKEN as of 2026-08-18: tests/ does not exist at repo root (verified via `ls`).
# Either this was never carried over by the extraction or the directory was
# planned but never added — not determined this pass. Don't trust this command
# until someone locates/recreates the actual Base*-model tests it's meant to run.
uv run pytest tests/ -q

# storage (spool, s3_writer, static_gtfs) + compaction — from gtfs-rt-pipeline,
# no Django settings or real S3/MinIO needed (local tmpdir + mocks)
uv run --no-project --with pandas --with pyarrow --with duckdb --with pytest --with requests \
  python -m pytest rt_pipeline/storage/tests rt_pipeline/compaction/tests -q

# sink + rt_source (need a full Django env) — from gtfs-rt-pipeline
DJANGO_SETTINGS_MODULE=ingestproj.settings PYTHONPATH="..:$(pwd)" \
  uv run --extra dev python -m pytest \
  rt_pipeline/test_s3_sink.py ../feature_engineering/tests/test_rt_source.py -q

# django system check — from gtfs-rt-pipeline
uv run python manage.py check

# ETA wiring — from gtfs-eta (repo root)
PYTHONPATH="$(pwd)" uv run --with pytest python -m pytest models/tests/test_eta_wiring.py -q
```

**Running `feature_engineering/tests/` inside the running Docker container** (verified
2026-08-18 — 38 tests, all passing after the perf fixes above; this is the fastest way to
exercise `dataset_builder.py`/`spatial.py` against a real Django-loaded environment without
a full local `uv sync --extra train`). The image doesn't ship `pytest` — install it once per
container lifetime — and anything importing `dataset_builder.py` needs Django apps loaded
first (`sch_pipeline.models` pulls in `gtfs.models` transitively, which needs `django.setup()`
called before import, not just `DJANGO_SETTINGS_MODULE` set):
```bash
cd gtfs-rt-pipeline
docker compose exec web pip install -q pytest
docker compose exec web bash -c "cd /app/gtfs-eta && DJANGO_SETTINGS_MODULE=ingestproj.settings PYTHONPATH=/app/gtfs-eta:/app/gtfs-eta/gtfs-rt-pipeline python -c \"
import django; django.setup()
import pytest
raise SystemExit(pytest.main(['feature_engineering/tests/', '-q']))
\""
```
`feature_engineering/` and `core/` are bind-mounted into the `web`/`celery-worker`
containers (see `docker-compose.yml`), so edits to those files on the host are live inside
the container immediately — no rebuild needed to re-run tests after a change. A full image
rebuild (`docker compose build web`) is only needed for `gtfs-rt-pipeline/` changes
(Dockerfile `COPY`s that directory, not bind-mounts it) or dependency changes.

## Troubleshooting (issues seen in practice)
- **`Temporary failure in name resolution` (Error -3)** for `redis`/`postgres`
  inside containers → Docker embedded DNS wedged. `sudo systemctl restart docker`,
  then `docker compose down --remove-orphans && docker compose up -d`. Or use the
  local-app flow above (talks to `localhost:15432/16379`, no container DNS).
- **`port is already allocated` (15432/16379)** → leftover container:
  `docker compose down --remove-orphans`; if still held,
  `docker rm -f $(docker ps -aq --filter publish=15432)`.
- **Legacy Postgres dual-write sink (`S3_VP_SINK_ENABLED`)** — off by
  default and not the active path; `poll_vehicle_positions_s3` (the only
  scheduled VP task) writes to the spool only, never Postgres. If you've
  turned the sink on for some reason: it swallows errors, so check a VP
  parse log for `s3_rows`. Most common cause of Postgres-fills-S3-stays-empty:
  a malformed `AWS_ENDPOINT_URL` in `.env`.
- **TripUpdates is defined but NOT scheduled** — only `poll-vehicle-positions-s3`,
  `flush-vp-spool`, `compact-vp-day`, and `snapshot-static-gtfs` are in
  `celery_app.conf.beat_schedule` (`rt_pipeline/tasks.py`). Roadmap 0.5.
- **`queue=` on a `@shared_task` decorator overrides `task_routes` in
  `celery.py`** — setting a route there alone does nothing if the decorator
  also sets `queue=`. Bit this project once for real: the hourly flush ran
  on the wrong worker (no spool volume mounted), DuckDB silently created an
  empty database, and it reported `{'flushed': 0}` as success. Anything
  that must share a worker with polling (DuckDB is single-writer) needs
  `queue="fetch"` on the decorator itself.
- **Status files unreadable (`cat`: Permission denied)** — the container
  runs as root; `tempfile.mkstemp` (used for the atomic write in
  `rt_pipeline/status.py`) creates files `0600`, unreadable from the host
  as an ordinary user. Fixed with `os.fchmod(fd, 0o644)` before the
  `os.replace`; if a status file is unreadable again, check that fix is
  still in place rather than re-deriving it.
- **A collector's status name isn't necessarily its agency name** — bUCR's
  status files are `navsat.json`/`.txt`/`.events.log` (the collector's
  package name), not `bucr.*`. A write under the wrong name is silently
  invisible to both `simovi-status` and the collector — this happened once
  with the static-GTFS snapshot task; check `simovi-status` actually shows
  a new field after adding one, don't just trust the write succeeded.
- **Known dedup gap: same `(vehicle_id, ts)` under two different routes.**
  The natural key `(feed_name, vehicle_id, ts)` deliberately excludes
  `route_id`, but the curated layout partitions data *by* `route_id` and
  compaction dedups per-partition — so it can't see across routes. In
  practice this is a live mid-trip reassignment in MBTA's feed (two polls,
  20s apart, same vehicle+ts, different route_id/trip_id/current_status).
  Full 28-day scan (2026-08-14): 32,192,234 rows, 1,027 residual duplicate
  keys total (~0.0032%), present in every single day at 7–55 keys/day —
  small, and not something 0.4b's backfill could have caught, since it
  isn't a bug in that backfill. See RESEARCH_ROADMAP.md for the research
  framing.
- **GeoDjango/GDAL** — the Django image installs `gdal-bin`/`libgdal-dev`/`binutils`;
  required for the PostGIS backend used by `sch_pipeline` models.
