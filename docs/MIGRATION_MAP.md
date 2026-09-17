# Migration map

Where everything from `gtfs-eta` and `etaval` went. History was rewritten with
`git filter-repo`, so `git log --follow <new path>` shows the full history.

Sources: `gtfs-eta@feat/lstm-segment-model (c3e3dff)`, `etaval@feat/ops-console-eval-repro (4156710)`.

## gtfs-eta

| From | To | Import path |
|---|---|---|
| `core/*.py` | `packages/core/src/stoptimes/core/` | `stoptimes.core.*` |
| `core/tests/` | `packages/core/tests/` | |
| `feature_engineering/spatial.py` | `packages/core/.../core/spatial/geo.py` | `stoptimes.core.spatial.geo` |
| `feature_engineering/temporal.py` | `packages/core/.../core/features/temporal.py` | `stoptimes.core.features.temporal` |
| `feature_engineering/segment_core.py` | `packages/core/.../core/segments.py` | `stoptimes.core.segments` |
| `feature_engineering/{navsat_*,bucr_*}.py` | `packages/train/.../train/sources/` | `stoptimes.train.sources.*` |
| `feature_engineering/*.py` (rest) | `packages/train/.../train/datasets/` | `stoptimes.train.datasets.*` |
| `feature_engineering/tests/` | `packages/train/tests/` (spatial/temporal/segment tests → `packages/core/tests/`) | |
| `models/common/`, `models/<family>/` | `packages/models/src/stoptimes/models/` | `stoptimes.models.*` |
| `models/evaluation/` | `packages/train/.../train/evaluation/` | `stoptimes.train.evaluation.*` |
| `models/train_all_models.py` | `packages/train/.../train/train_all_models.py` | `stoptimes.train.train_all_models` |
| `models/tests/` | `packages/models/tests/` (`test_cost`, `test_stats` → train; `test_eta_wiring` → serve) | |
| `eta_service/` | `packages/serve/src/stoptimes/serve/` | `stoptimes.serve.estimator` |
| `eta_service/test_*.py`, `eta_service/tests/` | `packages/serve/tests/` | |
| `datasets/sample.parquet` | `packages/serve/tests/fixtures/sample.parquet` | |
| `gtfs_eta/seed_baseline_model.py` | `packages/train/.../train/seed_baseline_model.py` | `stoptimes.train.seed_baseline_model` |
| `gtfs_eta/__init__.py` (alias shim) | `legacy/gtfs-eta/gtfs_eta/` | retired |
| `gtfs-rt-pipeline/rt_pipeline/{storage,compaction}/` | `packages/record/src/stoptimes/record/` | `stoptimes.record.{storage,compaction}` |
| `gtfs-rt-pipeline/rt_pipeline/{tu_records,status}.py` | `packages/record/src/stoptimes/record/` | `stoptimes.record.*` |
| `gtfs-rt-pipeline/` (Django app, rest) | `legacy/collector/` | reference only |
| `prefect/`, `mqtt-subscriber/`, `docker/`, `scratchpad/` | `legacy/` | reference only |
| `Dockerfile`, compose files, `Makefile`, `pyproject.toml`, `uv.lock` | `legacy/gtfs-eta/` | reference only |
| `docs/` | `docs/research/` | |
| `README.md`, `CHANGELOG.md` | `docs/history/` | |
| `LICENSE`, `CITATION.cff` | root | |

## etaval

| From | To | Import path |
|---|---|---|
| `etaval/domain/models.py` | `packages/core/.../core/domain/models.py` | `stoptimes.core.domain.models` |
| `etaval/spatial/` | `packages/core/.../core/spatial/` | `stoptimes.core.spatial.polyline` |
| `etaval/feeds/static.py` | `packages/core/.../core/gtfs/static.py` | `stoptimes.core.gtfs.static` |
| `etaval/feeds/gtfsrt_json.py` | `packages/core/.../core/realtime/gtfsrt_json.py` | `stoptimes.core.realtime.gtfsrt_json` |
| `etaval/**` (rest) | `packages/val/src/stoptimes/val/` | `stoptimes.val.*` |
| `tests/test_spatial.py` | `packages/core/tests/test_polyline.py` | |
| `tests/` (rest) | `packages/val/tests/` | |
| `web/` | `web/` | |
| `plans/` | `docs/val/plans/` | |
| `README.md` | `packages/val/README.md` | |
| `pyproject.toml`, `.gitignore` | `legacy/etaval/` | reference only |

## CLI equivalents

| Before | Now |
|---|---|
| `etaval <cmd>` | `stoptimes val <cmd>` |
| `python -m rt_pipeline.compaction.cli` | `stoptimes record compact` |
| `python models/train_all_models.py` | `stoptimes train legacy-all` |
| `python -m gtfs_eta.seed_baseline_model` | `stoptimes train seed-baseline` |
| `python models/check_registry.py` | `stoptimes models check` |
