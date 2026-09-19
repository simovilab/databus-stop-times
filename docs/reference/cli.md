# CLI reference

Generated from the `stoptimes` Typer app's own `--help` output by
`docs/scripts/gen_cli_reference.py` — run that script after any CLI change;
do not hand-edit this file.

Every block below is the literal output of `stoptimes <path> --help` in this
checkout. Options shown as `[required]` have no default and must be passed.


## `stoptimes`

```text
Usage: stoptimes [OPTIONS] COMMAND [ARGS]...

  stoptimes — train, serve and validate transit arrival-time models on GTFS
  data.

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  serve     Run the serving pipeline and publish GTFS-RT TripUpdates...
  diagnose  Report GTFS / GTFS-RT feed quality problems before training...
  data      Load and inspect GTFS static + realtime data.
  record    Record live GTFS-RT feeds into a compacted archive.
  train     Build datasets and train models.
  models    List, inspect and promote trained models.
  val       Validate arrival predictions against ground truth (etaval).
```

## `stoptimes data`

```text
Usage: stoptimes data [OPTIONS] COMMAND [ARGS]...

  Load and inspect GTFS static + realtime data.

Options:
  --help  Show this message and exit.

Commands:
  add       Register GTFS static / realtime / AVL sources (protobuf,...
  list      List registered datasets.
  inspect   Summarize coverage of registered data (trips matched,...
  remove    Remove a dataset from the catalog (does not delete the...
  arrivals  Derive actual stop arrivals from vehicle positions...
  frame     Build the training frame: observations + arrivals + static...
```

### `stoptimes data add`

```text
Usage: stoptimes data add [OPTIONS] {name} {location}

  Register GTFS static / realtime / AVL sources (protobuf, JSON, parquet,
  CSV).

  `--json`: the registered `DatasetEntry` as a dict
  (`stoptimes.core.catalog`).

Arguments:
  name      Name to register this dataset under in the catalog.  [required]
  location  Local path, s3:// URI, or feed URL for this dataset.  [required]

Options:
  --source <str>            Source id (agency) this dataset belongs to.
                            [required]
  --kind <str>              observations | arrivals | static  [default:
                            observations]
  --format <str>            One of ('gtfsrt_json', 'gtfsrt_pb',
                            'archive_parquet', 'avl_csv') (observations only;
                            auto-detected if omitted).
  --observation-kind <str>  vehicle_position | trip_update (observations
                            only).  [default: vehicle_position]
  --mapping <str>           Column-mapping YAML file (avl_csv format only).
  --start-date <str>        Filter: archive_parquet only.
  --end-date <str>          Filter: archive_parquet only.
  --overwrite               Replace an existing entry with this name.
  --json                    Print the registered entry as JSON.
  --help                    Show this message and exit.
```

### `stoptimes data arrivals`

```text
Usage: stoptimes data arrivals [OPTIONS]

  Derive actual stop arrivals from vehicle positions (map-matching).

  The training target and the validation ground truth: observations in, one
  `Arrival` row per (trip, service day, stop) out.

  `--json`: ``{"source", "observations", "arrivals", "by_method",
  "static_tables", "out", "registered"}``.

Options:
  --observations <str>  Observations to derive from: a catalog dataset name,
                        or a path/URI.  [required]
  --static <str>        GTFS static .zip (path or URL).  [required]
  --source <str>        Source id (agency) these observations came from.
                        [required]
  --out <str>           Where to write the Arrival parquet.  [required]
  --sources-dir <str>   Directory of agency YAMLs (default: STOPTIMES_SOURCES
                        -> ./examples/agencies).
  --register <str>      Also register the result in the catalog under this
                        name.
  --overwrite           Replace an existing catalog entry / output file.
  --json                Print a machine-readable summary.
  --help                Show this message and exit.
```

### `stoptimes data frame`

```text
Usage: stoptimes data frame [OPTIONS]

  Build the training frame: observations + arrivals + static GTFS -> one row
  per (vehicle position, upcoming stop), ready for `train run` (via the
  feature registry, which adds temporal features on top).

  `--json`: ``{"source", "observations", "arrivals", "rows", "labeled_rows",
  "static_tables", "out", "registered"}``.

Options:
  --observations <str>     Observations to build from: a catalog dataset name,
                           or a path/URI.  [required]
  --arrivals <str>         The Arrival table (target): a catalog dataset name,
                           or a path/URI to a parquet file.  [required]
  --static <str>           GTFS static .zip (path or URL).  [required]
  --source <str>           Source id (agency) these observations/arrivals came
                           from.  [required]
  --out <str>              Where to write the training frame parquet.
                           [required]
  --max-stops-ahead <int>  Upcoming stops emitted per vehicle-position row.
                           [default: 5]
  --keep-unlabeled         Keep rows with no matching arrival (target left
                           NaN) instead of dropping them.
  --sources-dir <str>      Directory of agency YAMLs (default:
                           STOPTIMES_SOURCES -> ./examples/agencies).
  --register <str>         Also register the result in the catalog under this
                           name.
  --overwrite              Replace an existing catalog entry / output file.
  --json                   Print a machine-readable summary.
  --help                   Show this message and exit.
```

### `stoptimes data inspect`

```text
Usage: stoptimes data inspect [OPTIONS] {name}

  Summarize coverage of registered data (trips matched, arrivals present,
  gaps).

  `--json`: a flat `{"kind": str, ...}` dict; the remaining keys depend on
  `kind` (`static`/`arrivals`/an observation kind) -- see
  `stoptimes.core.catalog.inspect.build_coverage_report`.

Arguments:
  name  A dataset name registered with `data add`.  [required]

Options:
  --static <str>             A GTFS static .zip to check trip coverage
                             against.
  --gap-threshold-s <float>  Gap size (s) counted in gap_count.  [default:
                             300.0]
  --json
  --help                     Show this message and exit.
```

### `stoptimes data list`

```text
Usage: stoptimes data list [OPTIONS]

  List registered datasets.

  `--json`: a list of `DatasetEntry` dicts (`stoptimes.core.catalog`), one per
  registered dataset.

Options:
  --json
  --help  Show this message and exit.
```

### `stoptimes data remove`

```text
Usage: stoptimes data remove [OPTIONS] {name}

  Remove a dataset from the catalog (does not delete the underlying file).

  `--json`: ``{"name": str, "removed": bool}``.

Arguments:
  name  A dataset name registered with `data add`.  [required]

Options:
  --json
  --help  Show this message and exit.
```

## `stoptimes diagnose`

```text
Usage: stoptimes diagnose [OPTIONS]

  Report GTFS / GTFS-RT feed quality problems before training on them.

  Exits non-zero only on a usage error (a location that can't be read). A feed
  with quality problems is a successful run that reports them.

Options:
  --static <str>                  GTFS static .zip (path or URL). Omit to
                                  diagnose a feed with no static schedule at
                                  all.
  --vehicle-positions <str>       Vehicle-position observations: a `data add`
                                  dataset name, path, s3:// archive dir, or
                                  feed URL.
  --vehicle-positions-format <str>
                                  One of ('gtfsrt_json', 'gtfsrt_pb',
                                  'archive_parquet', 'avl_csv') (auto-detected
                                  if omitted).
  --trip-updates <str>            Trip-update observations: a `data add`
                                  dataset name, path, s3:// archive dir, or
                                  feed URL.
  --trip-updates-format <str>     One of ('gtfsrt_json', 'gtfsrt_pb',
                                  'archive_parquet', 'avl_csv') (auto-detected
                                  if omitted).
  --from <str>                    Window start (YYYY-MM-DD or full ISO 8601,
                                  day precision) — same spelling as `val
                                  replay`. Bounds an archive_parquet read to
                                  matching partitions before any file is
                                  opened; a live feed URL always reads its one
                                  current snapshot regardless. Recorded in the
                                  output so it's clear what was and wasn't
                                  examined.
  --to <str>                      Window end (YYYY-MM-DD or full ISO 8601, day
                                  precision). See --from.
  --source <str>                  Source id (agency). Tags the read
                                  observations, and — with --sources-dir —
                                  fills in --static/--vehicle-
                                  positions/--trip-updates from the agency
                                  file when they're omitted.
  --sources-dir <str>             Directory of agency YAMLs (default:
                                  STOPTIMES_SOURCES -> ./examples/agencies).
  --gap-threshold-s <float>       Vehicle-position sampling gap (s) counted as
                                  a coverage gap.  [default: 300.0]
  --stale-threshold-s <float>     ingested_at - ts (s) beyond which a record
                                  counts as stale.  [default: 120.0]
  --majority-threshold <float>    Affected share at/above which a mismatch
                                  finding is a blocker rather than degraded.
                                  [default: 0.5]
  --max-samples <int>             Affected-sample identifiers kept per
                                  finding.  [default: 10]
  --json                          Print the DiagnosticReport as JSON.
  --help                          Show this message and exit.
```

## `stoptimes models`

```text
Usage: stoptimes models [OPTIONS] COMMAND [ARGS]...

  List, inspect and promote trained models.

Options:
  --help  Show this message and exit.

Commands:
  check    Check the model registry (migrated gtfs-eta check_registry).
  list     List artifacts in a registry directory (contracts.artifact...
  card     Show a model card (data, features, scores by horizon).
  promote  Pin `artifact_id` into `source`'s agency profile...
  migrate  Migrate a legacy gtfs-eta model registry (`{key}.pkl` +...
```

### `stoptimes models card`

```text
Usage: stoptimes models card [OPTIONS] [artifact_id]

  Show a model card (data, features, scores by horizon).

  `--json`: ``{"artifact_id": str, "card_path": str, "card": str}`` (the
  card's raw markdown text, unparsed).

Arguments:
  artifact_id  Artifact id (registry subdirectory name).

Options:
  --registry <path>  Artifact registry directory.  [default: artifacts/models]
  --json             Print {'artifact_id', 'card_path', 'card'} as JSON
                     instead of raw markdown.
  --help             Show this message and exit.
```

### `stoptimes models check`

```text
Usage: stoptimes models check [OPTIONS]

  Check the model registry (migrated gtfs-eta check_registry).

Options:
  --help  Show this message and exit.
```

### `stoptimes models list`

```text
Usage: stoptimes models list [OPTIONS]

  List artifacts in a registry directory (contracts.artifact layout): scope,
  headline metric, and creation time.

  `--json`: a list of ``{"artifact_id", "family", "scope", "routes", "metric",
  "created"}``.

Options:
  --registry <path>  Artifact registry directory.  [default: artifacts/models]
  --source <str>     Filter to this Source.id
                     (contracts.artifact.ModelScope.source_id).
  --route <str>      Filter to artifacts scoped to this route.
  --family <str>     Filter to this model family.
  --json             Print as JSON instead of a table.
  --help             Show this message and exit.
```

### `stoptimes models migrate`

```text
Usage: stoptimes models migrate [OPTIONS] {legacy_dir} {registry_dir}

  Migrate a legacy gtfs-eta model registry (`{key}.pkl` + `{key}_meta.json`)
  into the `stoptimes.core.contracts.artifact` layout.

  Idempotent: an artifact already present at the destination is skipped unless
  `--overwrite` is given. Never raises on one bad artifact (e.g. one needing
  an uninstalled optional library like xgboost/torch) -- that artifact is
  reported skipped and the rest of the run continues. Exits 1 if any artifact
  failed outright.

Arguments:
  legacy_dir    Legacy gtfs-eta registry directory (*.pkl + *_meta.json
                pairs).  [required]
  registry_dir  Destination artifact registry directory (contracts/artifact.py
                layout).  [required]

Options:
  --dry-run    Report what would happen without writing anything.
  --overwrite  Re-migrate artifacts that already exist at the destination.
  --json       Print the report as JSON.
  --help       Show this message and exit.
```

### `stoptimes models promote`

```text
Usage: stoptimes models promote [OPTIONS] {artifact_id}

  Pin `artifact_id` into `source`'s agency profile
  (`serving.selection.pins[as_pin]`), per the
  `stoptimes.core.contracts.selection` contract. Refuses if the artifact
  doesn't exist in `registry`.

  `--json`: ``{"artifact_id": str, "pin": str, "source": str, "profile_path":
  str}``.

Arguments:
  artifact_id  Artifact id to promote.  [required]

Options:
  --as <str>            Pin key (contracts.selection.SelectionPolicy.pins):
                        "global", "route:<id>", "route:<id>/<direction>", or
                        "trip:<id>".  [required]
  --source <str>        Agency id whose profile file to update.  [required]
  --registry <path>     Artifact registry directory.  [default:
                        artifacts/models]
  --sources-dir <path>  Agency profile directory. Defaults to
                        STOPTIMES_SOURCES or ./examples/agencies.
  --json                Print the pin as JSON.
  --help                Show this message and exit.
```

## `stoptimes record`

```text
Usage: stoptimes record [OPTIONS] COMMAND [ARGS]...

  Record live GTFS-RT feeds into a compacted archive.

Options:
  --help  Show this message and exit.

Commands:
  start    Poll a Source's realtime feeds and spool them to the archive...
  compact  Compact spooled data into the curated archive (existing...
  status   Recorder status: spool size, last flush, last poll per feed,...
```

### `stoptimes record compact`

```text
Usage: stoptimes record compact [OPTIONS]

  Compact spooled data into the curated archive (existing implementation).

Options:
  --help  Show this message and exit.
```

### `stoptimes record start`

```text
Usage: stoptimes record start [OPTIONS]

  Poll a Source's realtime feeds and spool them to the archive (standalone, no
  Django/Celery).

  `--json`: ``{"stopped_reason": str, "stats": {feed_kind: {"polls",
  "written", "skipped_unchanged", "errors"}}}``.

Options:
  --source <str>       Source id (matches examples/agencies/<id>.yaml).
  --out <str>          Archive root: a local path or s3://bucket/prefix.
                       Overrides the source's configured archive.
  --feeds <str>        Comma-separated feed kinds to poll (vp,tu). Default:
                       every feed the source configures.
  --duration <float>   Stop automatically after this many seconds. Default:
                       run until SIGINT/SIGTERM.
  --spool-dir <path>   Local DuckDB spool directory.  [default: spool]
  --sources-dir <str>  Directory of <id>.yaml source configs (default:
                       $STOPTIMES_SOURCES or ./examples/agencies).
  --status-dir <str>   Where to write <source>.json/.txt/.events.log status
                       snapshots.
  --dry-run            Poll and parse but never write to the spool or archive.
  --json               Print the run summary as JSON instead of text.
  --help               Show this message and exit.
```

### `stoptimes record status`

```text
Usage: stoptimes record status [OPTIONS]

  Recorder status: spool size, last flush, last poll per feed, archive size.

  Generic replacement for SIMOVI's ops-only `simovi-status` script — no agency
  hostnames or paths baked in; everything comes from `--source` / `--spool-
  dir` / `--status-dir`.

  `--json`: ``{"source", "vehicle_positions_spool", "trip_updates_spool",
  "last_status", "archive_bytes"}``; the two spool keys are each ``{"rows",
  "oldest_ts", "newest_ts", "bytes"}`` or null if unstarted.

Options:
  --source <str>       Source id.
  --spool-dir <path>   [default: spool]
  --sources-dir <str>
  --status-dir <str>
  --json               Emit machine-readable JSON.
  --help               Show this message and exit.
```

## `stoptimes serve`

```text
Usage: stoptimes serve [OPTIONS]

  Run the serving pipeline and publish GTFS-RT TripUpdates (file, HTTP, MQTT).

  `--json`: one line per run (repeated if `--interval` loops): ``{"source",
  "artifact_id", "generated_at", "trip_update_count", "errors",
  "published"}``.

Options:
  --source <str>        Agency id, e.g. 'mbta' (examples/agencies/<id>.yaml).
  --sources-dir <path>  Directory of agency YAMLs (default: --sources-dir ->
                        STOPTIMES_SOURCES -> ./examples/agencies).
  --registry <path>     Model registry directory (one artifact.json per
                        subdir).
  --model <str>         Artifact id, or a BYO spec: pickle:PATH | onnx:PATH |
                        callable:module:func | http(s)://URL.
  --artifact <str>      Alias for --model.
  --pipeline <str>      Custom Pipeline: 'module:attr' or
                        'path/to/file.py:attr'.
  --sink <str>          file:PATH | http(s)://... | mqtt://host/topic
  --format <str>        Feed encoding: pb | json.  [default: json]
  --once                Produce one feed and exit (the Phase 2 e2e gate uses
                        this).
  --interval <float>    Seconds between runs; loops until interrupted.
  --json                Print a machine-readable run summary.
  --help                Show this message and exit.
```

## `stoptimes train`

```text
Usage: stoptimes train [OPTIONS] COMMAND [ARGS]...

  Build datasets and train models.

Options:
  --help  Show this message and exit.

Commands:
  run            Train from an experiment.yaml: resolve data, build...
  ui             Interactive experiment builder (TUI): choose target,...
  legacy-all     Train every model family on a prepared dataset (migrated...
  seed-baseline  Seed the synthetic global baseline model into the...
```

### `stoptimes train legacy-all`

```text
Usage: stoptimes train legacy-all [OPTIONS]

  Train every model family on a prepared dataset (migrated gtfs-eta trainer).

Options:
  --help  Show this message and exit.
```

### `stoptimes train run`

```text
Usage: stoptimes train run [OPTIONS] [experiment_path]

  Train from an experiment.yaml: resolve data, build features, split, fit the
  model plus its baselines, score by horizon, and write an artifact + card.
  `--dry-run` resolves and prints the plan without touching data.

  `--json` (ignored with `--dry-run`, which always prints the plan as JSON):
  ``{"artifact_id": str, "family": str, "metrics": {...}}``.

Arguments:
  experiment_path  experiment.yaml
                   (stoptimes.core.contracts.experiment.Experiment).

Options:
  --registry <path>     Artifact registry directory. Defaults to the
                        experiment's outputs.registry_dir.
  --dataset-dir <path>  Directory holding named datasets (overrides
                        STOPTIMES_DATASET_DIR / ./datasets).
  --dry-run             Print the resolved plan (data, features, split, model,
                        outputs) and exit.
  --overwrite           Replace an existing artifact with the same id instead
                        of refusing.
  --json                Print the trained artifact's id/family/metrics as
                        JSON.
  --help                Show this message and exit.
```

### `stoptimes train seed-baseline`

```text
Usage: stoptimes train seed-baseline [OPTIONS]

  Seed the synthetic global baseline model into the registry (migrated gtfs-
  eta helper).

  A regular command, not a passthrough: the underlying script takes no
  arguments and runs on import as __main__, so passing `--help` through would
  execute it.

Options:
  --help  Show this message and exit.
```

### `stoptimes train ui`

```text
Usage: stoptimes train ui [OPTIONS] [experiment_path]

  Interactive experiment builder (TUI): choose target, model family, features,
  hyperparameters and split, then write (or edit) an experiment.yaml. Never
  trains anything itself -- run the file it writes with `stoptimes train run
  <file>`.

Arguments:
  experiment_path  experiment.yaml to open for editing, or where to save a new
                   one. Omitted: starts a blank experiment; the save path is
                   entered inside the TUI.

Options:
  --help  Show this message and exit.
```

## `stoptimes val`

```text
Usage: stoptimes val [OPTIONS] COMMAND [ARGS]...

  Validate arrival predictions against ground truth (etaval).

Options:
  --help  Show this message and exit.

Commands:
  version       Print the stoptimes val version.
  sources       List all configured GTFS-RT sources.
  run           Run headless ETA validation against a live GTFS-RT...
  replay        Score predictions from a recorded archive instead of a...
  report        Render a metrics report from a Parquet results file.
  walk-forward  Walk-forward backtest: per-window MAE + CI over an...
  figures       Render publication figures (300 dpi PNG) from a Parquet...
  api           Run the REST + WebSocket API (was the `etaval-api`...
  tui           Launch the terminal ops console (was the `etaval-tui`...
```

### `stoptimes val api`

```text
Usage: stoptimes val api [OPTIONS]

  Run the REST + WebSocket API (was the `etaval-api` console script).

Options:
  --help  Show this message and exit.
```

### `stoptimes val figures`

```text
Usage: stoptimes val figures [OPTIONS]

  Render publication figures (300 dpi PNG) from a Parquet results file.

Options:
  --help  Show this message and exit.
```

### `stoptimes val replay`

```text
Usage: stoptimes val replay [OPTIONS]

  Score predictions from a recorded archive instead of a live feed.

Options:
  --help  Show this message and exit.
```

### `stoptimes val report`

```text
Usage: stoptimes val report [OPTIONS]

  Render a metrics report from a Parquet results file.

Options:
  --help  Show this message and exit.
```

### `stoptimes val run`

```text
Usage: stoptimes val run [OPTIONS]

  Run headless ETA validation against a live GTFS-RT source, or replay an
  archive.

Options:
  --help  Show this message and exit.
```

### `stoptimes val sources`

```text
Usage: stoptimes val sources [OPTIONS]

  List all configured GTFS-RT sources.

Options:
  --help  Show this message and exit.
```

### `stoptimes val tui`

```text
Usage: stoptimes val tui [OPTIONS]

  Launch the terminal ops console (was the `etaval-tui` console script).

Options:
  --help  Show this message and exit.
```

### `stoptimes val version`

```text
Usage: stoptimes val version [OPTIONS]

  Print the stoptimes val version.

Options:
  --help  Show this message and exit.
```

### `stoptimes val walk-forward`

```text
Usage: stoptimes val walk-forward [OPTIONS]

  Walk-forward backtest: per-window MAE + CI over an archived replay window.

Options:
  --help  Show this message and exit.
```


<!-- generated 2026-09-18 -->
