# Agency profiles

One YAML file per agency, named `<agency-id>.yaml` (e.g. `mbta.yaml`,
`bucr.yaml`). Each file does double duty:

- **Source config** (`stoptimes.val`): the feed URLs and ground-truth
  detector parameters `stoptimes val` needs to poll and score a GTFS-RT
  source. These are the fields that existed here before Phase 1 (`name`,
  `gtfs_static_url`, `vehicle_positions_url`, `trip_updates_url`,
  `poll_interval_s`, `headers`, `auth`, `arrival_radius_m`, `dwell_s`,
  `shape_threshold_m`).
- **Agency profile** (`stoptimes.core.config`): optional top-level keys that
  `stoptimes.serve` and `stoptimes.train` resolve timezone, holiday
  calendar, and served-model pins through:

  ```yaml
  timezone: America/New_York       # IANA tz name for temporal features
  holiday_region: US_MA            # passed to the `holidays` package
  serving:
    primary_model: null            # registry key served by default, or null
    fallback_model: null           # tried if primary_model is missing
  ```

  Both `stoptimes.serve.estimator` and `stoptimes.train.datasets.dataset_builder`
  resolve through the same table (`stoptimes.core.config.AGENCY_TEMPORAL_DEFAULTS`
  / `ProjectConfig`), so a given timestamp produces identical temporal
  features whether it's being trained on or predicted against.

None of these agency-profile keys are required. An agency with no profile
keys at all still works as a plain `stoptimes val` source; `stoptimes.core`
falls back to generic defaults (timezone `UTC`, no holiday region, no
pinned model).

## Discovery: the sources search path

Both `stoptimes val` and `stoptimes serve`/`stoptimes train` look for these
files in the same three places, in order:

1. An explicit directory -- e.g. `stoptimes val sources --sources-dir <dir>`,
   or `STOPTIMES_AGENCY_FILE=<path>` to point straight at one file.
2. `STOPTIMES_SOURCES`, an `os.pathsep`-separated list of directories.
3. `./examples/agencies`, relative to the current working directory.

(`stoptimes.core.config.sources_search_path()` is the shared resolver both
packages call.)

## Selecting an agency

Set `STOPTIMES_AGENCY=<id>` (an agency id, looked up by name on the search
path above) or `STOPTIMES_AGENCY_FILE=<path>` (an explicit file, skipping
the search). The deprecated `ETA_AGENCY` name still works, with a
`DeprecationWarning`, and is removed in Phase 4.

## Env vars

| Current | Deprecated | Meaning |
|---|---|---|
| `STOPTIMES_AGENCY` | `ETA_AGENCY` | Active agency id, looked up on the sources search path |
| `STOPTIMES_AGENCY_FILE` | -- | Explicit profile file path, skipping the search path |
| `STOPTIMES_SOURCES` | -- | `os.pathsep`-separated list of directories to search |
| `STOPTIMES_TIMEZONE` | `ETA_TIMEZONE` | Overrides the resolved profile's timezone |
| `STOPTIMES_HOLIDAY_REGION` | `ETA_HOLIDAY_REGION` | Overrides the resolved profile's holiday region |
| `STOPTIMES_MODEL_KEY` | `ETA_MODEL_KEY` | Overrides the resolved profile's `serving.primary_model` |
| `STOPTIMES_MODEL_FALLBACK_KEY` | `ETA_MODEL_FALLBACK_KEY` | Overrides the resolved profile's `serving.fallback_model` |
| `STOPTIMES_MODEL_REGISTRY_DIR` | `MODEL_REGISTRY_DIR` | Trained-model registry location |
| `STOPTIMES_HOME` | -- | Base directory for stoptimes state (registry defaults to `$STOPTIMES_HOME/models`) |

The new name always wins when both are set.

## Deprecated builtin fallback

If no profile file can be found anywhere on the search path, `stoptimes.core.config`
falls back to a small hardcoded table (`_LEGACY_BUILTIN_PROFILES` in
`packages/core/src/stoptimes/core/config.py`) for `mbta`/`bucr` only, so
existing SIMOVI production deploys that set `ETA_AGENCY=bucr` with no
`examples/agencies/` checked out keep serving the pinned bUCR model. It
emits a `DeprecationWarning` and is deleted in Phase 4 (see `docs/PLAN.md`
§8 G1) -- ship a profile file (or point `STOPTIMES_SOURCES`/
`STOPTIMES_AGENCY_FILE` at one) instead.
