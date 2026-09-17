# P0-A notes (models + serve)

## Environment quirk: iCloud-hidden `.venv` editable-install `.pth` files

This worktree (like the whole repo) lives under `~/Desktop`, which iCloud
syncs. During this track's work, the uv-generated
`.venv/lib/python3.12/site-packages/_editable_impl_stoptimes*.pth` files
repeatedly picked up the macOS `UF_HIDDEN` file flag (`chflags hidden`),
apparently reapplied by the iCloud sync daemon shortly after any write
touches them. CPython's `site.addpackage` silently **skips a hidden `.pth`
file** (see `site.py`: `if st_flags & stat.UF_HIDDEN: ... return`), so any
fresh interpreter start (including the `python -m pytest` process itself,
and subprocesses `test_lazy_xgboost_import.py` spawns via
`sys.executable`) would fail with `ModuleNotFoundError: No module named
'stoptimes'` even though nothing in the workspace was broken.

Workaround used during this track:
`chflags nohidden .venv/lib/python3.12/site-packages/_editable_impl_stoptimes*.pth`
run immediately before each `uv run`/pytest invocation (the flag reappears
within roughly a second, so it must be the *last* command before the run,
chained in the same shell invocation).

Also hardened `packages/models/tests/test_lazy_xgboost_import.py`'s `_run`
helper to pass the parent process's `sys.path` through as `PYTHONPATH` to
the subprocess, rather than relying solely on the child re-deriving it from
the (potentially transiently hidden) `.pth` files. This does not weaken the
test's intent -- the subprocess still starts with a fresh `sys.modules`,
which is what the "xgboost not imported" assertions depend on.

**Not fixed** (outside owned paths / not a code issue): if this recurs for
other tracks/CI, the durable fix is likely one of: disable iCloud sync for
`.venv` directories specifically (e.g. via `.nosync` suffix or an iCloud
exclusion), or have CI create venvs outside any iCloud-synced path.

## Legacy pickle module names: bare form, not `models.*`-prefixed

The task description's examples (`models.xgb.train`, `core.*`,
`eta_service.*`, `feature_engineering.*`) suggested pickled legacy classes
record module paths prefixed with the top-level package (`models.`). A
manual, read-only check against the real registry at
`/Users/dotj/Desktop/SIMOVI/git.no_sync/gtfs-eta/models/trained/` (32
artifacts; copied to a `/tmp` scratch dir, `MODEL_REGISTRY_DIR` pointed at
the copy, original directory never modified) showed the actual pickled
`__module__` values are **bare**, e.g. `historical_mean.train`,
`polyreg_distance.train`, `ewma.train`, `xgb.train`, `lstm.train` -- not
`models.historical_mean.train` etc. This matches how gtfs-eta's trainers
actually ran (`sys.path.append(<models dir>)` then `from xgb.train import
...`, so `xgb.train` was genuinely a top-level module at pickle time, not a
submodule of `models`).

`stoptimes.models.common.legacy_unpickler.remap_legacy_module` handles
**both** forms (the bare one real artifacts use, and the `models.`-prefixed
one named in the task, in case some artifact was pickled a different way).

Manual verification result: sampled 6 real artifacts (historical_mean,
polyreg_distance, ewma, xgboost families) from the copied registry and
loaded each through `ModelRegistry.load_model` (which uses
`LegacyModuleUnpickler`) -- all 6 loaded successfully and remapped to their
`stoptimes.models.<family>.train` classes. sklearn/xgboost emitted their
usual `InconsistentVersionWarning`/version-mismatch warnings (the artifacts
were trained under older sklearn/xgboost than this workspace's), which is
expected and unrelated to the module-path remapping. The original
`gtfs-eta/models/trained/` directory was only read from, never written to;
the temp copy was deleted after the check.

## Registry default location changed

`packages/core/src/stoptimes/core/config.py`'s `model_registry_dir` no
longer resolves relative to a discovered "project root". New resolution
order: `MODEL_REGISTRY_DIR` env var > `$STOPTIMES_HOME/models` >
`./artifacts/models` (relative to CWD). Any other track/consumer that
relied on the old implicit default (`<project_root>/models/trained`) needs
to set `MODEL_REGISTRY_DIR` or `STOPTIMES_HOME` explicitly, or accept the
new `./artifacts/models` default. No other Phase-0 track's code referenced
`get_config().project_root`/`models_dir`/etc. (verified by grep across
`packages/`), so this should be a no-op for everyone else in Phase 0.
