# P0-B notes (train)

Things noticed outside `packages/train/**` while doing P0-B, or that the
orchestrator should be aware of. No files outside `packages/train/**` were
edited.

## `stoptimes train seed-baseline` has no `--help` gate

`packages/train/src/stoptimes/train/seed_baseline_model.py:main()` is called
unconditionally by `stoptimes.cli.main` when the CLI passes through
`--help` (it has no `argparse`, so `--help` isn't special-cased). Running
`uv run --no-sync stoptimes train seed-baseline --help` to verify the CLI
wiring therefore actually seeds a model and writes:

    packages/core/models/trained/{registry.json,polyreg_distance_global_baseline_v0.pkl,polyreg_distance_global_baseline_v0_meta.json}

This directory was untracked (not part of any package's owned paths); it has
been deleted (`rm -rf packages/core/models`) so it doesn't leak into a
commit. The orchestrator or CLI track (P0-D) may want `seed_baseline_model.py`
to check `sys.argv` for `--help`/no-args, or for the CLI's passthrough
wrapper to special-case `--help` generically, so `--help` never has a side
effect for any legacy-passthrough command.

## Pandas dtype drift fixed locally

`packages/train/tests/test_navsat_adapter.py::test_ts_is_utc_from_cr_datetime_utc`
failed against the pinned pandas version here (`pd.to_datetime` now infers
`datetime64[us, UTC]` for some inputs instead of the historical `ns`
resolution). Fixed inside owned code by pinning the cast explicitly in
`stoptimes.train.sources.navsat_adapter.navsat_to_vehicle_positions`
(`.astype("datetime64[ns, UTC]")`) rather than touching any pyproject/lock
file. Flagging in case other tracks hit the same pandas-version symptom
elsewhere (e.g. `stoptimes.record.storage` timestamp columns) and want a
shared fix instead of per-package patches.

## iCloud sets the macOS "hidden" flag on `.venv`'s `.pth` files -> `import stoptimes` breaks

Environmental gotcha, not caused by any change in this track, but it will hit
every track's worktree (repo lives under `~/Desktop`, synced by iCloud —
`AGENTS.md`/`PLAN.md` §10 already warn about `* 2.py`-style duplicates from
the same sync). Symptom: `uv run --no-sync python -c "import stoptimes..."`
suddenly fails with `ModuleNotFoundError: No module named 'stoptimes'`, and
`pytest` collection fails the same way, even though `uv sync` succeeded and
nothing under `packages/` changed. Cause: `ls -lO
.venv/lib/python3.12/site-packages/*.pth` shows the `hidden` BSD flag
(`UF_HIDDEN`) set on the editable-install `.pth` files (`_editable_impl_*.pth`,
`_virtualenv.pth`, `a1_coverage.pth`) — Python 3.12's `site` module skips any
`.pth` file with that flag (`-v` shows "Skipping hidden .pth file: ..."), so
none of the workspace packages' `src/` dirs get added to `sys.path`. iCloud
appears to (re-)apply this flag on its own, even seconds after clearing it.

Workaround used here: `chflags nohidden
.venv/lib/python3.12/site-packages/*.pth` immediately before the python/pytest
invocation, in the same shell command (a separate `ls` in between was enough
time for iCloud to re-hide them). If a track's tests suddenly can't import
`stoptimes` for no code reason, check `ls -lO .venv/lib/python3.12/site-packages/*.pth`
for `hidden` before assuming a real regression. A more permanent fix (move
`.venv`/the repo off an iCloud-synced path, or a pre-test hook that clears the
flag) is outside this track's owned paths.
