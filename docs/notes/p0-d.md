# P0-D notes

Track: CLI, val docs, CI, web (`packages/cli/**`, `packages/val/README.md`, `web/**`, `.github/**`).

## For P0-B (train): `train_all_models.py` top-level bare imports are broken today

`packages/train/src/stoptimes/train/train_all_models.py` (not owned by this track) still has,
at module top level:

```python
sys.path.append(str(Path(__file__).parent))          # models/
sys.path.append(str(Path(__file__).parent.parent))   # eta_prediction/ (for feature_engineering)

from common.data import load_dataset
from historical_mean.train import train_historical_mean
from polyreg_distance.train import train_polyreg_distance
from polyreg_time.train import train_polyreg_time
from ewma.train import train_ewma
from xgb.train import train_xgboost
```

None of `common`, `historical_mean`, `polyreg_distance`, `polyreg_time`, `ewma`, or `xgb` exist
as bare top-level packages anywhere in this workspace (they now live under
`stoptimes.models.*`) — so `stoptimes train legacy-all` (and any `--help` on it) reliably fails
with `ModuleNotFoundError: No module named 'common'` at import time, before argparse ever runs.
This is exactly PLAN §4.3 P0-B's listed item ("`train_all_models.py`: replace bare
`from ewma…`/`from common…`/`sys.path` hacks with package imports"), so it's left for that
track. `packages/cli/tests/test_cli.py::test_train_legacy_all_passthrough_reaches_train_all_models`
accepts either a clean argparse help *or* this specific known error, so it still verifies the
CLI passthrough plumbing (which this track does own) without blocking on P0-B's fix. Once P0-B
lands, that test will only exercise the "clean help" branch; no test change needed then.

`stoptimes train seed-baseline` has no such problem (it imports properly via
`stoptimes.models.common.registry`) and is exercised by an end-to-end passthrough test that
actually runs it (fixed seed, synthetic data, `MODEL_REGISTRY_DIR` pointed at `tmp_path`).

## Local dev-machine environment quirk (not a code bug, not filed against any track)

On this machine, `uv`'s editable-install `.pth` files under
`.venv/lib/python3.12/site-packages/_editable_impl_stoptimes*.pth` intermittently carry the
macOS Finder "hidden" flag (`UF_HIDDEN`). cpython 3.12's `site.addpackage()` silently skips
hidden `.pth` files, so a freshly spawned interpreter (including the installed `stoptimes`
console script, or a bare `.venv/bin/python -c "import stoptimes"`) sometimes raises
`ModuleNotFoundError: No module named 'stoptimes'` even though the workspace is synced and
correct. Symptom: `uv run --no-sync stoptimes ...` or `uv run --no-sync python -m pytest ...`
occasionally fails outright at import time; running `chflags nohidden
.venv/lib/python3.12/site-packages/_editable_impl_stoptimes*.pth` immediately before the command
reliably works around it for that one invocation. Root cause looks specific to this repo living
under an iCloud-synced `~/Desktop` tree combined with `uv`'s venv layout; it does not reproduce
on a normal Linux CI runner (no `UF_HIDDEN` there), so `.github/workflows/ci.yml` is unaffected.

To make `packages/cli/tests/test_cli.py`'s subprocess-based passthrough tests immune to this on
any machine, `_uv_run()` sets `PYTHONPATH` explicitly to every workspace member's `src/`
directory before spawning, instead of depending on the venv's `.pth` resolution at all. This is
a test-only change (`packages/cli/tests/test_cli.py`), harmless in CI, and does not touch any
tracked `.venv` state (that directory isn't part of the repo).

## Web

`web/` (Nuxt app) already talks to `stoptimes.val.api` correctly and needed no code changes:
`NUXT_PUBLIC_API_BASE` defaults to `http://localhost:8010`, matching
`stoptimes.val.api.config.get_settings()`'s default port, and
`RunProvenance.to_dict()`'s field names (`etaval_git_sha`, `gtfs_eta_git_sha`, etc.) are
unchanged, matching `web/app/types/api.ts`. Only `web/README.md`'s backend-launch instructions
were stale (`etaval-api` console script, a `docs/eta-val/` repo-root path) and have been updated
to `stoptimes val api` from the uv workspace root.
