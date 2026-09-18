"""`stoptimes --help` and every `<group> --help` must return in under 300ms
(docs/PLAN.md §7, track U2). This is a real subprocess/wall-clock test,
mirroring how a user actually experiences startup: `runner.invoke` reuses the
already-imported process and wouldn't catch a regression where a heavy
import creeps back into module scope.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_SRC_DIRS = [str(p) for p in sorted(_REPO_ROOT.glob("packages/*/src"))]

BUDGET_S = 0.3
# A cold interpreter can be slower on a loaded CI box; take the best of a few
# runs so the assertion is about the CLI's own cost, not scheduler noise.
ATTEMPTS = 3


def _env() -> dict[str, str]:
    full_env = dict(os.environ)
    existing = full_env.get("PYTHONPATH")
    src_dirs = [*_WORKSPACE_SRC_DIRS, existing] if existing else _WORKSPACE_SRC_DIRS
    full_env["PYTHONPATH"] = os.pathsep.join(src_dirs)
    return full_env


def _best_help_time(*args: str) -> float:
    best = float("inf")
    for _ in range(ATTEMPTS):
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "stoptimes.cli.main", *args, "--help"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            env=_env(),
        )
        elapsed = time.monotonic() - start
        assert result.returncode == 0, result.stdout + result.stderr
        best = min(best, elapsed)
    return best


@pytest.mark.parametrize(
    "group", ["", "data", "record", "train", "models", "serve", "val"]
)
def test_help_is_fast(group: str) -> None:
    args = (group,) if group else ()
    elapsed = _best_help_time(*args)
    label = f"stoptimes {group} --help".strip() if group else "stoptimes --help"
    assert elapsed < BUDGET_S, f"`{label}` took {elapsed:.3f}s, budget is {BUDGET_S}s"
