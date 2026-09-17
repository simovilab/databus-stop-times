"""
Tests that stoptimes.models stays light: importing it (or its registry /
predict modules) never pulls in xgboost, a training-only dependency (the
`train` extra in pyproject.toml).

Dependency direction (see docs/PLAN.md §1): core <- models <- serve <- train.
stoptimes.models must not depend on stoptimes.train, even lazily. A
previous version of stoptimes.models.__init__ had a PEP 562 __getattr__
that lazily imported stoptimes.train.train_all_models on attribute access
(`stoptimes.models.train_all_models`) -- which inverted that dependency
(and chained into xgb.train -> xgboost, only lazily). That accessor has
been removed outright: models has no train_all_models attribute at all,
lazy or otherwise. A caller that wants it imports it directly from
stoptimes.train (owned by a separate track/package).

Each check here runs in a fresh subprocess (via `python -c`) rather than
in-process, because sys.modules is a process-global cache: once anything in
the same pytest run imports xgboost, later in-process assertions about "not
in sys.modules" would be false regardless of whether THIS import path is
lazy. A subprocess is the only way to observe a clean import.

Run with: pytest packages/models/tests/test_lazy_xgboost_import.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    # Pass this (working) process's sys.path down as PYTHONPATH rather than
    # relying on the child re-deriving it from the venv's editable-install
    # .pth files. This repo lives under an iCloud-synced ~/Desktop tree,
    # where those .pth files can transiently pick up the macOS "hidden"
    # (UF_HIDDEN) flag from the sync daemon; site.py silently skips a
    # hidden .pth file (see CPython site.addpackage), which would make a
    # freshly spawned interpreter fail to find `stoptimes` at all -- a
    # host-environment flake, not a real import-graph regression. Explicit
    # PYTHONPATH sidesteps that path entirely while still exercising a
    # genuinely fresh sys.modules for the "not in sys.modules" assertions.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


class TestPlainImportStaysLight:
    def test_import_stoptimes_models_does_not_import_xgboost(self):
        result = _run(
            "import sys; import stoptimes.models; "
            "assert 'xgboost' not in sys.modules, sys.modules.keys()"
        )
        assert result.returncode == 0, result.stderr

    def test_prediction_paths_import_without_xgboost(self):
        # The live collector's actual import surface: registry + all four
        # predict modules. None of these should need xgboost.
        result = _run(
            "import sys\n"
            "import stoptimes.models.common.registry\n"
            "import stoptimes.models.historical_mean.predict\n"
            "import stoptimes.models.ewma.predict\n"
            "import stoptimes.models.polyreg_distance.predict\n"
            "assert 'xgboost' not in sys.modules, sys.modules.keys()\n"
        )
        assert result.returncode == 0, result.stderr


class TestNoLazyTrainAttribute:
    """models must not depend on train, not even lazily via __getattr__."""

    def test_train_all_models_is_not_an_attribute_of_models(self):
        result = _run(
            "import stoptimes.models\n"
            "stoptimes.models.train_all_models\n"
        )
        assert result.returncode != 0
        assert "AttributeError" in result.stderr

    def test_unknown_attribute_still_raises_attribute_error(self):
        result = _run(
            "import stoptimes.models\n"
            "stoptimes.models.definitely_not_a_real_attribute\n"
        )
        assert result.returncode != 0
        assert "AttributeError" in result.stderr
