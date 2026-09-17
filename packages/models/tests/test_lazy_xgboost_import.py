"""
Tests for lazy-loading models.train_all_models (and therefore xgboost).

xgboost is a training-only dependency (the `train` extra in pyproject.toml).
models/__init__.py used to import train_all_models eagerly, which chains
into xgb.train -> xgboost, forcing every inference-only consumer (the live
gtfs-rt-pipeline collector, or a downstream project that only needs
models.common.registry / models.*.predict) to install xgboost just to run
`import models`.

Each check here runs in a fresh subprocess (via `python -c`) rather than
in-process, because sys.modules is a process-global cache: once anything in
the same pytest run imports xgboost, later in-process assertions about "not
in sys.modules" would be false regardless of whether THIS import path is
lazy. A subprocess is the only way to observe a clean import.

Run with: pytest models/tests/test_lazy_xgboost_import.py -v
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestPlainImportStaysLight:
    def test_import_models_does_not_import_xgboost(self):
        result = _run(
            "import sys; import models; "
            "assert 'xgboost' not in sys.modules, sys.modules.keys()"
        )
        assert result.returncode == 0, result.stderr

    def test_import_gtfs_eta_models_does_not_import_xgboost(self):
        result = _run(
            "import sys; import gtfs_eta.models; "
            "assert 'xgboost' not in sys.modules, sys.modules.keys()"
        )
        assert result.returncode == 0, result.stderr

    def test_prediction_paths_import_without_xgboost(self):
        # The live collector's actual import surface: registry + all four
        # predict modules. None of these should need xgboost.
        result = _run(
            "import sys\n"
            "import models.common.registry\n"
            "import models.historical_mean.predict\n"
            "import models.ewma.predict\n"
            "import models.polyreg_distance.predict\n"
            "assert 'xgboost' not in sys.modules, sys.modules.keys()\n"
        )
        assert result.returncode == 0, result.stderr


class TestTrainingPathStillWorks:
    def test_from_models_import_train_all_models_still_works(self):
        result = _run(
            "import sys\n"
            "from models import train_all_models\n"
            "assert callable(train_all_models)\n"
            "assert 'xgboost' in sys.modules\n"
        )
        assert result.returncode == 0, result.stderr

    def test_models_train_all_models_attribute_access_still_works(self):
        result = _run(
            "import sys\n"
            "import models\n"
            "assert 'xgboost' not in sys.modules\n"
            "fn = models.train_all_models\n"
            "assert callable(fn)\n"
            "assert 'xgboost' in sys.modules\n"
        )
        assert result.returncode == 0, result.stderr

    def test_unknown_attribute_still_raises_attribute_error(self):
        result = _run(
            "import models\n"
            "models.definitely_not_a_real_attribute\n"
        )
        assert result.returncode != 0
        assert "AttributeError" in result.stderr
