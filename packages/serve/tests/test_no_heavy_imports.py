"""`import stoptimes.serve` must stay light: no xgboost, no torch, no
onnxruntime, no paho-mqtt. Runs in a subprocess so already-imported modules
in the test process (e.g. from another test file) can't hide a regression.
"""

from __future__ import annotations

import subprocess
import sys


def _import_check(heavy_module: str) -> subprocess.CompletedProcess:
    code = (
        "import sys\n"
        "import stoptimes.serve\n"
        f"assert {heavy_module!r} not in sys.modules, {heavy_module!r} + ' was imported by stoptimes.serve'\n"
        "print('ok')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_importing_serve_does_not_import_xgboost():
    result = _import_check("xgboost")
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_serve_does_not_import_torch():
    result = _import_check("torch")
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_serve_does_not_import_onnxruntime():
    result = _import_check("onnxruntime")
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_serve_pipeline_and_model_loading_stay_light():
    code = (
        "import sys\n"
        "import stoptimes.serve.pipeline\n"
        "import stoptimes.serve.model_loading\n"
        "import stoptimes.serve.tripupdates\n"
        "import stoptimes.serve.sinks\n"
        "for heavy in ('xgboost', 'torch', 'onnxruntime', 'paho'):\n"
        "    assert heavy not in sys.modules, heavy\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
