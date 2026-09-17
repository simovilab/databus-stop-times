"""Smoke test: the package imports and exposes a version."""

import etaval


def test_version():
    assert etaval.__version__
