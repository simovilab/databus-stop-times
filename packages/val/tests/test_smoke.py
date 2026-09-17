"""Smoke test: the package imports and exposes a version."""

import stoptimes.val


def test_version():
    assert stoptimes.val.__version__
