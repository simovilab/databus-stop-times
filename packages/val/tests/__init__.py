"""etaval test suite.

This must be a real package: importing ``etaval`` transitively puts
``gtfs-eta/models`` on ``sys.path``, whose ``models/tests`` would otherwise
shadow a namespace-style ``tests`` package and break collection on a fresh
checkout. Combined with ``--import-mode=importlib`` (pyproject) this keeps
``uv run pytest`` working.
"""
