"""Storage protocol — single source of the storage contract.

Re-exports `Storage` from `etaval.domain.interfaces` so that downstream
implementations (e.g. `storage/sqlite.py`) and any future backends import
from one stable, well-known location.
"""

from etaval.domain.interfaces import Storage

__all__ = ["Storage"]
