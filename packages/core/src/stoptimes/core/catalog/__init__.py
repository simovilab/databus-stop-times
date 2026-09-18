"""Local data catalog: a JSON registry of datasets, no database.

See `store.Catalog` (add/get/list/remove/resolve) and `models.DatasetEntry`.
`inspect.build_coverage_report` is the pure function behind `stoptimes data
inspect`.
"""

from __future__ import annotations

from .models import DatasetEntry, DatasetKind
from .store import Catalog, DatasetExistsError, DatasetNotFoundError, catalog_path

__all__ = [
    "Catalog",
    "DatasetEntry",
    "DatasetKind",
    "DatasetExistsError",
    "DatasetNotFoundError",
    "catalog_path",
]
