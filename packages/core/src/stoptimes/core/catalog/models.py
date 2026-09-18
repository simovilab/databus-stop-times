"""A dataset entry: one row of the local data catalog.

No database — `stoptimes.core.catalog.store.Catalog` persists a dict of
these as JSON under `$STOPTIMES_HOME` (see `core.config`). Kept intentionally
generic: `kind` covers everything T1 (observations, static) and T2
(arrivals) register, so the CLI and the catalog format don't grow a new
shape per track.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetKind(str, Enum):
    """What a catalog entry's `location` holds."""

    OBSERVATIONS = "observations"
    ARRIVALS = "arrivals"
    STATIC = "static"
    FRAME = "frame"
    """A prepared training frame: observations x upcoming stops, features + target."""


class DatasetEntry(BaseModel):
    """One registered dataset.

    `content_hash` is filled only when cheap to compute (e.g. a single local
    file); it is `None` for a live URL or a large archive directory rather
    than force a full scan.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source_id: str | None = None
    kind: DatasetKind
    location: str
    date_start: str | None = None
    date_end: str | None = None
    row_count: int | None = None
    added_at: datetime
    content_hash: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    """Reader-specific bookkeeping the CLI needs to re-read this dataset
    later — e.g. `{"format": "avl_csv", "mapping": {...}}` for an AVL CSV
    dataset, or `{"tables": [...], "missing_required": [...]}` for a static
    one."""
