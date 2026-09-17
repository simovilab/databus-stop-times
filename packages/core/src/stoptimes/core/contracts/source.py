"""Contract: feed and agency descriptions.

A `Source` names one agency/feed: where its static GTFS lives, which realtime
feeds it publishes, what timezone it runs in, where its archive is kept, and
the parameters a ground-truth detector needs to turn vehicle positions into
actual arrivals. Nothing here is agency-specific — bUCR and MBTA are two
values of this contract, not two code paths.

Back-compat: the val package's source YAMLs (``packages/val/.../config/sources/{mbta,bucr}.yaml``,
today parsed into ``stoptimes.core.domain.models.SourceConfig``) are flat and
predate this contract. They are not rewritten by C1 — the val YAML shape is
adapted onto ``Source`` by ``contracts.adapters.source_from_val_yaml``, not
absorbed as optional fields here. See docs/CONTRACTS.md.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedKind(str, Enum):
    """Which GTFS-RT entity a realtime feed carries."""

    VEHICLE_POSITIONS = "vehicle_positions"
    TRIP_UPDATES = "trip_updates"
    ALERTS = "alerts"


class FeedFormat(str, Enum):
    """Wire format of a realtime feed."""

    PROTOBUF = "pb"
    JSON = "json"


class RealtimeFeed(BaseModel):
    """One realtime endpoint for a source."""

    model_config = ConfigDict(extra="forbid")

    kind: FeedKind
    url: str
    format: FeedFormat = FeedFormat.JSON
    poll_interval_s: float = 15.0
    headers: dict[str, str] = Field(default_factory=dict)
    """Static request headers (e.g. ``{"Accept": "application/x-protobuf"}``)."""

    auth_env: dict[str, str] = Field(default_factory=dict)
    """Header name -> environment variable NAME (never a secret value) whose
    contents are sent as that header at request time. e.g.
    ``{"x-api-key": "MBTA_API_KEY"}``."""


class ArchiveKind(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class Archive(BaseModel):
    """Where recorded realtime data for this source lives."""

    model_config = ConfigDict(extra="forbid")

    kind: ArchiveKind = ArchiveKind.LOCAL
    uri: str
    """A local filesystem path (``kind="local"``) or an ``s3://bucket/prefix`` URI (``kind="s3"``)."""


class GroundTruthDetector(BaseModel):
    """Parameters for deriving actual stop arrivals from vehicle positions.

    Field names and defaults match the parameters already used by
    ``stoptimes.core.domain.models.SourceConfig`` / etaval's ground-truth
    detector, so existing tuned values carry over unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    arrival_radius_m: float = 40.0
    dwell_s: float = 0.0
    shape_threshold_m: float = 30.0


class Source(BaseModel):
    """A GTFS agency/feed: static schedule, realtime feeds, archive, detector.

    General on purpose: any agency is describable with this shape. Agency
    identity (bUCR, MBTA, ...) is a value of `id`/`name`, never a subclass.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str
    timezone: str = "UTC"
    gtfs_static_url: str
    """URL or local filesystem path to the agency's GTFS static .zip."""

    realtime_feeds: list[RealtimeFeed] = Field(default_factory=list)
    archive: Archive | None = None
    detector: GroundTruthDetector = Field(default_factory=GroundTruthDetector)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_most_one_feed_per_kind(self) -> "Source":
        seen: set[FeedKind] = set()
        for feed in self.realtime_feeds:
            if feed.kind in seen:
                raise ValueError(f"duplicate realtime feed kind: {feed.kind}")
            seen.add(feed.kind)
        return self

    def feed(self, kind: FeedKind) -> RealtimeFeed | None:
        """Return the realtime feed of *kind*, or None if this source doesn't publish it."""
        for feed in self.realtime_feeds:
            if feed.kind == kind:
                return feed
        return None


def load_source(path: str | Path) -> Source:
    """Load a `Source` from a YAML file written in this contract's own shape.

    For the legacy flat val-source YAML shape, use
    ``contracts.adapters.source_from_val_yaml`` instead (or on top of this:
    try `load_source` first and fall back to the adapter).
    """
    with Path(path).open() as fh:
        data = yaml.safe_load(fh)
    return Source.model_validate(data)


def dump_source(source: Source, path: str | Path) -> Path:
    """Write *source* to *path* as YAML, round-trippable with `load_source`."""
    target = Path(path)
    target.write_text(
        yaml.safe_dump(source.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return target
