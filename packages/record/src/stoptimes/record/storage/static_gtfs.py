"""Weekly dated snapshots of each agency's static GTFS feed.

Static GTFS (stops/routes/trips/shapes) changes only occasionally, but every
realtime observation collected during the 90-day replication window needs to
be matchable back to "what schedule was in effect" at the time it was
recorded. Each snapshot is the upstream zip stored verbatim (no parsing) at
``feeds/<agency>/gtfs_static/<ISO date>.zip`` -- see docs/S3_LAYOUT.md and
roadmap 0.2.

Uploads go through `mc pipe` rather than DuckDB's httpfs (used elsewhere in
this package for Parquet) because this is one opaque binary blob, not a
columnar dataset. Credential resolution is delegated to
``rt_pipeline.compaction.credentials`` rather than reimplemented here.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import zipfile
from io import BytesIO

import requests

# Present in every real GTFS feed regardless of agency/protocol; their
# absence means the URL returned something other than a GTFS zip (an error
# page, an empty response, a redirect to a login wall) that should not be
# uploaded under a dated key that later steps would trust.
REQUIRED_MEMBERS = ("stops.txt", "routes.txt")


class StaticGtfsError(RuntimeError):
    """The upstream feed did not return a usable GTFS zip, or upload failed."""


def fetch(url: str, *, timeout: int = 60) -> bytes:
    """Download and sanity-check a static GTFS zip.

    Raises `StaticGtfsError` on anything that isn't a real GTFS feed rather
    than silently uploading garbage under a dated key.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    try:
        zf = zipfile.ZipFile(BytesIO(content))
        bad_member = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise StaticGtfsError(f"{url} did not return a valid zip") from exc
    if bad_member is not None:
        raise StaticGtfsError(f"{url}: corrupt member {bad_member!r} in zip")
    missing = [m for m in REQUIRED_MEMBERS if m not in zf.namelist()]
    if missing:
        raise StaticGtfsError(f"{url}: zip is missing required GTFS files: {missing}")
    return content


def list_snapshots(
    agency: str,
    *,
    alias: str = "simovilab",
    bucket: str = "transit",
) -> list[dt.date]:
    """List the available snapshot dates for `agency`, sorted ascending.

    Returns `[]` if the prefix does not exist or has no snapshots yet.
    """
    from ..compaction.credentials import load_credentials

    load_credentials(alias)
    prefix = f"feeds/{agency}/gtfs_static/"
    target = f"{alias}/{bucket}/{prefix}"
    proc = subprocess.run(["mc", "ls", target], capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        stderr_lower = stderr.lower()
        if (
            "not found" in stderr_lower
            or "no such" in stderr_lower
            or "does not exist" in stderr_lower
        ):
            return []
        raise StaticGtfsError(f"mc ls failed for {target}: {stderr}")

    dates: list[dt.date] = []
    for line in proc.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.split()[-1]
        if not name.endswith(".zip"):
            continue
        try:
            dates.append(dt.date.fromisoformat(name[: -len(".zip")]))
        except ValueError:
            continue
    return sorted(dates)


def get_snapshot(
    agency: str,
    snapshot_date: dt.date,
    *,
    alias: str = "simovilab",
    bucket: str = "transit",
) -> bytes:
    """Fetch the bytes of one dated snapshot.

    Raises `StaticGtfsError` if that date's object does not exist in S3.
    """
    from ..compaction.credentials import load_credentials

    load_credentials(alias)
    key = f"feeds/{agency}/gtfs_static/{snapshot_date.isoformat()}.zip"
    target = f"{alias}/{bucket}/{key}"
    proc = subprocess.run(["mc", "cat", target], capture_output=True)
    if proc.returncode != 0:
        raise StaticGtfsError(
            f"mc cat failed for {target}: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout


def latest_snapshot_on_or_before(
    agency: str,
    target_date: dt.date,
    *,
    alias: str = "simovilab",
    bucket: str = "transit",
) -> dt.date | None:
    """The newest available snapshot date `<= target_date`, matching a
    realtime observation back to the schedule version in effect then.

    Returns `None` if no snapshot exists on or before `target_date` (e.g. a
    vehicle position recorded before the first snapshot was taken).
    """
    candidates = [
        d
        for d in list_snapshots(agency, alias=alias, bucket=bucket)
        if d <= target_date
    ]
    return max(candidates) if candidates else None


def put_snapshot(
    content: bytes,
    agency: str,
    snapshot_date: dt.date,
    *,
    alias: str = "simovilab",
    bucket: str = "transit",
) -> str:
    """Upload `content` to feeds/<agency>/gtfs_static/<ISO date>.zip.

    Returns the bucket-relative key. Resolves `mc` credentials the same way
    `rt_pipeline.compaction.run_compaction` does (an `mc` alias, else
    environment variables).
    """
    from ..compaction.credentials import load_credentials

    load_credentials(alias)
    key = f"feeds/{agency}/gtfs_static/{snapshot_date.isoformat()}.zip"
    target = f"{alias}/{bucket}/{key}"
    proc = subprocess.run(["mc", "pipe", target], input=content, capture_output=True)
    if proc.returncode != 0:
        raise StaticGtfsError(
            f"mc pipe failed for {target}: {proc.stderr.decode(errors='replace').strip()}"
        )
    return key
