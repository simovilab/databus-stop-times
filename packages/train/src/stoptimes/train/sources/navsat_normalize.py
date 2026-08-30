"""Pure front-step: normalize raw BUCR navsat column names/timestamps.

The real BUCR navsat export (``s3://transit/feeds/bucr/navsat/...``) does not
match the column contract ``navsat_cleaning``/``navsat_adapter`` were built
against: it names coordinates ``latitude``/``longitude`` (not ``lat``/``lon``),
and its ``cr_datetime_utc`` column is either absent entirely or partially
null, with only ``cr_datetime`` (a naive local Costa Rica time string)
reliably populated. This module is the single place that reconciles that --
everything downstream (``navsat_cleaning.clean_navsat_trace``,
``navsat_adapter.navsat_to_vehicle_positions``) keeps assuming the
canonical ``lat``/``lon``/``cr_datetime_utc`` names, unchanged.

Pipeline order becomes:

    normalize_raw_navsat(raw) -> clean_navsat_trace(normalized)
        -> navsat_to_vehicle_positions(cleaned) -> infer_bucr_trips(vp, candidates)

Pure and deterministic: no I/O, no network, no mutation of the input frame.
Mirrors the error-handling style of ``navsat_adapter.py``/``navsat_cleaning.py``.
"""

from __future__ import annotations

from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

# Costa Rica has used a fixed UTC-6 offset with no DST since 1998, so
# interpreting a naive ``cr_datetime`` string in this zone and converting to
# UTC is an exact conversion, not an approximation -- using ZoneInfo instead
# of a hardcoded ``+ 6 hours`` keeps that fact self-documenting and correct
# even if that ever changes.
COSTA_RICA_TZ = ZoneInfo("America/Costa_Rica")

# A normalizable frame must have enough information to produce lat/lon and
# some usable timestamp -- either name of the coordinate columns is
# accepted (see _resolve_coord_columns), but at least one of
# cr_datetime_utc / cr_datetime must be present.
_COORD_ALIASES = {"lat": ["lat", "latitude"], "lon": ["lon", "longitude"]}


class NavsatSchemaError(ValueError):
    """The input frame lacks any usable coordinate or timestamp column."""


def _resolve_coord_columns(df: pd.DataFrame) -> "tuple[str, str]":
    """Pick which columns supply lat/lon, preferring already-canonical names.

    Raises:
        NavsatSchemaError: neither a canonical nor real-schema name is present
            for latitude, or for longitude.
    """
    resolved: List[str] = []
    for canonical, aliases in _COORD_ALIASES.items():
        present = [a for a in aliases if a in df.columns]
        if not present:
            raise NavsatSchemaError(
                f"navsat frame has none of the expected columns for {canonical!r}: {aliases}"
            )
        # Prefer the canonical name itself if both happen to be present.
        resolved.append(canonical if canonical in present else present[0])
    return resolved[0], resolved[1]


def _derive_cr_datetime_utc(df: pd.DataFrame) -> pd.Series:
    """Derive UTC timestamps from naive local ``cr_datetime`` strings.

    Rows where ``cr_datetime`` is null/unparseable produce ``NaT`` -- left
    for ``clean_navsat_trace``'s bad-timestamp rule to drop, not this
    function's job to raise on.
    """
    local_naive = pd.to_datetime(df["cr_datetime"], errors="coerce")
    # tz_localize interprets the naive values AS being in COSTA_RICA_TZ, then
    # tz_convert re-expresses the same instant in UTC -- this is a genuine
    # timezone conversion (not a relabeling), exact because CR has no DST.
    localized = local_naive.dt.tz_localize(COSTA_RICA_TZ, nonexistent="NaT", ambiguous="NaT")
    return localized.dt.tz_convert("UTC")


def normalize_raw_navsat(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile real-schema BUCR navsat columns to the canonical contract.

    Steps (see module docstring for full rationale):
      1. Renames ``latitude``/``longitude`` -> ``lat``/``lon`` if present;
         idempotent -- a frame that already has ``lat``/``lon`` passes
         through unchanged on this step.
      2. Ensures ``cr_datetime_utc``: keeps existing non-null values, and
         derives the rest from ``cr_datetime`` (interpreted in
         ``America/Costa_Rica``, converted to UTC). Rows with a null/
         unparseable ``cr_datetime`` get a null ``cr_datetime_utc`` --
         cleaning drops those via its bad-timestamp rule, this function
         never raises on them.
      3. All other columns (``ingested_at_utc``, ``provided_by``,
         ``odometer_km``, ``estado``, ``speed_kmh``, ``plate_number``, ...)
         pass through untouched.

    Args:
        df: Raw navsat rows. Must have a resolvable lat/lon column pair
            (``lat``/``lon`` or ``latitude``/``longitude``) and at least one
            of ``cr_datetime_utc``/``cr_datetime``.

    Returns:
        A new DataFrame (the input is never mutated) with canonical
        ``lat``, ``lon``, ``cr_datetime_utc`` columns, ``cr_datetime_utc``
        tz-aware UTC. Row order and all other columns are preserved.

    Raises:
        TypeError: if ``df`` is not a ``pandas.DataFrame``.
        NavsatSchemaError: no resolvable lat/lon columns, or neither
            ``cr_datetime_utc`` nor ``cr_datetime`` is present at all.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"normalize_raw_navsat expects a DataFrame, got {type(df)!r}")

    if df.empty:
        # Still validate column presence so callers get a clear error even
        # on an empty frame with the wrong schema, mirroring the sibling
        # modules' empty-input handling (see their _missing_columns checks).
        _resolve_coord_columns(df)
        if "cr_datetime_utc" not in df.columns and "cr_datetime" not in df.columns:
            raise NavsatSchemaError(
                "navsat frame has neither 'cr_datetime_utc' nor 'cr_datetime'"
            )
        out = df.copy(deep=True)
        lat_col, lon_col = _resolve_coord_columns(df)
        if lat_col != "lat":
            out["lat"] = out.pop(lat_col)
        if lon_col != "lon":
            out["lon"] = out.pop(lon_col)
        if "cr_datetime_utc" not in out.columns:
            out["cr_datetime_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        return out

    lat_col, lon_col = _resolve_coord_columns(df)

    if "cr_datetime_utc" not in df.columns and "cr_datetime" not in df.columns:
        raise NavsatSchemaError(
            "navsat frame has neither 'cr_datetime_utc' nor 'cr_datetime'"
        )

    out = df.copy(deep=True)

    if lat_col != "lat":
        out["lat"] = out.pop(lat_col)
    if lon_col != "lon":
        out["lon"] = out.pop(lon_col)

    existing_utc = (
        pd.to_datetime(out["cr_datetime_utc"], utc=True, errors="coerce")
        if "cr_datetime_utc" in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )

    if "cr_datetime" in out.columns:
        derived_utc = _derive_cr_datetime_utc(out)
        # Prefer existing non-null UTC values; fall back to the derived
        # value only where the existing column is null/absent.
        out["cr_datetime_utc"] = existing_utc.where(existing_utc.notna(), derived_utc)
    else:
        out["cr_datetime_utc"] = existing_utc

    return out
