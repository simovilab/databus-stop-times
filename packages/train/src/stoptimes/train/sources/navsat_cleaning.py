"""Pure row-filtering cleaner over raw BUCR navsat rows (roadmap 2.5).

This module operates on the RAW navsat frame -- i.e. *before*
``navsat_adapter.navsat_to_vehicle_positions`` runs -- because the staleness
check needs ``ingested_at_utc``, a column the adapter deliberately does not
emit (see ``navsat_adapter.py`` module docstring). Pipeline order is:

    clean_navsat_trace(raw) -> navsat_to_vehicle_positions(cleaned)

Only row filtering happens here: stale device fixes and null/zero/
out-of-range coordinates are dropped. NO smoothing, NO interpolation, NO trip
logic -- cross-track/anomaly rejection is a later step (trip inference,
roadmap Step 3), not this one's job.

Pure and deterministic: no I/O, no network, no DB, no mutation of the input
frame. Same input always produces the same output frame and the same report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

# Raw BUCR navsat columns this cleaner requires. Mirrors
# ``navsat_adapter.REQUIRED_RAW_COLUMNS`` but adds ``ingested_at_utc``, which
# the adapter does not need but staleness detection does.
REQUIRED_RAW_COLUMNS: List[str] = [
    "cr_datetime_utc",
    "ingested_at_utc",
    "lat",
    "lon",
]

# Default staleness threshold: a device fix is dropped if
# |cr_datetime_utc - ingested_at_utc| exceeds this many seconds.
#
# Reasoning: BUCR devices buffer fixes locally and flush them in batches when
# connectivity returns, so some upload lag is normal and NOT a quality
# problem -- only fixes that arrive long after they were recorded indicate
# the device was offline (tunnel, dead zone, power cycle) long enough that
# the fix is stale relative to the vehicle's current position. 15 minutes
# (900s) is chosen as a defensible, generous cutoff: short enough to catch
# fixes that are clearly unusable for real-time-shaped trip inference (a bus
# on a ~20-40 min BUCR route can traverse most of its route in 15 minutes),
# long enough to tolerate normal store-and-forward upload jitter over
# cellular data in Costa Rica without discarding good fixes. Recorded here
# so it is citable in the Step-4 quality report.
DEFAULT_STALENESS_THRESHOLD_SECONDS: float = 900.0

# Plausible geographic bounds. Anything outside is a broken/garbled fix,
# never a real position.
_LAT_MAX_ABS = 90.0
_LON_MAX_ABS = 180.0


@dataclass(frozen=True)
class TraceCleaningReport:
    """Immutable, deterministic summary of one ``clean_navsat_trace`` call.

    Per-reason counts follow the precedence order documented on
    ``clean_navsat_trace``: a row droppable for multiple reasons is counted
    under only the first reason that applies, so
    ``dropped_total == sum(per-reason counts)`` and
    ``rows_in == rows_kept + dropped_total`` always hold.
    """

    rows_in: int
    dropped_bad_timestamp: int
    dropped_stale: int
    dropped_null_coord: int
    dropped_zero_coord: int
    dropped_out_of_range: int
    rows_kept: int
    staleness_threshold_seconds: float

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_bad_timestamp
            + self.dropped_stale
            + self.dropped_null_coord
            + self.dropped_zero_coord
            + self.dropped_out_of_range
        )

    @property
    def drop_rate(self) -> float:
        """Fraction of input rows dropped, in [0.0, 1.0]. 0.0 for an empty input."""
        if self.rows_in == 0:
            return 0.0
        return self.dropped_total / self.rows_in


def _missing_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]


def _empty_report(threshold: float) -> TraceCleaningReport:
    return TraceCleaningReport(
        rows_in=0,
        dropped_bad_timestamp=0,
        dropped_stale=0,
        dropped_null_coord=0,
        dropped_zero_coord=0,
        dropped_out_of_range=0,
        rows_kept=0,
        staleness_threshold_seconds=threshold,
    )


def clean_navsat_trace(
    df: pd.DataFrame,
    staleness_threshold_seconds: float = DEFAULT_STALENESS_THRESHOLD_SECONDS,
) -> "tuple[pd.DataFrame, TraceCleaningReport]":
    """Drop stale-fix and bad-coordinate rows from a raw BUCR navsat frame.

    Precedence order (a row droppable for multiple reasons is counted under
    only the FIRST reason below that applies -- this makes per-reason counts
    sum exactly to ``dropped_total`` with no double counting):

        1. bad/unparseable timestamp (``cr_datetime_utc`` or
           ``ingested_at_utc`` is null or fails to parse as a datetime) --
           checked first because staleness cannot be computed without both
           timestamps.
        2. stale fix (``|cr_datetime_utc - ingested_at_utc|`` exceeds
           ``staleness_threshold_seconds``) -- only evaluated once both
           timestamps are known-good.
        3. null coordinate (``lat`` or ``lon`` is null / non-numeric).
        4. zero coordinate (exact ``(0.0, 0.0)`` -- "null island", a classic
           bad GPS fix).
        5. out-of-range coordinate (``|lat| > 90`` or ``|lon| > 180``).

    No smoothing, no interpolation, no trip logic -- this is row filtering
    only. Cross-track/anomaly rejection is a later step (trip inference).

    Args:
        df: Raw navsat rows with (at least) ``REQUIRED_RAW_COLUMNS``. Row
            order is preserved among survivors.
        staleness_threshold_seconds: Named, documented threshold -- see
            ``DEFAULT_STALENESS_THRESHOLD_SECONDS`` for the default and the
            reasoning behind it. Must be non-negative.

    Returns:
        A ``(cleaned_df, report)`` tuple. ``cleaned_df`` is a new DataFrame
        (the input is never mutated) containing only surviving rows, with
        original columns and dtypes preserved and original row order kept.
        It is a valid input to ``navsat_adapter.navsat_to_vehicle_positions``
        whenever ``df`` was. ``report`` is a ``TraceCleaningReport``.

    Raises:
        TypeError: if ``df`` is not a ``pandas.DataFrame``.
        ValueError: if any of ``REQUIRED_RAW_COLUMNS`` is missing, or if
            ``staleness_threshold_seconds`` is negative.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"clean_navsat_trace expects a DataFrame, got {type(df)!r}")

    if staleness_threshold_seconds < 0:
        raise ValueError(
            f"staleness_threshold_seconds must be non-negative, got {staleness_threshold_seconds!r}"
        )

    missing = _missing_columns(df)
    if missing:
        raise ValueError(f"navsat frame missing required columns: {missing}")

    if df.empty:
        return df.copy(deep=True), _empty_report(staleness_threshold_seconds)

    n = len(df)

    cr_dt = pd.to_datetime(df["cr_datetime_utc"], utc=True, errors="coerce")
    ing_dt = pd.to_datetime(df["ingested_at_utc"], utc=True, errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")

    # Reason 1: bad/unparseable timestamp on either side.
    bad_timestamp = cr_dt.isna() | ing_dt.isna()

    # Reason 2: stale fix (only meaningful where both timestamps parsed).
    age_seconds = (cr_dt - ing_dt).abs().dt.total_seconds()
    stale = (~bad_timestamp) & (age_seconds > staleness_threshold_seconds)

    # Reason 3: null coordinate.
    null_coord = (~bad_timestamp) & (~stale) & (lat.isna() | lon.isna())

    # Reason 4: exact null-island zero coordinate.
    known_coord = ~(lat.isna() | lon.isna())
    zero_coord = (
        (~bad_timestamp) & (~stale) & (~null_coord) & known_coord & (lat == 0.0) & (lon == 0.0)
    )

    # Reason 5: out-of-plausible-range coordinate.
    out_of_range = (
        (~bad_timestamp)
        & (~stale)
        & (~null_coord)
        & (~zero_coord)
        & known_coord
        & ((lat.abs() > _LAT_MAX_ABS) | (lon.abs() > _LON_MAX_ABS))
    )

    drop_mask = bad_timestamp | stale | null_coord | zero_coord | out_of_range
    keep_mask = ~drop_mask

    cleaned = df.loc[keep_mask].copy(deep=True).reset_index(drop=True)

    report = TraceCleaningReport(
        rows_in=n,
        dropped_bad_timestamp=int(bad_timestamp.sum()),
        dropped_stale=int(stale.sum()),
        dropped_null_coord=int(null_coord.sum()),
        dropped_zero_coord=int(zero_coord.sum()),
        dropped_out_of_range=int(out_of_range.sum()),
        rows_kept=int(keep_mask.sum()),
        staleness_threshold_seconds=staleness_threshold_seconds,
    )

    return cleaned, report
