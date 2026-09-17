"""
Correct temporal splitting for ETA prediction datasets (roadmap 3.2).

Design goals, in order of importance:

1. **No leakage across time.** Splits are cut on calendar-boundary "service
   days," not arbitrary row fractions, so a model never trains on data that
   is chronologically adjacent to (or interleaved with) what it is evaluated
   on.
2. **No leakage across groups.** A `trip_id` (or any other identity that
   produces highly-correlated rows, e.g. `(trip_id, service_date)`) never
   appears in more than one split. Rows that would violate this are purged
   from the *earlier* split (earlier splits are cheaper to shrink than the
   held-out ones, which need to stay representative of "the future").
3. **Determinism.** Given the same input (regardless of row order) and the
   same parameters, the output is byte-identical. This matters because
   metadata/reports get compared across runs and because shuffled input
   (e.g. from a groupby or concurrent write) must not change the split.
4. **No torch/xgboost import.** This module is imported by every trainer;
   it must stay a thin, fast, pandas/numpy-only dependency.

This module replaces the ad-hoc `ETADataset.temporal_split` fraction split
and the informally re-implemented `day_grouped_split` that had accumulated
in scratchpad scripts (see `scratchpad/hpo_xgboost.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import pandas as pd


class SplitError(ValueError):
    """Raised when a split cannot be produced safely (empty input, no usable
    days, degenerate fractions, etc.). Distinct from ValueError so callers
    can catch split-specific failures without swallowing unrelated bugs."""


# --------------------------------------------------------------------------
# Service-day key
# --------------------------------------------------------------------------

def service_day(ts: pd.Series, offset_hours: int) -> pd.Series:
    """Return the local "service day" for each timestamp.

    Transit schedules don't reset at midnight: a trip that starts at 23:50
    and ends at 00:20 is still "yesterday's last trip," and a rider looking
    at the schedule for a given day expects to see it. GTFS itself encodes
    this by allowing `arrival_time` > 24:00:00 within a single service_date.

    We approximate this without needing the GTFS calendar by shifting every
    timestamp back by `offset_hours` before taking its date. Any hour before
    the offset (e.g. 00:00-02:59 for offset_hours=3) is folded into the
    *previous* calendar day, matching the operational "service day" transit
    agencies actually schedule against.

    Concretely, with offset_hours=7 (MBTA, ~3am ET local-to-UTC-ish margin
    used elsewhere in this codebase) a vp_ts of 00:15 becomes part of the
    previous day's service_day, not a lone one-row "day" of its own -- which
    is exactly the case the day-grouped split must get right (see
    test_splits.py::test_service_day_offset_moves_late_night_row).

    Args:
        ts: a datetime64 Series (must already be tz-consistent; this
            function does no tz handling of its own).
        offset_hours: hours to subtract before taking the date. MBTA uses 7
            (this repo's convention for MBTA's ET-ish service day cutoff),
            bUCR uses 6.

    Returns:
        A Series of `datetime.date` aligned to `ts.index`.
    """
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SplitError(f"service_day() requires a datetime64 Series, got dtype={ts.dtype}")
    return (ts - pd.Timedelta(hours=offset_hours)).dt.date


# --------------------------------------------------------------------------
# SplitResult
# --------------------------------------------------------------------------

@dataclass
class SplitResult:
    """The three splits plus a report dict suitable for stamping into model
    metadata (provenance)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    report: Dict = field(default_factory=dict)

    def as_tuple(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Back-compat shape: (train_df, val_df, test_df)."""
        return self.train, self.val, self.test


def _empty_check(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        raise SplitError(
            f"{name} split is empty after splitting -- widen the input, "
            f"lower min_rows_per_day, or reduce test_days/val_days/embargo_days."
        )


# --------------------------------------------------------------------------
# Group purge
# --------------------------------------------------------------------------

def _purge_group_overlap(
    splits: List[pd.DataFrame],
    names: List[str],
    group_cols: Sequence[str],
) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """Ensure no group (tuple of group_cols) appears in more than one split.

    Splits are given oldest-first (e.g. [train, val, test]). For any group
    that appears in more than one split, we keep it only in the *latest*
    split it appears in and drop the offending rows from every earlier
    split. This favors keeping the held-out (later, "future") splits intact
    since they define what "generalization" is measured against.

    Returns the purged splits (same order) and a dict of rows dropped per
    split name.
    """
    missing = [c for c in group_cols if not all(c in df.columns for df in splits if len(df))]
    if missing:
        raise SplitError(f"group_cols {missing} not present in all non-empty splits")

    # Map each group key -> set of split indices it appears in.
    group_to_splits: Dict[tuple, set] = {}
    for i, df in enumerate(splits):
        if df.empty:
            continue
        keys = df[list(group_cols)].drop_duplicates()
        for row in keys.itertuples(index=False, name=None):
            group_to_splits.setdefault(row, set()).add(i)

    overlapping = {g: idxs for g, idxs in group_to_splits.items() if len(idxs) > 1}

    dropped = {name: 0 for name in names}
    if not overlapping:
        return splits, dropped

    purged = list(splits)
    for i, df in enumerate(splits):
        if df.empty:
            continue
        # For this split, drop rows whose group overlaps AND a later split
        # also holds that group (keep in the latest split it appears in).
        keys_in_this_split = set(
            df[list(group_cols)].drop_duplicates().itertuples(index=False, name=None)
        )
        to_drop = {
            g for g in keys_in_this_split
            if g in overlapping and max(overlapping[g]) > i
        }
        if not to_drop:
            continue
        mask = df[list(group_cols)].apply(tuple, axis=1).isin(to_drop)
        dropped[names[i]] += int(mask.sum())
        purged[i] = df.loc[~mask].copy()

    return purged, dropped


def assert_no_group_overlap(split_result: SplitResult, group_cols: Sequence[str]) -> None:
    """Raise SplitError if any group in group_cols appears in more than one
    of train/val/test."""
    frames = {"train": split_result.train, "val": split_result.val, "test": split_result.test}
    seen: Dict[tuple, str] = {}
    for name, df in frames.items():
        if df.empty:
            continue
        for row in df[list(group_cols)].drop_duplicates().itertuples(index=False, name=None):
            if row in seen and seen[row] != name:
                raise SplitError(
                    f"group overlap on {group_cols}={row}: appears in both "
                    f"'{seen[row]}' and '{name}' splits"
                )
            seen[row] = name


def verify(split_result: SplitResult, group_cols: Sequence[str] = ("trip_id",),
           ts_col: str = "vp_ts") -> Dict[str, object]:
    """Run a battery of sanity checks on a SplitResult and return a dict of
    results (rather than raising), so callers can log/report a full picture.

    Checks:
      - no split is empty
      - no group overlap across splits (per group_cols)
      - chronological ordering: max(train.ts) <= min(val.ts) and
        max(val.ts) <= min(test.ts) (best-effort; embargoed gaps make this
        strict rather than approximate)
    """
    checks: Dict[str, object] = {}

    checks["train_nonempty"] = not split_result.train.empty
    checks["val_nonempty"] = not split_result.val.empty
    checks["test_nonempty"] = not split_result.test.empty

    try:
        assert_no_group_overlap(split_result, group_cols)
        checks["no_group_overlap"] = True
    except SplitError as exc:
        checks["no_group_overlap"] = False
        checks["group_overlap_error"] = str(exc)

    ordering_ok = True
    if not split_result.train.empty and not split_result.val.empty:
        ordering_ok &= bool(split_result.train[ts_col].max() <= split_result.val[ts_col].min())
    if not split_result.val.empty and not split_result.test.empty:
        ordering_ok &= bool(split_result.val[ts_col].max() <= split_result.test[ts_col].min())
    checks["chronological_order"] = ordering_ok

    return checks


# --------------------------------------------------------------------------
# day_grouped_split
# --------------------------------------------------------------------------

def day_grouped_split(
    df: pd.DataFrame,
    *,
    ts_col: str = "vp_ts",
    offset_hours: int,
    test_days: int,
    val_days: int,
    min_rows_per_day: int = 0,
    embargo_days: int = 0,
    purge: bool = True,
    group_cols: Sequence[str] = ("trip_id",),
) -> SplitResult:
    """Calendar-boundary temporal split on whole service days.

    Days are sorted ascending. test = the last `test_days` usable days;
    val = the `val_days` immediately before that; train = every usable day
    before that, minus any embargoed days.

    Args:
        df: input DataFrame.
        ts_col: timestamp column (must be datetime64).
        offset_hours: passed to `service_day()`.
        test_days: number of trailing usable days assigned to test.
        val_days: number of usable days assigned to val, immediately
            preceding the test days.
        min_rows_per_day: days with fewer than this many rows are dropped
            entirely before assignment (too sparse to be a usable "day" --
            e.g. a partial first/last day of data collection).
        embargo_days: number of usable days dropped from train immediately
            preceding val, and from val immediately preceding test. This
            creates a temporal gap so no training row sits adjacent in time
            to an evaluated row (guards against short-range autocorrelation
            leaking across the boundary).
        purge: if True, additionally guarantee no group in group_cols spans
            more than one split (see `_purge_group_overlap`).
        group_cols: columns identifying a "group" that must not be split
            across train/val/test, e.g. ("trip_id",) or
            ("trip_id", "service_date").

    Returns:
        SplitResult with train/val/test DataFrames and a report dict.
    """
    if df.empty:
        raise SplitError("day_grouped_split() received an empty DataFrame")
    if test_days <= 0 or val_days <= 0:
        raise SplitError("test_days and val_days must both be positive")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise SplitError(f"{ts_col} must be datetime64; got {df[ts_col].dtype}")

    work = df.copy()
    work["_service_day"] = service_day(work[ts_col], offset_hours)

    counts = work.groupby("_service_day").size()
    usable_days = sorted(counts[counts >= min_rows_per_day].index.tolist())
    dropped_sparse_days = sorted(set(counts.index) - set(usable_days))

    needed = test_days + val_days
    if len(usable_days) <= needed:
        raise SplitError(
            f"only {len(usable_days)} usable day(s) (min_rows_per_day={min_rows_per_day}) "
            f"but test_days={test_days} + val_days={val_days} = {needed} requires more; "
            f"train would be empty. Lower min_rows_per_day or test_days/val_days."
        )

    test_day_list = usable_days[-test_days:]
    val_day_list = usable_days[-(test_days + val_days):-test_days]
    train_day_list = usable_days[: -(test_days + val_days)]

    # Embargo: drop the embargo_days usable days immediately preceding val
    # (out of train) and immediately preceding test (out of val).
    embargoed_from_train: List = []
    embargoed_from_val: List = []
    if embargo_days > 0:
        if embargo_days >= len(train_day_list):
            raise SplitError(
                f"embargo_days={embargo_days} >= available train days "
                f"({len(train_day_list)}); train would be emptied by embargo alone"
            )
        embargoed_from_train = train_day_list[-embargo_days:]
        train_day_list = train_day_list[:-embargo_days]

        if embargo_days >= len(val_day_list):
            raise SplitError(
                f"embargo_days={embargo_days} >= available val days "
                f"({len(val_day_list)}); val would be emptied by embargo alone"
            )
        embargoed_from_val = val_day_list[-embargo_days:]
        val_day_list = val_day_list[:-embargo_days]

    train_df = work[work["_service_day"].isin(train_day_list)].drop(columns=["_service_day"])
    val_df = work[work["_service_day"].isin(val_day_list)].drop(columns=["_service_day"])
    test_df = work[work["_service_day"].isin(test_day_list)].drop(columns=["_service_day"])

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    rows_purged = {"train": 0, "val": 0, "test": 0}
    if purge and any(c in df.columns for c in group_cols):
        (train_df, val_df, test_df), rows_purged = _purge_group_overlap(
            [train_df, val_df, test_df], ["train", "val", "test"], group_cols
        )

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        _empty_check(name, split_df)

    def _day_bounds(days: List) -> Tuple[object, object]:
        return (days[0], days[-1]) if days else (None, None)

    report = {
        "method": "day_grouped_split",
        "params": {
            "ts_col": ts_col,
            "offset_hours": offset_hours,
            "test_days": test_days,
            "val_days": val_days,
            "min_rows_per_day": min_rows_per_day,
            "embargo_days": embargo_days,
            "purge": purge,
            "group_cols": list(group_cols),
        },
        "day_boundaries": {
            "train": _day_bounds(train_day_list),
            "val": _day_bounds(val_day_list),
            "test": _day_bounds(test_day_list),
        },
        "row_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "group_counts": {
            split_name: (
                int(split[list(group_cols)].drop_duplicates().shape[0])
                if all(c in split.columns for c in group_cols) and not split.empty
                else 0
            )
            for split_name, split in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "days_dropped_sparse": len(dropped_sparse_days),
        "days_embargoed": {
            "train": len(embargoed_from_train),
            "val": len(embargoed_from_val),
        },
        "rows_purged": rows_purged,
    }

    return SplitResult(train=train_df, val=val_df, test=test_df, report=report)


# --------------------------------------------------------------------------
# fraction_split (drop-in replacement for the old temporal_split)
# --------------------------------------------------------------------------

def fraction_split(
    df: pd.DataFrame,
    *,
    ts_col: str = "vp_ts",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    group_cols: Sequence[str] = ("trip_id",),
    purge: bool = True,
    row_id_col: str = None,
) -> SplitResult:
    """Fraction-based temporal split, done correctly.

    This replaces the old `ETADataset.temporal_split`: rows are sorted by
    time with a stable, deterministic secondary key so that (a) ties on
    `ts_col` don't depend on input row order, and (b) the same input always
    produces the same split regardless of how it was assembled (concat
    order, groupby order, etc). The cut points are row-fractions of the
    sorted data; a group purge is then applied so groups don't straddle the
    train/val/test boundary.

    Args:
        df: input DataFrame.
        ts_col: timestamp column (must be datetime64).
        train_frac: fraction of (sorted) rows assigned to train.
        val_frac: fraction assigned to val; the remainder goes to test.
            Requires train_frac + val_frac < 1 (test must be non-empty).
        group_cols: columns identifying a group that must not straddle a
            split boundary (see `day_grouped_split`).
        purge: if True, apply the group purge after cutting.
        row_id_col: optional column to use as the final tiebreaker in the
            sort key (must be unique per row). If not given, a temporary
            deterministic key derived from the DataFrame's own content hash
            is used, so re-ordering the *same* rows still yields the same
            sort order (true row-order independence), while relying on
            pandas' stable sort ("mergesort") for any residual ties.

    Returns:
        SplitResult with train/val/test DataFrames and a report dict.
    """
    if df.empty:
        raise SplitError("fraction_split() received an empty DataFrame")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise SplitError("train_frac and val_frac must each be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise SplitError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) >= 1 -- test split would be empty"
        )
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise SplitError(f"{ts_col} must be datetime64; got {df[ts_col].dtype}")

    work = df.copy()

    if row_id_col is not None:
        if row_id_col not in work.columns:
            raise SplitError(f"row_id_col={row_id_col!r} not found in DataFrame columns")
        sort_key_col = row_id_col
    else:
        # Deterministic, content-derived tiebreaker: independent of input
        # row order, so shuffling the input yields byte-identical splits.
        # pandas.util.hash_pandas_object hashes each row's values; ties on
        # (ts_col, *group_cols, content_hash) are astronomically unlikely
        # for real data, and a final stable mergesort resolves any that
        # remain in a row-order-independent way (identical rows really are
        # interchangeable).
        sort_key_col = "__row_hash__"
        cols_for_hash = [c for c in work.columns if c != sort_key_col]
        work[sort_key_col] = pd.util.hash_pandas_object(work[cols_for_hash], index=False)

    sort_cols = [ts_col, *[c for c in group_cols if c in work.columns], sort_key_col]
    work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if row_id_col is None:
        work = work.drop(columns=[sort_key_col])

    n = len(work)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = work.iloc[:train_end].copy()
    val_df = work.iloc[train_end:val_end].copy()
    test_df = work.iloc[val_end:].copy()

    rows_purged = {"train": 0, "val": 0, "test": 0}
    if purge and any(c in df.columns for c in group_cols):
        (train_df, val_df, test_df), rows_purged = _purge_group_overlap(
            [train_df, val_df, test_df], ["train", "val", "test"], group_cols
        )

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        _empty_check(name, split_df)

    report = {
        "method": "fraction_split",
        "params": {
            "ts_col": ts_col,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "group_cols": list(group_cols),
            "purge": purge,
        },
        "row_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "group_counts": {
            split_name: (
                int(split[list(group_cols)].drop_duplicates().shape[0])
                if all(c in split.columns for c in group_cols) and not split.empty
                else 0
            )
            for split_name, split in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "rows_purged": rows_purged,
    }

    return SplitResult(train=train_df, val=val_df, test=test_df, report=report)
