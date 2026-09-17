"""Parquet export/import of validation_results. Phase P6 (pandas + pyarrow).

Public API
----------
- export_results(results, path) -> None
- load_results_parquet(path) -> list[ValidationResult]
- read_results_df(path) -> DataFrame  (for ad-hoc analysis / report command)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stoptimes.core.domain.models import ValidationResult

# Column order mirrors ValidationResult field order exactly.
_COLUMNS: list[str] = [
    "trip_key",
    "predictor_id",
    "stop_id",
    "stop_sequence",
    "route_id",
    "error_s",
    "horizon_s",
    "dist_remaining_m",
    "stops_remaining",
    "predicted_arrival_epoch",
    "actual_arrival_epoch",
    "made_at_epoch",
]


def export_results(results: list[ValidationResult], path: str | Path) -> None:
    """Write ValidationResults to a Parquet file at *path*.

    One row per result; all ValidationResult fields become columns.
    An empty list writes a valid Parquet file with the expected schema
    and zero rows (no crash, round-trippable).

    Args:
        results: ValidationResults to serialise.
        path:    Destination file path (created or overwritten).
    """
    dest = Path(path)
    if results:
        df = pd.DataFrame([r.model_dump() for r in results])
    else:
        # Empty export: preserve the schema so round-trip works.
        df = pd.DataFrame(columns=_COLUMNS)
    df.to_parquet(dest, index=False)


def load_results_parquet(path: str | Path) -> list[ValidationResult]:
    """Read a Parquet file produced by :func:`export_results`.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of :class:`~stoptimes.core.domain.models.ValidationResult` in file order.
        Returns an empty list if the file has no rows.

    Raises:
        FileNotFoundError: Propagated from ``pd.read_parquet`` if *path* does
            not exist.
    """
    df = pd.read_parquet(Path(path))
    if df.empty:
        return []

    results: list[ValidationResult] = []
    for _, row in df.iterrows():
        results.append(
            ValidationResult(
                trip_key=str(row["trip_key"]),
                predictor_id=str(row["predictor_id"]),
                stop_id=str(row["stop_id"]),
                stop_sequence=int(row["stop_sequence"]),
                route_id=(str(row["route_id"]) if pd.notna(row["route_id"]) else None),
                error_s=float(row["error_s"]),
                horizon_s=float(row["horizon_s"]),
                dist_remaining_m=(
                    float(row["dist_remaining_m"])
                    if pd.notna(row["dist_remaining_m"])
                    else None
                ),
                stops_remaining=(
                    int(row["stops_remaining"]) if pd.notna(row["stops_remaining"]) else None
                ),
                predicted_arrival_epoch=float(row["predicted_arrival_epoch"]),
                actual_arrival_epoch=float(row["actual_arrival_epoch"]),
                made_at_epoch=float(row["made_at_epoch"]),
            )
        )
    return results


def read_results_df(path: str | Path) -> pd.DataFrame:
    """Read a Parquet results file as a raw DataFrame (for ad-hoc analysis).

    Args:
        path: Path to the Parquet file.

    Returns:
        Raw :class:`pandas.DataFrame` with one row per ValidationResult.
    """
    return pd.read_parquet(Path(path))
