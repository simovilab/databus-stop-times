"""`stoptimes data` — load and inspect GTFS static + realtime data.

Heavy imports (pandas, pyarrow, the readers module) happen inside each
command body, never at module scope, so `stoptimes data --help` stays fast
(docs/PLAN.md §6 T1 requirement) even when those optional parts aren't
installed.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer
from stoptimes.cli._util import print_table

app = typer.Typer(help="Load and inspect GTFS static + realtime data.", no_args_is_help=True)

_FORMAT_CHOICES = ("gtfsrt_json", "gtfsrt_pb", "archive_parquet", "avl_csv")


def _guess_format(location: str, mapping: Path | None) -> str:
    if mapping is not None:
        return "avl_csv"
    lower = location.lower()
    if lower.endswith(".csv"):
        return "avl_csv"
    if lower.endswith(".pb") or lower.endswith(".bin"):
        return "gtfsrt_pb"
    if lower.endswith(".json"):
        return "gtfsrt_json"
    return "archive_parquet"


def _now_utc():
    import pandas as pd

    return pd.Timestamp.now(tz="UTC").to_pydatetime()


@app.command("add")
def data_add(
    name: str = typer.Argument(..., help="Name to register this dataset under in the catalog."),
    location: str = typer.Argument(
        ..., help="Local path, s3:// URI, or feed URL for this dataset."
    ),
    source: str = typer.Option(..., "--source", help="Source id (agency) this dataset belongs to."),
    kind: str = typer.Option("observations", "--kind", help="observations | arrivals | static"),
    format: str | None = typer.Option(
        None,
        "--format",
        help=f"One of {_FORMAT_CHOICES} (observations only; auto-detected if omitted).",
    ),
    observation_kind: str = typer.Option(
        "vehicle_position",
        "--observation-kind",
        help="vehicle_position | trip_update (observations only).",
    ),
    mapping: str | None = typer.Option(
        None, "--mapping", help="Column-mapping YAML file (avl_csv format only)."
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Filter: archive_parquet only."
    ),
    end_date: str | None = typer.Option(None, "--end-date", help="Filter: archive_parquet only."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing entry with this name."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the registered entry as JSON."),
) -> None:
    """Register GTFS static / realtime / AVL sources (protobuf, JSON, parquet, CSV).

    `--json`: the registered `DatasetEntry` as a dict (`stoptimes.core.catalog`).
    """
    from stoptimes.core import readers as readers_module
    from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetExistsError, DatasetKind
    from stoptimes.core.contracts.data import ObservationKind

    try:
        dataset_kind = DatasetKind(kind)
    except ValueError:
        valid = [k.value for k in DatasetKind]
        typer.secho(f"✗ unknown --kind {kind!r}; expected one of {valid}", fg="red", err=True)
        raise typer.Exit(1) from None

    row_count: int | None = None
    date_start = start_date
    date_end = end_date
    extra: dict = {}

    if dataset_kind is DatasetKind.STATIC:
        _, static_feed = readers_module.read_static(location, source)
        extra = {"tables": static_feed.tables, "missing_required": static_feed.missing_required()}
        if static_feed.missing_required():
            typer.secho(
                f"⚠ {location!r} is missing required static table(s): "
                f"{static_feed.missing_required()}",
                fg="yellow",
                err=True,
            )
    elif dataset_kind is DatasetKind.ARRIVALS:
        # T2 owns Arrival derivation (`data arrivals`); registering here only
        # records where a prepared Arrival dataset lives.
        extra = {}
    else:
        reader_format = format or _guess_format(location, mapping)
        if reader_format not in _FORMAT_CHOICES:
            typer.secho(
                f"✗ unknown --format {reader_format!r}; expected one of {_FORMAT_CHOICES}",
                fg="red",
                err=True,
            )
            raise typer.Exit(1)
        try:
            obs_kind = ObservationKind(observation_kind)
        except ValueError:
            typer.secho(f"✗ unknown --observation-kind {observation_kind!r}", fg="red", err=True)
            raise typer.Exit(1) from None

        mapping_obj = None
        mapping_dump = None
        if reader_format == "avl_csv":
            if mapping is None:
                typer.secho("✗ --format avl_csv requires --mapping FILE", fg="red", err=True)
                raise typer.Exit(1)
            if not Path(mapping).is_file():
                typer.secho(f"✗ --mapping file not found: {mapping!r}", fg="red", err=True)
                raise typer.Exit(1)
            mapping_obj = readers_module.ColumnMapping.from_file(mapping)
            mapping_dump = mapping_obj.model_dump()

        try:
            df = readers_module.read_observations(
                location,
                format=reader_format,
                kind=obs_kind,
                source_id=source,
                mapping=mapping_obj,
                start_date=start_date,
                end_date=end_date,
            )
        except (ValueError, ImportError, OSError) as exc:
            typer.secho(f"✗ could not read {location!r}: {exc}", fg="red", err=True)
            raise typer.Exit(1) from None

        row_count = int(len(df))
        if row_count and "ts" in df.columns:
            usable = df["ts"].dropna()
            if not usable.empty:
                date_start = date_start or str(usable.min())
                date_end = date_end or str(usable.max())
        extra = {"format": reader_format, "observation_kind": obs_kind.value}
        if mapping_dump is not None:
            extra["mapping"] = mapping_dump

    entry = DatasetEntry(
        name=name,
        source_id=source,
        kind=dataset_kind,
        location=location,
        date_start=date_start,
        date_end=date_end,
        row_count=row_count,
        added_at=_now_utc(),
        extra=extra,
    )

    try:
        Catalog().add(entry, overwrite=overwrite)
    except DatasetExistsError as exc:
        typer.secho(f"✗ {exc} (pass --overwrite to replace it)", fg="red", err=True)
        raise typer.Exit(1) from None

    if json_output:
        typer.echo(_json.dumps(entry.model_dump(mode="json"), indent=2))
    else:
        rows_desc = str(row_count) if row_count is not None else "—"
        typer.secho(
            f"✓ added dataset {name!r} ({dataset_kind.value}, {rows_desc} rows)", fg="green"
        )


@app.command("list")
def data_list(json_output: bool = typer.Option(False, "--json")) -> None:
    """List registered datasets.

    `--json`: a list of `DatasetEntry` dicts (`stoptimes.core.catalog`), one per registered dataset.
    """
    from stoptimes.core.catalog import Catalog

    entries = Catalog().list()

    if json_output:
        typer.echo(_json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return

    rows = [
        [
            entry.name,
            entry.source_id or "—",
            entry.kind.value,
            str(entry.row_count) if entry.row_count is not None else "—",
            f"{entry.date_start or '—'} .. {entry.date_end or '—'}",
            entry.location,
        ]
        for entry in entries
    ]
    print_table("stoptimes datasets", ["name", "source", "kind", "rows", "date range", "location"], rows)


@app.command("inspect")
def data_inspect(
    name: str = typer.Argument(..., help="A dataset name registered with `data add`."),
    static: str | None = typer.Option(
        None, "--static", help="A GTFS static .zip to check trip coverage against."
    ),
    gap_threshold_s: float = typer.Option(
        300.0, "--gap-threshold-s", help="Gap size (s) counted in gap_count."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Summarize coverage of registered data (trips matched, arrivals present, gaps).

    `--json`: a flat `{"kind": str, ...}` dict; the remaining keys depend on
    `kind` (`static`/`arrivals`/an observation kind) -- see
    `stoptimes.core.catalog.inspect.build_coverage_report`.
    """
    from stoptimes.core import readers as readers_module
    from stoptimes.core.catalog import Catalog
    from stoptimes.core.catalog.inspect import build_coverage_report
    from stoptimes.core.contracts.data import ObservationKind

    entry = Catalog().get(name)
    if entry is None:
        typer.secho(f"✗ no such dataset: {name!r}", fg="red", err=True)
        raise typer.Exit(1)

    if entry.kind.value == "static":
        _, feed = readers_module.read_static(entry.location, entry.source_id or "unknown")
        report = {
            "kind": "static",
            "tables": feed.tables,
            "missing_required": feed.missing_required(),
        }
    elif entry.kind.value == "arrivals":
        report = {
            "kind": "arrivals",
            "note": "coverage for arrivals is T2's `data arrivals` (docs/PLAN.md).",
        }
    else:
        obs_kind = ObservationKind(entry.extra.get("observation_kind", "vehicle_position"))
        reader_format = entry.extra.get("format", "archive_parquet")
        mapping_obj = None
        if entry.extra.get("mapping"):
            mapping_obj = readers_module.ColumnMapping.model_validate(entry.extra["mapping"])
        try:
            df = readers_module.read_observations(
                entry.location,
                format=reader_format,
                kind=obs_kind,
                source_id=entry.source_id or "unknown",
                mapping=mapping_obj,
            )
        except (ValueError, ImportError, OSError) as exc:
            typer.secho(f"✗ could not re-read {entry.location!r}: {exc}", fg="red", err=True)
            raise typer.Exit(1) from None

        static_trip_ids = None
        if static is not None:
            if not Path(static).is_file():
                typer.secho(f"✗ --static file not found: {static!r}", fg="red", err=True)
                raise typer.Exit(1)
            loader, static_feed = readers_module.read_static(static, entry.source_id or "unknown")
            if loader is not None:
                static_trip_ids = set(loader.trips["trip_id"].astype(str))
            else:
                typer.secho(
                    f"⚠ {static} is missing required static table(s): "
                    f"{static_feed.missing_required()}",
                    fg="yellow",
                    err=True,
                )

        report = build_coverage_report(
            df, obs_kind.value, static_trip_ids=static_trip_ids, gap_threshold_s=gap_threshold_s
        )
        report = {"kind": obs_kind.value, **report}

    if json_output:
        typer.echo(_json.dumps(report, indent=2, default=str))
        return

    rows = [[key, str(value)] for key, value in report.items()]
    print_table(f"coverage: {name}", ["metric", "value"], rows)


@app.command("remove")
def data_remove(
    name: str = typer.Argument(..., help="A dataset name registered with `data add`."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Remove a dataset from the catalog (does not delete the underlying file).

    `--json`: ``{"name": str, "removed": bool}``.
    """
    from stoptimes.core.catalog import Catalog

    removed = Catalog().remove(name)

    if json_output:
        typer.echo(_json.dumps({"name": name, "removed": removed}))
        if not removed:
            raise typer.Exit(1)
        return

    if removed:
        typer.secho(f"✓ removed {name!r}", fg="green")
    else:
        typer.secho(f"✗ no such dataset: {name!r}", fg="red", err=True)
        raise typer.Exit(1)


@app.command("arrivals")
def data_arrivals(
    observations: str = typer.Option(
        ..., "--observations", help="Observations to derive from: a catalog dataset name, or a path/URI."
    ),
    static: str = typer.Option(..., "--static", help="GTFS static .zip (path or URL)."),
    source: str = typer.Option(..., "--source", help="Source id (agency) these observations came from."),
    out: str = typer.Option(..., "--out", help="Where to write the Arrival parquet."),
    sources_dir: str | None = typer.Option(
        None, "--sources-dir", help="Directory of agency YAMLs (default: STOPTIMES_SOURCES -> ./examples/agencies)."
    ),
    register: str | None = typer.Option(
        None, "--register", help="Also register the result in the catalog under this name."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing catalog entry / output file."),
    json_output: bool = typer.Option(False, "--json", help="Print a machine-readable summary."),
) -> None:
    """Derive actual stop arrivals from vehicle positions (map-matching).

    The training target and the validation ground truth: observations in,
    one `Arrival` row per (trip, service day, stop) out.

    `--json`: ``{"source", "observations", "arrivals", "by_method", "static_tables", "out", "registered"}``.
    """
    from stoptimes.core.arrivals import derive_arrivals
    from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetExistsError, DatasetKind

    src = _resolve_source(source, sources_dir)
    frame = _load_observations(observations, source)
    loader, static_feed = _read_static_or_exit(static, source)

    out_path = Path(out)
    if out_path.exists() and not overwrite:
        typer.secho(f"✗ {out} already exists (pass --overwrite to replace it)", fg="red", err=True)
        raise typer.Exit(1)

    arrivals = derive_arrivals(frame, loader, source=src)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrivals.to_parquet(out_path, index=False)

    date_start = date_end = None
    if not arrivals.empty:
        date_start = str(arrivals["arrival_time"].min())
        date_end = str(arrivals["arrival_time"].max())

    if register:
        entry = DatasetEntry(
            name=register,
            source_id=source,
            kind=DatasetKind.ARRIVALS,
            location=str(out_path),
            date_start=date_start,
            date_end=date_end,
            row_count=int(len(arrivals)),
            added_at=_now_utc(),
            extra={"derived_from": observations, "static": static},
        )
        try:
            Catalog().add(entry, overwrite=overwrite)
        except DatasetExistsError as exc:
            typer.secho(f"✗ {exc} (pass --overwrite to replace it)", fg="red", err=True)
            raise typer.Exit(1) from None

    by_method = {str(k): int(v) for k, v in arrivals["method"].value_counts().items()} if not arrivals.empty else {}
    summary = {
        "source": source,
        "observations": int(len(frame)),
        "arrivals": int(len(arrivals)),
        "by_method": by_method,
        "static_tables": static_feed.tables,
        "out": str(out_path),
        "registered": register,
    }

    if json_output:
        typer.echo(_json.dumps(summary, indent=2))
        return

    typer.secho(
        f"✓ {summary['arrivals']} arrival(s) from {summary['observations']} observation(s) → {out_path}",
        fg="green",
    )
    for method, count in by_method.items():
        typer.echo(f"  {method}: {count}")


@app.command("frame")
def data_frame(
    observations: str = typer.Option(
        ..., "--observations", help="Observations to build from: a catalog dataset name, or a path/URI."
    ),
    arrivals: str = typer.Option(
        ..., "--arrivals", help="The Arrival table (target): a catalog dataset name, or a path/URI to a parquet file."
    ),
    static: str = typer.Option(..., "--static", help="GTFS static .zip (path or URL)."),
    source: str = typer.Option(..., "--source", help="Source id (agency) these observations/arrivals came from."),
    out: str = typer.Option(..., "--out", help="Where to write the training frame parquet."),
    max_stops_ahead: int = typer.Option(
        5, "--max-stops-ahead", help="Upcoming stops emitted per vehicle-position row."
    ),
    keep_unlabeled: bool = typer.Option(
        False, "--keep-unlabeled", help="Keep rows with no matching arrival (target left NaN) instead of dropping them."
    ),
    sources_dir: str | None = typer.Option(
        None, "--sources-dir", help="Directory of agency YAMLs (default: STOPTIMES_SOURCES -> ./examples/agencies)."
    ),
    register: str | None = typer.Option(
        None, "--register", help="Also register the result in the catalog under this name."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing catalog entry / output file."),
    json_output: bool = typer.Option(False, "--json", help="Print a machine-readable summary."),
) -> None:
    """Build the training frame: observations + arrivals + static GTFS -> one
    row per (vehicle position, upcoming stop), ready for `train run`
    (via the feature registry, which adds temporal features on top).

    `--json`: ``{"source", "observations", "arrivals", "rows", "labeled_rows",
    "static_tables", "out", "registered"}``.
    """
    from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetExistsError, DatasetKind
    from stoptimes.train.frame import FrameParams, build_training_frame

    src = _resolve_source(source, sources_dir)
    obs_frame = _load_observations(observations, source)
    arrivals_frame = _load_arrivals(arrivals)
    loader, static_feed = _read_static_or_exit(static, source)

    out_path = Path(out)
    if out_path.exists() and not overwrite:
        typer.secho(f"✗ {out} already exists (pass --overwrite to replace it)", fg="red", err=True)
        raise typer.Exit(1)

    params = FrameParams(max_stops_ahead=max_stops_ahead, keep_unlabeled=keep_unlabeled)
    result = build_training_frame(obs_frame, arrivals_frame, loader, source=src, params=params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)

    date_start = date_end = None
    if not result.empty:
        date_start = str(result["vp_ts"].min())
        date_end = str(result["vp_ts"].max())

    if register:
        entry = DatasetEntry(
            name=register,
            source_id=source,
            kind=DatasetKind.FRAME,
            location=str(out_path),
            date_start=date_start,
            date_end=date_end,
            row_count=int(len(result)),
            added_at=_now_utc(),
            extra={
                "derived_from_observations": observations,
                "derived_from_arrivals": arrivals,
                "static": static,
                "max_stops_ahead": max_stops_ahead,
            },
        )
        try:
            Catalog().add(entry, overwrite=overwrite)
        except DatasetExistsError as exc:
            typer.secho(f"✗ {exc} (pass --overwrite to replace it)", fg="red", err=True)
            raise typer.Exit(1) from None

    n_labeled = int(result["time_to_arrival_seconds"].notna().sum()) if not result.empty else 0
    summary = {
        "source": source,
        "observations": int(len(obs_frame)),
        "arrivals": int(len(arrivals_frame)),
        "rows": int(len(result)),
        "labeled_rows": n_labeled,
        "static_tables": static_feed.tables,
        "out": str(out_path),
        "registered": register,
    }

    if json_output:
        typer.echo(_json.dumps(summary, indent=2))
        return

    typer.secho(
        f"✓ {summary['rows']} row(s) ({summary['labeled_rows']} labeled) from "
        f"{summary['observations']} observation(s) + {summary['arrivals']} arrival(s) → {out_path}",
        fg="green",
    )


def _load_arrivals(reference: str):
    """`reference` is a catalog dataset name if one is registered, else a parquet path/URI."""
    import pandas as pd

    from stoptimes.core.catalog import Catalog

    entry = Catalog().get(reference)
    location = entry.location if entry is not None else reference
    try:
        return pd.read_parquet(location)
    except (OSError, ValueError) as exc:
        typer.secho(f"✗ could not read arrivals {location!r}: {exc}", fg="red", err=True)
        raise typer.Exit(1) from None


def _resolve_source(source_id: str, sources_dir: str | None):
    """Load an agency file the same way `serve` and `record` do."""
    import os

    import yaml
    from stoptimes.core.config import sources_search_path
    from stoptimes.core.contracts.adapters import source_from_val_yaml

    candidates = (
        [Path(sources_dir)] if sources_dir else sources_search_path(os.environ.get("STOPTIMES_SOURCES"))
    )
    for directory in candidates:
        path = Path(directory) / f"{source_id}.yaml"
        if path.is_file():
            return source_from_val_yaml(yaml.safe_load(path.read_text(encoding="utf-8")))
    searched = ", ".join(str(c) for c in candidates)
    typer.secho(f"✗ no agency file for --source {source_id!r} (searched: {searched})", fg="red", err=True)
    raise typer.Exit(1)


def _load_observations(reference: str, source_id: str):
    """`reference` is a catalog dataset name if one is registered, else a path/URI."""
    from stoptimes.core import readers as readers_module
    from stoptimes.core.catalog import Catalog
    from stoptimes.core.contracts.data import ObservationKind

    entry = Catalog().get(reference)
    location = entry.location if entry is not None else reference
    extra = entry.extra if entry is not None else {}
    reader_format = extra.get("format") or _guess_format(location, None)
    kind = ObservationKind(extra.get("observation_kind", ObservationKind.VEHICLE_POSITION.value))
    mapping = None
    if reader_format == "avl_csv" and extra.get("mapping"):
        mapping = readers_module.ColumnMapping(**extra["mapping"])

    try:
        return readers_module.read_observations(
            location, format=reader_format, kind=kind, source_id=source_id, mapping=mapping
        )
    except (ValueError, ImportError, OSError) as exc:
        typer.secho(f"✗ could not read {location!r}: {exc}", fg="red", err=True)
        raise typer.Exit(1) from None


def _read_static_or_exit(location: str, source_id: str):
    from stoptimes.core import readers as readers_module

    try:
        loader, feed = readers_module.read_static(location, source_id)
    except (OSError, ValueError) as exc:
        typer.secho(f"✗ could not read static feed {location!r}: {exc}", fg="red", err=True)
        raise typer.Exit(1) from None
    if loader is None:
        typer.secho(
            f"✗ {location!r} is missing required static table(s): {feed.missing_required()}",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)
    return loader, feed
