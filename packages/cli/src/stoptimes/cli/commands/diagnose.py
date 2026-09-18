"""`stoptimes diagnose` — feed quality report: can this GTFS static +
realtime data support ETA modelling at all, before investing a week finding
out it can't (docs/PLAN.md §7.1's Newcomer story, track U3).

Heavy imports (pandas, the readers module, `stoptimes.core.diagnose`) happen
inside the command body, never at module scope, so `stoptimes diagnose
--help` stays fast even when optional parts (protobuf, parquet) aren't
installed — same rule `data.py` follows.

A feed with problems is a *successful* run reporting problems: this command
exits non-zero only on a usage error (a location that can't be read at all,
an unknown format). `--json` prints `{"window": ..., **DiagnosticReport}` so
the report is data a caller can consume, not just prose this command prints.

This module is wired into `stoptimes.cli.main` as a single top-level
command (`app.command("diagnose")(diagnose.diagnose)`), not a `typer.Typer`
sub-app — `diagnose` has no sub-commands.

Large-archive note: `--from`/`--to` (same spelling as `val replay`) bound an
`archive_parquet` read to the files whose Hive partition falls in that
window — `stoptimes.core.readers.read_archive_parquet` prunes the file
*list* before opening anything, so a bounded window really does mean fewer
files read, not just fewer rows kept after the fact. Neither bound narrows
a live URL feed (there's only ever one snapshot to read) or the *columns*
read from each matched parquet file — `read_observations`/
`read_archive_parquet` have no column-projection parameter today (see
docs/notes/p3-u3.md); an unbounded `--vehicle-positions`/`--trip-updates`
archive read still loads every matched file's full row set into memory.
"""

from __future__ import annotations

import json as _json

import typer

_FORMAT_CHOICES = ("gtfsrt_json", "gtfsrt_pb", "archive_parquet", "avl_csv")


def _guess_format(location: str) -> str:
    lower = location.lower()
    if lower.endswith(".csv"):
        return "avl_csv"
    if lower.endswith(".pb") or lower.endswith(".bin"):
        return "gtfsrt_pb"
    if lower.endswith(".json"):
        return "gtfsrt_json"
    if location.startswith("http://") or location.startswith("https://"):
        # An extensionless GTFS-RT URL is far more often JSON than protobuf
        # among the example agencies this toolkit ships with; --*-format
        # overrides this guess when it's wrong.
        return "gtfsrt_json"
    return "archive_parquet"


def _resolve_source_yaml(source_id: str, sources_dir: str | None):
    """Load an agency file the same way `data`/`serve`/`record` do, so
    `--source` alone (no explicit locations) can diagnose a configured
    agency's feeds directly. Returns `None` if nothing matches — not fatal,
    since every location can also be given explicitly."""
    import os
    from pathlib import Path

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
    return None


def _load_observation_frame(
    *,
    location: str | None,
    format_override: str | None,
    kind,
    feed_kind,
    src,
    source_id: str,
    start_date: str | None,
    end_date: str | None,
):
    """Read one Observation frame for `diagnose_feed`.

    `location` (a catalog dataset name, path, s3:// prefix, or feed URL)
    wins when given; otherwise, if `--source` resolved an agency, its
    matching `RealtimeFeed` URL is used. Returns `None` (not an error) when
    neither is available — `diagnose_feed` reports that as a finding, not a
    crash, per docs/PLAN.md's "run on a feed with missing pieces" requirement.

    `start_date`/`end_date` are forwarded to the reader either way;
    `read_archive_parquet` is the only reader that acts on them (pruning the
    file list before opening anything — see the module docstring's
    large-archive note), so passing them for a live/JSON/CSV read is a
    harmless no-op there.
    """
    from stoptimes.core import readers as readers_module
    from stoptimes.core.catalog import Catalog

    if location is None and src is None:
        return None
    try:
        if location is not None:
            entry = Catalog().get(location)
            loc = entry.location if entry is not None else location
            extra = entry.extra if entry is not None else {}
            reader_format = format_override or extra.get("format") or _guess_format(loc)
            mapping_obj = None
            if reader_format == "avl_csv" and extra.get("mapping"):
                mapping_obj = readers_module.ColumnMapping(**extra["mapping"])
            if reader_format == "archive_parquet" and start_date is None and end_date is None:
                typer.secho(
                    f"⚠ reading all of {loc!r} with no --from/--to bound — this loads every "
                    "matched parquet file's full row set into memory. Pass --from/--to to scan "
                    "only part of a large archive.",
                    fg="yellow",
                    err=True,
                )
            return readers_module.read_observations(
                loc,
                format=reader_format,
                kind=kind,
                source_id=source_id,
                mapping=mapping_obj,
                start_date=start_date,
                end_date=end_date,
            )
        return readers_module.read_observations_for_source(
            src, feed_kind, start_date=start_date, end_date=end_date
        )
    except (ValueError, ImportError, OSError) as exc:
        label = location or f"the {source_id!r} agency's feed"
        typer.secho(f"✗ could not read observations from {label!r}: {exc}", fg="red", err=True)
        raise typer.Exit(1) from None


_SEVERITY_COLOR = {"blocker": "red", "degraded": "yellow", "info": "cyan"}
_SEVERITY_ORDER = {"blocker": 0, "degraded": 1, "info": 2}


def _render(report, window: dict[str, str | None]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    color = _SEVERITY_COLOR[report.verdict_severity.value]
    console.print(f"[bold {color}]{report.verdict}[/bold {color}]")
    console.print(
        f"window: {window['from'] or '(start)'} .. {window['to'] or '(end)'}"
        " (applies to archive_parquet reads only — see --help)"
    )
    console.print(
        f"static: {'present (' + ', '.join(report.static_tables) + ')' if report.static_present else 'MISSING'}"
        f"  ·  vehicle positions: {report.vp_rows} row(s)"
        f"  ·  trip updates: {report.tu_rows} row(s)"
    )

    if not report.findings:
        console.print("\nNo findings — nothing to report.")
        return

    table = Table(title="findings")
    for column in ("severity", "id", "title", "count", "total", "share"):
        table.add_column(column)
    ordered = sorted(report.findings, key=lambda f: _SEVERITY_ORDER[f.severity.value])
    for f in ordered:
        table.add_row(
            f"[{_SEVERITY_COLOR[f.severity.value]}]{f.severity.value}[/]",
            f.id,
            f.title,
            str(f.count),
            str(f.total) if f.total is not None else "—",
            f"{f.share:.0%}" if f.share is not None else "—",
        )
    console.print(table)

    for f in ordered:
        if f.severity.value == "info":
            continue
        console.print(f"\n[{_SEVERITY_COLOR[f.severity.value]}]{f.id}[/]: {f.explanation}")
        console.print(f"  blocks: {f.blocks}")
        console.print(f"  do: {f.recommendation}")
        if f.samples:
            console.print(f"  e.g. {', '.join(f.samples[:5])}")


def diagnose(
    static: str | None = typer.Option(
        None, "--static", help="GTFS static .zip (path or URL). Omit to diagnose a feed with no static schedule at all."
    ),
    vehicle_positions: str | None = typer.Option(
        None,
        "--vehicle-positions",
        help="Vehicle-position observations: a `data add` dataset name, path, s3:// archive dir, or feed URL.",
    ),
    vehicle_positions_format: str | None = typer.Option(
        None, "--vehicle-positions-format", help=f"One of {_FORMAT_CHOICES} (auto-detected if omitted)."
    ),
    trip_updates: str | None = typer.Option(
        None,
        "--trip-updates",
        help="Trip-update observations: a `data add` dataset name, path, s3:// archive dir, or feed URL.",
    ),
    trip_updates_format: str | None = typer.Option(
        None, "--trip-updates-format", help=f"One of {_FORMAT_CHOICES} (auto-detected if omitted)."
    ),
    from_: str | None = typer.Option(
        None,
        "--from",
        help="Window start (YYYY-MM-DD or full ISO 8601, day precision) — same spelling as `val "
        "replay`. Bounds an archive_parquet read to matching partitions before any file is "
        "opened; a live feed URL always reads its one current snapshot regardless. Recorded in "
        "the output so it's clear what was and wasn't examined.",
    ),
    to: str | None = typer.Option(
        None,
        "--to",
        help="Window end (YYYY-MM-DD or full ISO 8601, day precision). See --from.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Source id (agency). Tags the read observations, and — with --sources-dir — fills in "
        "--static/--vehicle-positions/--trip-updates from the agency file when they're omitted.",
    ),
    sources_dir: str | None = typer.Option(
        None,
        "--sources-dir",
        help="Directory of agency YAMLs (default: STOPTIMES_SOURCES -> ./examples/agencies).",
    ),
    gap_threshold_s: float = typer.Option(
        300.0, "--gap-threshold-s", help="Vehicle-position sampling gap (s) counted as a coverage gap."
    ),
    stale_threshold_s: float = typer.Option(
        120.0, "--stale-threshold-s", help="ingested_at - ts (s) beyond which a record counts as stale."
    ),
    majority_threshold: float = typer.Option(
        0.5,
        "--majority-threshold",
        help="Affected share at/above which a mismatch finding is a blocker rather than degraded.",
    ),
    max_samples: int = typer.Option(
        10, "--max-samples", help="Affected-sample identifiers kept per finding."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the DiagnosticReport as JSON."),
) -> None:
    """Report GTFS / GTFS-RT feed quality problems before training on them.

    Exits non-zero only on a usage error (a location that can't be read).
    A feed with quality problems is a successful run that reports them.
    """
    from stoptimes.core import readers as readers_module
    from stoptimes.core.contracts.data import ObservationKind
    from stoptimes.core.contracts.source import FeedKind
    from stoptimes.core.diagnose import DiagnoseParams, diagnose_feed

    source_id = source or "diagnose"
    src = _resolve_source_yaml(source, sources_dir) if source else None

    static_location = static or (src.gtfs_static_url if src is not None else None)
    static_loader = None
    static_feed = None
    if static_location is not None:
        try:
            static_loader, static_feed = readers_module.read_static(static_location, source_id)
        except (OSError, ValueError) as exc:
            typer.secho(f"✗ could not read static feed {static_location!r}: {exc}", fg="red", err=True)
            raise typer.Exit(1) from None

    vp_df = _load_observation_frame(
        location=vehicle_positions,
        format_override=vehicle_positions_format,
        kind=ObservationKind.VEHICLE_POSITION,
        feed_kind=FeedKind.VEHICLE_POSITIONS,
        src=src,
        source_id=source_id,
        start_date=from_,
        end_date=to,
    )
    tu_df = _load_observation_frame(
        location=trip_updates,
        format_override=trip_updates_format,
        kind=ObservationKind.TRIP_UPDATE,
        feed_kind=FeedKind.TRIP_UPDATES,
        src=src,
        source_id=source_id,
        start_date=from_,
        end_date=to,
    )

    params = DiagnoseParams(
        gap_threshold_s=gap_threshold_s,
        stale_threshold_s=stale_threshold_s,
        majority_threshold=majority_threshold,
        max_samples=max_samples,
    )
    report = diagnose_feed(
        vp_df=vp_df, tu_df=tu_df, static_loader=static_loader, static_feed=static_feed, params=params
    )

    window = {"from": from_, "to": to}
    if json_output:
        typer.echo(_json.dumps({"window": window, **report.model_dump(mode="json")}, indent=2))
        return
    _render(report, window)
