"""Walk-forward backtest over an archived replay. A6/M6.

Splits a replay window ``[start, end]`` into contiguous, non-overlapping
sub-windows (either an equal COUNT of windows via ``n_windows``, or a fixed
WIDTH in seconds via ``width``), runs a fresh :func:`~etaval.engine.assembly.
build_replay_run` + :func:`~etaval.engine.runtime.run_engine` for each window
(same shared ``seed`` every window, for reproducibility), and computes each
window's per-predictor MAE with a bootstrap confidence interval.

Public API
----------
split_windows(start, end, *, n_windows=None, width=None) -> list[(float, float)]
walk_forward(cfg, *, vp_path, ..., start, end, predictor_ids, ...) -> WalkForwardResult
WindowResult        dataclass — one window's per-predictor MAE + CI
WalkForwardResult    dataclass — all windows + aggregate() + to_dict()/write_json()

M6 note
-------
gtfs-eta's trained models (``gtfs_eta:*`` predictors) are fit on a canonical
chronological train/validation/test split of their own training data (see
``../gtfs-eta/models/common/data.py``'s ``temporal_split``, which reserves the
most-recent slice of the timeline as a held-out test set). etaval never
modifies gtfs-eta or its split. For a walk-forward run here to be a valid
held-out evaluation of any ``gtfs_eta:*`` predictor, every backtest window
must fall inside that reserved test slice — see :data:`SPLIT_NOTE` and the
"Walk-forward backtest and the 70/10/20 split" section in the project README.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from etaval.domain.models import SourceConfig
from etaval.engine.assembly import DEFAULT_PREDICTOR_IDS, build_detectors, build_replay_run
from etaval.engine.runtime import run_engine
from etaval.provenance import RunProvenance, build_run_provenance
from etaval.stats.bootstrap import CIResult, bootstrap_ci
from etaval.storage.sqlite import SqliteStorage

EpochLike = float | int | str | dt.datetime | None

SPLIT_NOTE: str = (
    "gtfs-eta trains its per-agency models on a canonical chronological "
    "70/10/20 train/validation/test split (see ../gtfs-eta/models/common/"
    "data.py's temporal_split, which reserves the most-recent slice of the "
    "training timeline as a held-out test set). etaval never modifies "
    "gtfs-eta or re-derives that split -- it only replays archived VP/TU "
    "data through the engine. For THIS walk-forward backtest's windows to "
    "be a valid, leakage-free held-out evaluation of any gtfs_eta:* "
    "predictor under test, every window's [window_start, window_end] here "
    "must fall entirely inside gtfs-eta's reserved test slice for that "
    "model; a window overlapping the train/val slices would let the model "
    "be evaluated on data it (directly or via a nearby timestamp) was "
    "fitted or tuned on, inflating apparent accuracy. Predictors with no "
    "training split (gtfsrt, baseline:*) have no such constraint. It is "
    "the CALLER's responsibility to choose --replay-start/--replay-end (or "
    "the start/end passed to walk_forward()) so they lie within the "
    "trained model's reserved test window -- this note documents the "
    "alignment requirement, it does not verify it."
)


def _to_epoch(value: EpochLike) -> float:
    """Coerce an epoch number or ISO 8601 string to epoch seconds (UTC)."""
    if value is None:
        raise ValueError("start/end are required")
    if isinstance(value, bool):
        raise TypeError(f"Cannot convert bool {value!r} to epoch seconds")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dt.datetime):
        d = value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
        return d.timestamp()
    if isinstance(value, str):
        s = value.strip()
        try:
            return float(s)
        except ValueError:
            pass
        iso = s.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.timestamp()
    raise TypeError(f"Cannot convert {value!r} to epoch seconds")


# ── Window splitting ─────────────────────────────────────────────────────────


def split_windows(
    start: float,
    end: float,
    *,
    n_windows: int | None = None,
    width: float | None = None,
) -> list[tuple[float, float]]:
    """Split ``[start, end]`` into contiguous, non-overlapping windows.

    Exactly one of *n_windows* (equal-COUNT splitting) or *width* (fixed
    seconds-WIDTH splitting) must be given.

    Args:
        start: Range start (epoch seconds).
        end:   Range end (epoch seconds); must be > start.
        n_windows: Split into exactly this many equal-length windows.
        width: Split into fixed-width windows of this many seconds; the
            final window is truncated to *end* if it would overrun.

    Returns:
        List of ``(window_start, window_end)`` tuples, ascending, covering
        ``[start, end]`` exactly with no gaps or overlaps.

    Raises:
        ValueError: start >= end, neither/both of n_windows/width given,
            n_windows < 1, or width <= 0.
    """
    if start >= end:
        raise ValueError(f"start ({start}) must be < end ({end})")
    if (n_windows is None) == (width is None):
        raise ValueError("exactly one of n_windows or width must be given")

    if n_windows is not None:
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        total = end - start
        edges = [start + total * i / n_windows for i in range(n_windows + 1)]
        edges[-1] = end  # avoid float drift on the final edge
        return [(edges[i], edges[i + 1]) for i in range(n_windows)]

    assert width is not None
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    windows: list[tuple[float, float]] = []
    cur = start
    while cur < end:
        nxt = min(cur + width, end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


# ── Results ───────────────────────────────────────────────────────────────────


@dataclass
class WindowResult:
    """One walk-forward window's outcome: boundaries + per-predictor MAE/CI."""

    window_start: float
    window_end: float
    eval_fraction: float  # this window's share of the full [start, end] range
    mae_by_predictor: dict[str, CIResult]
    n_results: int
    ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "eval_fraction": self.eval_fraction,
            "n_results": self.n_results,
            "ticks": self.ticks,
            "mae_by_predictor": {
                pid: {
                    "point": ci.point,
                    "lo": ci.lo,
                    "hi": ci.hi,
                    "n": ci.n,
                    "n_boot": ci.n_boot,
                    "ci": ci.ci,
                }
                for pid, ci in self.mae_by_predictor.items()
            },
        }


@dataclass
class WalkForwardResult:
    """Full walk-forward outcome: per-window results + aggregate + provenance."""

    windows: list[WindowResult] = field(default_factory=list)
    seed: int | None = None
    provenance: RunProvenance | None = None
    split_note: str = SPLIT_NOTE

    def aggregate(self) -> dict[str, dict[str, float]]:
        """Per-predictor mean +/- std of MAE (point estimate) across windows."""
        by_predictor: dict[str, list[float]] = {}
        for w in self.windows:
            for pid, ci in w.mae_by_predictor.items():
                if math.isnan(ci.point):
                    continue
                by_predictor.setdefault(pid, []).append(ci.point)

        result: dict[str, dict[str, float]] = {}
        for pid, values in by_predictor.items():
            arr = np.asarray(values, dtype=np.float64)
            result[pid] = {
                "mean_mae": float(arr.mean()),
                "std_mae": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                "n_windows": float(len(arr)),
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": [w.to_dict() for w in self.windows],
            "aggregate": self.aggregate(),
            "seed": self.seed,
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
            "split_note": self.split_note,
        }

    def write_json(self, path: str | Path) -> Path:
        """Write :meth:`to_dict` as pretty JSON to *path* and return it."""
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


# ── Main entry point ──────────────────────────────────────────────────────────


async def walk_forward(
    cfg: SourceConfig,
    *,
    vp_path: str | Path,
    tu_path: str | Path | None = None,
    start: EpochLike,
    end: EpochLike,
    predictor_ids: list[str] | None = None,
    n_windows: int | None = None,
    width: float | None = None,
    seed: int = 0,
    routes: list[str] | None = None,
    static_gtfs_zip: str | Path | None = None,
    static_gtfs_dir: str | Path | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> WalkForwardResult:
    """Run a walk-forward backtest over ``[start, end]``.

    For each window (see :func:`split_windows`), builds a FRESH
    :func:`~etaval.engine.assembly.build_replay_run` + runs
    :func:`~etaval.engine.runtime.run_engine` against an in-memory
    :class:`~etaval.storage.sqlite.SqliteStorage`, scoped exactly to that
    window (via ``build_replay_run``'s ``start``/``end``), using the SAME
    ``seed`` every window. Each window's persisted ValidationResults are
    reduced to a per-predictor MAE with a bootstrap CI
    (:func:`~etaval.stats.bootstrap.bootstrap_ci`).

    Args:
        cfg: SourceConfig (detector params + provenance source name).
        vp_path, tu_path: Archived VP/TU parquet paths (see build_replay_run).
        start, end: Overall backtest range (epoch seconds or ISO 8601 str).
        predictor_ids: Predictor IDs to evaluate; None -> DEFAULT_PREDICTOR_IDS.
        n_windows, width: Exactly one required — see split_windows.
        seed: Shared run seed for every window (reproducibility + provenance).
        routes: Optional route ID filter.
        static_gtfs_zip, static_gtfs_dir: Static GTFS resolution (required —
            one of the two, exactly as build_replay_run requires).
        n_boot, ci: Bootstrap CI parameters, forwarded to bootstrap_ci.

    Returns:
        :class:`WalkForwardResult` with one WindowResult per window, the
        cross-window aggregate, run provenance, seed, and the M6 split_note.

    Raises:
        ValueError: invalid start/end/window arguments, or (per-window) no
            archived rows matched a window's bounds.
        FileNotFoundError: static GTFS snapshot could not be resolved.
    """
    start_epoch = _to_epoch(start)
    end_epoch = _to_epoch(end)
    windows = split_windows(start_epoch, end_epoch, n_windows=n_windows, width=width)

    ids = list(predictor_ids) if predictor_ids is not None else list(DEFAULT_PREDICTOR_IDS)
    total_span = end_epoch - start_epoch

    window_results: list[WindowResult] = []
    for win_start, win_end in windows:
        deps = build_replay_run(
            cfg,
            vp_path=vp_path,
            tu_path=tu_path,
            start=win_start,
            end=win_end,
            routes=routes,
            predictor_ids=ids,
            static_gtfs_zip=static_gtfs_zip,
            static_gtfs_dir=static_gtfs_dir,
        )
        with SqliteStorage(":memory:") as storage:
            summary = await run_engine(
                deps.feed_source,
                deps.static_loader,
                deps.predictors,
                deps.detectors,
                storage,
                source_name=cfg.name,
                seed=seed,
            )
            results = storage.load_results()

        errors_by_predictor: dict[str, list[float]] = {}
        for r in results:
            errors_by_predictor.setdefault(r.predictor_id, []).append(r.error_s)

        mae_by_predictor: dict[str, CIResult] = {
            pid: bootstrap_ci(errors_by_predictor.get(pid, []), n_boot=n_boot, ci=ci, seed=seed)
            for pid in ids
        }

        eval_fraction = (win_end - win_start) / total_span if total_span else 0.0
        window_results.append(
            WindowResult(
                window_start=win_start,
                window_end=win_end,
                eval_fraction=eval_fraction,
                mae_by_predictor=mae_by_predictor,
                n_results=len(results),
                ticks=summary.ticks,
            )
        )

    gtfs_snapshot = (
        str(static_gtfs_zip)
        if static_gtfs_zip is not None
        else (str(static_gtfs_dir) if static_gtfs_dir is not None else None)
    )
    provenance = build_run_provenance(
        source_name=cfg.name,
        source_config=cfg,
        detectors=build_detectors(cfg),
        predictor_ids=ids,
        seed=seed,
        gtfs_snapshot=gtfs_snapshot,
    )

    return WalkForwardResult(
        windows=window_results,
        seed=seed,
        provenance=provenance,
        split_note=SPLIT_NOTE,
    )
