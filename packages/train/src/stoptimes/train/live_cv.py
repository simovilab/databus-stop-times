"""Leak-free cross-validation of live-frame XGBoost models.

A live frame's ``split`` is fixed when it is built: the segment statistics behind
``prior_rem`` and the route vocabulary behind ``route_code`` are fitted on its training
days.  Re-filtering that parquet into other folds would let a fold's test days leak into
its own features, so every fold here *refits* both on its own fit days.

Per fold the days fall in three disjoint groups:

* ``fit`` - the model is fitted on them; their ``prior_rem`` is out-of-fold by day
  (:func:`~stoptimes.core.features.prior.fit_out_of_fold`) exactly as in the production frame;
* ``es`` - early-stopping days, drawn from the training side, never from test;
* ``test`` - scored.

The statistics and vocabulary are fitted on the ``fit`` days only, and ``es`` and ``test``
rows both get the all-``fit``-days statistics.  That is :func:`build_live_dataset` with
``train_days=fit`` and ``val_days=es + test``: early-stopping rows are then distributed like
test rows (statistics fitted without them), which is what the stopping criterion needs to
track, and the fold's assets are exactly the asset that would ship from those fit days.

Cost: trip records, arrivals, segments and the 17 asset-free features of every day are
computed once (:func:`build_frame_cache`); per fold only the statistics are refitted and
``prior_rem`` / ``route_code`` / ``split`` recomputed (:func:`fold_frame`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stoptimes.core.features.identity import RouteVocabulary
from stoptimes.core.features.live import LIVE_FEATURES, LiveAssets
from stoptimes.core.features.prior import fit_out_of_fold, segments_from_arrivals
from stoptimes.train.live_frame import (
    TARGET_COLUMN,
    LiveFrameParams,
    TripRecord,
    arrivals_table,
    build_day_frame,
)

__all__ = [
    "Fold",
    "FrameCache",
    "HORIZON_BUCKETS_S",
    "aggregate",
    "build_frame_cache",
    "day_grouped_kfold",
    "fold_frame",
    "holdout",
    "rolling_origin_by_week",
    "run_cv",
    "score_fold",
]

_log = logging.getLogger(__name__)

ASSET_FEATURES: tuple[str, ...] = ("prior_rem", "route_code")
BASE_FEATURES: tuple[str, ...] = tuple(f for f in LIVE_FEATURES if f not in ASSET_FEATURES)
HORIZON_BUCKETS_S: tuple[tuple[str, float, float], ...] = (
    ("lt2", 0.0, 120.0),
    ("2_5", 120.0, 300.0),
    ("5_10", 300.0, 600.0),
    ("10_20", 600.0, 1200.0),
    ("20p", 1200.0, math.inf),
)
"""Model MAE buckets on the true horizon (``time_to_arrival_seconds``), in minutes."""
MAX_N_JOBS = 6
_ALIGN_KEYS = ("trip_id", "t", "stop_idx")


# ---------------------------------------------------------------------------
# Fold plans (pure functions of a sorted list of YYYYMMDD service days)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One fold: ``fit``, ``es`` and ``test`` days (``es_is_test``: early stopping on test)."""

    name: str
    fit_days: tuple[str, ...]
    es_days: tuple[str, ...]
    test_days: tuple[str, ...]
    es_is_test: bool = False

    def __post_init__(self) -> None:
        if not self.fit_days or not self.test_days or not self.es_days:
            raise ValueError(
                f"fold {self.name}: fit, early-stopping and test days must be non-empty"
            )
        if set(self.fit_days) & (set(self.es_days) | set(self.test_days)):
            raise ValueError(f"fold {self.name}: fit days overlap early-stopping/test days")
        if not self.es_is_test and set(self.es_days) & set(self.test_days):
            raise ValueError(f"fold {self.name}: early-stopping days overlap test days")


def _checked_days(days: Sequence[str]) -> list[str]:
    out = [str(d) for d in days]
    if out != sorted(set(out)):
        raise ValueError("days must be sorted and unique")
    return out


def _iso_week(day: str) -> tuple[int, int]:
    year, week, _ = pd.Timestamp(day).isocalendar()
    return int(year), int(week)


def rolling_origin_by_week(days: Sequence[str], min_train_weeks: int = 4) -> list[Fold]:
    """Fold k tests on ISO week k and trains on every earlier week; the last training week
    is the early-stopping set, the weeks before it are fitted."""
    if min_train_weeks < 2:
        raise ValueError("min_train_weeks must be >= 2 (one fit week + one early-stopping week)")
    weeks: dict[tuple[int, int], list[str]] = {}
    for day in _checked_days(days):
        weeks.setdefault(_iso_week(day), []).append(day)
    keys = list(weeks)
    folds = []
    for k in range(min_train_weeks, len(keys)):
        fit = tuple(d for key in keys[: k - 1] for d in weeks[key])
        year, week = keys[k]
        folds.append(
            Fold(f"{year}-W{week:02d}", fit, tuple(weeks[keys[k - 1]]), tuple(weeks[keys[k]]))
        )
    return folds


def day_grouped_kfold(
    days: Sequence[str], k: int = 5, seed: int = 42, es_fraction: float = 0.15
) -> list[Fold]:
    """Whole days shuffled into ``k`` test folds; per fold a seeded random ``es_fraction`` of
    the training days (at least one) is held out for early stopping."""
    days = _checked_days(days)
    if k < 2 or len(days) < k + 2:
        raise ValueError(f"need k >= 2 and at least k + 2 days, got k={k}, {len(days)} days")
    if not 0.0 < es_fraction < 1.0:
        raise ValueError("es_fraction must be in (0, 1)")
    order = np.random.default_rng(seed).permutation(len(days))
    folds = []
    for i, part in enumerate(np.array_split(order, k)):
        test = sorted(days[j] for j in part)
        held = set(test)
        train = [d for d in days if d not in held]
        n_es = min(len(train) - 1, max(1, round(es_fraction * len(train))))
        es = set(np.random.default_rng([seed, i]).choice(train, size=n_es, replace=False).tolist())
        fit = tuple(d for d in train if d not in es)
        folds.append(Fold(f"fold{i}", fit, tuple(sorted(es)), tuple(test)))
    return folds


def holdout(days: Sequence[str], test_from: str) -> list[Fold]:
    """The production frame's split: fit on days before ``test_from``, test on the rest, and
    early-stop *on the test days* (as ``stoptimes train run`` does) - a reproduction check,
    not an unbiased estimate."""
    days = _checked_days(days)
    fit = tuple(d for d in days if d < test_from)
    test = tuple(d for d in days if d >= test_from)
    return [Fold(f"holdout-{test_from}", fit, test, test, es_is_test=True)]


# ---------------------------------------------------------------------------
# Per-fold frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameCache:
    """Everything about a day that does not depend on the fitted assets."""

    records: Mapping[str, Sequence[TripRecord]]
    base: Mapping[str, pd.DataFrame]  # asset-free features + target + identity, per day
    segments: pd.DataFrame  # segments_from_arrivals over every cached day
    routes: Mapping[str, frozenset[str]]
    params: LiveFrameParams


def build_frame_cache(
    days: Sequence[str],
    load_records: Callable[[str], Sequence[TripRecord]],
    *,
    tz: str,
    params: LiveFrameParams,
) -> FrameCache:
    """Records, segments and asset-free feature rows of every day, computed once.

    Segments are per trip instance, so the segments of a subset of days are exactly the
    cached segments filtered to those days (which ``fit_out_of_fold`` does)."""
    records = {d: list(load_records(d)) for d in days}
    base = {}
    for day, recs in records.items():
        base[day] = build_day_frame(recs, LiveAssets(), params, day, "", features=BASE_FEATURES)
        _log.info("cached %s: %d rows", day, len(base[day]))
    tables = [arrivals_table(r) for r in records.values()]
    arrivals = pd.concat(tables, ignore_index=True) if tables else arrivals_table([])
    routes = {d: frozenset(r.ctx.route_id for r in recs) for d, recs in records.items()}
    return FrameCache(records, base, segments_from_arrivals(arrivals, tz), routes, params)


def _iso(day: str) -> str:
    return pd.Timestamp(day).date().isoformat()


def fold_frame(cache: FrameCache, fold: Fold) -> pd.DataFrame:
    """The fold's frame, ``split`` in {fit, es, test}; assets fitted on ``fold.fit_days`` only.

    With ``es_is_test`` the test rows double as the early-stopping set (``split='test'``)."""
    oof = fit_out_of_fold(cache.segments, [_iso(d) for d in fold.fit_days])
    vocabulary = RouteVocabulary.fit(sorted(set().union(*(cache.routes[d] for d in fold.fit_days))))
    plan = [(d, "fit") for d in fold.fit_days] + [(d, "test") for d in fold.test_days]
    if not fold.es_is_test:
        plan += [(d, "es") for d in fold.es_days]
    parts = []
    for day, split in plan:
        base = cache.base[day]
        if base.empty:
            continue
        stats = oof.for_day(_iso(day)) if split == "fit" else oof.full
        prior = build_day_frame(
            cache.records[day],
            LiveAssets(segment_stats=stats),
            cache.params,
            day,
            split,
            features=("prior_rem",),
        )
        _check_aligned(base, prior, day)
        codes = vocabulary.encode(base["route_id"].astype(str)).astype(np.float32)
        parts.append(
            base.assign(prior_rem=prior["prior_rem"].to_numpy(), route_code=codes, split=split)
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _check_aligned(base: pd.DataFrame, other: pd.DataFrame, day: str) -> None:
    same = len(base) == len(other) and all(
        np.array_equal(base[k].to_numpy(), other[k].to_numpy()) for k in _ALIGN_KEYS
    )
    if not same:
        raise RuntimeError(f"{day}: asset-dependent rows do not align with the cached rows")


# ---------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------


def _mae(err: np.ndarray) -> float:
    err = err[np.isfinite(err)]
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def score_fold(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """Model, timetable (``sched_rem``) and segment-history (``prior_rem``) errors on test rows."""
    y = test[TARGET_COLUMN].to_numpy(float)
    err = np.asarray(pred, float) - y
    prior = test["prior_rem"].to_numpy(float)
    out = {
        "model_mae": _mae(err),
        "model_medae": float(np.median(np.abs(err))),
        "model_bias": float(np.mean(err)),
        "sched_mae": _mae(test["sched_rem"].to_numpy(float) - y),
        "prior_mae": _mae(prior - y),
        "prior_coverage": float(np.isfinite(prior).mean()),
    }
    for name, lo, hi in HORIZON_BUCKETS_S:
        mask = (y >= lo) & (y < hi)
        out[f"model_mae_h{name}"] = _mae(err[mask])
        out[f"n_h{name}"] = float(mask.sum())
    return out


def _train_and_score(
    frame: pd.DataFrame, fold: Fold, features: list[str], params: dict[str, Any], seed: int
):
    from stoptimes.train.live_experiment import _fit

    fit = frame[frame["split"] == "fit"]
    test = frame[frame["split"] == "test"]
    es = test if fold.es_is_test else frame[frame["split"] == "es"]
    if fit.empty or es.empty or test.empty:
        raise ValueError(f"fold {fold.name}: empty fit/early-stopping/test rows")
    model = _fit(fit, es, features, params, seed)
    pred = np.asarray(model.predict(test[features]), dtype=float)
    row: dict[str, Any] = {
        "fold": fold.name,
        "test_start": fold.test_days[0],
        "test_end": fold.test_days[-1],
        "n_fit_days": len(fold.fit_days),
        "n_es_days": len(fold.es_days),
        "n_test_days": len(fold.test_days),
        "n_fit": len(fit),
        "n_es": len(es),
        "n_test": len(test),
        "best_iteration": int(model.best_iteration),
        **score_fold(test, pred),
    }
    y = test[TARGET_COLUMN].to_numpy(float)
    sums = {
        "model": (float(np.abs(pred - y).sum()), len(y)),
        "sched": _abs_sum(test["sched_rem"].to_numpy(float) - y),
        "prior": _abs_sum(test["prior_rem"].to_numpy(float) - y),
    }
    return row, sums


def _abs_sum(err: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(err)
    return float(np.abs(err[ok]).sum()), int(ok.sum())


def aggregate(
    rows: Sequence[Mapping[str, Any]], sums: Sequence[Mapping[str, tuple[float, int]]]
) -> dict[str, Any]:
    """Mean / std / min / max of each per-fold MAE, plus the pooled MAE over every test row."""
    table = pd.DataFrame(list(rows))
    out: dict[str, Any] = {"n_folds": len(table)}
    for col in [c for c in table.columns if "mae" in c]:
        values = table[col].astype(float)
        out[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for name in ("model", "sched", "prior"):
        total = sum(s[name][0] for s in sums)
        n = sum(s[name][1] for s in sums)
        out[f"{name}_mae"]["pooled"] = total / n if n else float("nan")
    return out


def run_cv(
    cache: FrameCache,
    folds: Sequence[Fold],
    features: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and score every fold; ``(per-fold table, aggregate)``."""
    unknown = [f for f in features if f not in LIVE_FEATURES]
    if unknown:
        raise ValueError(f"not live features: {unknown}")
    rows, sums = [], []
    for fold in folds:
        row, s = _train_and_score(fold_frame(cache, fold), fold, list(features), dict(params), seed)
        _log.info(
            "fold %s: model %.1f s, sched %.1f s, prior %.1f s, best_it %d",
            fold.name,
            row["model_mae"],
            row["sched_mae"],
            row["prior_mae"],
            row["best_iteration"],
        )
        rows.append(row)
        sums.append(s)
    return pd.DataFrame(rows), aggregate(rows, sums)


# ---------------------------------------------------------------------------
# Entry point: python -m stoptimes.train.live_cv
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m stoptimes.train.live_cv", description=__doc__.splitlines()[0]
    )
    p.add_argument("--scheme", choices=("rolling", "kfold", "holdout"), required=True)
    p.add_argument(
        "--experiment", required=True, help="experiment YAML (model.params, features, seed)"
    )
    p.add_argument("--observations", required=True, help="observation archive (archive_parquet)")
    p.add_argument("--static", required=True, help="GTFS static zip")
    p.add_argument(
        "--source", default="bucr", help="agency id (agency YAML: timezone, holiday region)"
    )
    p.add_argument("--sources-dir", default=None)
    p.add_argument("--days", required=True, help="START:END (YYYY-MM-DD, inclusive)")
    p.add_argument(
        "--all-days", action="store_true", help="keep weekends too (default: weekdays only)"
    )
    p.add_argument("--arrival-radius-m", type=float, default=None)
    p.add_argument("--rows-per-day", type=int, default=None)
    p.add_argument("--min-train-weeks", type=int, default=4, help="rolling")
    p.add_argument("--k", type=int, default=5, help="kfold")
    p.add_argument("--es-fraction", type=float, default=0.15, help="kfold")
    p.add_argument("--test-from", default=None, help="holdout: first test day, YYYY-MM-DD")
    p.add_argument("--params-json", default=None, help="JSON object merged over model.params")
    p.add_argument(
        "--n-jobs", type=int, default=None, help=f"xgboost threads (capped at {MAX_N_JOBS})"
    )
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def _load_source(source: str, sources_dir: str | None) -> Any:
    import os

    import yaml
    from stoptimes.core.config import sources_search_path
    from stoptimes.core.contracts.adapters import source_from_val_yaml

    dirs = (
        [Path(sources_dir)]
        if sources_dir
        else sources_search_path(os.environ.get("STOPTIMES_SOURCES"))
    )
    for d in dirs:
        path = Path(d) / f"{source}.yaml"
        if path.is_file():
            return source_from_val_yaml(yaml.safe_load(path.read_text(encoding="utf-8")))
    raise SystemExit(f"no agency file for {source!r} in {[str(d) for d in dirs]}")


def _requested_days(spec: str, weekdays_only: bool) -> list[str]:
    start, _, end = spec.partition(":")
    days = pd.date_range(start, end, freq="D")
    return [d.strftime("%Y%m%d") for d in days if not weekdays_only or d.dayofweek < 5]


def _model_params(experiment: Any, params_json: str | None, n_jobs: int | None) -> dict[str, Any]:
    params = dict(experiment.model.params)
    if params_json:
        params.update(json.loads(params_json))
    jobs = n_jobs if n_jobs is not None else params.get("n_jobs", 4)
    params["n_jobs"] = max(1, min(int(jobs), MAX_N_JOBS))
    return params


def _folds(args: argparse.Namespace, days: list[str], seed: int) -> list[Fold]:
    if args.scheme == "rolling":
        return rolling_origin_by_week(days, args.min_train_weeks)
    if args.scheme == "kfold":
        return day_grouped_kfold(days, args.k, seed, args.es_fraction)
    if not args.test_from:
        raise SystemExit("--scheme holdout needs --test-from")
    return holdout(days, pd.Timestamp(args.test_from).strftime("%Y%m%d"))


def _print_table(table: pd.DataFrame, summary: Mapping[str, Any]) -> None:
    cols = [
        "fold",
        "test_start",
        "test_end",
        "n_fit",
        "n_es",
        "n_test",
        "best_iteration",
        "model_mae",
        "sched_mae",
        "prior_mae",
        "model_medae",
        "model_bias",
    ]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(table[cols].to_string(index=False))
    for name in ("model_mae", "sched_mae", "prior_mae"):
        s = summary[name]
        print(
            f"{name:10s} mean {s['mean']:.1f} ± {s['std']:.1f}  [{s['min']:.1f}, {s['max']:.1f}]  "
            f"pooled {s['pooled']:.1f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    from stoptimes.core import readers
    from stoptimes.core.contracts.data import ObservationKind
    from stoptimes.core.contracts.experiment import load_experiment
    from stoptimes.train.live_frame import prepare_trip_records

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv)
    experiment = load_experiment(args.experiment)
    features = list(experiment.features) if experiment.features else list(LIVE_FEATURES)
    params = _model_params(experiment, args.params_json, args.n_jobs)
    src = _load_source(args.source, args.sources_dir)
    obs = readers.read_observations(
        args.observations,
        format="archive_parquet",
        kind=ObservationKind.VEHICLE_POSITION,
        source_id=src.id,
    )
    loader, feed = readers.read_static(args.static, src.id)
    if loader is None:
        raise SystemExit(f"{args.static}: missing static tables {feed.missing_required()}")
    frame_params = LiveFrameParams(
        rows_per_day=args.rows_per_day, seed=experiment.seed, arrival_radius_m=args.arrival_radius_m
    )
    requested = _requested_days(args.days, weekdays_only=not args.all_days)
    by_day, skipped = prepare_trip_records(
        obs,
        loader,
        config_timezone=src.timezone,
        holiday_region=src.holiday_region,
        params=frame_params,
        service_dates=requested,
    )
    days = [d for d in requested if by_day.get(d)]
    _log.info(
        "%d of %d requested days have trips (skipped: %s)", len(days), len(requested), skipped
    )
    folds = _folds(args, days, experiment.seed)
    tz = loader.agency_timezone() or src.timezone
    cache = build_frame_cache(days, lambda d: by_day[d], tz=tz, params=frame_params)
    table, summary = run_cv(cache, folds, features, params, experiment.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "folds.csv", index=False)
    summary = {
        **summary,
        "scheme": args.scheme,
        "experiment": experiment.name,
        "features": features,
        "params": params,
        "seed": experiment.seed,
        "days": [days[0], days[-1], len(days)],
        "frame_params": asdict(frame_params),
        "folds": [asdict(f) for f in folds],
        "assets_fitted_on": "fit days (fit rows out-of-fold by day; es, test rows all fit days)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _print_table(table, summary)


if __name__ == "__main__":
    main()
