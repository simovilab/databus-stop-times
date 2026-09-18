"""Run an experiment.yaml end to end: dataset -> features -> fit -> artifact + card.

Phase 2, track T3 (docs/PLAN.md). `run_experiment` is the one function `stoptimes
train run` calls; everything else here is a piece of it, exposed separately so
`--dry-run` and tests can exercise the plan/split/score steps without a full run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import (
    ArtifactMetadata,
    DataWindow,
    HorizonScore,
    ModelScope,
    Provenance,
    ScopeLevel,
    artifact_dir,
    write_metadata,
)
from stoptimes.core.contracts.experiment import DataSelection, Experiment, SplitStrategy
from stoptimes.core.features import registry as feature_registry

# Legacy datasets (stoptimes.models.common.data.ETADataset) carry the target
# under this name; contracts.experiment.Experiment.target defaults to
# "arrival_delay_s", which no shipped dataset has yet. See _resolve_target_column.
_LEGACY_TARGET_COLUMN = "time_to_arrival_seconds"

# Horizon buckets for scores_by_horizon: seconds-to-arrival at prediction
# time (the target itself, for a row-level ETA dataset), same edges as
# stoptimes.train.evaluation.stats's default horizon_bins.
HORIZON_BINS_S: tuple[float, ...] = (0, 60, 180, 360, 600, float("inf"))

_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")


class ArtifactExistsError(FileExistsError):
    """Raised instead of silently overwriting an existing artifact directory."""


def _sanitize_artifact_id(name: str) -> str:
    if _ARTIFACT_ID_PATTERN.match(name):
        return name
    sanitized = re.sub(r"[^A-Za-z0-9._=-]", "-", name).lstrip("._=-") or "experiment"
    if not sanitized[0].isalnum():
        sanitized = f"a-{sanitized}"
    return sanitized


def resolve_dataset(selection: DataSelection, *, dataset_dir: str | Path | None = None):
    """Turn an `Experiment.data` selection into a loaded `ETADataset`.

    **Seam**: T1's local data catalog (`core/catalog/**`) doesn't exist yet.
    `selection.datasets` (named, already-prepared parquet datasets) is
    resolved today via `stoptimes.models.common.data.load_dataset` — the
    same lookup every model family's own `train_*` function uses.
    `selection.sources` (raw agency/feed ids, meant to route through the
    catalog once T1 builds it) has no resolver yet: this raises
    `NotImplementedError` naming exactly what's missing, rather than
    guessing at a dataset, so T1 has one function to extend
    (`docs/notes/p2-t3.md` has the details).
    """
    from stoptimes.models.common.data import load_dataset

    if selection.datasets:
        if len(selection.datasets) > 1:
            raise NotImplementedError(
                f"resolve_dataset: only one dataset name is supported today, got {selection.datasets!r}"
            )
        return load_dataset(selection.datasets[0], dataset_dir=dataset_dir)

    raise NotImplementedError(
        f"resolve_dataset: DataSelection.sources={selection.sources!r} has no resolver yet "
        "-- T1's data catalog isn't built. Pass `data.datasets: [<name>]` naming a prepared "
        "parquet dataset instead (see stoptimes.models.common.data.load_dataset), or extend "
        "resolve_dataset() once catalog lookup exists."
    )


def _resolve_target_column(target: str, columns: pd.Index) -> str:
    if target in columns:
        return target
    if target == "arrival_delay_s" and _LEGACY_TARGET_COLUMN in columns:
        return _LEGACY_TARGET_COLUMN
    raise KeyError(
        f"target column {target!r} not found in dataset columns {sorted(columns)} "
        f"(tried legacy column {_LEGACY_TARGET_COLUMN!r} too)"
    )


@dataclass
class SplitOutcome:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    notes: list[str]


def split_dataset(dataset, split: SplitStrategy, *, seed: int) -> SplitOutcome:
    """Apply an `Experiment.split` strategy to a loaded `ETADataset`.

    - "time": `ETADataset.temporal_split`, no group purging.
    - "trip": `ETADataset.temporal_split` with `purge=True` on `trip_id`,
      using MBTA's service-day offset (7h) as the default -- see the note in
      `ETADataset.temporal_split` about why an offset is required; no
      per-agency config exists yet at this layer (`docs/notes/p2-t3.md`).
    - "kfold": one held-out fold (the last `1/n_folds` of rows, time-ordered)
      is the test set, the rest is train. A true k-fold *rotation* would
      train `n_folds` models; `contracts.experiment` is explicit that one
      `Experiment` trains one model, so this reports one fold's split and
      notes the simplification.
    """
    notes: list[str] = []
    if split.kind == "kfold":
        n = len(dataset.df)
        assert split.n_folds is not None
        fold_size = max(1, n // split.n_folds)
        df_sorted = dataset.df.sort_values("vp_ts").reset_index(drop=True)
        train_df = df_sorted.iloc[: n - fold_size].copy()
        test_df = df_sorted.iloc[n - fold_size :].copy()
        notes.append(
            f"kfold split: trained on one held-out fold (last 1/{split.n_folds} of rows, "
            "time-ordered) rather than a full k-fold rotation -- one Experiment trains one model."
        )
        return SplitOutcome(train_df, test_df, notes)

    train_frac = split.train_fraction
    test_frac = split.test_fraction
    assert train_frac is not None and test_frac is not None
    # `ETADataset.temporal_split` (-> `splits.fraction_split`) requires
    # val_frac strictly in (0, 1) -- it always produces a 3-way split. A
    # 2-way Experiment split (the common case: train_fraction + test_fraction
    # == 1.0) still needs a small, nonzero val slice to satisfy that; it's
    # folded back into train immediately below.
    remainder = 1.0 - train_frac - test_frac
    val_frac = remainder if remainder > 1e-9 else min(0.05, max(1.0 - train_frac, 1e-6))

    if split.kind == "trip":
        service_offset_h = 7
        notes.append(f"trip split: purged on trip_id with service_offset_h={service_offset_h} (MBTA default).")
        train_df, val_df, test_df = dataset.temporal_split(
            train_frac=train_frac,
            val_frac=val_frac,
            purge=True,
            service_offset_h=service_offset_h,
        )
    else:
        train_df, val_df, test_df = dataset.temporal_split(train_frac=train_frac, val_frac=val_frac)

    if len(val_df):
        # Model contract has no separate validation split; fold it into train
        # (LSTM's adapter carves its own validation slice back out of X/y).
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        notes.append(f"folded {len(val_df)} validation rows into train (Model contract has no separate val split).")

    return SplitOutcome(train_df, test_df, notes)


def _horizon_scores(y_true: np.ndarray, y_pred: np.ndarray) -> list[HorizonScore]:
    scores: list[HorizonScore] = []
    edges = HORIZON_BINS_S
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_true >= lo) & (y_true < hi) if np.isfinite(hi) else (y_true >= lo)
        n = int(mask.sum())
        if n == 0:
            continue
        err = np.abs(y_true[mask] - y_pred[mask])
        scores.append(
            HorizonScore(
                horizon_min_s=int(lo),
                horizon_max_s=int(hi) if np.isfinite(hi) else None,
                n=n,
                mae_s=float(err.mean()),
                rmse_s=float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))),
                median_ae_s=float(np.median(err)),
                p90_ae_s=float(np.percentile(err, 90)),
            )
        )
    return scores


def plan_experiment(experiment: Experiment) -> dict[str, Any]:
    """The resolved plan `stoptimes train run --dry-run` prints: data, features,
    split, model, baselines and outputs, without touching any data or model
    library. Pure and side-effect-free."""
    return {
        "name": experiment.name,
        "target": experiment.target,
        "data": {
            "sources": list(experiment.data.sources),
            "datasets": list(experiment.data.datasets),
            "start_date": experiment.data.start_date.isoformat() if experiment.data.start_date else None,
            "end_date": experiment.data.end_date.isoformat() if experiment.data.end_date else None,
            "routes": experiment.data.routes,
        },
        "features": list(experiment.features),
        "split": {
            "kind": experiment.split.kind,
            "train_fraction": experiment.split.train_fraction,
            "test_fraction": experiment.split.test_fraction,
            "n_folds": experiment.split.n_folds,
        },
        "model": {"family": experiment.model.family, "params": dict(experiment.model.params)},
        "baselines": list(experiment.baselines),
        "seed": experiment.seed,
        "outputs": {"registry_dir": experiment.outputs.registry_dir, "tags": list(experiment.outputs.tags)},
    }


def run_experiment(
    experiment: Experiment,
    *,
    registry_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    overwrite: bool = False,
) -> ArtifactMetadata:
    """Run `experiment` end to end and write its artifact.

    Steps: resolve the dataset (`resolve_dataset`), filter to
    `data.routes` if given, build `features` through the feature registry,
    split (`split_dataset`), fit `model` plus every name in `baselines`
    (scored but not written as separate artifacts — their numbers go in the
    card, matching docs/PLAN.md rule 5), score overall and by horizon, then
    write `<registry_dir>/<artifact_id>/{artifact.json,model.<ext>,card.md}`.

    Deterministic given `experiment.seed`: the split is a pure function of
    the (sorted) data and the fractions/folds, and every family/baseline is
    constructed fresh per run. Refuses to overwrite an existing artifact
    directory unless `overwrite=True`.
    """
    from stoptimes.models.common.contract_adapter import get_model_class
    from stoptimes.models.common.metrics import compute_all_metrics
    from stoptimes.models.common.provenance import git_sha, lib_versions

    registry_root = Path(registry_dir) if registry_dir is not None else Path(experiment.outputs.registry_dir)
    artifact_id = _sanitize_artifact_id(experiment.name)
    target_dir = artifact_dir(registry_root, artifact_id)
    if target_dir.exists() and not overwrite:
        raise ArtifactExistsError(
            f"artifact {artifact_id!r} already exists at {target_dir} (pass overwrite=True / --overwrite to replace)"
        )

    dataset = resolve_dataset(experiment.data, dataset_dir=dataset_dir)
    dataset.clean_data()

    if experiment.data.routes:
        dataset.df = dataset.df[dataset.df["route_id"].isin(experiment.data.routes)].copy()

    target_col = _resolve_target_column(experiment.target, dataset.df.columns)

    if experiment.features:
        dataset.df = feature_registry.build(experiment.features, dataset.df)

    split_outcome = split_dataset(dataset, experiment.split, seed=experiment.seed)
    train_df, test_df = split_outcome.train_df, split_outcome.test_df

    feature_cols = list(experiment.features) if experiment.features else [
        c for c in train_df.columns if c != target_col
    ]
    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy(dtype=float)
    X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy(dtype=float)

    model_cls = get_model_class(experiment.model.family)
    params = {k: v for k, v in experiment.model.params.items() if isinstance(v, (int, float, str, bool))}
    model = model_cls(**params)
    fitted = model.fit(X_train, y_train, seed=experiment.seed)
    preds_test = fitted.predict(X_test)
    metrics = compute_all_metrics(y_test, preds_test, prefix="test_")

    baseline_metrics: dict[str, dict[str, float]] = {}
    for baseline_name in experiment.baselines:
        baseline_cls = get_model_class(baseline_name)
        baseline = baseline_cls()
        baseline.fit(train_df, y_train)
        baseline_preds = baseline.predict(test_df)
        baseline_metrics[baseline_name] = compute_all_metrics(y_test, baseline_preds, prefix="test_")

    scope = ModelScope(
        level=ScopeLevel.ROUTE if experiment.data.routes else ScopeLevel.GLOBAL,
        source_id=experiment.data.sources[0] if experiment.data.sources else None,
        route_ids=list(experiment.data.routes) if experiment.data.routes else [],
    )

    versions = lib_versions()
    provenance = Provenance(
        created_at=datetime.now(timezone.utc),
        git_sha=git_sha(),
        python_version=versions.get("python"),
        lib_versions=versions,
        seed=experiment.seed,
    )

    dataset_name = experiment.data.datasets[0] if experiment.data.datasets else None
    data_window = DataWindow(
        dataset=dataset_name,
        sources=list(experiment.data.sources),
        start=datetime.combine(experiment.data.start_date, datetime.min.time(), tzinfo=timezone.utc)
        if experiment.data.start_date
        else None,
        end=datetime.combine(experiment.data.end_date, datetime.min.time(), tzinfo=timezone.utc)
        if experiment.data.end_date
        else None,
        n_samples=len(train_df) + len(test_df),
        n_train=len(train_df),
        n_test=len(test_df),
    )

    meta = ArtifactMetadata(
        artifact_id=artifact_id,
        family=experiment.model.family,
        target=experiment.target,
        scope=scope,
        features=feature_cols,
        params=dict(experiment.model.params),
        data=data_window,
        provenance=provenance,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        scores_by_horizon=_horizon_scores(y_test, np.asarray(preds_test, dtype=float)),
        tags=list(experiment.outputs.tags),
    )

    _write_artifact(fitted, meta, registry_root, target_dir, split_outcome.notes, baseline_metrics)
    return meta


def _write_artifact(
    model: Any,
    meta: ArtifactMetadata,
    registry_root: Path,
    target_dir: Path,
    split_notes: list[str],
    baseline_metrics: dict[str, dict[str, float]],
) -> None:
    import shutil
    import tempfile

    registry_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=registry_root, prefix=f".train-{meta.artifact_id}-"))
    try:
        model.save(tmp_dir)
        write_metadata(meta, tmp_dir)
        (tmp_dir / "card.md").write_text(
            render_card(meta, baseline_metrics=baseline_metrics, notes=split_notes), encoding="utf-8"
        )

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        import os

        os.rename(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def render_card(
    meta: ArtifactMetadata,
    *,
    baseline_metrics: dict[str, dict[str, float]] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Render `card.md` from an `ArtifactMetadata` (+ baseline comparison
    numbers, which don't live in the contract itself)."""
    baseline_metrics = baseline_metrics or {}
    lines = [
        f"# {meta.artifact_id}",
        "",
        f"- **family**: `{meta.family}`",
        f"- **target**: `{meta.target}`",
        f"- **scope**: {meta.scope.level.value}"
        + (f" (routes: {', '.join(meta.scope.route_ids)})" if meta.scope.route_ids else ""),
        f"- **created**: {meta.provenance.created_at.isoformat()}",
        f"- **git sha**: {meta.provenance.git_sha or 'unknown'}",
        f"- **seed**: {meta.provenance.seed}",
        "",
        "## Data",
        "",
        f"- dataset: {meta.data.dataset or '(unnamed)'}",
        f"- sources: {', '.join(meta.data.sources) or '(none)'}",
        f"- samples: {meta.data.n_samples} (train {meta.data.n_train}, test {meta.data.n_test})",
        "",
        "## Features",
        "",
        *(
            [f"- `{f}`" for f in meta.features]
            if meta.features
            else ["(none — model saw every non-target column)"]
        ),
        "",
        "## Params",
        "",
        "```",
        *(f"{k}: {v!r}" for k, v in meta.params.items()),
        "```",
        "",
        "## Scores",
        "",
        "| model | test_mae_seconds | test_rmse_seconds | test_r2 |",
        "|---|---|---|---|",
        _score_row(meta.family, meta.metrics),
        *(_score_row(name, m) for name, m in baseline_metrics.items()),
        "",
        "## Scores by horizon",
        "",
        "| horizon (s) | n | mae_s | rmse_s | median_ae_s | p90_ae_s |",
        "|---|---|---|---|---|---|",
        *(_horizon_row(h) for h in meta.scores_by_horizon),
    ]
    if notes:
        lines += ["", "## Notes", "", *(f"- {n}" for n in notes)]
    return "\n".join(lines) + "\n"


def _horizon_row(h: HorizonScore) -> str:
    def fmt(value: float | None) -> str:
        return f"{value:.1f}" if isinstance(value, (int, float)) else ""

    upper = str(h.horizon_max_s) if h.horizon_max_s is not None else "+"
    return (
        f"| {h.horizon_min_s}-{upper} | {h.n} | {fmt(h.mae_s)} | {fmt(h.rmse_s)} "
        f"| {fmt(h.median_ae_s)} | {fmt(h.p90_ae_s)} |"
    )


def _score_row(name: str, metrics: dict[str, float | None]) -> str:
    def fmt(key: str) -> str:
        value = metrics.get(key)
        return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"

    return f"| {name} | {fmt('test_mae_seconds')} | {fmt('test_rmse_seconds')} | {fmt('test_r2')} |"
