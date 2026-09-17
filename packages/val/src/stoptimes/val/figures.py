"""Publication figures for validation results. A9.

Headless matplotlib (Agg backend, set at import — no display needed) using a
colorblind-safe Okabe-Ito-derived palette, 300 dpi PNG output, no seaborn.

Public API
----------
error_vs_horizon(df, out_path)          -> Path | None
error_distribution(df, out_path)        -> Path | None
convergence_curve(df, out_path)         -> Path | None
per_route_mae(df, out_path)             -> Path | None
calibration_diagram(df, out_path)       -> Path | None
cost_accuracy_pareto(df, latency, out_path) -> Path | None  (None w/o latency)
load_latency_from_summary(summary_path) -> dict[str, dict[str, float]]
render_all_figures(results_path, out_dir, *, summary_path=None) -> dict[str, Path | None]

Every figure function takes an already-loaded results DataFrame (the schema
written by ``etaval.storage.parquet_export.export_results`` /
``read_results_df`` — one row per ValidationResult) and an output path; it
writes a PNG and returns the path, or ``None`` when there is nothing to plot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etaval.stats.bootstrap import bootstrap_ci  # noqa: E402
from etaval.stats.calibration import calibration_by_horizon  # noqa: E402

# ── Okabe-Ito colorblind-safe palette (black held back as a neutral/last) ────
_PALETTE: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

_DPI = 300

_BUCKET_LABELS: list[str] = ["<2m", "2-5m", "5-10m", ">10m"]


def _color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def _assign_bucket(horizon_s: float) -> str:
    if horizon_s < 120.0:
        return "<2m"
    if horizon_s < 300.0:
        return "2-5m"
    if horizon_s < 600.0:
        return "5-10m"
    return ">10m"


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=_DPI)
    plt.close(fig)
    return p


# ── error_vs_horizon ──────────────────────────────────────────────────────────


def error_vs_horizon(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Mean |error| vs horizon bucket, one line per predictor, with a bootstrap CI band."""
    if df.empty or "predictor_id" not in df.columns:
        return None

    work = df.copy()
    work["abs_error"] = work["error_s"].abs()
    work["bucket"] = work["horizon_s"].apply(_assign_bucket)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for i, (pid, grp) in enumerate(work.groupby("predictor_id")):
        xs: list[int] = []
        means: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for j, label in enumerate(_BUCKET_LABELS):
            bgrp = grp[grp["bucket"] == label]
            if bgrp.empty:
                continue
            ci_res = bootstrap_ci(bgrp["abs_error"].to_numpy())
            xs.append(j)
            means.append(ci_res.point)
            los.append(ci_res.lo)
            his.append(ci_res.hi)
        if not xs:
            continue
        plotted = True
        color = _color(i)
        ax.plot(xs, means, marker="o", color=color, label=str(pid))
        ax.fill_between(xs, los, his, color=color, alpha=0.2, linewidth=0)

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xticks(range(len(_BUCKET_LABELS)))
    ax.set_xticklabels(_BUCKET_LABELS)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Mean |error| (s)")
    ax.set_title("Error vs Horizon")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


# ── error_distribution ────────────────────────────────────────────────────────


def error_distribution(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Per-predictor absolute-error distribution (boxplot)."""
    if df.empty or "predictor_id" not in df.columns:
        return None

    work = df.copy()
    work["abs_error"] = work["error_s"].abs()
    predictor_ids = sorted(work["predictor_id"].astype(str).unique())
    if not predictor_ids:
        return None
    data = [work.loc[work["predictor_id"].astype(str) == pid, "abs_error"].to_numpy()
            for pid in predictor_ids]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=predictor_ids, patch_artist=True, showfliers=False)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_color(i))
        box.set_alpha(0.6)

    ax.set_ylabel("|error| (s)")
    ax.set_title("Error Distribution by Predictor")
    fig.tight_layout()
    return _save(fig, out_path)


# ── convergence_curve ─────────────────────────────────────────────────────────


def convergence_curve(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Mean |error| vs time-to-arrival (horizon), one line per predictor.

    Unlike :func:`error_vs_horizon`'s fixed buckets, this bins horizon_s into
    decile bands per predictor to show the continuous convergence shape as
    horizon shrinks toward zero.
    """
    if df.empty or "predictor_id" not in df.columns or "horizon_s" not in df.columns:
        return None

    work = df.copy()
    work["abs_error"] = work["error_s"].abs()

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for i, (pid, grp) in enumerate(work.groupby("predictor_id")):
        if len(grp) < 2:
            continue
        n_bins = min(10, grp["horizon_s"].nunique())
        if n_bins < 2:
            continue
        try:
            binned = pd.qcut(grp["horizon_s"], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        agg = grp.groupby(binned, observed=True).agg(
            horizon=("horizon_s", "mean"), abs_error=("abs_error", "mean")
        ).sort_values("horizon")
        if agg.empty:
            continue
        plotted = True
        color = _color(i)
        ax.plot(agg["horizon"], agg["abs_error"], marker="o", color=color, label=str(pid))

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Horizon (s, time-to-arrival)")
    ax.set_ylabel("Mean |error| (s)")
    ax.set_title("Convergence Curve")
    ax.invert_xaxis()  # convergence reads left (far) -> right (near)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


# ── per_route_mae ──────────────────────────────────────────────────────────────


def per_route_mae(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Grouped bar chart of MAE by route, one bar group per route, bars per predictor."""
    if df.empty or "route_id" not in df.columns:
        return None

    work = df[df["route_id"].notna()].copy()
    if work.empty:
        return None
    work["abs_error"] = work["error_s"].abs()

    pivot = work.groupby(["route_id", "predictor_id"])["abs_error"].mean().unstack("predictor_id")
    if pivot.empty:
        return None
    pivot = pivot.sort_index()

    routes = [str(r) for r in pivot.index]
    predictor_ids = [str(c) for c in pivot.columns]
    n_pred = len(predictor_ids)
    x = np.arange(len(routes))
    width = 0.8 / max(n_pred, 1)

    fig, ax = plt.subplots(figsize=(max(7, len(routes) * 0.6), 5))
    for i, pid in enumerate(predictor_ids):
        values = pivot[pid].to_numpy(dtype=np.float64)
        offset = (i - (n_pred - 1) / 2.0) * width
        ax.bar(x + offset, np.nan_to_num(values, nan=0.0), width=width, color=_color(i), label=pid)

    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=45, ha="right")
    ax.set_ylabel("MAE (s)")
    ax.set_title("MAE by Route")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


# ── calibration_diagram ───────────────────────────────────────────────────────


def calibration_diagram(df: pd.DataFrame, out_path: str | Path) -> Path | None:
    """Nominal vs empirical coverage by horizon band (per predictor)."""
    required = {"predicted_arrival_epoch", "actual_arrival_epoch", "horizon_s", "predictor_id"}
    if df.empty or not required.issubset(df.columns):
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for i, (pid, grp) in enumerate(df.groupby("predictor_id")):
        bins = calibration_by_horizon(
            grp["predicted_arrival_epoch"].to_numpy(),
            grp["actual_arrival_epoch"].to_numpy(),
            grp["horizon_s"].to_numpy(),
        )
        if not bins:
            continue
        nominal = [b.nominal_coverage for b in bins]
        empirical = [b.empirical_coverage for b in bins]
        plotted = True
        ax.scatter(nominal, empirical, color=_color(i), label=str(pid), s=40)

    if not plotted:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration Diagram")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


# ── cost_accuracy_pareto ──────────────────────────────────────────────────────


def cost_accuracy_pareto(
    df: pd.DataFrame,
    latency_by_predictor: dict[str, dict[str, Any]] | None,
    out_path: str | Path,
) -> Path | None:
    """Latency (cost, p50_s) vs MAE (accuracy) scatter — a cost/accuracy Pareto view.

    Returns ``None`` when *latency_by_predictor* is falsy (no latency data
    available) — there is nothing meaningful to plot on the cost axis.
    """
    if not latency_by_predictor:
        return None
    if df.empty or "predictor_id" not in df.columns:
        return None

    work = df.copy()
    work["abs_error"] = work["error_s"].abs()
    mae_by_predictor = work.groupby("predictor_id")["abs_error"].mean().to_dict()

    points: list[tuple[str, float, float]] = []
    for pid, stats in latency_by_predictor.items():
        if pid not in mae_by_predictor:
            continue
        p50 = stats.get("p50_s")
        if p50 is None:
            continue
        points.append((str(pid), float(p50), float(mae_by_predictor[pid])))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (pid, latency, mae) in enumerate(points):
        ax.scatter(latency, mae, color=_color(i), s=80, label=pid)
        ax.annotate(pid, (latency, mae), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Inference latency p50 (s)")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Cost / Accuracy Pareto")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


def load_latency_from_summary(summary_path: str | Path) -> dict[str, dict[str, float]]:
    """Read the ``inference_latency`` block from a ``<...>.summary.json`` sidecar.

    The sidecar is written by the CLI (``etaval run --out <path>``) as
    ``RunSummary.to_dict()`` — see ``etaval.cli.main._write_sidecars_if_requested``.

    Returns:
        ``{predictor_id: {"count": ..., "p50_s": ..., "p95_s": ..., "p99_s": ...}}``,
        or an empty dict if the file has no ``inference_latency`` block.
    """
    data = json.loads(Path(summary_path).read_text())
    return dict(data.get("inference_latency") or {})


# ── Entrypoint ─────────────────────────────────────────────────────────────────

_FIGURE_BUILDERS: dict[str, Any] = {
    "error_vs_horizon": error_vs_horizon,
    "error_distribution": error_distribution,
    "convergence_curve": convergence_curve,
    "per_route_mae": per_route_mae,
    "calibration_diagram": calibration_diagram,
}


def render_all_figures(
    results_path: str | Path,
    out_dir: str | Path,
    *,
    summary_path: str | Path | None = None,
) -> dict[str, Path | None]:
    """Load a results Parquet file and write all applicable figures under *out_dir*.

    Args:
        results_path: Parquet file written by ``etaval.storage.parquet_export.
            export_results`` (one row per ValidationResult).
        out_dir: Destination directory (created if missing); each figure is
            written as ``<name>.png``.
        summary_path: Optional ``<...>.summary.json`` sidecar; when given, its
            inference-latency block enables ``cost_accuracy_pareto``.

    Returns:
        ``{figure_name: Path | None}`` — ``None`` for any figure with nothing
        to plot (including ``cost_accuracy_pareto`` when *summary_path* is
        omitted or carries no latency data).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(Path(results_path))

    written: dict[str, Path | None] = {}
    for name, builder in _FIGURE_BUILDERS.items():
        written[name] = builder(df, out_dir / f"{name}.png")

    latency: dict[str, dict[str, float]] = {}
    if summary_path is not None:
        latency = load_latency_from_summary(summary_path)
    written["cost_accuracy_pareto"] = cost_accuracy_pareto(
        df, latency, out_dir / "cost_accuracy_pareto.png"
    )
    return written
