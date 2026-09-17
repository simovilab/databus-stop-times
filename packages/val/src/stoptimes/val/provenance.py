"""Run provenance + inference-latency tracking.

Mirrors the spirit of gtfs-eta's ``models/common/provenance.py``: small,
dependency-light helpers that record enough about a run to reproduce it
later — git commits (etaval's own, plus the sibling gtfs-eta checkout it
may have pulled trained models from), the source config actually used, the
ground-truth detector thresholds, the predictor IDs, an optional seed, and
key package versions.

Every helper here is defensive: a missing git binary, a repo that isn't a
git checkout, a missing/uninstalled package, or any other failure must
never raise — everything degrades to ``None`` (or an empty/partial
structure) so that building provenance never fails a run.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_GIT_TIMEOUT_S = 10


# ── git helpers (defensive; never raise) ────────────────────────────────────


def _git_rev_parse_head(repo_dir: Path) -> str | None:
    """``git rev-parse HEAD`` in *repo_dir*. None on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
        if result.returncode != 0:
            return None
        sha = result.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


def _git_is_dirty(repo_dir: Path) -> bool | None:
    """True if *repo_dir* has uncommitted changes. None if undeterminable."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def _etaval_repo_root() -> Path | None:
    """Best-effort root of the etaval source checkout containing this file.

    Only meaningful when running from source (editable install / pytest);
    in a non-editable installed wheel this simply won't be a git repo and
    the git helpers above degrade to None, which is fine — there is no
    other sound anchor for "etaval's own git sha" from an installed wheel.
    """
    try:
        return Path(__file__).resolve().parent.parent
    except Exception:  # noqa: BLE001
        return None


def _gtfs_eta_repo_root() -> Path | None:
    """Locate the gtfs-eta source checkout via the same resolution chain
    ``GtfsEtaModelPredictor._resolve_model_registry_dir`` uses (a verified
    ``models/trained`` dir), then walk up two levels to the repo root.

    Defensive: any import failure or resolution failure degrades to None.
    """
    try:
        from stoptimes.val.predictors.gtfs_eta_models import GtfsEtaModelPredictor

        registry_dir = GtfsEtaModelPredictor._resolve_model_registry_dir()
    except Exception:  # noqa: BLE001
        return None
    if registry_dir is None:
        return None
    try:
        # registry_dir is .../gtfs-eta/models/trained
        return registry_dir.parent.parent
    except Exception:  # noqa: BLE001
        return None


def _versions() -> dict[str, str | None]:
    """Versions of key packages, plus the running Python version.

    Each package is looked up individually and guarded — an uninstalled
    or unresolvable package records None rather than raising.
    """
    import sys

    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for pkg_name in ("etaval", "gtfs-eta", "numpy", "pandas", "scipy", "fastapi"):
        versions[pkg_name] = _safe_pkg_version(pkg_name)
    return versions


def _safe_pkg_version(pkg_name: str) -> str | None:
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:  # noqa: BLE001
        return None


def _sha256_of_json_obj(obj: Any) -> str | None:
    """Stable sha256 hex digest of a JSON-serializable object.

    Uses ``sort_keys=True`` so field order never affects the digest.
    None if *obj* is not JSON-serializable.
    """
    try:
        import hashlib

        payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _source_config_to_dict(source_config: Any) -> dict[str, Any]:
    """Best-effort conversion of a SourceConfig (or dict) to a plain dict."""
    if isinstance(source_config, dict):
        return dict(source_config)
    try:
        # pydantic BaseModel
        return source_config.model_dump()
    except Exception:  # noqa: BLE001
        pass
    try:
        return dict(vars(source_config))
    except Exception:  # noqa: BLE001
        return {}


def _detectors_to_dict(
    detectors: list[Any] | dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize *detectors* into ``method_id -> thresholds dict``.

    Accepts either an already-built mapping, or a list of detector
    instances (duck-typed: ``.method`` for the id, remaining public
    non-callable attributes as thresholds). Defensive per-item — a
    detector that can't be introspected is simply omitted.
    """
    if isinstance(detectors, dict):
        return {str(k): dict(v) for k, v in detectors.items()}

    result: dict[str, dict[str, Any]] = {}
    for detector in detectors or []:
        try:
            method = getattr(detector, "method", None) or type(detector).__name__
            thresholds = {
                k: v
                for k, v in vars(detector).items()
                if not k.startswith("_") and not callable(v)
            }
            result[str(method)] = thresholds
        except Exception:  # noqa: BLE001
            continue
    return result


# ── RunProvenance ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunProvenance:
    """Everything needed to reproduce (or at least explain) one validation run."""

    etaval_git_sha: str | None
    etaval_dirty: bool | None
    gtfs_eta_git_sha: str | None
    gtfs_eta_dirty: bool | None
    source_name: str
    source_config_sha256: str | None
    gtfs_snapshot: str | None
    detectors: dict[str, dict[str, Any]]
    seed: int | None
    predictor_ids: list[str]
    versions: dict[str, str | None]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation for persistence / API / CLI serialization."""
        return {
            "etaval_git_sha": self.etaval_git_sha,
            "etaval_dirty": self.etaval_dirty,
            "gtfs_eta_git_sha": self.gtfs_eta_git_sha,
            "gtfs_eta_dirty": self.gtfs_eta_dirty,
            "source_name": self.source_name,
            "source_config_sha256": self.source_config_sha256,
            "gtfs_snapshot": self.gtfs_snapshot,
            "detectors": {k: dict(v) for k, v in self.detectors.items()},
            "seed": self.seed,
            "predictor_ids": list(self.predictor_ids),
            "versions": dict(self.versions),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunProvenance:
        """Reconstruct a RunProvenance from :meth:`to_dict` output."""
        return cls(
            etaval_git_sha=data.get("etaval_git_sha"),
            etaval_dirty=data.get("etaval_dirty"),
            gtfs_eta_git_sha=data.get("gtfs_eta_git_sha"),
            gtfs_eta_dirty=data.get("gtfs_eta_dirty"),
            source_name=data.get("source_name", ""),
            source_config_sha256=data.get("source_config_sha256"),
            gtfs_snapshot=data.get("gtfs_snapshot"),
            detectors={
                k: dict(v) for k, v in (data.get("detectors") or {}).items()
            },
            seed=data.get("seed"),
            predictor_ids=list(data.get("predictor_ids") or []),
            versions=dict(data.get("versions") or {}),
            created_at=data.get("created_at", ""),
        )


def build_run_provenance(
    *,
    source_name: str,
    source_config: Any,
    detectors: list[Any] | dict[str, dict[str, Any]],
    predictor_ids: list[str],
    seed: int | None,
    gtfs_snapshot: str | None = None,
) -> RunProvenance:
    """Assemble a :class:`RunProvenance` for the run about to start.

    Every sub-lookup is individually defensive; a failure in any one of
    them (no git binary, gtfs-eta checkout unresolvable, config not
    serializable, ...) degrades that field to ``None`` rather than
    raising and aborting the run.
    """
    etaval_root = _etaval_repo_root()
    etaval_sha = _git_rev_parse_head(etaval_root) if etaval_root is not None else None
    etaval_dirty = _git_is_dirty(etaval_root) if etaval_root is not None else None

    gtfs_eta_root = _gtfs_eta_repo_root()
    gtfs_eta_sha = (
        _git_rev_parse_head(gtfs_eta_root) if gtfs_eta_root is not None else None
    )
    gtfs_eta_dirty = (
        _git_is_dirty(gtfs_eta_root) if gtfs_eta_root is not None else None
    )

    config_dict = _source_config_to_dict(source_config)
    config_sha256 = _sha256_of_json_obj(config_dict) if config_dict else None

    return RunProvenance(
        etaval_git_sha=etaval_sha,
        etaval_dirty=etaval_dirty,
        gtfs_eta_git_sha=gtfs_eta_sha,
        gtfs_eta_dirty=gtfs_eta_dirty,
        source_name=source_name,
        source_config_sha256=config_sha256,
        gtfs_snapshot=gtfs_snapshot,
        detectors=_detectors_to_dict(detectors),
        seed=seed,
        predictor_ids=list(predictor_ids),
        versions=_versions(),
        created_at=datetime.now(tz=UTC).isoformat(),
    )


def write_provenance_sidecar(path: str | Path, prov: RunProvenance) -> Path:
    """Write ``<path>.provenance.json`` (pretty JSON) and return its path.

    *path* may or may not already end in an extension — the sidecar name
    is always ``str(path) + ".provenance.json"``.
    """
    sidecar_path = Path(f"{path}.provenance.json")
    sidecar_path.write_text(json.dumps(prov.to_dict(), indent=2, sort_keys=True))
    return sidecar_path


# ── Inference latency ────────────────────────────────────────────────────────

_MAX_SAMPLES_PER_PREDICTOR = 10_000


@dataclass(frozen=True)
class LatencyStats:
    """Summary latency stats for one predictor's ``predict()`` calls."""

    count: int
    p50_s: float
    p95_s: float
    p99_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "p50_s": self.p50_s,
            "p95_s": self.p95_s,
            "p99_s": self.p99_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyStats:
        return cls(
            count=int(data.get("count", 0)),
            p50_s=float(data.get("p50_s", 0.0)),
            p95_s=float(data.get("p95_s", 0.0)),
            p99_s=float(data.get("p99_s", 0.0)),
        )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list.

    Same convention as numpy's default ``interpolation="linear"``.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct / 100.0 * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


class LatencyRecorder:
    """Cheap per-predictor inference-latency recorder.

    Retains up to the first ``_MAX_SAMPLES_PER_PREDICTOR`` duration samples
    per predictor (so percentile computation stays bounded on very long
    runs) alongside a running total count of every call actually made,
    which may exceed the retained sample count.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}

    def record(self, predictor_id: str, duration_s: float) -> None:
        """Record one ``predict()`` call's duration for *predictor_id*."""
        self._counts[predictor_id] = self._counts.get(predictor_id, 0) + 1
        bucket = self._samples.setdefault(predictor_id, [])
        if len(bucket) < _MAX_SAMPLES_PER_PREDICTOR:
            bucket.append(duration_s)

    def summary(self) -> dict[str, LatencyStats]:
        """Return per-predictor LatencyStats over the retained samples."""
        result: dict[str, LatencyStats] = {}
        for predictor_id, samples in self._samples.items():
            ordered = sorted(samples)
            result[predictor_id] = LatencyStats(
                count=self._counts.get(predictor_id, len(samples)),
                p50_s=_percentile(ordered, 50.0),
                p95_s=_percentile(ordered, 95.0),
                p99_s=_percentile(ordered, 99.0),
            )
        return result
