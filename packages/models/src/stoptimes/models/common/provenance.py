"""
Training provenance helpers.

Small, dependency-light utilities used by the registry (and directly by
tests) to record enough about a training run to reproduce it later: the git
commit it was trained from, the library/interpreter versions in play, a
content hash of the dataset file, the random seed, and a timestamp.

Every helper here is defensive: a missing git binary, an uninstalled
optional dependency (xgboost/sklearn are behind extras), or a missing
dataset file must never raise -- they degrade to `None` so that saving a
model never fails because provenance couldn't be fully computed.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import get_config

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def git_sha() -> Optional[str]:
    """
    Short SHA of the current HEAD commit (`git rev-parse --short HEAD`),
    run with cwd = the project root. Returns None (never raises) if git
    is unavailable, this isn't a git checkout, or anything else goes
    wrong.
    """
    try:
        repo_root = get_config().project_root
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


def lib_versions() -> Dict[str, Optional[str]]:
    """
    Versions of the libraries that materially affect model reproducibility,
    plus the running Python version. Each library is imported lazily and
    guarded individually -- an uninstalled optional dependency (xgboost,
    sklearn) records None rather than raising, since inference environments
    may not have training extras installed.
    """
    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
    }

    for lib_name in ("sklearn", "xgboost", "numpy", "pandas"):
        versions[lib_name] = _safe_lib_version(lib_name)

    return versions


def _safe_lib_version(lib_name: str) -> Optional[str]:
    try:
        module = __import__(lib_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def file_content_hash(path: Any) -> Optional[str]:
    """
    sha256 hex digest of a file's bytes, computed in chunks. Returns None
    if `path` doesn't exist (or isn't a regular file) rather than raising.
    """
    try:
        file_path = Path(path)
        if not file_path.is_file():
            return None

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def _resolve_dataset_path(dataset_name: str) -> Path:
    repo_root = get_config().project_root
    return repo_root / "datasets" / f"{dataset_name}.parquet"


def build_provenance(
    dataset_name: Optional[str] = None,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble a provenance block suitable for embedding in saved model
    metadata: git commit, library/python versions, a content hash of the
    training dataset, the random seed used, and a creation timestamp.

    Args:
        dataset_name: Dataset name (without extension); resolved to
            `datasets/<dataset_name>.parquet` relative to the project root.
            None if the dataset is unknown/not applicable.
        seed: Random seed used for training, or None for deterministic
            (RNG-free) trainers.
        extra: Optional additional fields to merge in (e.g. trainer-specific
            provenance already computed by the caller). Does not override
            the core keys below.

    Returns:
        Dict with keys: git_sha, lib_versions, python_version, dataset_name,
        dataset_sha256, seed, created_at.
    """
    dataset_sha256: Optional[str] = None
    if dataset_name:
        dataset_sha256 = file_content_hash(_resolve_dataset_path(dataset_name))

    versions = lib_versions()

    provenance: Dict[str, Any] = {
        "git_sha": git_sha(),
        "lib_versions": versions,
        "python_version": versions["python"],
        "dataset_name": dataset_name,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "created_at": datetime.now().isoformat(),
    }

    if extra:
        for key, value in extra.items():
            provenance.setdefault(key, value)

    return provenance
