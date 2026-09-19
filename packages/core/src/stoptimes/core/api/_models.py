"""`stoptimes models list/card/promote` as functions.

These only ever read `artifact.json` files (`stoptimes.core.contracts.artifact`)
and, for `promote_model`, an agency profile YAML -- no dependency on
`stoptimes.models` or `stoptimes.serve` is needed for any of the three, so
they live here rather than in a higher package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from stoptimes.core.contracts.artifact import ArtifactMetadata, METADATA_FILENAME, artifact_dir, read_metadata

__all__ = ["list_models", "model_card", "promote_model"]

_DEFAULT_REGISTRY = Path("artifacts/models")


def list_models(
    *,
    registry: str | Path = _DEFAULT_REGISTRY,
    source: str | None = None,
    route: str | None = None,
    family: str | None = None,
) -> list[dict[str, Any]]:
    """List artifacts in a registry directory: scope, headline metric, creation time.

    Equivalent to `stoptimes models list --json`.

    Example:
        >>> from stoptimes.core.api import list_models  # doctest: +SKIP
        >>> rows = list_models(registry="artifacts/models")  # doctest: +SKIP
        >>> sorted(rows[0])  # doctest: +SKIP
        ['artifact_id', 'created', 'family', 'metric', 'routes', 'scope']
    """
    registry = Path(registry)
    rows: list[dict[str, Any]] = []
    if not registry.exists():
        return rows
    for entry in sorted(registry.iterdir()):
        if not (entry / METADATA_FILENAME).exists():
            continue
        try:
            meta: ArtifactMetadata = read_metadata(entry)
        except Exception:  # noqa: BLE001 - one bad artifact shouldn't kill the listing
            continue
        if family and meta.family != family:
            continue
        if source and meta.scope.source_id not in (None, source):
            continue
        if route and route not in meta.scope.route_ids:
            continue
        rows.append(
            {
                "artifact_id": meta.artifact_id,
                "family": meta.family,
                "scope": meta.scope.level.value,
                "routes": meta.scope.route_ids,
                "metric": meta.metrics.get("test_mae_seconds"),
                "created": meta.provenance.created_at.isoformat(),
            }
        )
    return rows


def model_card(artifact_id: str, *, registry: str | Path = _DEFAULT_REGISTRY) -> str:
    """The raw markdown text of an artifact's model card.

    Equivalent to `stoptimes models card`.

    Raises:
        FileNotFoundError: no `card.md` for `artifact_id` in `registry`.
    """
    card_path = artifact_dir(Path(registry), artifact_id) / "card.md"
    if not card_path.exists():
        raise FileNotFoundError(f"no card.md for artifact {artifact_id!r} at {card_path}")
    return card_path.read_text(encoding="utf-8")


def promote_model(
    artifact_id: str,
    *,
    as_pin: str,
    source: str,
    registry: str | Path = _DEFAULT_REGISTRY,
    sources_dir: str | Path | None = None,
) -> Path:
    """Pin `artifact_id` into `source`'s agency profile for serving selection.

    Equivalent to `stoptimes models promote`. Writes
    `serving.selection.pins[as_pin] = artifact_id` into
    `<sources_dir>/<source>.yaml` (`stoptimes.core.contracts.selection`).

    Args:
        artifact_id: the artifact to promote; must exist in `registry`.
        as_pin: a pin key, e.g. `"global"`, `"route:1"`, `"route:1/0"`, or
            `"trip:t1"` (`contracts.selection.SelectionPolicy.pins`).
        source: the agency id whose profile file to update.
        registry: the artifact registry directory `artifact_id` lives in.
        sources_dir: agency profile directory; defaults to
            `$STOPTIMES_SOURCES` or `./examples/agencies`.

    Returns:
        The path of the profile file that was updated.

    Raises:
        FileNotFoundError: `artifact_id` isn't in `registry`, or the agency
            profile file doesn't exist.
    """
    registry = Path(registry)
    target_dir = artifact_dir(registry, artifact_id)
    if not (target_dir / METADATA_FILENAME).exists():
        raise FileNotFoundError(f"artifact {artifact_id!r} not found at {target_dir}")
    read_metadata(target_dir)  # fail loudly if artifact.json is corrupt, before touching the profile

    profile_dir = Path(sources_dir) if sources_dir else Path(os.environ.get("STOPTIMES_SOURCES", "examples/agencies"))
    profile_path = profile_dir / f"{source}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"agency profile not found: {profile_path}")

    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    serving = data.setdefault("serving", {})
    selection = serving.setdefault("selection", {})
    pins = selection.setdefault("pins", {})
    pins[as_pin] = artifact_id
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return profile_path
