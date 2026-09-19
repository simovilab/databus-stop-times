"""Promote/rollback: the pointer serving reads, with a way back.

Per docs/PLAN.md U7 ("production must always have a way back") and §8's
served-model discipline, `stoptimes models promote` (CLI, `commands/models.py`)
needs more than "write a pin and hope": an operator must be able to see what
is live and undo a bad promotion quickly. This module is the smallest honest
version of that -- no deployment system, no registry mutation, just:

- **The pointer**: unchanged from before this track -- `serving.selection.pins`
  in an agency's profile YAML (`contracts.selection.SelectionPolicy.pins`),
  which `resolve_artifact`/`select_artifact` already read. This module never
  changes that shape.
- **The history**: a small sidecar JSON file next to the profile,
  `<profile>.promotions.json`, recording every artifact a pin has pointed at,
  in order. `rollback` pops the current entry and restores the one before it.

The sidecar is deliberately *not* folded into the profile YAML itself: that
YAML is parsed by `stoptimes.core.contracts.adapters.source_from_val_yaml`
into `ServingDefaults`/`SelectionPolicy`, both `extra="forbid"` (core,
out of this track's owned paths) -- adding a history key there would break
every other reader of that file the moment this track's CLI touched it.
Keeping history in its own file means promote/rollback need nothing from
core to change.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMOTIONS_SUFFIX = ".promotions.json"


class RollbackError(RuntimeError):
    """Raised when there is nothing to roll back to for a given pin."""


def promotions_path(profile_path: Path) -> Path:
    """Sidecar history file for an agency profile, e.g.
    `examples/agencies/mbta.yaml` -> `examples/agencies/mbta.promotions.json`."""
    profile_path = Path(profile_path)
    return profile_path.with_name(profile_path.stem + PROMOTIONS_SUFFIX)


def load_history(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the sidecar history file, or an empty history if it doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pins", {})


def save_history(path: Path, history: dict[str, list[dict[str, Any]]]) -> None:
    path = Path(path)
    path.write_text(json.dumps({"pins": history}, indent=2) + "\n", encoding="utf-8")


def record_promotion(path: Path, pin: str, artifact_id: str) -> dict[str, list[dict[str, Any]]]:
    """Append `artifact_id` as the new current value for `pin`'s history.

    Idempotent-ish: promoting the same artifact twice in a row still appends
    (the timestamp differs and rollback should still land you where you
    were), matching "promote" being an explicit, logged action rather than a
    no-op.
    """
    history = load_history(path)
    entries = history.setdefault(pin, [])
    entries.append({"artifact_id": artifact_id, "at": datetime.now(timezone.utc).isoformat()})
    save_history(path, history)
    return history


def current_promotion(history: dict[str, list[dict[str, Any]]], pin: str) -> dict[str, Any] | None:
    """The most recent history entry for `pin`, or None if it was never promoted through this history."""
    entries = history.get(pin) or []
    return entries[-1] if entries else None


def rollback_target(path: Path, pin: str) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """Pop `pin`'s current history entry and return the artifact id to restore
    (the entry now on top), plus the updated history.

    Raises `RollbackError` if `pin` has fewer than two recorded promotions --
    there is nothing before the current one to go back to. The popped entry
    is dropped, not kept as a redo log: rolling back and then promoting again
    is how you'd move forward again, same as a real deployment pointer.
    """
    history = load_history(path)
    entries = history.get(pin) or []
    if len(entries) < 2:
        raise RollbackError(
            f"no earlier promotion recorded for pin {pin!r} in {path} "
            f"({len(entries)} entr{'y' if len(entries) == 1 else 'ies'}) -- nothing to roll back to"
        )
    entries.pop()  # drop the current one
    restored = entries[-1]["artifact_id"]
    save_history(path, history)
    return restored, history
