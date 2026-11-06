"""Identity features: agency-agnostic categorical encodings fitted at training time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _clean(values: Iterable[Any]) -> list[str | None]:
    out: list[str | None] = []
    for v in values:
        out.append(None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v))
    return out


@dataclass(frozen=True)
class RouteVocabulary:
    """Training-time vocabulary of GTFS ``route_id`` strings.

    ``encode`` yields the index of each id in the sorted vocabulary as a float;
    ids unseen at fit time (or missing) become NaN, which tree models treat as missing.
    Works for any id shape ("Red", "CR-Fitchburg", "R-12", "12").
    """

    route_ids: tuple[str, ...] = ()

    @classmethod
    def fit(cls, route_ids: Iterable[Any]) -> "RouteVocabulary":
        return cls(tuple(sorted({r for r in _clean(route_ids) if r is not None})))

    def encode(self, route_ids: Iterable[Any]) -> np.ndarray:
        index = {r: float(i) for i, r in enumerate(self.route_ids)}
        return np.array([index.get(r, np.nan) if r is not None else np.nan for r in _clean(route_ids)], dtype=float)

    def __len__(self) -> int:
        return len(self.route_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"route_ids": list(self.route_ids)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RouteVocabulary":
        ids = tuple(str(r) for r in d["route_ids"])
        if list(ids) != sorted(set(ids)):
            raise ValueError("route vocabulary must be sorted and unique")
        return cls(ids)
