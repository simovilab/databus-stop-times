"""Small shared helpers for the diagnose checks — kept out of `models.py` (a
pure contract module) and out of each checks module (to avoid duplicating
sample-bounding / severity-by-share logic three times).
"""

from __future__ import annotations

import pandas as pd
from stoptimes.core.diagnose.models import DiagnoseParams, Severity


def bounded_samples(values: pd.Series | list, params: DiagnoseParams) -> list[str]:
    """Return up to `params.max_samples` distinct values as strings, in their
    original order — never the whole affected set, however large it is."""
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.append(text)
        if len(seen) >= params.max_samples:
            break
    return seen


def severity_by_share(share: float, params: DiagnoseParams) -> Severity:
    """`blocker` once the affected share reaches `majority_threshold`, else
    `degraded`. Never `info` — a mismatch/gap finding that fires at all costs
    something; `info` is reserved for findings that don't."""
    return Severity.BLOCKER if share >= params.majority_threshold else Severity.DEGRADED
