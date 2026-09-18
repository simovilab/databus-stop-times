"""Contracts for `stoptimes diagnose`: `Severity` / `Finding` / `DiagnoseParams` /
`DiagnosticReport`.

Kept next to, not inside, `stoptimes.core.contracts` — a diagnose report is
this command's own output shape, not a canonical table passed between
tracks (docs/CONTRACTS.md scope). Same style as `contracts.data`: pydantic,
`extra="forbid"`, no mutation — a report is built once, not edited in place.

The report is data, not printing: every check function in this package
returns `Finding`s (or `None`), and `stoptimes.core.diagnose.diagnose_feed`
assembles a `DiagnosticReport`. Rendering (human table or `--json`) belongs
to the CLI (`packages/cli/src/stoptimes/cli/commands/diagnose.py`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """How much a finding should worry a newcomer deciding whether to invest
    a week in this feed. Deliberately three levels, not a numeric score —
    a score invites false precision this data doesn't support."""

    BLOCKER = "blocker"
    """Modelling is not possible at all until this is fixed (e.g. no GTFS
    static feed, or a required static table is missing)."""
    DEGRADED = "degraded"
    """Modelling is possible but will be measurably less accurate or cover
    less data than the raw row counts suggest."""
    INFO = "info"
    """Worth knowing about the feed; does not by itself block or degrade
    modelling."""


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.DEGRADED: 1,
    Severity.BLOCKER: 2,
}
"""Ordering for "worst finding wins" verdict aggregation — not a numeric score,
just a comparison key."""


class Finding(BaseModel):
    """One diagnostic finding.

    `count`/`total`/`share` carry the reproducible magnitude (e.g. 214 of
    900 rows, share 0.238); every value here must be derivable from the
    input data by a named, documented rule — no unexplained cutoffs.
    `samples` is a bounded list of concrete affected identifiers (trip_ids,
    stop_ids, vehicle_ids, ...), capped at `DiagnoseParams.max_samples` so a
    finding on a huge archive doesn't try to enumerate every offender.
    `detail` carries extra structured numbers a renderer may want (a gap
    histogram, a per-route or per-hour breakdown) — still data, never a
    pre-rendered string.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    """Stable short code, e.g. `"vp_sampling_gaps"` — safe to key off of in
    automation, unlike `title`."""
    severity: Severity
    title: str
    explanation: str
    """What this finding means and, where non-obvious, how it was computed."""
    blocks: str
    """What this specifically prevents or degrades, in one sentence — the
    "so what" a newcomer needs to decide whether to act on it."""
    recommendation: str
    """What to do about it, in one sentence."""
    count: int = 0
    total: int | None = None
    share: float | None = None
    samples: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class DiagnoseParams(BaseModel):
    """Named, documented thresholds. Every default here is a deliberate,
    explained choice, not a magic number — see each field's docstring.
    Echoed back on `DiagnosticReport.params` so a finding is reproducible
    from the report alone.
    """

    model_config = ConfigDict(extra="forbid")

    gap_threshold_s: float = 300.0
    """A vehicle-position sampling gap wider than this counts as a coverage
    gap. Matches `stoptimes.core.catalog.inspect.DEFAULT_GAP_THRESHOLD_S`,
    so `data inspect` and `diagnose` agree on what "a gap" means."""
    stale_threshold_s: float = 120.0
    """How far a record's `ingested_at - ts` may drift before it counts as
    stale. 120s is two polling cycles at the common 60s GTFS-RT refresh
    interval used across the example agencies (`examples/agencies/*.yaml`,
    `RealtimeFeed.poll_interval_s` default 15s is finer-grained still) — a
    record already this old when it was fetched was stale in the upstream
    feed, not merely delayed by polling jitter."""
    majority_threshold: float = 0.5
    """Share of affected trips/rows at or above which a mismatch finding is
    promoted from `degraded` to `blocker`: "most of this data is unusable"
    rather than "some of it is". 0.5 is the natural "more than half"
    breakpoint for that distinction."""
    max_samples: int = 10
    """Affected-sample identifiers kept per finding, so a finding on an
    archive with a million rows doesn't try to enumerate every offender."""


class DiagnosticReport(BaseModel):
    """Feed-quality report: what's present, every finding, and an overall
    verdict a newcomer can act on without reading the findings list first."""

    model_config = ConfigDict(extra="forbid")

    static_present: bool
    vp_present: bool
    tu_present: bool
    static_tables: list[str] = Field(default_factory=list)
    vp_rows: int = 0
    tu_rows: int = 0
    findings: list[Finding] = Field(default_factory=list)
    verdict: str
    verdict_severity: Severity
    params: DiagnoseParams

    def findings_by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]
