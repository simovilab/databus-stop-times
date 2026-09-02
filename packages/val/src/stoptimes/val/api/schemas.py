"""Pydantic v2 request/response schemas for the etaval REST API."""

from __future__ import annotations

import pydantic

from etaval.engine.assembly import parse_duration


class RunRequest(pydantic.BaseModel):
    """Body for POST /api/runs."""

    source: str
    predictors: list[str] = []
    routes: list[str] = []
    duration: str = "30m"
    seed: int | None = None

    @pydantic.field_validator("duration")
    @classmethod
    def _check_duration(cls, v: str) -> str:
        parse_duration(v)  # raises ValueError → Pydantic 422 if malformed
        return v


class RunInfo(pydantic.BaseModel):
    """Response body for GET /api/runs and GET /api/runs/{id}."""

    id: str
    params: dict
    status: str
    summary: dict | None = None
    provenance: dict | None = None
    db_path: str
    started_at: float
    ended_at: float | None = None
    error: str | None = None


class MetricsResponse(pydantic.BaseModel):
    """Response body for GET /api/runs/{id}/metrics.

    Wraps the output of ``MetricsReport.summary()``.
    """

    per_predictor: list[dict]
    horizon_buckets: list[dict]
    convergence: list[dict]
    head_to_head: list[dict]


class ResultRow(pydantic.BaseModel):
    """One ValidationResult row, for GET /api/runs/{id}/results."""

    trip_key: str
    predictor_id: str
    stop_id: str
    stop_sequence: int
    route_id: str | None = None
    error_s: float
    horizon_s: float
    dist_remaining_m: float | None = None
    stops_remaining: int | None = None
    predicted_arrival_epoch: float
    actual_arrival_epoch: float
    made_at_epoch: float
