"""SQLite storage. Phase P0.

Three tables: prediction_samples, actual_arrivals, validation_results;
indexed by (trip_key, predictor_id). Implements the Storage protocol.

Usage:
    # File-backed
    store = SqliteStorage("etaval.db")

    # In-memory (tests)
    with SqliteStorage(":memory:") as store:
        store.save_samples([...])
        results = store.load_results()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from etaval.domain.models import ActualArrival, PredictionSample, ValidationResult

# ── DDL ───────────────────────────────────────────────────────────────────────

_TABLES = """
CREATE TABLE IF NOT EXISTS prediction_samples (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_key                  TEXT    NOT NULL,
    predictor_id              TEXT    NOT NULL,
    stop_id                   TEXT    NOT NULL,
    stop_sequence             INTEGER NOT NULL,
    predicted_arrival_epoch   REAL    NOT NULL,
    made_at_epoch             REAL    NOT NULL,
    vehicle_progress_m_at_made REAL,
    dist_remaining_m          REAL,
    stops_remaining           INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ps_trip_predictor
    ON prediction_samples (trip_key, predictor_id);

CREATE TABLE IF NOT EXISTS actual_arrivals (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_key              TEXT    NOT NULL,
    stop_id               TEXT    NOT NULL,
    stop_sequence         INTEGER NOT NULL,
    actual_arrival_epoch  REAL    NOT NULL,
    method                TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_aa_trip_stop
    ON actual_arrivals (trip_key, stop_id);

CREATE TABLE IF NOT EXISTS validation_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_key                TEXT    NOT NULL,
    predictor_id            TEXT    NOT NULL,
    stop_id                 TEXT    NOT NULL,
    stop_sequence           INTEGER NOT NULL,
    route_id                TEXT,
    error_s                 REAL    NOT NULL,
    horizon_s               REAL    NOT NULL,
    dist_remaining_m        REAL,
    stops_remaining         INTEGER,
    predicted_arrival_epoch REAL    NOT NULL,
    actual_arrival_epoch    REAL    NOT NULL,
    made_at_epoch           REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vr_trip_predictor
    ON validation_results (trip_key, predictor_id);
"""

# ── INSERT statements ─────────────────────────────────────────────────────────

_INSERT_SAMPLE = """
INSERT INTO prediction_samples
    (trip_key, predictor_id, stop_id, stop_sequence,
     predicted_arrival_epoch, made_at_epoch,
     vehicle_progress_m_at_made, dist_remaining_m, stops_remaining)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ARRIVAL = """
INSERT INTO actual_arrivals
    (trip_key, stop_id, stop_sequence, actual_arrival_epoch, method)
VALUES (?, ?, ?, ?, ?)
"""

_INSERT_RESULT = """
INSERT INTO validation_results
    (trip_key, predictor_id, stop_id, stop_sequence, route_id,
     error_s, horizon_s, dist_remaining_m, stops_remaining,
     predicted_arrival_epoch, actual_arrival_epoch, made_at_epoch)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class SqliteStorage:
    """SQLite-backed storage implementing the Storage protocol."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ── schema ────────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        for stmt in _TABLES.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> SqliteStorage:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ── write ─────────────────────────────────────────────────────────────────

    def save_samples(self, samples: list[PredictionSample]) -> None:
        """Persist a batch of PredictionSamples."""
        self._conn.executemany(
            _INSERT_SAMPLE,
            [
                (
                    s.trip_key,
                    s.predictor_id,
                    s.stop_id,
                    s.stop_sequence,
                    s.predicted_arrival_epoch,
                    s.made_at_epoch,
                    s.vehicle_progress_m_at_made,
                    s.dist_remaining_m,
                    s.stops_remaining,
                )
                for s in samples
            ],
        )
        self._conn.commit()

    def save_arrivals(self, arrivals: list[ActualArrival]) -> None:
        """Persist a batch of ActualArrivals."""
        self._conn.executemany(
            _INSERT_ARRIVAL,
            [
                (
                    a.trip_key,
                    a.stop_id,
                    a.stop_sequence,
                    a.actual_arrival_epoch,
                    a.method,
                )
                for a in arrivals
            ],
        )
        self._conn.commit()

    def save_results(self, results: list[ValidationResult]) -> None:
        """Persist a batch of ValidationResults."""
        self._conn.executemany(
            _INSERT_RESULT,
            [
                (
                    r.trip_key,
                    r.predictor_id,
                    r.stop_id,
                    r.stop_sequence,
                    r.route_id,
                    r.error_s,
                    r.horizon_s,
                    r.dist_remaining_m,
                    r.stops_remaining,
                    r.predicted_arrival_epoch,
                    r.actual_arrival_epoch,
                    r.made_at_epoch,
                )
                for r in results
            ],
        )
        self._conn.commit()

    # ── read ──────────────────────────────────────────────────────────────────

    def load_samples(self, predictor_id: str | None = None) -> list[PredictionSample]:
        """Load PredictionSamples, optionally filtered by predictor_id.

        Used by the A5 detector-sensitivity CLI path (``etaval report
        --sensitivity --db``), which re-scores every persisted sample against
        each detector method's already-persisted arrivals — see
        ``etaval.scoring.sensitivity`` for why raw position history (and
        therefore true re-detection) is not available from storage.
        """
        if predictor_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM prediction_samples WHERE predictor_id = ?",
                (predictor_id,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM prediction_samples").fetchall()

        return [
            PredictionSample(
                trip_key=row["trip_key"],
                predictor_id=row["predictor_id"],
                stop_id=row["stop_id"],
                stop_sequence=row["stop_sequence"],
                predicted_arrival_epoch=row["predicted_arrival_epoch"],
                made_at_epoch=row["made_at_epoch"],
                vehicle_progress_m_at_made=row["vehicle_progress_m_at_made"],
                dist_remaining_m=row["dist_remaining_m"],
                stops_remaining=row["stops_remaining"],
            )
            for row in rows
        ]

    def load_arrivals(self, method: str | None = None) -> list[ActualArrival]:
        """Load ActualArrivals, optionally filtered by detector method.

        The engine persists arrivals from *every* configured detector each
        tick (not just the ``primary_method`` used for scoring), so this
        returns arrivals for every detector method that ran during the
        original ``etaval run`` — see ``etaval.scoring.sensitivity``.
        """
        if method is not None:
            rows = self._conn.execute(
                "SELECT * FROM actual_arrivals WHERE method = ?",
                (method,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM actual_arrivals").fetchall()

        return [
            ActualArrival(
                trip_key=row["trip_key"],
                stop_id=row["stop_id"],
                stop_sequence=row["stop_sequence"],
                actual_arrival_epoch=row["actual_arrival_epoch"],
                method=row["method"],
            )
            for row in rows
        ]

    def load_results(self, predictor_id: str | None = None) -> list[ValidationResult]:
        """Load ValidationResults, optionally filtered by predictor_id."""
        if predictor_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM validation_results WHERE predictor_id = ?",
                (predictor_id,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM validation_results").fetchall()

        return [
            ValidationResult(
                trip_key=row["trip_key"],
                predictor_id=row["predictor_id"],
                stop_id=row["stop_id"],
                stop_sequence=row["stop_sequence"],
                route_id=row["route_id"],
                error_s=row["error_s"],
                horizon_s=row["horizon_s"],
                dist_remaining_m=row["dist_remaining_m"],
                stops_remaining=row["stops_remaining"],
                predicted_arrival_epoch=row["predicted_arrival_epoch"],
                actual_arrival_epoch=row["actual_arrival_epoch"],
                made_at_epoch=row["made_at_epoch"],
            )
            for row in rows
        ]
