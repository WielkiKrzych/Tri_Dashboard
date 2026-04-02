"""
Database module for storing training sessions.
SQLite-based persistent storage for historical data.
"""

import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from modules.db.base import BaseStore

logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Represents a single training session record."""

    id: Optional[int] = None
    date: str = ""
    filename: str = ""
    duration_sec: int = 0
    tss: float = 0.0
    np: float = 0.0
    if_factor: float = 0.0
    avg_watts: float = 0.0
    avg_hr: float = 0.0
    max_hr: float = 0.0
    work_kj: float = 0.0
    avg_cadence: float = 0.0
    # MMP values
    mmp_5s: Optional[float] = None
    mmp_1m: Optional[float] = None
    mmp_5m: Optional[float] = None
    mmp_20m: Optional[float] = None
    # HRV
    avg_rmssd: Optional[float] = None
    # Alerts count
    alerts_count: int = 0
    # JSON for additional metrics
    extra_metrics: str = "{}"
    session_type: str = "unknown"  # ramp, steady, intervals, race, unknown
    athlete_id: str = "default"
    test_validity: str = "valid"  # valid, conditional, invalid
    vo2max_estimated: float = 0.0
    cp_estimated: float = 0.0
    smo2_quality_score: float = 0.0


class SessionStore(BaseStore):
    """SQLite-based session storage with CRUD operations."""

    table_name = "sessions"
    _migration_columns = {
        "session_type": "TEXT DEFAULT 'unknown'",
        "athlete_id": "TEXT DEFAULT 'default'",
        "test_validity": "TEXT DEFAULT 'valid'",
        "vo2max_estimated": "REAL",
        "cp_estimated": "REAL",
        "smo2_quality_score": "REAL",
    }
    _schema_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            filename TEXT NOT NULL,
            duration_sec INTEGER DEFAULT 0,
            tss REAL DEFAULT 0,
            np REAL DEFAULT 0,
            if_factor REAL DEFAULT 0,
            avg_watts REAL DEFAULT 0,
            avg_hr REAL DEFAULT 0,
            max_hr REAL DEFAULT 0,
            work_kj REAL DEFAULT 0,
            avg_cadence REAL DEFAULT 0,
            mmp_5s REAL,
            mmp_1m REAL,
            mmp_5m REAL,
            mmp_20m REAL,
            avg_rmssd REAL,
            alerts_count INTEGER DEFAULT 0,
            extra_metrics TEXT DEFAULT '{}',
            session_type TEXT DEFAULT 'unknown',
            athlete_id TEXT DEFAULT 'default',
            test_validity TEXT DEFAULT 'valid',
            vo2max_estimated REAL,
            cp_estimated REAL,
            smo2_quality_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, filename)
        )
    """

    def add_session(self, record: SessionRecord) -> int:
        """Add or update a session record. Returns session ID."""
        return (
            self.upsert(
                """
            INSERT INTO sessions (
                date, filename, duration_sec, tss, np, if_factor,
                avg_watts, avg_hr, max_hr, work_kj, avg_cadence,
                mmp_5s, mmp_1m, mmp_5m, mmp_20m, avg_rmssd,
                alerts_count, extra_metrics,
                session_type, athlete_id, test_validity,
                vo2max_estimated, cp_estimated, smo2_quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, filename) DO UPDATE SET
                duration_sec = excluded.duration_sec,
                tss = excluded.tss,
                np = excluded.np,
                if_factor = excluded.if_factor,
                avg_watts = excluded.avg_watts,
                avg_hr = excluded.avg_hr,
                max_hr = excluded.max_hr,
                work_kj = excluded.work_kj,
                avg_cadence = excluded.avg_cadence,
                mmp_5s = excluded.mmp_5s,
                mmp_1m = excluded.mmp_1m,
                mmp_5m = excluded.mmp_5m,
                mmp_20m = excluded.mmp_20m,
                avg_rmssd = excluded.avg_rmssd,
                alerts_count = excluded.alerts_count,
                extra_metrics = excluded.extra_metrics,
                session_type = excluded.session_type,
                athlete_id = excluded.athlete_id,
                test_validity = excluded.test_validity,
                vo2max_estimated = excluded.vo2max_estimated,
                cp_estimated = excluded.cp_estimated,
                smo2_quality_score = excluded.smo2_quality_score
        """,
                (
                    record.date,
                    record.filename,
                    record.duration_sec,
                    record.tss,
                    record.np,
                    record.if_factor,
                    record.avg_watts,
                    record.avg_hr,
                    record.max_hr,
                    record.work_kj,
                    record.avg_cadence,
                    record.mmp_5s,
                    record.mmp_1m,
                    record.mmp_5m,
                    record.mmp_20m,
                    record.avg_rmssd,
                    record.alerts_count,
                    record.extra_metrics,
                    record.session_type,
                    record.athlete_id,
                    record.test_validity,
                    record.vo2max_estimated,
                    record.cp_estimated,
                    record.smo2_quality_score,
                ),
            )
            or 0
        )

    def get_sessions(self, days: int = 90) -> List[SessionRecord]:
        """Get sessions from the last N days."""
        rows = self.query(
            """
            SELECT id, date, filename, duration_sec, tss, np, if_factor,
                   avg_watts, avg_hr, max_hr, work_kj, avg_cadence,
                   mmp_5s, mmp_1m, mmp_5m, mmp_20m, avg_rmssd,
                   alerts_count, extra_metrics,
                   session_type, athlete_id, test_validity,
                   vo2max_estimated, cp_estimated, smo2_quality_score
            FROM sessions
            WHERE date >= date('now', ?)
            ORDER BY date DESC
        """,
            (f"-{days} days",),
        )
        return [self._row_to_session(r) for r in rows]

    def get_all_tss(self, days: int = 90) -> List[tuple]:
        """Get (date, tss) tuples for training load calculation."""
        rows = self.query(
            """
            SELECT date, SUM(tss) as daily_tss
            FROM sessions
            WHERE date >= date('now', ?)
            GROUP BY date
            ORDER BY date ASC
        """,
            (f"-{days} days",),
        )
        return [(r["date"], r["daily_tss"]) for r in rows]

    def get_session_count(self) -> int:
        """Get total number of stored sessions."""
        row = self.query_one("SELECT COUNT(*) as cnt FROM sessions")
        return row["cnt"] if row else 0

    def delete_session(self, session_id: int) -> bool:
        """Delete a session by ID."""
        return self.delete("id", session_id)

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionRecord:
        """Convert a sqlite3.Row to a SessionRecord."""
        return SessionRecord(
            id=row["id"],
            date=row["date"],
            filename=row["filename"],
            duration_sec=row["duration_sec"],
            tss=row["tss"],
            np=row["np"],
            if_factor=row["if_factor"],
            avg_watts=row["avg_watts"],
            avg_hr=row["avg_hr"],
            max_hr=row["max_hr"],
            work_kj=row["work_kj"],
            avg_cadence=row["avg_cadence"],
            mmp_5s=row["mmp_5s"],
            mmp_1m=row["mmp_1m"],
            mmp_5m=row["mmp_5m"],
            mmp_20m=row["mmp_20m"],
            avg_rmssd=row["avg_rmssd"],
            alerts_count=row["alerts_count"],
            extra_metrics=row["extra_metrics"],
            session_type=row["session_type"],
            athlete_id=row["athlete_id"],
            test_validity=row["test_validity"],
            vo2max_estimated=row["vo2max_estimated"]
            if row["vo2max_estimated"] is not None
            else 0.0,
            cp_estimated=row["cp_estimated"] if row["cp_estimated"] is not None else 0.0,
            smo2_quality_score=row["smo2_quality_score"]
            if row["smo2_quality_score"] is not None
            else 0.0,
        )
