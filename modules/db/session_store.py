"""
Database module for storing training sessions.
SQLite-based persistent storage for historical data.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from modules.config import Config

logger = logging.getLogger(__name__)

# Columns that were added after the initial schema — must be migrated on legacy DBs.
_MIGRATION_COLUMNS: dict[str, str] = {
    "session_type": "TEXT DEFAULT 'unknown'",
    "athlete_id": "TEXT DEFAULT 'default'",
    "test_validity": "TEXT DEFAULT 'valid'",
    "vo2max_estimated": "REAL",
    "cp_estimated": "REAL",
    "smo2_quality_score": "REAL",
}


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


class SessionStore:
    """SQLite-based session storage with CRUD operations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Config.DB_PATH
        if self.db_path and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self._migrate_db()

    def _init_db(self) -> None:
        """Initialize database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            """)
            conn.commit()

    def _migrate_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            for col_name, col_def in _MIGRATION_COLUMNS.items():
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
                    logger.info("Migration: added column %s to sessions", col_name)
            conn.commit()

    def add_session(self, record: SessionRecord) -> int:
        """Add or update a session record. Returns session ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
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
            conn.commit()
            return cursor.lastrowid

    def get_sessions(self, days: int = 90) -> List[SessionRecord]:
        """Get sessions from the last N days using optimized batch fetch."""
        with sqlite3.connect(self.db_path) as conn:
            # Use custom row factory for direct SessionRecord creation
            def session_record_factory(cursor, row):
                # Map row directly to SessionRecord - O(1) per row
                return SessionRecord(
                    id=row[0],
                    date=row[1],
                    filename=row[2],
                    duration_sec=row[3],
                    tss=row[4],
                    np=row[5],
                    if_factor=row[6],
                    avg_watts=row[7],
                    avg_hr=row[8],
                    max_hr=row[9],
                    work_kj=row[10],
                    avg_cadence=row[11],
                    mmp_5s=row[12],
                    mmp_1m=row[13],
                    mmp_5m=row[14],
                    mmp_20m=row[15],
                    avg_rmssd=row[16],
                    alerts_count=row[17],
                    extra_metrics=row[18],
                    session_type=row[19],
                    athlete_id=row[20],
                    test_validity=row[21],
                    vo2max_estimated=row[22] if row[22] is not None else 0.0,
                    cp_estimated=row[23] if row[23] is not None else 0.0,
                    smo2_quality_score=row[24] if row[24] is not None else 0.0,
                )

            conn.row_factory = session_record_factory
            cursor = conn.execute(
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

            # Single batch fetch - O(n) total
            return cursor.fetchall()

    def get_all_tss(self, days: int = 90) -> List[tuple]:
        """Get (date, tss) tuples for training load calculation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT date, SUM(tss) as daily_tss 
                FROM sessions 
                WHERE date >= date('now', ?)
                GROUP BY date
                ORDER BY date ASC
            """,
                (f"-{days} days",),
            )
            return cursor.fetchall()

    def get_session_count(self) -> int:
        """Get total number of stored sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]

    def delete_session(self, session_id: int) -> bool:
        """Delete a session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
