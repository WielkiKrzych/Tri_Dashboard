"""
Database module for storing training sessions.
SQLite-based persistent storage for historical data.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
from modules.config import Config


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


class SessionStore:
    """SQLite-based session storage with CRUD operations."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Config.DB_PATH
        # Ensure directory exists
        if self.db_path and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        self._init_db()
    
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
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, filename)
                )
            """)
            conn.commit()
    
    def add_session(self, record: SessionRecord) -> int:
        """Add or update a session record. Returns session ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO sessions (
                    date, filename, duration_sec, tss, np, if_factor,
                    avg_watts, avg_hr, max_hr, work_kj, avg_cadence,
                    mmp_5s, mmp_1m, mmp_5m, mmp_20m, avg_rmssd,
                    alerts_count, extra_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    extra_metrics = excluded.extra_metrics
            """, (
                record.date, record.filename, record.duration_sec,
                record.tss, record.np, record.if_factor,
                record.avg_watts, record.avg_hr, record.max_hr,
                record.work_kj, record.avg_cadence,
                record.mmp_5s, record.mmp_1m, record.mmp_5m, record.mmp_20m,
                record.avg_rmssd, record.alerts_count, record.extra_metrics
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_sessions(self, days: int = 90) -> List[SessionRecord]:
        """Get sessions from the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sessions 
                WHERE date >= date('now', ?)
                ORDER BY date DESC
            """, (f'-{days} days',))
            
            results = []
            for row in cursor.fetchall():
                results.append(SessionRecord(
                    id=row['id'],
                    date=row['date'],
                    filename=row['filename'],
                    duration_sec=row['duration_sec'],
                    tss=row['tss'],
                    np=row['np'],
                    if_factor=row['if_factor'],
                    avg_watts=row['avg_watts'],
                    avg_hr=row['avg_hr'],
                    max_hr=row['max_hr'],
                    work_kj=row['work_kj'],
                    avg_cadence=row['avg_cadence'],
                    mmp_5s=row['mmp_5s'],
                    mmp_1m=row['mmp_1m'],
                    mmp_5m=row['mmp_5m'],
                    mmp_20m=row['mmp_20m'],
                    avg_rmssd=row['avg_rmssd'],
                    alerts_count=row['alerts_count'],
                    extra_metrics=row['extra_metrics']
                ))
            return results
    
    def get_all_tss(self, days: int = 90) -> List[tuple]:
        """Get (date, tss) tuples for training load calculation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT date, SUM(tss) as daily_tss 
                FROM sessions 
                WHERE date >= date('now', ?)
                GROUP BY date
                ORDER BY date ASC
            """, (f'-{days} days',))
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
