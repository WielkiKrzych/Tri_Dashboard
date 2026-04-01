"""Tests for SQLite schema migration — ensures legacy DBs are upgraded safely."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from modules.db.session_store import SessionStore, SessionRecord


LEGACY_CREATE = """
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
"""

EXPECTED_MIGRATION_COLS = [
    "session_type",
    "athlete_id",
    "test_validity",
    "vo2max_estimated",
    "cp_estimated",
    "smo2_quality_score",
]


def _create_legacy_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(LEGACY_CREATE)
        conn.execute(
            "INSERT INTO sessions (date, filename, duration_sec, tss) VALUES (?, ?, ?, ?)",
            ("2026-01-15", "legacy_test.csv", 3600, 120.0),
        )
        conn.commit()


def _get_column_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}


class TestMigrationFromLegacySchema:
    def test_missing_columns_are_added(self, tmp_path: Path):
        db_path = tmp_path / "legacy.db"
        _create_legacy_db(db_path)

        SessionStore(db_path)

        columns = _get_column_names(db_path)
        for col in EXPECTED_MIGRATION_COLS:
            assert col in columns, f"Column {col} was not added during migration"

    def test_existing_data_preserved(self, tmp_path: Path):
        db_path = tmp_path / "legacy.db"
        _create_legacy_db(db_path)

        SessionStore(db_path)

        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT date, filename, tss FROM sessions").fetchone()

        assert row == ("2026-01-15", "legacy_test.csv", 120.0)

    def test_add_session_no_longer_raises(self, tmp_path: Path):
        db_path = tmp_path / "legacy.db"
        _create_legacy_db(db_path)

        store = SessionStore(db_path)

        record = SessionRecord(
            date="2026-03-01",
            filename="new_session.fit",
            duration_sec=1800,
            tss=85.0,
            session_type="ramp",
            vo2max_estimated=65.2,
        )
        session_id = store.add_session(record)

        assert session_id is not None
        assert isinstance(session_id, int)
        assert store.get_session_count() == 2

    def test_migration_is_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "legacy.db"
        _create_legacy_db(db_path)

        SessionStore(db_path)
        SessionStore(db_path)

        columns = _get_column_names(db_path)
        for col in EXPECTED_MIGRATION_COLS:
            assert col in columns


class TestFreshSchema:
    def test_fresh_db_has_all_columns(self, tmp_path: Path):
        db_path = tmp_path / "fresh.db"
        store = SessionStore(db_path)

        columns = _get_column_names(db_path)
        for col in EXPECTED_MIGRATION_COLS:
            assert col in columns

    def test_fresh_db_add_session_works(self, tmp_path: Path):
        db_path = tmp_path / "fresh.db"
        store = SessionStore(db_path)

        record = SessionRecord(
            date="2026-03-01",
            filename="test.fit",
            session_type="intervals",
            cp_estimated=280.0,
        )
        session_id = store.add_session(record)
        assert session_id is not None

        sessions = store.get_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_type == "intervals"
        assert sessions[0].cp_estimated == 280.0
