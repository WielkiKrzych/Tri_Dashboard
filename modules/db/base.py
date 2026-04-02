"""Abstract base class for SQLite-backed stores."""

from __future__ import annotations

import logging
import sqlite3
from abc import ABC
from pathlib import Path
from typing import Any, Optional

from modules.config import Config

logger = logging.getLogger(__name__)


class BaseStore(ABC):
    """Base class for all SQLite stores. Provides shared connection handling,
    schema bootstrap, migrations, and generic CRUD helpers.

    Subclasses set:
      - table_name: str
      - _schema_sql: str  (CREATE TABLE IF NOT EXISTS ...)
      - _migration_columns: dict[str, str] (optional, {col: type_def})
    """

    table_name: str
    _schema_sql: str
    _migration_columns: dict[str, str] = {}

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path: Path = db_path or Config.DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(self._schema_sql)
            conn.commit()
        if self._migration_columns:
            self._migrate()

    def _migrate(self) -> None:
        with self.connect() as conn:
            existing = {
                row[1] for row in conn.execute(f"PRAGMA table_info({self.table_name})").fetchall()
            }
            for col_name, col_def in self._migration_columns.items():
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {col_name} {col_def}")
                    logger.info("Migration: added %s.%s", self.table_name, col_name)
            conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self.connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return row

    def upsert(self, sql: str, params: tuple[Any, ...] = ()) -> int | None:
        with self.connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.lastrowid

    def delete(self, pk_col: str, pk_value: Any) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(f"DELETE FROM {self.table_name} WHERE {pk_col} = ?", (pk_value,))
            conn.commit()
            return cursor.rowcount > 0
