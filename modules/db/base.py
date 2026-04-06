"""Abstract base class for SQLite-backed stores."""

from __future__ import annotations

import logging
import re
import sqlite3
from abc import ABC
from pathlib import Path
from typing import Any, Optional

from modules.config import Config

logger = logging.getLogger(__name__)

# Allowlist pattern for SQL identifiers — alphanumeric + underscore only
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    """Validate that *name* is a safe SQL identifier (no injection risk).

    Raises ValueError if the name contains anything other than
    ``[a-zA-Z0-9_]`` or does not start with a letter/underscore.
    """
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL {kind}: {name!r}")
    return name


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
        self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _validate_identifier(self.table_name, "table name")
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
        table = _validate_identifier(self.table_name, "table name")
        with self.connect() as conn:
            existing = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col_name, col_def in self._migration_columns.items():
                col = _validate_identifier(col_name, "column name")
                # col_def is a type like "TEXT", "REAL DEFAULT 0" — validate each word
                for token in col_def.split():
                    if not _SAFE_IDENTIFIER_RE.match(token):
                        # Allow numeric defaults like "0", "0.0"
                        try:
                            float(token)
                        except ValueError:
                            raise ValueError(f"Unsafe SQL column definition token: {token!r}")
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                    logger.info("Migration: added %s.%s", table, col)
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
        table = _validate_identifier(self.table_name, "table name")
        col = _validate_identifier(pk_col, "primary key column")
        with self.connect() as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (pk_value,))
            conn.commit()
            return cursor.rowcount > 0
