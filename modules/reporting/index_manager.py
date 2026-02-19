"""
Index Manager — CSV index operations for ramp test reports.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "session_id",
    "test_date",
    "athlete_id",
    "method_version",
    "json_path",
    "pdf_path",
    "source_file",
]


def _check_source_file_exists(base_dir: str, source_file: str) -> bool:
    """
    Check if a source file has already been saved in the index.

    Used for deduplication - prevents saving multiple reports for the same CSV file.

    Args:
        base_dir: Base directory containing index.csv
        source_file: Filename to check (e.g., 'ramp_test_2026-01-03.csv')

    Returns:
        True if source_file already exists in index, False otherwise
    """
    index_path = Path(base_dir) / "index.csv"

    if not index_path.exists():
        return False

    try:
        with open(index_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_source = row.get("source_file", "")
                if existing_source and existing_source == source_file:
                    return True
    except Exception as e:
        logger.warning("Failed to check deduplication: %s", e)
        return False

    return False


def _update_index(
    base_dir: str,
    metadata: Dict,
    file_path: str,
    pdf_path: Optional[str] = None,
    source_file: Optional[str] = None,
):
    """
    Update CSV index with new test record.

    Columns: session_id, test_date, athlete_id, method_version, json_path, pdf_path, source_file
    """
    index_path = Path(base_dir) / "index.csv"
    file_exists = index_path.exists()

    row = {
        "session_id": metadata.get("session_id", ""),
        "test_date": metadata.get("test_date", ""),
        "athlete_id": metadata.get("athlete_id") or "anonymous",
        "method_version": metadata.get("method_version", ""),
        "json_path": file_path,
        "pdf_path": pdf_path or "",
        "source_file": source_file or "",
    }

    if len(row) != len(INDEX_COLUMNS):
        logger.error(
            "Invalid record length for index. Expected %d, got %d.",
            len(INDEX_COLUMNS),
            len(row),
        )
        return

    if not row["session_id"] or not row["json_path"]:
        logger.error(
            "Missing critical data for index (session_id or json_path). Record not saved."
        )
        return

    try:
        with open(index_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Ramp Test indexed: %s", row["session_id"])
    except Exception as e:
        logger.error("Failed to write to index at %s: %s", index_path, e)


def update_index_pdf_path(base_dir: str, session_id: str, pdf_path: str):
    """
    Update existing index row with PDF path.

    PDF can be regenerated, so this updates an existing row.
    JSON is never modified (immutable).

    Args:
        base_dir: Base directory containing index.csv
        session_id: Session ID to update
        pdf_path: Path to generated PDF
    """
    index_path = Path(base_dir) / "index.csv"

    if not index_path.exists():
        logger.warning("Index not found at %s", index_path)
        return

    rows = []

    with open(index_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("session_id") == session_id:
                row["pdf_path"] = pdf_path

            if all(k in row for k in INDEX_COLUMNS):
                rows.append(row)
            else:
                logger.warning(
                    "Skipping malformed index row for session %s", row.get("session_id")
                )

    try:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Updated PDF path for session %s", session_id)
    except Exception as e:
        logger.error("Failed to update index at %s: %s", index_path, e)
