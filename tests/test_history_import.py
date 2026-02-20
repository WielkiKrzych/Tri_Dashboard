"""Tests for modules/history_import.py — date extraction and file listing."""

import pytest
import tempfile
from pathlib import Path
from modules.history_import import extract_date_from_filename, get_available_files


class TestExtractDateFromFilename:
    def test_yyyy_mm_dd(self):
        assert extract_date_from_filename("2024-12-28_trening.csv") == "2024-12-28"

    def test_dd_mm_yyyy(self):
        assert extract_date_from_filename("trening_28.12.2024.csv") == "2024-12-28"

    def test_yyyymmdd(self):
        assert extract_date_from_filename("20241228.csv") == "2024-12-28"

    def test_yyyymmdd_in_session(self):
        assert extract_date_from_filename("session_20241228_120000.csv") == "2024-12-28"

    def test_no_date(self):
        assert extract_date_from_filename("random_file.csv") is None

    def test_bad_year_yyyymmdd(self):
        """Year outside 2020-2030 range should not match YYYYMMDD pattern."""
        assert extract_date_from_filename("19991228.csv") is None

    def test_yyyy_mm_dd_takes_priority(self):
        """If multiple patterns match, first one wins."""
        result = extract_date_from_filename("2024-01-15_session.csv")
        assert result == "2024-01-15"


class TestGetAvailableFiles:
    def test_nonexistent_folder(self):
        result = get_available_files(Path("/nonexistent/path"))
        assert result == []

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_available_files(Path(tmpdir))
            assert result == []

    def test_finds_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "2024-01-15_session.csv").write_text("watts,hr\n200,120\n")
            (Path(tmpdir) / "2024-02-20_session.csv").write_text("watts,hr\n250,130\n")
            (Path(tmpdir) / "readme.txt").write_text("not a csv")

            result = get_available_files(Path(tmpdir))
            assert len(result) == 2
            assert all(r["name"].endswith(".csv") for r in result)
            assert "date" in result[0]
            assert "size" in result[0]
