"""
Ramp Test Report Persistence — thin facade re-exporting all public symbols.

Handles saving of analysis results to filesystem in canonical JSON format.
Per methodology/ramp_test/10_canonical_json_spec.md.

Sub-modules:
  report_io.py        — NumpyEncoder, load_ramp_test_report, check_git_tracking
  index_manager.py    — _check_source_file_exists, _update_index, update_index_pdf_path
  pdf_generator.py    — _auto_generate_pdf, generate_and_save_pdf, generate_ramp_test_pdf
  report_generator.py — save_ramp_test_report, _get_limiter_interpretation
"""

from .report_io import (
    NumpyEncoder,
    load_ramp_test_report,
    check_git_tracking,
    CANONICAL_SCHEMA,
    CANONICAL_VERSION,
    METHOD_VERSION,
)
from .index_manager import (
    INDEX_COLUMNS,
    _check_source_file_exists,
    _update_index,
    update_index_pdf_path,
)
from .pdf_generator import (
    _auto_generate_pdf,
    generate_and_save_pdf,
    generate_ramp_test_pdf,
)
from .report_generator import (
    save_ramp_test_report,
    _get_limiter_interpretation,
)

__all__ = [
    "NumpyEncoder",
    "load_ramp_test_report",
    "check_git_tracking",
    "CANONICAL_SCHEMA",
    "CANONICAL_VERSION",
    "METHOD_VERSION",
    "INDEX_COLUMNS",
    "_check_source_file_exists",
    "_update_index",
    "update_index_pdf_path",
    "_auto_generate_pdf",
    "generate_and_save_pdf",
    "generate_ramp_test_pdf",
    "save_ramp_test_report",
    "_get_limiter_interpretation",
]
