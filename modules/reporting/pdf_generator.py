"""
PDF Generator — auto-generation and manual regeneration of ramp test PDFs.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import streamlit as st

from .report_io import load_ramp_test_report
from .index_manager import update_index_pdf_path

logger = logging.getLogger(__name__)


def _auto_generate_pdf(
    json_path: str,
    report_data: Dict,
    is_conditional: bool = False,
    source_df=None,
    manual_overrides=None,
) -> Optional[str]:
    """
    Auto-generate PDF from JSON report.

    Called automatically after save_ramp_test_report.
    PDF is saved next to JSON with same basename.

    Args:
        json_path: Absolute path to saved JSON
        report_data: The report data dictionary
        is_conditional: If True, PDF will include conditional warning
        source_df: Optional DataFrame with raw data for chart generation
        manual_overrides: Dict of manual threshold values from session_state

    Returns:
        PDF path if successful, None otherwise
    """
    if not st.session_state.get("report_generation_requested", False):
        logger.info("[PDF GATING] PDF generation NOT requested (Hard Trigger). Aborting.")
        return None

    from .pdf import generate_ramp_pdf, PDFConfig
    from .figures import generate_all_ramp_figures
    import tempfile

    json_path = Path(json_path)
    pdf_path = json_path.with_suffix(".pdf")

    temp_dir = tempfile.mkdtemp()
    method_version = report_data.get("metadata", {}).get("method_version", "1.0.0")
    fig_config = {"method_version": method_version}

    figure_paths = generate_all_ramp_figures(report_data, temp_dir, fig_config, source_df=source_df)

    pdf_config = PDFConfig(is_conditional=is_conditional)

    generate_ramp_pdf(
        report_data, figure_paths, str(pdf_path), pdf_config, manual_overrides=manual_overrides
    )

    try:
        from .docx_builder import build_ramp_docx

        docx_path = pdf_path.with_suffix(".docx")
        build_ramp_docx(report_data, figure_paths, str(docx_path))
        logger.info("Ramp Test DOCX generated: %s", docx_path)
    except Exception as e:
        logger.error("DOCX generation failed: %s", e)

    logger.info("Ramp Test PDF generated: %s", pdf_path)

    st.session_state["report_generation_requested"] = False

    return str(pdf_path.absolute())


def generate_and_save_pdf(
    json_path: Union[str, Path],
    output_base_dir: str = "reports/ramp_tests",
    is_conditional: bool = False,
    manual_overrides: Optional[Dict] = None,
) -> Optional[str]:
    """
    Generate PDF from existing JSON report and save alongside it.

    - PDF is saved next to the JSON with .pdf extension
    - PDF can be regenerated (overwritten)
    - JSON is NEVER modified (immutable)
    - Index is updated with PDF path

    Args:
        json_path: Path to the canonical JSON report
        output_base_dir: Base directory for index update
        is_conditional: Whether to include conditional warning
        manual_overrides: Dict of manual threshold values from session_state

    Returns:
        Path to generated PDF or None on failure
    """
    from .pdf import generate_ramp_pdf, PDFConfig
    from .figures import generate_all_ramp_figures
    import tempfile

    json_path = Path(json_path)

    if not json_path.exists():
        logger.error("JSON report not found: %s", json_path)
        return None

    report_data = load_ramp_test_report(json_path)

    temp_dir = tempfile.mkdtemp()
    fig_config = {"method_version": report_data.get("metadata", {}).get("method_version", "1.0.0")}
    figure_paths = generate_all_ramp_figures(
        report_data, temp_dir, fig_config, source_df=None, manual_overrides=manual_overrides
    )

    pdf_path = json_path.with_suffix(".pdf")

    pdf_config = PDFConfig(is_conditional=is_conditional)

    generate_ramp_pdf(
        report_data, figure_paths, str(pdf_path), pdf_config, manual_overrides=manual_overrides
    )

    try:
        from .docx_builder import build_ramp_docx

        docx_path = pdf_path.with_suffix(".docx")
        build_ramp_docx(report_data, figure_paths, str(docx_path))
        logger.info("DOCX generated: %s", docx_path)
    except Exception as e:
        logger.error("DOCX failure: %s", e)

    session_id = report_data.get("metadata", {}).get("session_id", "")
    if session_id:
        update_index_pdf_path(output_base_dir, session_id, str(pdf_path.absolute()))

    logger.info("PDF generated: %s", pdf_path)

    return str(pdf_path.absolute())


def generate_ramp_test_pdf(
    session_id: str,
    output_base_dir: str = "reports/ramp_tests",
    manual_overrides: Optional[Dict] = None,
) -> Optional[str]:
    """
    Ręczne generowanie raportu PDF na podstawie session_id.

    1. Znajduje json_path w index.csv
    2. Wczytuje JSON
    3. Generuje PDF (z opcjonalnymi wartościami manualnymi) i aktualizuje index

    Args:
        session_id: ID sesji raportu
        output_base_dir: Katalog bazowy raportów
        manual_overrides: Dict z manualnymi wartościami progów (VT1/VT2/SmO2/CP)
    """
    logger.info("Generating PDF for session_id: %s", session_id)
    if manual_overrides:
        logger.debug("With manual overrides: %s", list(manual_overrides.keys()))

    index_path = Path(output_base_dir) / "index.csv"
    if not index_path.exists():
        logger.error("Index not found at %s", index_path)
        return None

    json_path = None
    with open(index_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("session_id") == session_id:
                json_path = row.get("json_path")
                break

    if not json_path:
        logger.error("JSON path not found in index for session %s", session_id)
        return None

    pdf_path_str = generate_and_save_pdf(
        json_path, output_base_dir, manual_overrides=manual_overrides
    )

    if pdf_path_str:
        logger.info("PDF saved to: %s", pdf_path_str)
        logger.info("index.csv updated for session_id: %s", session_id)
        return pdf_path_str

    return None
