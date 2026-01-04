"""
Ramp Test Report Persistence.

Handles saving of analysis results to filesystem in canonical JSON format.
Per methodology/ramp_test/10_canonical_json_spec.md.
"""
import json
import os
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import subprocess
import streamlit as st

from models.results import RampTestResult
from modules.calculations.version import RAMP_METHOD_VERSION

# canonical version of the JSON structure
CANONICAL_SCHEMA = "ramp_test_result_v1.json"
CANONICAL_VERSION = "1.0.0"
METHOD_VERSION = RAMP_METHOD_VERSION  # Pipeline version

# Index structure
INDEX_COLUMNS = ["session_id", "test_date", "athlete_id", "method_version", "json_path", "pdf_path", "source_file"]


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
    import csv
    
    index_path = Path(base_dir) / "index.csv"
    
    if not index_path.exists():
        return False
    
    try:
        with open(index_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_source = row.get("source_file", "")
                if existing_source and existing_source == source_file:
                    return True
    except Exception as e:
        print(f"Warning: Failed to check deduplication: {e}")
        return False
    
    return False


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_ramp_test_report(
    result: RampTestResult,
    output_base_dir: str = "reports/ramp_tests",
    athlete_id: Optional[str] = None,
    notes: Optional[str] = None,
    dev_mode: bool = False,
    session_type = None,
    ramp_confidence: float = 0.0,
    source_file: Optional[str] = None,
    source_df = None
) -> Dict:
    """
    Save Ramp Test result to JSON file.
    
    GATING: Only saves if session_type is RAMP_TEST and confidence >= threshold.
    
    Generates path: {output_base_dir}/YYYY/MM/ramp_test_{date}_{uuid}.json
    Enriches result with metadata (UUID, timestamps).
    
    Safety:
    - By default, writes with 'x' mode (exclusive creation).
    - Checks for file existence to prevent overwrites.
    - 'dev_mode=True' allows overwriting.
    
    Args:
        result: Analysis result object
        output_base_dir: Base directory for reports
        athlete_id: Optional athlete identifier
        notes: Optional analysis notes
        dev_mode: If True, allows overwriting existing files
        session_type: SessionType enum (must be RAMP_TEST to save)
        ramp_confidence: Classification confidence (must be >= threshold)
        source_file: Original CSV filename for deduplication
        source_df: Optional source DataFrame for chart generation
        
    Returns:
        Dict with path, session_id, or None if gated
        
    Raises:
        ValueError: If called without RAMP_TEST session type
    """
    # --- DEDUPLICATION: Check if source_file already exists in index ---
    if source_file:
        if _check_source_file_exists(output_base_dir, source_file):
            print(f"[Dedup] Source file '{source_file}' already exists in index. Skipping save.")
            return {"gated": True, "reason": f"Source file '{source_file}' already saved", "deduplicated": True}
    # --- GATING: Check SessionType and confidence ---
    from modules.domain import SessionType
    
    # Allowed types for saving
    ALLOWED_TYPES = [SessionType.RAMP_TEST, SessionType.RAMP_TEST_CONDITIONAL]
    
    if session_type is not None and session_type not in ALLOWED_TYPES:
        # NOT a ramp test - do not save
        return {"gated": True, "reason": f"SessionType is {session_type}, not a Ramp Test"}
    
    # Minimum confidence for any ramp save is 0.5 (2/4 criteria)
    if ramp_confidence > 0 and ramp_confidence < 0.5:
        # Confidence too low even for conditional
        return {"gated": True, "reason": f"Confidence {ramp_confidence:.2f} too low to save report"}
    # 1. Prepare data dictionary
    data = result.to_dict()
    
    # 1.1 Add time series if source_df is available (for regeneration support)
    if source_df is not None and len(source_df) > 0:
        df_ts = source_df.copy()
        df_ts.columns = df_ts.columns.str.lower().str.strip()
        
        # Mapping: df_column -> json_key
        # We save raw values to keep the JSON canonical, 
        # but limited to key metrics to keep it reasonably sized.
        ts_map = {
            'watts': 'power_watts',
            'power': 'power_watts',
            'hr': 'hr_bpm',
            'heartrate': 'hr_bpm',
            'heart_rate': 'hr_bpm',
            'smo2': 'smo2_pct',
            'smo2_pct': 'smo2_pct',
            'tymeventilation': 've_lmin',
            've': 've_lmin',
            'torque': 'torque_nm',
            'cadence': 'cadence_rpm',
            'cad': 'cadence_rpm'
        }
        
        ts_data = {}
        # Always try to get time
        if 'time' in df_ts.columns:
            ts_data['time_sec'] = df_ts['time'].tolist()
        elif 'seconds' in df_ts.columns:
            ts_data['time_sec'] = df_ts['seconds'].tolist()
        else:
            ts_data['time_sec'] = list(range(len(df_ts)))
            
        for df_col, json_key in ts_map.items():
            if df_col in df_ts.columns and json_key not in ts_data:
                ts_data[json_key] = df_ts[df_col].fillna(0).tolist()
        
        data['time_series'] = ts_data
    
    # 2. Enrich metadata
    now = datetime.now()
    analysis_timestamp = now.isoformat()
    session_id = str(uuid.uuid4())
    
    # Parse test date for directory structure
    try:
        test_date = datetime.strptime(result.test_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        # Fallback if date missing or invalid format
        test_date = now.date()
    
    # Update/Enrich metadata section
    if "metadata" not in data:
        data["metadata"] = {}
        
    data["metadata"].update({
        "test_date": result.test_date or test_date.isoformat(),
        "analysis_timestamp": analysis_timestamp,
        "method_version": METHOD_VERSION,
        "session_id": session_id,
        "athlete_id": athlete_id,
        "notes": notes,
        "analyzer": "Tri_Dashboard/ramp_pipeline"
    })
    
    # 3. Add canonical header fields
    final_json = {
        "$schema": CANONICAL_SCHEMA,
        "version": CANONICAL_VERSION,
        **data
    }
    
    # 4. Generate path
    year_str = test_date.strftime("%Y")
    month_str = test_date.strftime("%m")
    
    # Directory: reports/ramp_tests/2026/01/
    save_dir = Path(output_base_dir) / year_str / month_str
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename: ramp_test_2026-01-02_abc123.json
    # Use short UUID (first 8 chars) for readability
    short_uuid = session_id[:8]
    filename = f"ramp_test_{test_date.isoformat()}_{short_uuid}.json"
    
    file_path = save_dir / filename
    
    # 5. Save file (Immutable by default)
    mode = 'w' if dev_mode else 'x'
    
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"Ramp Test JSON saved: {session_id}")
    except FileExistsError:
        # Should be rare given UUID, but protects against collision/logic errors
        if not dev_mode:
            # Regenerate UUID and try one more time or raise
            # Simple strategy: Raise to indicate safety mechanism worked
            raise FileExistsError(f"Ramp Test Report already exists and immutable: {file_path}")
    
    # 6. Check validity for PDF generation
    validity_section = final_json.get("test_validity", {})
    test_validity_status = validity_section.get("status", "unknown")
    
    # Block PDF ONLY if test_validity_status == "invalid"
    should_generate_pdf = test_validity_status != "invalid"
    
    # Determine if conditional (for PDF warning)
    is_conditional = session_type == SessionType.RAMP_TEST_CONDITIONAL
    
    pdf_path = None
    
    # 7. Auto-generate PDF if valid
    if should_generate_pdf:
        try:
            pdf_path = _auto_generate_pdf(str(file_path.absolute()), final_json, is_conditional, source_df=source_df)
        except Exception as e:
            # PDF failure does NOT affect JSON or index
            print(f"Warning: PDF generation failed for {session_id}: {e}")
    
    # 8. Update Index (CSV)
    try:
        _update_index(output_base_dir, final_json["metadata"], str(file_path.absolute()), pdf_path, source_file)
        print(f"Ramp Test indexed: {session_id}")
    except Exception as e:
        print(f"Warning: Failed to update report index: {e}")
        
    return {
        "path": str(file_path.absolute()),
        "pdf_path": pdf_path,
        "session_id": session_id,
        "uuid": session_id  # alias
    }


def _auto_generate_pdf(json_path: str, report_data: Dict, is_conditional: bool = False, source_df = None) -> Optional[str]:
    """
    Auto-generate PDF from JSON report.
    
    Called automatically after save_ramp_test_report.
    PDF is saved next to JSON with same basename.
    
    Args:
        json_path: Absolute path to saved JSON
        report_data: The report data dictionary
        is_conditional: If True, PDF will include conditional warning
        source_df: Optional DataFrame with raw data for chart generation
        
    Returns:
        PDF path if successful, None otherwise
    """
    from .pdf import generate_ramp_pdf, PDFConfig
    from .figures import generate_all_ramp_figures
    import tempfile
    
    json_path = Path(json_path)
    pdf_path = json_path.with_suffix(".pdf")
    
    # Generate figures in temp directory
    temp_dir = tempfile.mkdtemp()
    method_version = report_data.get("metadata", {}).get("method_version", "1.0.0")
    fig_config = {"method_version": method_version}
    
    # Pass source_df for chart generation
    # If source_df is missing (regeneration from index), charts will try to use report_data['time_series']
    figure_paths = generate_all_ramp_figures(report_data, temp_dir, fig_config, source_df=source_df)
    
    # Configure PDF with conditional flag
    pdf_config = PDFConfig(is_conditional=is_conditional)
    
    # Generate PDF
    generate_ramp_pdf(report_data, figure_paths, str(pdf_path), pdf_config)
    
    # Generate DOCX (optional)
    try:
        from .docx_builder import build_ramp_docx
        docx_path = pdf_path.with_suffix(".docx")
        build_ramp_docx(report_data, figure_paths, str(docx_path))
        print(f"Ramp Test DOCX generated: {docx_path}")
    except Exception as e:
        print(f"DOCX generation failed: {e}")
    
    print(f"Ramp Test PDF generated: {pdf_path}")
    
    return str(pdf_path.absolute())



def _update_index(base_dir: str, metadata: Dict, file_path: str, pdf_path: Optional[str] = None, source_file: Optional[str] = None):
    """
    Update CSV index with new test record.
    
    Columns: session_id, test_date, athlete_id, method_version, json_path, pdf_path, source_file
    """
    import csv
    
    index_path = Path(base_dir) / "index.csv"
    file_exists = index_path.exists()
    
    row = {
        "session_id": metadata.get("session_id", ""),
        "test_date": metadata.get("test_date", ""),
        "athlete_id": metadata.get("athlete_id") or "anonymous",
        "method_version": metadata.get("method_version", ""),
        "json_path": file_path,
        "pdf_path": pdf_path or "",
        "source_file": source_file or ""
    }
    
    # Validation: Ensure all columns are present and no empty critical fields
    if len(row) != len(INDEX_COLUMNS):
        print(f"Error: Invalid record length for index. Expected {len(INDEX_COLUMNS)}, got {len(row)}.")
        return

    if not row["session_id"] or not row["json_path"]:
        print(f"Error: Missing critical data for index (session_id or json_path). Record not saved.")
        return

    try:
        with open(index_path, 'a', newline='', encoding='utf-8') as f:
            # quote_all ensures paths (and other strings) are in quotes as requested
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Ramp Test indexed: {row['session_id']}")
    except Exception as e:
        print(f"Error: Failed to write to index at {index_path}: {e}")


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
    import csv
    
    index_path = Path(base_dir) / "index.csv"
    
    if not index_path.exists():
        print(f"Warning: Index not found at {index_path}")
        return
    
    # Read all rows
    rows = []
    
    with open(index_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("session_id") == session_id:
                row["pdf_path"] = pdf_path
            
            # Basic validation for existing row
            if all(k in row for k in INDEX_COLUMNS):
                rows.append(row)
            else:
                print(f"Warning: Skipping malformed index row for session {row.get('session_id')}")
    
    # Write back with updated row
    try:
        with open(index_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated PDF path for session {session_id}")
    except Exception as e:
        print(f"Error: Failed to update index at {index_path}: {e}")


def generate_and_save_pdf(
    json_path: Union[str, Path],
    output_base_dir: str = "reports/ramp_tests",
    is_conditional: bool = False
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
        
    Returns:
        Path to generated PDF or None on failure
    """
    from .pdf import generate_ramp_pdf, PDFConfig
    from .figures import generate_all_ramp_figures
    import tempfile
    
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"Error: JSON report not found: {json_path}")
        return None
    
    # Load JSON report
    report_data = load_ramp_test_report(json_path)
    
    # Generate figure paths
    # NOTE: Charts will show "Brak danych" since source_df is not available
    # during regeneration from JSON. Full charts are only generated during 
    # initial save when DataFrame is available.
    temp_dir = tempfile.mkdtemp()
    fig_config = {"method_version": report_data.get("metadata", {}).get("method_version", "1.0.0")}
    figure_paths = generate_all_ramp_figures(report_data, temp_dir, fig_config, source_df=None)
    
    # Generate PDF path (same name as JSON but .pdf)
    pdf_path = json_path.with_suffix(".pdf")
    
    # Configure PDF
    pdf_config = PDFConfig(is_conditional=is_conditional)
    
    # Generate PDF (can overwrite existing)
    generate_ramp_pdf(report_data, figure_paths, str(pdf_path), pdf_config)
    
    # Generate DOCX (optional)
    try:
        from .docx_builder import build_ramp_docx
        docx_path = pdf_path.with_suffix(".docx")
        build_ramp_docx(report_data, figure_paths, str(docx_path))
        print(f"DOCX generated: {docx_path}")
    except Exception as e:
        print(f"DOCX failure: {e}")
    
    # Update index with PDF path
    session_id = report_data.get("metadata", {}).get("session_id", "")
    if session_id:
        update_index_pdf_path(output_base_dir, session_id, str(pdf_path.absolute()))
    
    print(f"PDF generated: {pdf_path}")
    
    return str(pdf_path.absolute())


def generate_ramp_test_pdf(session_id: str, output_base_dir: str = "reports/ramp_tests") -> Optional[str]:
    """
    Ręczne generowanie raportu PDF na podstawie session_id.
    
    1. Znajduje json_path w index.csv
    2. Wczytuje JSON
    3. Generuje PDF i aktualizuje index
    """
    import csv
    print(f"Generating PDF for session_id: {session_id}")
    
    index_path = Path(output_base_dir) / "index.csv"
    if not index_path.exists():
        print(f"Error: Index not found at {index_path}")
        return None
        
    json_path = None
    with open(index_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("session_id") == session_id:
                json_path = row.get("json_path")
                break
                
    if not json_path:
        print(f"Error: JSON path not found in index for session {session_id}")
        return None
        
    # Re-use existing logic for generation
    pdf_path_str = generate_and_save_pdf(json_path, output_base_dir)
    
    if pdf_path_str:
        print(f"PDF saved to: {pdf_path_str}")
        print(f"index.csv updated for session_id: {session_id}")
        return pdf_path_str
        
    return None



def load_ramp_test_report(file_path: Union[str, Path]) -> Dict:
    """
    Load a Ramp Test report from JSON.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with report data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_git_tracking(directory: str = "reports/ramp_tests"):
    """
    Check if a directory contains any files tracked by git.
    Display a warning in Streamlit if tracked files are found.
    
    This is a safeguard against accidental committing of sensitive subject data.
    """
    # Only check in local development environment (could verify env vars but simple check is enough)
    if not os.path.exists(".git"):
        return
        
    try:
        # Check if any files in the directory are tracked
        # git ls-files returns output if files are tracked
        result = subprocess.run(
            ["git", "ls-files", directory],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Tracked files found!
            st.error(
                f"🚨 **SECURITY WARNING**: Folder `{directory}` zawiera pliki śledzone przez Git!\n\n"
                "Dane badanych mogą trafić do repozytorium. "
                "Usuń je z historii gita:\n"
                "```bash\n"
                f"git rm --cached -r {directory}\n"
                "```"
            )
            
    except Exception:
        # Git command failed or not available - ignore
        pass
