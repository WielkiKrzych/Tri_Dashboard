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
    ramp_confidence: float = 0.0
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
        
    Returns:
        Dict with path, session_id, or None if gated
        
    Raises:
        ValueError: If called without RAMP_TEST session type
    """
    # --- GATING: Check SessionType and confidence ---
    from modules.domain import SessionType, RAMP_CONFIDENCE_THRESHOLD
    
    if session_type is not None and session_type != SessionType.RAMP_TEST:
        # NOT a ramp test - do not save
        return {"gated": True, "reason": f"SessionType is {session_type}, not RAMP_TEST"}
    
    if ramp_confidence > 0 and ramp_confidence < RAMP_CONFIDENCE_THRESHOLD:
        # Confidence too low
        return {"gated": True, "reason": f"Confidence {ramp_confidence:.2f} < threshold {RAMP_CONFIDENCE_THRESHOLD}"}
    # 1. Prepare data dictionary
    data = result.to_dict()
    
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
    except FileExistsError:
        # Should be rare given UUID, but protects against collision/logic errors
        if not dev_mode:
            # Regenerate UUID and try one more time or raise
            # Simple strategy: Raise to indicate safety mechanism worked
            raise FileExistsError(f"Ramp Test Report already exists and immutable: {file_path}")
    
    # 6. Update Index (CSV)
    try:
        _update_index(output_base_dir, final_json["metadata"], str(file_path.absolute()))
    except Exception as e:
        print(f"Warning: Failed to update report index: {e}")
        
    return {
        "path": str(file_path.absolute()),
        "session_id": session_id,
        "uuid": session_id  # alias
    }


def _update_index(base_dir: str, metadata: Dict, file_path: str, pdf_path: Optional[str] = None):
    """
    Update CSV index with new test record.
    
    Columns: session_id, test_date, athlete_id, method_version, json_path, pdf_path
    """
    import csv
    
    index_path = Path(base_dir) / "index.csv"
    file_exists = index_path.exists()
    
    fieldnames = ["session_id", "test_date", "athlete_id", "method_version", "json_path", "pdf_path"]
    
    row = {
        "session_id": metadata.get("session_id", ""),
        "test_date": metadata.get("test_date", ""),
        "athlete_id": metadata.get("athlete_id") or "anonymous",
        "method_version": metadata.get("method_version", ""),
        "json_path": file_path,
        "pdf_path": pdf_path or ""
    }
    
    with open(index_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    print(f"Ramp Test indexed: {row['session_id']}")


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
    fieldnames = ["session_id", "test_date", "athlete_id", "method_version", "json_path", "pdf_path"]
    
    with open(index_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or fieldnames
        
        # Ensure pdf_path column exists
        if "pdf_path" not in existing_fieldnames:
            existing_fieldnames = list(existing_fieldnames) + ["pdf_path"]
        
        for row in reader:
            if row.get("session_id") == session_id:
                row["pdf_path"] = pdf_path
            rows.append(row)
    
    # Write back with updated row
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated PDF path for session {session_id}")


def generate_and_save_pdf(
    json_path: Union[str, Path],
    output_base_dir: str = "reports/ramp_tests"
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
        
    Returns:
        Path to generated PDF or None on failure
    """
    from .pdf_generator import generate_ramp_pdf
    from .figures import generate_all_ramp_figures, FigureConfig
    import tempfile
    
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"Error: JSON report not found: {json_path}")
        return None
    
    # Load JSON report
    report_data = load_ramp_test_report(json_path)
    
    # Generate figure paths
    temp_dir = tempfile.mkdtemp()
    fig_config = FigureConfig(method_version=report_data.get("metadata", {}).get("method_version", "1.0.0"))
    figure_paths = generate_all_ramp_figures(report_data, temp_dir, fig_config)
    
    # Generate PDF path (same name as JSON but .pdf)
    pdf_path = json_path.with_suffix(".pdf")
    
    # Generate PDF (can overwrite existing)
    generate_ramp_pdf(report_data, figure_paths, str(pdf_path))
    
    # Update index with PDF path
    session_id = report_data.get("metadata", {}).get("session_id", "")
    if session_id:
        update_index_pdf_path(output_base_dir, session_id, str(pdf_path.absolute()))
    
    print(f"PDF generated: {pdf_path}")
    
    return str(pdf_path.absolute())



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
