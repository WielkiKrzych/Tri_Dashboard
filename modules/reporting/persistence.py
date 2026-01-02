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
    dev_mode: bool = False
) -> str:
    """
    Save Ramp Test result to JSON file.
    
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
        
    Returns:
        Absolute path to the saved file
    """
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
        
    return str(file_path.absolute())


def _update_index(base_dir: str, metadata: Dict, file_path: str):
    """
    Update CSV index with new test record.
    
    Columns: session_id, test_date, athlete_id, method_version, json_path
    """
    import csv
    
    index_path = Path(base_dir) / "index.csv"
    file_exists = index_path.exists()
    
    fieldnames = ["session_id", "test_date", "athlete_id", "method_version", "json_path"]
    
    row = {
        "session_id": metadata.get("session_id", ""),
        "test_date": metadata.get("test_date", ""),
        "athlete_id": metadata.get("athlete_id") or "anonymous",
        "method_version": metadata.get("method_version", ""),
        "json_path": file_path
    }
    
    with open(index_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
