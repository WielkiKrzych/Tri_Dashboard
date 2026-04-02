"""
Report I/O — JSON serialization, loading, and git tracking check.
"""

import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Union

import streamlit as st

from modules.calculations.version import RAMP_METHOD_VERSION

logger = logging.getLogger(__name__)

# Canonical schema constants
CANONICAL_SCHEMA = "ramp_test_result_v1.json"
CANONICAL_VERSION = "1.0.0"
METHOD_VERSION = RAMP_METHOD_VERSION


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


def load_ramp_test_report(file_path: Union[str, Path]) -> Dict:
    """
    Load a Ramp Test report from JSON.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with report data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_git_tracking(directory: str = "reports/ramp_tests"):
    """
    Check if a directory contains any files tracked by git.
    Display a warning in Streamlit if tracked files are found.

    This is a safeguard against accidental committing of sensitive subject data.
    """
    project_root = Path(__file__).parent.parent.parent
    if not (project_root / ".git").exists():
        return

    try:
        # nosec B603: subprocess call is safe - static command with controlled args
        result = subprocess.run(
            ["git", "ls-files", directory], capture_output=True, text=True, check=False
        )

        if result.returncode == 0 and result.stdout.strip():
            st.error(
                f"🚨 **SECURITY WARNING**: Folder `{directory}` zawiera pliki śledzone przez Git!\n\n"
                "Dane badanych mogą trafić do repozytorium. "
                "Usuń je z historii gita:\n"
                "```bash\n"
                f"git rm --cached -r {directory}\n"
                "```"
            )

    except Exception as e:
        logger.warning("Git tracking check failed: %s", e)
