"""
CSV export utilities for Tri_Dashboard session data.

Provides two export functions:
- export_session_csv : raw time-series data from df_plot
- export_metrics_csv : flat summary of the metrics dict

Both return bytes ready for Streamlit's st.download_button.
"""

from io import StringIO
from typing import Any, Dict

import pandas as pd


def export_session_csv(df: pd.DataFrame) -> bytes:
    """
    Serialize the session time-series DataFrame to CSV bytes.

    Columns are preserved as-is; the index is excluded.

    Args:
        df: Session DataFrame (df_plot from process_uploaded_session).

    Returns:
        UTF-8 encoded CSV bytes.
    """
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_metrics_csv(metrics: Dict[str, Any]) -> bytes:
    """
    Serialize the session metrics dict to a two-column CSV (metric, value).

    Nested dicts/lists are JSON-serialised into the value cell so every
    row remains flat and the file stays importable by Excel / Numbers.

    Private keys (starting with ``_``) are excluded automatically.

    Args:
        metrics: Metrics dict returned by process_uploaded_session.

    Returns:
        UTF-8 encoded CSV bytes.
    """
    import json

    rows = []
    for key, value in sorted(metrics.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append({"metric": key, "value": value})

    df = pd.DataFrame(rows, columns=["metric", "value"])
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
