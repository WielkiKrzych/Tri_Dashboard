"""
TrainingPeaks-compatible CSV export.

Produces a single-row CSV with session summary metrics and power zone time distribution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


# Power zone boundaries (fractions of CP) — 7-zone Coggan
_POWER_ZONE_BOUNDS: list[tuple[float, float]] = [
    (0.0, 0.55),
    (0.55, 0.75),
    (0.75, 0.90),
    (0.90, 1.05),
    (1.05, 1.20),
    (1.20, 1.50),
    (1.50, 999.0),
]


def export_trainingpeaks_csv(
    metrics: dict[str, Any],
    df_plot: pd.DataFrame,
    cp: float,
    rider_weight: float,
    session_date: str | None = None,
    filename: str = "session",
) -> bytes:
    """Export session summary as TrainingPeaks-compatible CSV.

    Args:
        metrics: Session metrics dict.
        df_plot: Session DataFrame with 'watts' column.
        cp: Critical Power in watts.
        rider_weight: Rider weight in kg.
        session_date: Override session date (YYYY-MM-DD). Defaults to today.
        filename: Base filename for the session.

    Returns:
        UTF-8 encoded CSV content (header + one data row).
    """
    if session_date is None:
        session_date = datetime.now().strftime("%Y-%m-%d")

    duration_min = len(df_plot) / 60.0
    np_val = metrics.get("normalized_power", metrics.get("np", 0))
    if_val = np_val / cp if cp > 0 else 0
    tss_val = metrics.get("tss", 0)
    avg_hr = metrics.get("avg_hr", 0)
    max_hr = metrics.get("max_hr", 0)
    avg_cad = metrics.get("avg_cadence", 0)
    vo2max = metrics.get("vo2_max_est", 0)

    # Time in power zones (seconds)
    watts_col = df_plot.get("watts", pd.Series(dtype=float))
    zones_time: dict[str, int] = {}
    for zi, (lo, hi) in enumerate(_POWER_ZONE_BOUNDS, 1):
        if cp > 0 and len(watts_col) > 0:
            mask = (watts_col >= cp * lo) & (watts_col < cp * hi)
        else:
            mask = pd.Series(False, index=watts_col.index)
        zones_time[f"pwr_zone_{zi}_sec"] = int(mask.sum())

    row: dict[str, Any] = {
        "date": session_date,
        "filename": filename,
        "duration_min": round(duration_min, 1),
        "tss": round(tss_val, 0),
        "if": round(if_val, 2),
        "np": round(np_val, 0),
        "avg_power": round(metrics.get("avg_power", 0), 0),
        "avg_hr": round(avg_hr, 0),
        "max_hr": round(max_hr, 0),
        "avg_cadence": round(avg_cad, 0),
        "vo2max_est": round(vo2max, 1),
        "work_kj": round(metrics.get("total_work", 0) / 1000, 1),
        "rider_weight_kg": rider_weight,
        "cp_watts": cp,
    }
    row.update(zones_time)

    header = ",".join(row.keys())
    values = ",".join(str(v) for v in row.values())
    return f"{header}\n{values}\n".encode("utf-8")
