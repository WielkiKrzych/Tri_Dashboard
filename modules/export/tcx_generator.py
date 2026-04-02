"""
TCX XML generator for Garmin/Strava import.

Produces TrainingCenterDatabase XML using stdlib xml.etree — no external deps.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def generate_tcx_bytes(
    df_plot: pd.DataFrame,
    metrics: dict[str, Any],
    cp: float,
    session_start: datetime | None = None,
    activity_type: str = "Bike",
) -> bytes:
    """Generate a TCX XML file from session data.

    Args:
        df_plot: Session DataFrame with at least a 'time' column.
        metrics: Session metrics dict (used for summary data).
        cp: Critical Power in watts.
        session_start: Override session start time. Defaults to now.
        activity_type: Garmin activity type (e.g. "Bike", "Run").

    Returns:
        UTF-8 encoded, pretty-printed TCX XML.
    """
    if session_start is None:
        session_start = datetime.now()

    from xml.dom import minidom
    from xml.etree.ElementTree import Element, SubElement, tostring

    ns = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"

    root = Element(f"{{{ns}}}TrainingCenterDatabase")
    root.set("xmlns", ns)
    activities = SubElement(root, f"{{{ns}}}Activities")
    activity = SubElement(activities, f"{{{ns}}}Activity")
    SubElement(activity, f"{{{ns}}}Id").text = session_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    SubElement(activity, f"{{{ns}}}Type").text = activity_type

    lap = SubElement(activity, f"{{{ns}}}Lap")
    lap.set("StartTime", session_start.strftime("%Y-%m-%dT%H:%M:%SZ"))
    SubElement(lap, f"{{{ns}}}Intensity").text = "Active"

    total_seconds = len(df_plot)
    total_watts = df_plot.get("watts", pd.Series(dtype=float)).sum()
    distance_m = metrics.get("distance_m", 0)

    SubElement(lap, f"{{{ns}}}TotalTimeSeconds").text = str(total_seconds)
    SubElement(lap, f"{{{ns}}}DistanceMeters").text = str(int(distance_m))
    SubElement(lap, f"{{{ns}}}Calories").text = str(int(total_watts / 1000 * 3.6))

    track = SubElement(lap, f"{{{ns}}}Track")

    time_col = df_plot.get("time", pd.Series(dtype=float))
    watts_col = df_plot.get("watts", pd.Series(dtype=float))
    hr_col = df_plot.get("heartrate", pd.Series(dtype=float))
    cadence_col = df_plot.get("cadence", pd.Series(dtype=float))

    max_points = min(len(df_plot), 36000)  # cap at ~10h of 1s data

    for i in range(max_points):
        tp = SubElement(track, f"{{{ns}}}Trackpoint")

        # Time
        if i < len(time_col) and pd.notna(time_col.iloc[i]):
            t = session_start + pd.Timedelta(seconds=float(time_col.iloc[i]))
        else:
            t = session_start + pd.Timedelta(seconds=i)
        SubElement(tp, f"{{{ns}}}Time").text = t.strftime("%Y-%m-%dT%H:%M:%SZ")

        # HeartRate
        if i < len(hr_col) and pd.notna(hr_col.iloc[i]):
            hr_elem = SubElement(tp, f"{{{ns}}}HeartRateBpm")
            SubElement(hr_elem, f"{{{ns}}}Value").text = str(int(hr_col.iloc[i]))

        # Cadence
        if i < len(cadence_col) and pd.notna(cadence_col.iloc[i]):
            SubElement(tp, f"{{{ns}}}Cadence").text = str(int(cadence_col.iloc[i]))

    rough = minidom.parseString(tostring(root, encoding="unicode"))
    return rough.toprettyxml(indent="  ", encoding="utf-8")
