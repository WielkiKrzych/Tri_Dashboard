"""
Power Zone Heatmap Module.

Provides functions for:
- Defining power zones based on FTP
- Assigning power values to zones
- Creating heatmaps of time spent in zones by hour or weekday+hour
- Exporting results to JSON
"""
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ============================================================
# Default Power Zones (7 zones, Coggan-style)
# ============================================================

DEFAULT_ZONE_EDGES = [
    (0.00, 0.55, "Z1 Recovery"),
    (0.55, 0.75, "Z2 Endurance"),
    (0.75, 0.90, "Z3 Tempo"),
    (0.90, 1.05, "Z4 Threshold"),
    (1.05, 1.20, "Z5 VO2max"),
    (1.20, 1.50, "Z6 Anaerobic"),
    (1.50, 3.00, "Z7 Neuromuscular"),
]

ZONE_COLORS = [
    "#3498db",  # Z1 - Blue (Recovery)
    "#2ecc71",  # Z2 - Green (Endurance)
    "#f1c40f",  # Z3 - Yellow (Tempo)
    "#e67e22",  # Z4 - Orange (Threshold)
    "#e74c3c",  # Z5 - Red (VO2max)
    "#9b59b6",  # Z6 - Purple (Anaerobic)
    "#1a1a2e",  # Z7 - Dark (Neuromuscular)
]


@dataclass
class PowerZone:
    """Power zone definition."""
    name: str
    lower_pct: float  # Lower bound as % of FTP
    upper_pct: float  # Upper bound as % of FTP
    lower_watts: float  # Absolute lower bound
    upper_watts: float  # Absolute upper bound


def get_power_zones(
    ftp: float,
    zone_edges: Optional[List[Tuple[float, float, str]]] = None
) -> List[PowerZone]:
    """Get power zones based on FTP.
    
    Args:
        ftp: Functional Threshold Power in watts
        zone_edges: List of (lower_pct, upper_pct, name) tuples
        
    Returns:
        List of PowerZone objects
    """
    if zone_edges is None:
        zone_edges = DEFAULT_ZONE_EDGES
    
    zones = []
    for lower_pct, upper_pct, name in zone_edges:
        zones.append(PowerZone(
            name=name,
            lower_pct=lower_pct,
            upper_pct=upper_pct,
            lower_watts=ftp * lower_pct,
            upper_watts=ftp * upper_pct
        ))
    
    return zones


def assign_power_zone(power: float, zones: List[PowerZone]) -> Optional[str]:
    """Assign a power value to a zone.
    
    Args:
        power: Power value in watts
        zones: List of PowerZone objects
        
    Returns:
        Zone name or None if power is invalid
    """
    if pd.isna(power) or power <= 0:
        return None
    
    for zone in zones:
        if zone.lower_watts <= power < zone.upper_watts:
            return zone.name
    
    # If above all zones, assign to highest
    if power >= zones[-1].upper_watts:
        return zones[-1].name
    
    return None


def power_zone_heatmap(
    df: pd.DataFrame,
    ftp: float,
    resolution: str = "hourly",
    zone_edges: Optional[List[Tuple[float, float, str]]] = None,
    timestamp_col: str = "timestamp",
    power_col: str = "watts"
) -> Tuple[pd.DataFrame, Dict]:
    """Compute power zone heatmap pivot table.
    
    Args:
        df: DataFrame with timestamp and power columns
        ftp: Functional Threshold Power
        resolution: "hourly" (0-23) or "weekday_hourly" (Mon-Sun × 0-23)
        zone_edges: Optional custom zone definitions
        timestamp_col: Name of timestamp column
        power_col: Name of power column
        
    Returns:
        Tuple of (pivot_table, metadata_dict)
    """
    if power_col not in df.columns:
        raise ValueError(f"Power column '{power_col}' not found in DataFrame")
    
    # Get zones
    zones = get_power_zones(ftp, zone_edges)
    zone_names = [z.name for z in zones]
    
    # Prepare data
    df_work = df.copy()
    
    # Handle timestamp
    if timestamp_col in df_work.columns:
        df_work['_ts'] = pd.to_datetime(df_work[timestamp_col], errors='coerce')
    elif 'time' in df_work.columns:
        # If only elapsed time, use current date + time
        base_date = pd.Timestamp.now().normalize()
        df_work['_ts'] = base_date + pd.to_timedelta(df_work['time'], unit='s')
    else:
        # Generate synthetic timestamps at 1Hz
        base_date = pd.Timestamp.now().normalize()
        df_work['_ts'] = pd.date_range(start=base_date, periods=len(df_work), freq='1s')
    
    # Extract hour and weekday
    df_work['_hour'] = df_work['_ts'].dt.hour
    df_work['_weekday'] = df_work['_ts'].dt.dayofweek  # 0=Mon, 6=Sun
    df_work['_weekday_name'] = df_work['_ts'].dt.day_name()
    
    # Assign zones
    df_work['_zone'] = df_work[power_col].apply(lambda p: assign_power_zone(p, zones))
    
    # Filter out unassigned zones
    df_valid = df_work[df_work['_zone'].notna()].copy()
    
    if len(df_valid) == 0:
        # Return empty pivot
        if resolution == "hourly":
            cols = list(range(24))
        else:
            cols = pd.MultiIndex.from_product([range(7), range(24)], names=['weekday', 'hour'])
        pivot = pd.DataFrame(0, index=zone_names, columns=cols)
        return pivot, {"total_seconds": 0, "ftp": ftp}
    
    # Compute time per zone per time bucket (assuming 1Hz data = 1 second per row)
    if resolution == "hourly":
        # Group by zone and hour
        grouped = df_valid.groupby(['_zone', '_hour']).size()
        pivot = grouped.unstack(fill_value=0)
        
        # Ensure all hours and zones are present
        for h in range(24):
            if h not in pivot.columns:
                pivot[h] = 0
        pivot = pivot.sort_index(axis=1)
        
        for zone in zone_names:
            if zone not in pivot.index:
                pivot.loc[zone] = 0
        pivot = pivot.reindex(zone_names)
        
    else:  # weekday_hourly
        # Group by zone, weekday, and hour
        grouped = df_valid.groupby(['_zone', '_weekday', '_hour']).size()
        
        # Create multi-index pivot
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        columns = []
        for wd in range(7):
            for h in range(24):
                columns.append(f"{weekday_names[wd]}_{h:02d}")
        
        pivot = pd.DataFrame(0, index=zone_names, columns=columns)
        
        for (zone, wd, hour), count in grouped.items():
            col_name = f"{weekday_names[wd]}_{hour:02d}"
            if zone in pivot.index and col_name in pivot.columns:
                pivot.loc[zone, col_name] = count
    
    # Metadata
    metadata = {
        "total_seconds": int(len(df_valid)),
        "ftp": ftp,
        "resolution": resolution,
        "zone_distribution": {
            zone: int(pivot.loc[zone].sum()) for zone in zone_names
        }
    }
    
    return pivot, metadata


def plot_power_zone_heatmap(
    pivot: pd.DataFrame,
    resolution: str = "hourly",
    title: str = "Power Zone Heatmap"
) -> go.Figure:
    """Create Plotly heatmap figure.
    
    Args:
        pivot: Pivot table from power_zone_heatmap
        resolution: "hourly" or "weekday_hourly"
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    # Convert to minutes for readability
    values = pivot.values / 60  # seconds to minutes
    
    # X-axis labels
    if resolution == "hourly":
        x_labels = [f"{h:02d}:00" for h in range(24)]
    else:
        # Weekday_hourly: show abbreviated labels
        x_labels = list(pivot.columns)
    
    # Y-axis labels (zones)
    y_labels = list(pivot.index)
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0, 'rgb(20, 20, 30)'],      # Dark for 0
            [0.25, 'rgb(30, 80, 120)'],  # Blue
            [0.5, 'rgb(50, 150, 100)'],  # Green
            [0.75, 'rgb(220, 180, 50)'], # Yellow
            [1, 'rgb(220, 80, 50)']      # Red for max
        ],
        colorbar=dict(
            title="Czas [min]",
            tickformat=".0f"
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Godzina: %{x}<br>"
            "Czas: <b>%{z:.1f} min</b>"
            "<extra></extra>"
        )
    ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title="Godzina" if resolution == "hourly" else "Dzień × Godzina",
            tickangle=-45 if resolution == "weekday_hourly" else 0,
            tickfont=dict(size=10 if resolution == "weekday_hourly" else 12)
        ),
        yaxis=dict(
            title="Strefa Mocy",
            autorange="reversed"  # Z1 at top
        ),
        template="plotly_dark",
        height=450,
        margin=dict(l=100, r=20, t=60, b=100 if resolution == "weekday_hourly" else 60)
    )
    
    return fig


def export_heatmap_json(
    pivot: pd.DataFrame,
    metadata: Dict,
    output_path: str
) -> str:
    """Export heatmap data to JSON file.
    
    Args:
        pivot: Pivot table from power_zone_heatmap
        metadata: Metadata dict from power_zone_heatmap
        output_path: Path for output JSON file
        
    Returns:
        Path to the created file
    """
    data = {
        "metadata": metadata,
        "heatmap": pivot.to_dict()
    }
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported heatmap to {path}")
    return str(path)
