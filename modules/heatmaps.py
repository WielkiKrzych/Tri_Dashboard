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
    zone_edges: Optional[List[Tuple[float, float, str]]] = None,
    power_col: str = "watts"
) -> Tuple[pd.DataFrame, Dict]:
    """Compute power zone heatmap pivot table (per minute).
    
    Args:
        df: DataFrame with power data
        ftp: Functional Threshold Power
        zone_edges: Optional custom zone definitions
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
    
    # Handle elapsed time in minutes
    if 'time' in df_work.columns:
        df_work['_minutes'] = (df_work['time'] // 60).astype(int)
    else:
        # If no time col, assume 1Hz and generate index
        df_work['_minutes'] = (np.arange(len(df_work)) // 60).astype(int)
    
    # Assign zones
    df_work['_zone'] = df_work[power_col].apply(lambda p: assign_power_zone(p, zones))
    
    # Filter out unassigned zones
    df_valid = df_work[df_work['_zone'].notna()].copy()
    
    if len(df_valid) == 0:
        pivot = pd.DataFrame(0, index=zone_names, columns=[0])
        return pivot, {"total_seconds": 0, "ftp": ftp}
    
    # Group by zone and minute
    grouped = df_valid.groupby(['_zone', '_minutes']).size()
    pivot_seconds = grouped.unstack(fill_value=0)
    
    # Ensure all zones are present
    for zone in zone_names:
        if zone not in pivot_seconds.index:
            pivot_seconds.loc[zone] = 0
    pivot_seconds = pivot_seconds.reindex(zone_names)
    
    # Ensure all minutes from 0 to max_min are present
    max_min = int(df_work['_minutes'].max())
    for m in range(max_min + 1):
        if m not in pivot_seconds.columns:
            pivot_seconds[m] = 0
    pivot_seconds = pivot_seconds.sort_index(axis=1)
    
    # Convert values to minutes
    pivot_minutes = pivot_seconds / 60.0
    
    # Metadata
    metadata = {
        "total_seconds": int(len(df_valid)),
        "ftp": ftp,
        "zone_distribution_min": {
            zone: float(pivot_minutes.loc[zone].sum()) for zone in zone_names
        }
    }
    
    return pivot_minutes, metadata


def plot_power_zone_heatmap(
    pivot: pd.DataFrame,
    title: str = "Rozkład Stref Mocy w Czasie"
) -> go.Figure:
    """Create Plotly heatmap figure (per minute).
    
    Args:
        pivot: Pivot table (minutes) from power_zone_heatmap
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    values = pivot.values
    
    # X-axis labels (minutes)
    x_labels = [f"{m}m" for m in pivot.columns]
    
    # Y-axis labels (zones)
    y_labels = list(pivot.index)
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0, 'rgb(20, 20, 30)'],
            [0.25, 'rgb(30, 80, 120)'],
            [0.5, 'rgb(50, 150, 100)'],
            [0.75, 'rgb(220, 180, 50)'],
            [1, 'rgb(220, 80, 50)']
        ],
        colorbar=dict(
            title="Minuty w strefie",
            tickformat=".1f"
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Minuta treningu: %{x}<br>"
            "Czas w strefie: <b>%{z:.2f} min</b>"
            "<extra></extra>"
        )
    ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title="Minuta Treningu",
            nticks=20
        ),
        yaxis=dict(
            title="Strefa Mocy",
            autorange="reversed"
        ),
        template="plotly_dark",
        height=400,
        margin=dict(l=100, r=20, t=60, b=60)
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
