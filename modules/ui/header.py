"""
Header UI Module.

Extracts sticky header and metric cards from app.py.
"""

import html
import warnings

import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_sticky_header(
    avg_power: float,
    avg_hr: float,
    avg_smo2: float,
    avg_cadence: float,
    avg_ve: float,
    duration_min: float,
) -> None:
    """Render the sticky metrics header.

    Args:
        avg_power: Average power in watts
        avg_hr: Average heart rate in bpm
        avg_smo2: Average SmO2 in %
        avg_cadence: Average cadence in rpm
        avg_ve: Average ventilation in L/min
        duration_min: Duration in minutes
    """
    warnings.warn(
        "render_sticky_header() is deprecated — use modules.frontend.components.UIComponents instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    st.markdown(
        f"""
    <div class="sticky-metrics">
        <h4>⚡ Live Training Summary</h4>
        <div class="metric-row">
            <div class="metric-box">
                <div class="label">Avg Power</div>
                <div class="value">{avg_power:.0f} <span class="unit">W</span></div>
            </div>
            <div class="metric-box">
                <div class="label">Avg HR</div>
                <div class="value">{avg_hr:.0f} <span class="unit">bpm</span></div>
            </div>
            <div class="metric-box">
                <div class="label">Avg SmO2</div>
                <div class="value">{avg_smo2:.1f} <span class="unit">%</span></div>
            </div>
            <div class="metric-box">
                <div class="label">Cadence</div>
                <div class="value">{avg_cadence:.0f} <span class="unit">rpm</span></div>
            </div>
            <div class="metric-box">
                <div class="label">Avg VE</div>
                <div class="value">{avg_ve:.0f} <span class="unit">L/min</span></div>
            </div>
            <div class="metric-box">
                <div class="label">Duration</div>
                <div class="value">{duration_min:.0f} <span class="unit">min</span></div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_cards(np_val: float, tss: float, if_val: float, work_kj: float) -> None:
    """Render the main metric cards row.

    Args:
        np_val: Normalized Power in watts
        tss: Training Stress Score
        if_val: Intensity Factor
        work_kj: Total work in kJ
    """
    warnings.warn(
        "render_metric_cards() is deprecated — use modules.frontend.components.UIComponents instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("NP (Norm. Power)", f"{np_val:.0f} W", help="Normalized Power (Coggan Formula)")
    m2.metric("TSS", f"{tss:.0f}", help=f"IF: {if_val:.2f}")
    m3.metric("Praca [kJ]", f"{work_kj:.0f}")


def show_breadcrumb(group: str, section: str = None) -> None:
    """Display breadcrumb navigation.

    Args:
        group: Current tab group name
        section: Current sub-section name (optional)
    """
    warnings.warn(
        "show_breadcrumb() is deprecated — use modules.frontend.components.UIComponents instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    safe_group = html.escape(group)
    if section:
        safe_section = html.escape(section)
        st.markdown(
            f"""
        <div class="breadcrumb-nav">
            🏠 Dashboard <span class="separator">›</span>
            {safe_group} <span class="separator">›</span>
            <span class="current">{safe_section}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="breadcrumb-nav">
            🏠 Dashboard <span class="separator">›</span>
            <span class="current">{safe_group}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


# DEPRECATED: Moved to services.session_analysis
# Re-exported for backward compatibility


def extract_header_data(df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract data needed for sticky header from DataFrame and metrics.

    Delegates to the canonical implementation in
    ``services.session_orchestrator.prepare_sticky_header_data`` and
    normalises empty-DataFrame edge-cases (NaN → 0) for backward compat.

    Args:
        df: Session DataFrame
        metrics: Calculated metrics dict

    Returns:
        Dict with header display values
    """
    from services.session_orchestrator import prepare_sticky_header_data

    result = prepare_sticky_header_data(df, metrics)
    # prepare_sticky_header_data does not guard len(df) for avg_smo2,
    # yielding NaN on empty DataFrames.  Normalise to 0 for compat.
    if len(df) == 0:
        result = {k: 0 for k in result}
    return result
