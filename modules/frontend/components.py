"""
Frontend Components Module.

Reusable UI components (widgets) for the application.
"""

import html

import streamlit as st
from typing import Dict, Any


class UIComponents:
    """Namespace for reusable UI components."""

    @staticmethod
    def show_breadcrumb(group: str, section: str = None) -> None:
        """Render a breadcrumb navigation aid."""
        safe_group = html.escape(str(group))
        safe_section = html.escape(str(section)) if section else None

        if safe_section:
            html_content = f"""
            <div class="breadcrumb-nav">
                🏠 Dashboard <span class="separator">›</span>
                {safe_group} <span class="separator">›</span>
                <span class="current">{safe_section}</span>
            </div>
            """
        else:
            html_content = f"""
            <div class="breadcrumb-nav">
                🏠 Dashboard <span class="separator">›</span>
                <span class="current">{safe_group}</span>
            </div>
            """
        st.markdown(html_content, unsafe_allow_html=True)

    @staticmethod
    def render_sticky_header(data: Dict[str, Any]) -> None:
        """Render the sticky metrics header."""
        if not data:
            return

        html = f"""
        <div class="sticky-metrics">
            <h4>⚡ Live Training Summary</h4>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="label">Avg Power</div>
                    <div class="value">{data.get("avg_power", 0):.0f} <span class="unit">W</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg HR</div>
                    <div class="value">{data.get("avg_hr", 0):.0f} <span class="unit">bpm</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg SmO2</div>
                    <div class="value">{data.get("avg_smo2", 0):.1f} <span class="unit">%</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Cadence</div>
                    <div class="value">{data.get("avg_cadence", 0):.0f} <span class="unit">rpm</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg VE</div>
                    <div class="value">{data.get("avg_ve", 0):.0f} <span class="unit">L/min</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Duration</div>
                    <div class="value">{data.get("duration_min", 0):.0f} <span class="unit">min</span></div>
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# STREAMLIT FRAGMENTS - Independent UI sections
# =============================================================================
# Fragments allow independent reruns - changes in one fragment don't trigger
# rerun of the entire page. Use for static sections that rarely change.


@st.fragment
def render_header_metrics_fragment(
    np_header: float, if_header: float, tss_header: float, df_plot
) -> None:
    """
    Render header metrics as independent fragment.

    This fragment only reruns when explicitly triggered, not when
    other parts of the page change.
    """
    m1, m2, m3 = st.columns(3)
    m1.metric("NP (Norm. Power)", f"{np_header:.0f} W")
    m2.metric("TSS", f"{tss_header:.0f}", help=f"IF: {if_header:.2f}")
    m3.metric("Praca [kJ]", f"{df_plot['watts'].sum() / 1000:.0f}")


@st.fragment
def render_export_buttons_fragment(
    safe_filename: str,
    metrics: dict,
    df_plot,
    df_plot_resampled,
    uploaded_file,
    cp_input: int,
    vt1_watts: int,
    vt2_watts: int,
    rider_weight: float,
    vt1_vent: float,
    vt2_vent: float,
    w_prime_input: int,
) -> None:
    """
    Render export buttons as independent fragment.

    Export buttons rarely change after data is loaded,
    so this fragment minimizes unnecessary reruns.
    """
    from modules.reports import export_all_charts_as_png

    try:
        zip_data = export_all_charts_as_png(
            df_plot,
            df_plot_resampled,
            cp_input,
            vt1_watts,
            vt2_watts,
            metrics,
            rider_weight,
            uploaded_file,
            None,
            None,
            None,
            None,
        )
        st.download_button("📸 PNG", zip_data, f"{safe_filename}.zip", mime="application/zip")
    except Exception as e:
        st.warning(f"PNG export issue: {e}")
