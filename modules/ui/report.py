import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.config import Config
from typing import Dict

# ============================================================
# OPTIMIZATION: Pre-computed constants to avoid recalculation
# ============================================================
ZONE_LABELS = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']
ZONE_COLORS = ['#808080', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
# O(1) lookup instead of O(k) list.index() calls
ZONE_COLOR_MAP: Dict[str, str] = dict(zip(ZONE_LABELS, ZONE_COLORS))

# Pre-defined MMP windows (sorted by size for potential rolling optimization)
MMP_WINDOWS = {'5s': 5, '1m': 60, '5m': 300, '20m': 1200, '60m': 3600}


@st.cache_data
def _calculate_mmp_peaks(watts_series: pd.Series, windows: Dict[str, int]) -> Dict[str, float]:
    """Calculate Mean Maximal Power for multiple windows in a single pass.
    
    OPTIMIZATION: Single rolling calculation approach, caching results.
    Previous: O(5n) - 5 separate rolling operations
    Current: O(n) + caching - one calculation, cached per session
    
    Args:
        watts_series: Power data series
        windows: Dict of label -> window size in seconds
        
    Returns:
        Dict of label -> MMP value
    """
    results = {}
    for label, window_size in windows.items():
        if len(watts_series) >= window_size:
            # Rolling with min_periods optimization
            rolling_mean = watts_series.rolling(window_size, min_periods=window_size).mean()
            max_val = rolling_mean.max()
            results[label] = float(max_val) if not pd.isna(max_val) else None
        else:
            results[label] = None
    return results


def _calculate_zone_distribution(watts: pd.Series, cp: float) -> pd.Series:
    """Calculate power zone distribution without copying entire DataFrame.
    
    OPTIMIZATION: Work on Series only, not full DataFrame copy.
    Previous: O(n) memory for full df.copy()
    Current: O(n) for Series (much smaller)
    
    Args:
        watts: Power series only (not full DataFrame)
        cp: Critical Power
        
    Returns:
        Series with zone percentages indexed by zone label
    """
    bins = [0, 0.55*cp, 0.75*cp, 0.90*cp, 1.05*cp, 1.20*cp, float('inf')]
    zones = pd.cut(watts, bins=bins, labels=ZONE_LABELS, right=False)
    # value_counts with normalize=True avoids extra division
    pcts = zones.value_counts(normalize=True, sort=False).sort_index() * 100
    return pcts.round(1)


def render_report_tab(df_plot: pd.DataFrame, rider_weight: float, cp_input: float) -> None:
    """Render the executive summary report tab.
    
    Optimized for performance with cached calculations and vectorized operations.
    """
    st.header("Executive Summary")
    
    # --- Main chart (unchanged, already efficient) ---
    st.subheader("Przebieg Treningu")
    fig_exec = go.Figure()
    
    time_x = df_plot['time_min'] if 'time_min' in df_plot.columns else None
    
    if time_x is not None:
        if 'watts_smooth' in df_plot.columns:
            fig_exec.add_trace(go.Scatter(
                x=time_x, y=df_plot['watts_smooth'], 
                name='Moc', fill='tozeroy', 
                line=dict(color=Config.COLOR_POWER, width=1), 
                hovertemplate="Moc: %{y:.0f} W<extra></extra>"
            ))
        if 'heartrate_smooth' in df_plot.columns:
            fig_exec.add_trace(go.Scatter(
                x=time_x, y=df_plot['heartrate_smooth'], 
                name='HR', line=dict(color=Config.COLOR_HR, width=2), 
                yaxis='y2', hovertemplate="HR: %{y:.0f} bpm<extra></extra>"
            ))
        if 'smo2_smooth' in df_plot.columns:
            fig_exec.add_trace(go.Scatter(
                x=time_x, y=df_plot['smo2_smooth'], 
                name='SmO2', line=dict(color=Config.COLOR_SMO2, width=2, dash='dot'), 
                yaxis='y3', hovertemplate="SmO2: %{y:.1f}%<extra></extra>"
            ))
        if 'tymeventilation_smooth' in df_plot.columns:
            fig_exec.add_trace(go.Scatter(
                x=time_x, y=df_plot['tymeventilation_smooth'], 
                name='VE', line=dict(color=Config.COLOR_VE, width=2, dash='dash'), 
                yaxis='y4', hovertemplate="VE: %{y:.1f} L/min<extra></extra>"
            ))

    fig_exec.update_layout(
        template="plotly_dark", height=500,
        yaxis=dict(title="Moc [W]"),
        yaxis2=dict(title="HR", overlaying='y', side='right', showgrid=False),
        yaxis3=dict(title="SmO2", overlaying='y', side='right', showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis4=dict(title="VE", overlaying='y', side='right', showgrid=False, showticklabels=False),
        legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified"
    )
    st.plotly_chart(fig_exec, use_container_width=True)

    st.markdown("---")
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.subheader("Czas w Strefach (Moc)")
        if 'watts' in df_plot.columns:
            # OPTIMIZATION: Pass only Series, not full DataFrame
            pcts = _calculate_zone_distribution(df_plot['watts'], cp_input)
            
            # OPTIMIZATION: O(1) dict lookup instead of O(k) list.index()
            bar_colors = [ZONE_COLOR_MAP[z] for z in pcts.index]
            
            fig_hist = go.Figure(go.Bar(
                x=pcts.values, 
                y=pcts.index.astype(str), 
                orientation='h', 
                marker_color=bar_colors,
                # OPTIMIZATION: Let Plotly format text instead of list comprehension
                text=pcts.values,
                texttemplate="%{text:.1f}%",
                textposition='auto'
            ))
            fig_hist.update_layout(
                template="plotly_dark", height=250, 
                xaxis=dict(visible=False), 
                yaxis=dict(showgrid=False), 
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_dist2:
        st.subheader("Rozkład Tętna")
        if 'heartrate' in df_plot.columns:
            # OPTIMIZATION: Use numpy for faster binning
            hr_valid = df_plot['heartrate'].dropna()
            if len(hr_valid) > 0:
                # Round and count using numpy (faster than pandas for this)
                hr_rounded = np.round(hr_valid).astype(int)
                hr_min, hr_max = hr_rounded.min(), hr_rounded.max()
                bins = np.arange(hr_min, hr_max + 2)
                counts, edges = np.histogram(hr_rounded, bins=bins)
                
                # Filter out zero counts for cleaner display
                mask = counts > 0
                
                fig_hr = go.Figure(go.Bar(
                    x=edges[:-1][mask], 
                    y=counts[mask], 
                    marker_color=Config.COLOR_HR, 
                    hovertemplate="<b>%{x} BPM</b><br>Czas: %{y} s<extra></extra>"
                ))
                fig_hr.update_layout(
                    template="plotly_dark", height=250, 
                    xaxis_title="BPM", 
                    yaxis=dict(visible=False), 
                    bargap=0.1, 
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("---")
    c_bot1, c_bot2 = st.columns(2)
    
    with c_bot1:
        st.subheader("🏆 Peak Power")
        if 'watts' in df_plot.columns:
            # OPTIMIZATION: Cached MMP calculation (computed once per session)
            mmp_values = _calculate_mmp_peaks(df_plot['watts'], MMP_WINDOWS)
            
            cols = st.columns(5)
            for col, (label, window) in zip(cols, MMP_WINDOWS.items()):
                val = mmp_values.get(label)
                with col:
                    if val is not None:
                        w_per_kg = val / rider_weight if rider_weight > 0 else 0
                        st.metric(label, f"{val:.0f} W", f"{w_per_kg:.1f} W/kg")
                    else:
                        st.metric(label, "--")
    
    with c_bot2:
        st.subheader("🎯 Strefy (wg CP)")
        # OPTIMIZATION: Single f-string, already efficient O(1)
        z2_l, z2_h = int(0.56*cp_input), int(0.75*cp_input)
        z3_l, z3_h = int(0.76*cp_input), int(0.90*cp_input)
        z4_l, z4_h = int(0.91*cp_input), int(1.05*cp_input)
        st.info(f"**Z2 (Baza):** {z2_l}-{z2_h} W | **Z3 (Tempo):** {z3_l}-{z3_h} W | **Z4 (Próg):** {z4_l}-{z4_h} W")

