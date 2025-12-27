import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.config import Config
from modules.calculations import calculate_vo2max

def render_report_tab(df_plot, rider_weight, cp_input):
    st.header("Executive Summary")
    
    st.subheader("Przebieg Treningu")
    fig_exec = go.Figure()
    
    if 'watts_smooth' in df_plot.columns:
        fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['watts_smooth'], name='Moc', fill='tozeroy', line=dict(color=Config.COLOR_POWER, width=1), hovertemplate="Moc: %{y:.0f} W<extra></extra>"))
    if 'heartrate_smooth' in df_plot.columns:
        fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['heartrate_smooth'], name='HR', line=dict(color=Config.COLOR_HR, width=2), yaxis='y2', hovertemplate="HR: %{y:.0f} bpm<extra></extra>"))
    if 'smo2_smooth' in df_plot.columns:
        fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['smo2_smooth'], name='SmO2', line=dict(color=Config.COLOR_SMO2, width=2, dash='dot'), yaxis='y3', hovertemplate="SmO2: %{y:.1f}%<extra></extra>"))
    if 'tymeventilation_smooth' in df_plot.columns:
        fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['tymeventilation_smooth'], name='VE', line=dict(color=Config.COLOR_VE, width=2, dash='dash'), yaxis='y4', hovertemplate="VE: %{y:.1f} L/min<extra></extra>"))

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
            bins = [0, 0.55*cp_input, 0.75*cp_input, 0.90*cp_input, 1.05*cp_input, 1.20*cp_input, 10000]
            labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']
            colors = ['#808080', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
            df_z = df_plot.copy()
            df_z['Zone'] = pd.cut(df_z['watts'], bins=bins, labels=labels, right=False)
            pcts = (df_z['Zone'].value_counts().sort_index() / len(df_z) * 100).round(1)
            fig_hist = go.Figure(go.Bar(x=pcts.values, y=labels, orientation='h', marker_color=colors, text=pcts.apply(lambda x: f"{x}%"), textposition='auto'))
            fig_hist.update_layout(template="plotly_dark", height=250, xaxis=dict(visible=False), yaxis=dict(showgrid=False), margin=dict(t=20, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_dist2:
        st.subheader("Rozkład Tętna")
        if 'heartrate' in df_plot.columns:
            hr_counts = df_plot['heartrate'].dropna().round().astype(int).value_counts().sort_index()
            fig_hr = go.Figure(go.Bar(x=hr_counts.index, y=hr_counts.values, marker_color=Config.COLOR_HR, hovertemplate="<b>%{x} BPM</b><br>Czas: %{y} s<extra></extra>"))
            fig_hr.update_layout(template="plotly_dark", height=250, xaxis_title="BPM", yaxis=dict(visible=False), bargap=0.1, margin=dict(t=20, b=20))
            st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("---")
    c_bot1, c_bot2 = st.columns(2)
    with c_bot1:
        st.subheader("🏆 Peak Power")
        mmp_windows = {'5s': 5, '1m': 60, '5m': 300, '20m': 1200, '60m': 3600}
        cols = st.columns(5)
        if 'watts' in df_plot.columns:
            for c, (l, s) in zip(cols, mmp_windows.items()):
                val = df_plot['watts'].rolling(s).mean().max()
                with c:
                    if not pd.isna(val): st.metric(l, f"{val:.0f} W", f"{val/rider_weight:.1f} W/kg")
                    else: st.metric(l, "--")
    
    with c_bot2:
        st.subheader("🎯 Strefy (wg CP)")
        z2_l, z2_h = int(0.56*cp_input), int(0.75*cp_input)
        z3_l, z3_h = int(0.76*cp_input), int(0.90*cp_input)
        z4_l, z4_h = int(0.91*cp_input), int(1.05*cp_input)
        z5_l, z5_h = int(1.06*cp_input), int(1.20*cp_input)
        st.info(f"**Z2 (Baza):** {z2_l}-{z2_h} W | **Z3 (Tempo):** {z3_l}-{z3_h} W | **Z4 (Próg):** {z4_l}-{z4_h} W")
