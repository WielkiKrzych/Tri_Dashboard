"""
Power Duration Curve (PDC) UI Module.

Displays advanced power analytics:
- Power Duration Curve visualization
- Fatigue Resistance Index (FRI)
- Match Burns counter
- Stamina Score
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Optional

from modules.calculations import (
    calculate_power_duration_curve,
    calculate_fatigue_resistance_index,
    count_match_burns,
    get_fri_interpretation,
    calculate_stamina_score,
    estimate_vlamax_from_pdc,
    get_stamina_interpretation,
    get_vlamax_interpretation,
    DEFAULT_PDC_DURATIONS,
    # NEW functions
    estimate_tte,
    classify_phenotype,
    get_phenotype_description,
    calculate_durability_index,
    get_durability_interpretation,
    calculate_recovery_score,
    get_recovery_recommendation,
)


def _format_duration(seconds: int) -> str:
    """Format seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}:{secs:02d}" if secs else f"{mins}min"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h{mins:02d}" if mins else f"{hours}h"


def _format_tte(seconds: float) -> str:
    """Format TTE to human-readable string."""
    if seconds == float('inf'):
        return "∞ (sustainable)"
    elif seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    else:
        return ">1h"


def _create_pdc_chart(pdc: Dict[int, float], cp: float) -> go.Figure:
    """Create Power Duration Curve chart.
    
    Args:
        pdc: Dict mapping duration (seconds) to power (watts)
        cp: Critical Power for reference line
        
    Returns:
        Plotly Figure
    """
    # Filter valid data points
    durations = sorted([d for d, p in pdc.items() if p is not None])
    powers = [pdc[d] for d in durations]
    
    if not durations:
        return go.Figure()
    
    # Create x-axis labels
    x_labels = [_format_duration(d) for d in durations]
    
    fig = go.Figure()
    
    # Main PDC curve
    fig.add_trace(go.Scatter(
        x=durations,
        y=powers,
        mode='lines+markers',
        name='Power Duration Curve',
        line=dict(color='#FFD700', width=3),
        marker=dict(size=10, color='#FFD700'),
        hovertemplate='%{text}: <b>%{y:.0f} W</b><extra></extra>',
        text=x_labels
    ))
    
    # CP reference line
    if cp > 0:
        fig.add_hline(
            y=cp, 
            line_dash="dash", 
            line_color="rgba(255,255,255,0.5)",
            annotation_text=f"CP: {cp}W",
            annotation_position="right"
        )
    
    # Zone shading
    if cp > 0:
        fig.add_hrect(
            y0=cp * 0.55, y1=cp * 0.75,
            fillcolor="rgba(0,200,200,0.1)",
            line_width=0,
            annotation_text="Z2",
            annotation_position="left"
        )
        fig.add_hrect(
            y0=cp * 0.90, y1=cp * 1.05,
            fillcolor="rgba(255,215,0,0.1)",
            line_width=0,
            annotation_text="Threshold",
            annotation_position="left"
        )
        fig.add_hrect(
            y0=cp * 1.05, y1=max(powers) * 1.1 if powers else cp * 1.5,
            fillcolor="rgba(255,69,0,0.1)",
            line_width=0,
            annotation_text="VO2max+",
            annotation_position="left"
        )
    
    fig.update_layout(
        template="plotly_dark",
        title="📊 Power Duration Curve (Krzywa Mocy)",
        xaxis=dict(
            title="Czas",
            type="log",
            tickvals=durations,
            ticktext=x_labels,
        ),
        yaxis=dict(title="Moc [W]"),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        showlegend=False
    )
    
    return fig


def render_pdc_tab(
    df_plot: pd.DataFrame,
    cp: float,
    w_prime: float,
    rider_weight: float,
    vo2max_est: float = 0
) -> None:
    """Render the Power Duration Curve tab.
    
    Args:
        df_plot: Session DataFrame with power data
        cp: Critical Power in watts
        w_prime: W' capacity in Joules
        rider_weight: Rider weight in kg
        vo2max_est: Estimated VO2max (optional)
    """
    st.header("📊 Power Duration Curve & Profil Mocy")
    
    if 'watts' not in df_plot.columns:
        st.warning("Brak danych mocy do analizy.")
        return
    
    # Calculate PDC
    pdc = calculate_power_duration_curve(df_plot)
    
    if not pdc or all(v is None for v in pdc.values()):
        st.warning("Za mało danych do wygenerowania krzywej mocy.")
        return
    
    # Get MMP values
    mmp_5s = pdc.get(5)
    mmp_1min = pdc.get(60)
    mmp_5min = pdc.get(300)
    mmp_20min = pdc.get(1200)
    
    # ===== PHENOTYPE BADGE (NEW) =====
    phenotype = classify_phenotype(pdc, rider_weight)
    emoji, name, desc = get_phenotype_description(phenotype)
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(255,215,0,0.2), transparent); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin:0;">{emoji} Twój Fenotyp: <span style="color: #FFD700;">{name}</span></h3>
        <p style="margin:5px 0 0 0; opacity: 0.8;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== PDC CHART =====
    fig_pdc = _create_pdc_chart(pdc, cp)
    st.plotly_chart(fig_pdc, use_container_width=True)
    
    # ===== KEY METRICS + TTE (NEW) =====
    st.subheader("📈 Kluczowe Metryki & Time to Exhaustion")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if mmp_5s:
            tte_5s = estimate_tte(mmp_5s, cp, w_prime)
            st.metric("⚡ MMP 5s", f"{mmp_5s:.0f} W", 
                     f"{mmp_5s/rider_weight:.1f} W/kg" if rider_weight > 0 else None)
            st.caption(f"TTE: {_format_tte(tte_5s)}")
        else:
            st.metric("⚡ MMP 5s", "—")
    
    with col2:
        if mmp_1min:
            tte_1min = estimate_tte(mmp_1min, cp, w_prime)
            st.metric("🔥 MMP 1min", f"{mmp_1min:.0f} W",
                     f"{mmp_1min/rider_weight:.1f} W/kg" if rider_weight > 0 else None)
            st.caption(f"TTE: {_format_tte(tte_1min)}")
        else:
            st.metric("🔥 MMP 1min", "—")
    
    with col3:
        if mmp_5min:
            tte_5min = estimate_tte(mmp_5min, cp, w_prime)
            st.metric("💪 MMP 5min", f"{mmp_5min:.0f} W",
                     f"{mmp_5min/rider_weight:.1f} W/kg" if rider_weight > 0 else None)
            st.caption(f"TTE: {_format_tte(tte_5min)}")
        else:
            st.metric("💪 MMP 5min", "—")
    
    with col4:
        if mmp_20min:
            tte_20min = estimate_tte(mmp_20min, cp, w_prime)
            st.metric("🏔️ MMP 20min", f"{mmp_20min:.0f} W",
                     f"{mmp_20min/rider_weight:.1f} W/kg" if rider_weight > 0 else None)
            st.caption(f"TTE: {_format_tte(tte_20min)}")
        else:
            st.metric("🏔️ MMP 20min", "—")
    
    st.divider()
    
    # ===== DURABILITY INDEX (NEW) =====
    st.subheader("🛡️ Durability Index")
    
    durability, avg_first, avg_second = calculate_durability_index(df_plot, min_duration_min=20)
    
    if durability is not None:
        durability_interp = get_durability_interpretation(durability)
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            delta_color = "normal" if durability >= 90 else "inverse"
            st.metric(
                "Durability Index",
                f"{durability:.1f}%",
                delta=f"{durability - 100:.1f}%" if durability < 100 else "+0%",
                delta_color=delta_color,
                help="Stosunek średniej mocy w 2. połowie do 1. połowy treningu"
            )
        
        with col_d2:
            st.metric("Śr. Moc (1. połowa)", f"{avg_first:.0f} W")
        
        with col_d3:
            st.metric("Śr. Moc (2. połowa)", f"{avg_second:.0f} W")
        
        st.info(f"**Interpretacja:** {durability_interp}")
    else:
        st.info("Potrzeba minimum 20 minut treningu do obliczenia Durability Index.")
    
    st.divider()
    
    # ===== RECOVERY SCORE (NEW) =====
    st.subheader("🔄 Recovery Score")
    
    if 'w_prime_balance' in df_plot.columns and w_prime > 0:
        w_bal_end = df_plot['w_prime_balance'].iloc[-1]
        recovery_score = calculate_recovery_score(w_bal_end, w_prime, time_since_effort_sec=0)
        zone_rec, zone_desc = get_recovery_recommendation(recovery_score)
        
        col_r1, col_r2 = st.columns([1, 2])
        
        with col_r1:
            st.metric(
                "Recovery Score",
                f"{recovery_score:.0f}/100",
                help="Gotowość do następnego treningu na podstawie W' Balance"
            )
        
        with col_r2:
            st.info(f"**{zone_rec}**\n\n{zone_desc}")
    else:
        st.info("Oblicz W' Balance w zakładce Power, aby zobaczyć Recovery Score.")
    
    st.divider()
    
    # ===== FATIGUE RESISTANCE INDEX =====
    st.subheader("🔋 Fatigue Resistance Index (FRI)")
    
    if mmp_5min and mmp_20min:
        fri = calculate_fatigue_resistance_index(mmp_5min, mmp_20min)
        fri_interpretation = get_fri_interpretation(fri)
        
        col_fri1, col_fri2 = st.columns([1, 2])
        
        with col_fri1:
            st.metric(
                "FRI (MMP20 / MMP5)", 
                f"{fri:.2f}",
                help="Stosunek mocy 20min do 5min. Im bliżej 1.0, tym lepsza wytrzymałość."
            )
        
        with col_fri2:
            st.info(f"**Interpretacja:** {fri_interpretation}")
        
        # FRI gauge
        fig_fri = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fri,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0.6, 1.0], 'tickwidth': 1},
                'bar': {'color': "#FFD700"},
                'steps': [
                    {'range': [0.6, 0.75], 'color': "#FF4500"},
                    {'range': [0.75, 0.85], 'color': "#FFA500"},
                    {'range': [0.85, 0.92], 'color': "#32CD32"},
                    {'range': [0.92, 1.0], 'color': "#00CED1"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': fri
                }
            },
            title={'text': "Fatigue Resistance"}
        ))
        fig_fri.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(l=30, r=30, t=50, b=10)
        )
        st.plotly_chart(fig_fri, use_container_width=True)
    else:
        st.warning("Potrzeba danych ≥5 minut i ≥20 minut dla obliczenia FRI.")
    
    st.divider()
    
    # ===== MATCH BURNS =====
    st.subheader("🔥 Spalone Zapałki (Match Burns)")
    
    if 'w_prime_balance' in df_plot.columns and w_prime > 0:
        w_bal = df_plot['w_prime_balance'].values
        burns = count_match_burns(w_bal, w_prime, threshold_pct=0.3)
        
        col_match1, col_match2 = st.columns([1, 2])
        
        with col_match1:
            st.metric(
                "🔥 Spalone Zapałki", 
                burns,
                help="Liczba razy, gdy W' spadło poniżej 30% pojemności"
            )
        
        with col_match2:
            if burns == 0:
                st.success("Zachowałeś wszystkie zapałki - trening regeneracyjny lub Z2.")
            elif burns <= 3:
                st.info(f"Spalono {burns} zapałek - umiarkowane wysiłki anaerobowe.")
            elif burns <= 6:
                st.warning(f"Spalono {burns} zapałek - intensywny trening interwałowy.")
            else:
                st.error(f"Spalono {burns} zapałek - ekstremalnie wymagający trening!")
        
        # Match burns over time visualization
        fig_matches = go.Figure()
        
        time_min = df_plot['time'] / 60 if 'time' in df_plot.columns else np.arange(len(w_bal)) / 60
        
        fig_matches.add_trace(go.Scatter(
            x=time_min,
            y=w_bal,
            name="W' Balance",
            line=dict(color='#00CED1', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,206,209,0.2)'
        ))
        
        # Threshold line
        threshold = w_prime * 0.3
        fig_matches.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Próg zapałki ({threshold/1000:.0f} kJ)"
        )
        
        fig_matches.update_layout(
            template="plotly_dark",
            title="W' Balance - Kiedy paliłeś zapałki?",
            xaxis=dict(title="Czas [min]"),
            yaxis=dict(title="W' Balance [J]"),
            height=350,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(fig_matches, use_container_width=True)
    else:
        st.info("Oblicz W' Balance w zakładce Power, aby zobaczyć analizę zapałek.")
    
    st.divider()
    
    # ===== STAMINA SCORE =====
    st.subheader("🏆 Stamina Score")
    
    if mmp_5min and mmp_20min and rider_weight > 0 and cp > 0:
        fri = calculate_fatigue_resistance_index(mmp_5min, mmp_20min)
        
        # Use provided VO2max or estimate from 5min power
        if vo2max_est <= 0:
            vo2max_est = (10.8 * mmp_5min / rider_weight) + 7
        
        stamina = calculate_stamina_score(vo2max_est, fri, w_prime, cp, rider_weight)
        stamina_interp = get_stamina_interpretation(stamina)
        
        # VLamax estimation
        vlamax = estimate_vlamax_from_pdc(pdc, rider_weight)
        vlamax_interp = get_vlamax_interpretation(vlamax) if vlamax else "Niewystarczające dane"
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric(
                "Stamina Score", 
                f"{stamina:.0f}/100",
                help="Composite metric: VO2max + FRI + CP/kg"
            )
            st.caption(stamina_interp)
        
        with col_s2:
            st.metric(
                "Est. VO2max",
                f"{vo2max_est:.1f} ml/kg/min",
                help="Szacowane z 5-minutowej mocy max"
            )
        
        with col_s3:
            if vlamax:
                st.metric(
                    "Est. VLamax",
                    f"{vlamax:.2f} mmol/L/s",
                    help="Szacowane z kształtu krzywej mocy"
                )
                st.caption(vlamax_interp)
            else:
                st.metric("Est. VLamax", "—")
        
        with st.expander("📚 Jak interpretować te metryki?"):
            st.markdown("""
            ### Stamina Score (0-100)
            Composite metric łącząca wydolność tlenową, zdolność do utrzymania mocy i moc względną.
            
            | Score | Poziom |
            |-------|--------|
            | 80+ | World Tour / Pro Continental |
            | 65-80 | Elitarny amator |
            | 50-65 | Wytrenowany kolarz klubowy |
            | 35-50 | Amator średni |
            | <35 | Początkujący |
            
            ### Fatigue Resistance Index (FRI)
            Stosunek MMP20 do MMP5. Im bliżej 1.0, tym lepiej utrzymujesz moc w czasie.
            
            - **0.95+**: Wyjątkowa wytrzymałość (diesele jak Froome)
            - **0.90-0.95**: Poziom Pro
            - **0.85-0.90**: Dobrze wytrenowany amator
            - **<0.80**: Profil sprinterski
            
            ### VLamax (Estimated)
            Maksymalna szybkość produkcji mleczanu. Niższa wartość = lepsza wydolność tlenowa.
            
            - **>0.9**: Sprinter - wysoka glikoliza
            - **0.5-0.7**: All-rounder
            - **<0.4**: Climber/TT specialist
            """)
    else:
        st.info("Potrzeba danych CP, wagi i minimum 20 minut treningu do obliczenia Stamina Score.")
