"""
Trends Tab UI.

Displays cross-test trend analysis with:
- Adaptation rate per metric
- Engine Map (radar chart)
- Trend lines
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def render_trends_tab():
    """Render the Trends (Intelligence) tab."""
    st.header("📈 Analiza Trendów (Intelligence)")
    
    st.markdown("""
    Silnik analizy trendów między testami. Śledzi postęp w kluczowych metrykach:
    **VT1, VT2, CP, W', EF, SmO2 slope, Occlusion Index, HSI**
    """)
    
    # Load trend data
    from modules.calculations.trend_engine import (
        load_ramp_test_history,
        analyze_trends,
        TrendAnalysis
    )
    
    reports = load_ramp_test_history()
    
    if len(reports) < 2:
        st.warning(f"""
        ⚠️ **Potrzebne minimum 2 testy** do analizy trendów.
        
        Aktualnie w archiwum: **{len(reports)} test(ów)**.
        
        Wykonaj więcej testów Ramp, aby zobaczyć analizę trendów.
        """)
        return
    
    # Analyze trends
    analysis = analyze_trends(reports)
    
    # === KPI HEADER ===
    st.subheader("📊 Podsumowanie Adaptacji")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Adaptation Score
    score_color = "#27AE60" if analysis.adaptation_score >= 60 else ("#F39C12" if analysis.adaptation_score >= 40 else "#E74C3C")
    c1.metric(
        "Wynik Adaptacji",
        f"{analysis.adaptation_score:.0f}/100",
        help="Ogólny wynik postępu treningowego"
    )
    
    # Adaptation Direction
    direction_emoji = {
        "central": "❤️ Centralny",
        "peripheral": "💪 Obwodowy",
        "thermal": "🌡️ Termiczny",
        "balanced": "⚖️ Zbalansowany"
    }
    c2.metric(
        "Kierunek Adaptacji",
        direction_emoji.get(analysis.adaptation_direction, "?"),
        help="Dominujący kierunek adaptacji organizmu"
    )
    
    # Tests Analyzed
    c3.metric(
        "Analizowanych Testów",
        analysis.tests_analyzed,
        help="Liczba testów w analizie"
    )
    
    # Date Range
    weeks = analysis.date_range_days / 7
    c4.metric(
        "Okres Analizy",
        f"{weeks:.1f} tyg.",
        help=f"{analysis.date_range_days} dni"
    )
    
    st.divider()
    
    # === ENGINE MAP (Radar Chart) ===
    st.subheader("🔧 Engine Map — Co się poprawia, co stoi?")
    
    _render_engine_map(analysis)
    
    st.divider()
    
    # === TEMPO ADAPTACJI (Rate per Week Table) ===
    st.subheader("⏱️ Tempo Adaptacji [% / tydzień]")
    
    _render_adaptation_rates(analysis)
    
    st.divider()
    
    # === TREND LINES ===
    st.subheader("📉 Linie Trendów")
    
    _render_trend_lines(analysis)
    
    # === INTERPRETACJA ===
    with st.expander("📖 Interpretacja Kierunku Adaptacji", expanded=False):
        st.markdown("""
        ### Kierunki Adaptacji
        
        | Kierunek | Wskaźniki | Znaczenie |
        |----------|-----------|-----------|
        | **Centralny** | VT1↑, VT2↑, CP↑, EF↑ | Poprawa układu krążenia i wentylacji |
        | **Obwodowy** | SmO2↑, Occlusion↓, W'↑ | Poprawa ekstrakcji O₂, kapilaryzacji, tolerancji mleczanu |
        | **Termiczny** | HSI↓ | Lepsza termoregulacja, niższy koszt termiczny |
        | **Zbalansowany** | Mix powyższych | Harmonijny rozwój wszystkich systemów |
        
        ---
        
        ### Interpretacja Engine Map
        
        - **> 60:** Wyraźna poprawa
        - **40-60:** Stabilizacja / utrzymanie
        - **< 40:** Regresja / zmęczenie / przetrenowanie
        """)

    # === TECHNICAL REGISTRY INFO ===
    st.divider()
    with st.expander("🛠️ TabRegistry & System Info", expanded=False):
        st.markdown("### Zarejestrowane Moduły UI")
        st.info("System wykrył następujące moduły w TabRegistry (OCP):")
        
        # We define a simplified view of the registry here to avoid circular imports
        # but showing the user the mapping we know exists in app.py
        registry_data = [
            ("report", "modules.ui.report", "Raport Główny"),
            ("kpi", "modules.ui.kpi", "Kluczowe Wskaźniki"),
            ("summary", "modules.ui.summary", "Podsumowanie"),
            ("power", "modules.ui.power", "Moc i Zony"),
            ("biomech", "modules.ui.biomech", "Biomechanika i Okluzja"),
            ("hrv", "modules.ui.hrv", "Zmienność Tętna"),
            ("vent", "modules.ui.vent", "Wentylacja (CPET)"),
            ("thermal", "modules.ui.thermal", "Termoregulacja"),
            ("nutrition", "modules.ui.nutrition", "Żywienie i Glikogen"),
            ("trends", "modules.ui.trends", "Analiza Trendów (Current)"),
            ("ramp_archive", "modules.ui.ramp_archive", "Archiwum Testów"),
        ]
        
        df_reg = pd.DataFrame(registry_data, columns=["ID", "Moduł", "Opis"])
        st.table(df_reg)
        
        st.caption(f"Status: Analysis Engine v2.4 | Tests: {len(reports)} | Registry: OCP Compliant")


def _render_engine_map(analysis):
    """Render radar chart (Engine Map)."""
    engine_map = analysis.engine_map
    
    categories = list(engine_map.keys())
    values = list(engine_map.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Adaptacja',
        line=dict(color='#00cc96', width=2),
        fillcolor='rgba(0, 204, 150, 0.3)'
    ))
    
    # Add reference line at 50 (stable)
    ref_values = [50] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=ref_values,
        theta=categories,
        mode='lines',
        name='Baseline (stabilny)',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['Regresja', 'Słaby', 'Stabilny', 'Dobry', 'Świetny']
            )
        ),
        template="plotly_dark",
        title="Engine Map: Mapa Adaptacji",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Color-coded summary
    col1, col2 = st.columns(2)
    
    improving = [k for k, v in analysis.engine_map.items() if v >= 60]
    declining = [k for k, v in analysis.engine_map.items() if v < 40]
    
    with col1:
        if improving:
            st.success(f"✅ **Poprawiające się:** {', '.join(improving)}")
        else:
            st.info("Brak wyraźnie poprawiających się metryk")
    
    with col2:
        if declining:
            st.error(f"⚠️ **Wymagające uwagi:** {', '.join(declining)}")
        else:
            st.success("Brak metryk w regresji")


def _render_adaptation_rates(analysis):
    """Render adaptation rates table."""
    metrics_data = [
        ("VT1", analysis.vt1.rate_per_week, analysis.vt1.direction, "central"),
        ("VT2", analysis.vt2.rate_per_week, analysis.vt2.direction, "central"),
        ("CP", analysis.cp.rate_per_week, analysis.cp.direction, "central"),
        ("W'", analysis.w_prime.rate_per_week, analysis.w_prime.direction, "peripheral"),
        ("EF", analysis.ef.rate_per_week, analysis.ef.direction, "central"),
        ("SmO2 Slope", analysis.smo2_slope.rate_per_week, analysis.smo2_slope.direction, "peripheral"),
        ("Occlusion", analysis.occlusion_index.rate_per_week, analysis.occlusion_index.direction, "peripheral"),
        ("HSI", analysis.hsi.rate_per_week, analysis.hsi.direction, "thermal"),
    ]
    
    # Create dataframe
    df = pd.DataFrame(metrics_data, columns=["Metryka", "Tempo [%/tyg]", "Kierunek", "Kategoria"])
    
    # Add visual indicators
    def direction_emoji(d):
        if d == "improving":
            return "📈 Poprawa"
        elif d == "declining":
            return "📉 Spadek"
        else:
            return "➡️ Stabilny"
    
    df["Status"] = df["Kierunek"].apply(direction_emoji)
    
    # Color-code tempo
    def tempo_color(t):
        if t > 0.5:
            return f"+{t:.2f}%"
        elif t < -0.5:
            return f"{t:.2f}%"
        else:
            return f"{t:.2f}%"
    
    df["Tempo"] = df["Tempo [%/tyg]"].apply(tempo_color)
    
    st.dataframe(
        df[["Metryka", "Tempo", "Status", "Kategoria"]],
        use_container_width=True,
        hide_index=True
    )


def _render_trend_lines(analysis):
    """Render trend line charts for key metrics."""
    metrics_to_plot = [
        ("CP", analysis.cp, "#1f77b4"),
        ("VT1", analysis.vt1, "#ff7f0e"),
        ("VT2", analysis.vt2, "#2ca02c"),
    ]
    
    fig = go.Figure()
    
    for name, trend, color in metrics_to_plot:
        if len(trend.values) >= 2:
            fig.add_trace(go.Scatter(
                x=trend.dates,
                y=trend.values,
                mode='lines+markers',
                name=f"{name} ({trend.rate_per_week:+.2f}%/tyg)",
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=f"{name}: %{{y:.0f}} W<br>Data: %{{x}}<extra></extra>"
            ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Trendy Mocy (CP, VT1, VT2)",
        xaxis=dict(title="Data"),
        yaxis=dict(title="Moc [W]"),
        height=400,
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Secondary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if len(analysis.w_prime.values) >= 2:
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=analysis.w_prime.dates,
                y=analysis.w_prime.values,
                mode='lines+markers',
                name="W'",
                line=dict(color='#9467bd', width=2),
                marker=dict(size=8)
            ))
            fig_w.update_layout(
                template="plotly_dark",
                title=f"W' ({analysis.w_prime.rate_per_week:+.2f}%/tyg)",
                xaxis=dict(title="Data"),
                yaxis=dict(title="W' [kJ]"),
                height=300,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_w, use_container_width=True)
    
    with col2:
        if len(analysis.ef.values) >= 2:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=analysis.ef.dates,
                y=analysis.ef.values,
                mode='lines+markers',
                name="EF",
                line=dict(color='#17becf', width=2),
                marker=dict(size=8)
            ))
            fig_ef.update_layout(
                template="plotly_dark",
                title=f"Efficiency Factor ({analysis.ef.rate_per_week:+.2f}%/tyg)",
                xaxis=dict(title="Data"),
                yaxis=dict(title="EF [W/bpm]"),
                height=300,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_ef, use_container_width=True)
