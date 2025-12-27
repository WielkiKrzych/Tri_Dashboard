import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.utils import load_data
from modules.calculations import process_data, calculate_metrics

def render_compare_dashboard(file1, file2, cp_input):
    st.header("⚔️ Tryb Porównania (vs)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📁 Plik A: {file1.name}")
    with col2:
        st.success(f"📁 Plik B: {file2.name}")

    with st.spinner("Przetwarzanie obu plików..."):
        try:
            # Load Data
            df1_raw = load_data(file1)
            df2_raw = load_data(file2)
            
            # Process Data
            df1 = process_data(df1_raw)
            df2 = process_data(df2_raw)
            
            # Metrics
            m1 = calculate_metrics(df1, cp_input)
            m2 = calculate_metrics(df2, cp_input)
            
        except Exception as e:
            st.error(f"Błąd przetwarzania plików: {e}")
            return

    # --- METRYKI ---
    st.subheader("1. Porównanie Metryk")
    
    metrics_list = [
        ("Średnia Moc [W]", 'avg_watts', "W", 0),
        ("Średnie Tętno [bpm]", 'avg_hr', "bpm", 0),
        ("NP (Norm. Power)", 'np_est', "W", 0),
        ("Średnia Kadencja", 'avg_cadence', "rpm", 0),
        ("Średnie SmO2", 'avg_smo2', "%", 1), # Custom handle for SmO2 not in m1 dict directly usually
        ("Efficiency (W/HR)", 'power_hr', "", 2)
    ]
    
    # Prepare custom metrics explicitly if needed
    m1['avg_smo2'] = df1['smo2'].mean() if 'smo2' in df1.columns else 0
    m2['avg_smo2'] = df2['smo2'].mean() if 'smo2' in df2.columns else 0

    # Display Metrics
    # We use a grid layout
    
    for i in range(0, len(metrics_list), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(metrics_list):
                label, key, unit, precision = metrics_list[i+j]
                val1 = m1.get(key, 0)
                val2 = m2.get(key, 0)
                if pd.isna(val1): val1 = 0
                if pd.isna(val2): val2 = 0
                
                diff = val2 - val1
                
                col.metric(
                    label,
                    f"{val1:.{precision}f} vs {val2:.{precision}f} {unit}",
                    f"{diff:+.{precision}f} {unit}",
                    delta_color="normal"
                )

    st.divider()

    # --- WYKRESY ---
    st.subheader("2. Overlay Wykresów")
    
    chart_type = st.radio("Wybierz Wykres", ["Moc (Power)", "Tętno (HR)", "SmO2", "Wentylacja (VE)"], horizontal=True)
    
    col_map = {
        "Moc (Power)": "watts_smooth",
        "Tętno (HR)": "heartrate_smooth",
        "SmO2": "smo2_smooth",
        "Wentylacja (VE)": "tymeventilation_smooth"
    }
    
    target_col = col_map[chart_type]
    
    fig = go.Figure()
    
    # Plik 1
    if target_col in df1.columns:
        fig.add_trace(go.Scatter(x=df1['time_min'], y=df1[target_col], name=f"A: {file1.name}", line=dict(color='cyan', width=1.5)))
    
    # Plik 2
    if target_col in df2.columns:
        fig.add_trace(go.Scatter(x=df2['time_min'], y=df2[target_col], name=f"B: {file2.name}", line=dict(color='magenta', width=1.5, dash='dash'))) # Dash for contrast
        
    fig.update_layout(
        template="plotly_dark",
        title=f"Porównanie: {chart_type}",
        xaxis_title="Czas [min]",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Plik A (Cyan) jest ciągły, Plik B (Magenta) jest przerywany.")
