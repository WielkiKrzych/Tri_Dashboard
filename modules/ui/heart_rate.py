import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_hr_tab(df):
    st.markdown("### ❤️ Analiza Tętna")
    
    if df is None or df.empty:
        st.error("Brak danych.")
        return

    # 1. Normalizacja danych
    df_chart = df.copy()
    # Ensure columns are lowercased and stripped (though load_data usually does this)
    df_chart.columns = df_chart.columns.str.lower().str.strip()
    
    # Aliasy dla HR
    if 'hr' not in df_chart.columns:
        for alias in ['heart_rate', 'heart rate', 'bpm', 'tętno']:
            if alias in df_chart.columns:
                df_chart.rename(columns={alias: 'hr'}, inplace=True)
                break
                
    if 'hr' not in df_chart.columns:
        st.warning("⚠️ Brak danych tętna (HR) w wczytanym pliku.")
        return

    # Upewnij się że time istnieje
    if 'time' not in df_chart.columns:
        # Spróbuj wygenerować czas z indexu jeśli brak
        df_chart['time'] = np.arange(len(df_chart))
    
    # 2. Selektor zakresu
    # Konwersja czasu na ładny format HH:MM:SS do wyświetlania w sliderze byłaby super, 
    # ale slider na sekundach jest bardziej precyzyjny/prostszy w kodzie.
    # Dodamy formatowanie czasu w opisie.
    
    min_time = int(df_chart['time'].min())
    max_time = int(df_chart['time'].max())
    
    st.markdown("#### Wybierz zakres analizy")
    
    # Dwustronny slider
    range_sel = st.slider("Zakres czasu (s)", 
                          min_value=min_time, max_value=max_time, 
                          value=(min_time, max_time))
    
    start_s, end_s = range_sel
    
    # Filtrowanie
    mask = (df_chart['time'] >= start_s) & (df_chart['time'] <= end_s)
    df_segment = df_chart.loc[mask]
    
    if df_segment.empty:
        st.info("Brak danych w wybranym zakresie.")
        return

    # Obliczenie czasu trwania
    duration = end_s - start_s
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    st.caption(f"Analizowany fragment: {minutes} min {seconds} s")

    # 3. Metryki
    avg_hr = df_segment['hr'].mean()
    min_hr = df_segment['hr'].min()
    max_hr = df_segment['hr'].max()
    
    # Wyświetlenie metryk
    # Stylizacja metryk w kontenerze
    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Średnie HR", f"{avg_hr:.0f} bpm")
        c2.metric("Min HR", f"{min_hr:.0f} bpm")
        c3.metric("Max HR", f"{max_hr:.0f} bpm")
    
    st.markdown("---")

    # 4. Wykres
    fig = go.Figure()
    
    # Linia HR
    # Dodajemy sformatowany czas do tooltipa
    df_segment['time_str'] = pd.to_datetime(df_segment['time'], unit='s').dt.strftime('%H:%M:%S')

    fig.add_trace(go.Scatter(
        x=df_segment['time'], 
        y=df_segment['hr'],
        mode='lines',
        name='HR',
        line=dict(color='#d62728', width=2),
        customdata=df_segment['time_str'],
        hovertemplate="<b>Czas:</b> %{customdata} (%{x}s)<br><b>HR:</b> %{y:.0f} bpm<extra></extra>"
    ))
    
    # Linia średniej wizualnie
    fig.add_hline(y=avg_hr, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Avg", annotation_position="bottom right")

    fig.update_layout(
        title="Wykres Tętna (HR)",
        xaxis_title="Czas (s)",
        yaxis_title="Tętno (bpm)",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
