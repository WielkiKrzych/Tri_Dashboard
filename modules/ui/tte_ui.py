"""
TTE (Time-to-Exhaustion) UI Module.

Displays TTE analysis for the current session and historical trends.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from modules.tte import (
    compute_tte_result,
    format_tte,
    export_tte_json,
    TTEResult,
    save_tte_to_db,
)


def render_tte_tab(df_plot: pd.DataFrame, ftp: float, uploaded_file_name: str = "manual_upload") -> None:
    """Render the TTE analysis tab.
    
    Args:
        df_plot: Session data with 'watts' column
        ftp: Functional Threshold Power
        uploaded_file_name: Original filename for record matching
    """
    st.header("⏱️ Time-to-Exhaustion (TTE)")
    st.markdown("""
    Analiza maksymalnego czasu, przez który utrzymałeś zadaną moc.
    TTE mierzy Twoją zdolność do utrzymania intensywności na poziomie progu (FTP).
    """)
    
    # Check for power data
    if 'watts' not in df_plot.columns:
        st.error("Brak danych mocy (watts) w pliku.")
        return
    
    # Configuration
    st.subheader("⚙️ Konfiguracja")
    col1, col2 = st.columns(2)
    
    with col1:
        target_pct = st.slider(
            "Docelowy % FTP",
            min_value=70,
            max_value=120,
            value=100,
            step=5,
            help="Procent FTP, który chcesz analizować"
        )
    
    with col2:
        tol_pct = st.slider(
            "Tolerancja [%]",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Dopuszczalne odchylenie od docelowej mocy"
        )
    
    # New: Manual history acceptance toggle
    accept_to_history = st.toggle(
        "⭐ Zalicz trening do historii TTE", 
        value=False,
        help="Włącz, aby trwale dodać wynik TTE z tej sesji do Twojej bazy treningów (training_history.db)."
    )
    
    # Compute TTE for current session
    power_series = df_plot['watts']
    result = compute_tte_result(
        power_series,
        target_pct=target_pct,
        ftp=ftp,
        tol_pct=tol_pct
    )
    
    # Update historical data if accepted
    if accept_to_history:
        # Extract date from first valid timestamp if possible
        session_date = datetime.now().strftime("%Y-%m-%d")
        if 'timestamp' in df_plot.columns and not df_plot['timestamp'].empty:
            try:
                ts = pd.to_datetime(df_plot['timestamp'].iloc[0])
                session_date = ts.strftime("%Y-%m-%d")
            except:
                pass
        
        success = save_tte_to_db(
            filename=uploaded_file_name,
            session_date=session_date,
            target_pct=target_pct,
            tte_seconds=result.tte_seconds
        )
        if success:
            st.toast(f"Wynik TTE ({target_pct}%) zapisany w bazie!", icon="✅")
        else:
            st.warning("Nie znaleziono treningu w bazie. Upewnij się, że został zaimportowany.")
    
    # Display results
    st.subheader("📊 Wyniki Sesji")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Maksymalny TTE",
            format_tte(result.tte_seconds),
            help="Najdłuższy ciągły czas utrzymania mocy w zadanym zakresie"
        )
    
    with col2:
        st.metric(
            "Zakres Mocy",
            f"{result.target_power_min:.0f} - {result.target_power_max:.0f} W",
            help=f"{target_pct}% FTP ± {tol_pct}%"
        )
    
    with col3:
        # Interpretation
        if result.tte_seconds >= 3600:
            status = "🏆 Elitarny"
            color = "🟢"
        elif result.tte_seconds >= 1800:
            status = "💪 Dobry"
            color = "🟢"
        elif result.tte_seconds >= 600:
            status = "📈 Rozwijający się"
            color = "🟡"
        else:
            status = "🔄 Do poprawy"
            color = "🟠"
        
        st.metric("Ocena", f"{color} {status}")
    
    # Power distribution chart
    _render_power_distribution_chart(df_plot, result)
    
    # Export section
    st.divider()
    with st.expander("📥 Eksport JSON"):
        json_data = export_tte_json(result)
        st.code(json_data, language="json")
        st.download_button(
            "Pobierz JSON",
            data=json_data,
            file_name=f"tte_{result.session_id}.json",
            mime="application/json"
        )
    
    # Theory section
    with st.expander("📚 Teoria TTE", expanded=False):
        st.markdown("""
        ### Czym jest Time-to-Exhaustion?
        
        **TTE** (Time-to-Exhaustion) to maksymalny czas, przez który sportowiec
        może utrzymać określoną intensywność wysiłku przed wyczerpaniem.
        
        #### Dlaczego to ważne?
        
        * **Planowanie wyścigów**: Wiedząc, jak długo możesz utrzymać 100% FTP,
          możesz lepiej planować tempo na trasie.
        * **Monitorowanie postępów**: Wzrost TTE przy tym samym % FTP oznacza
          poprawę wytrzymałości.
        * **Indywidualizacja treningu**: TTE pomaga dobrać długość interwałów.
        
        #### Typowe wartości TTE przy 100% FTP:
        
        | Poziom | TTE |
        |--------|-----|
        | Początkujący | 20-40 min |
        | Amator | 40-60 min |
        | Zaawansowany | 60-75 min |
        | Elita | 75+ min |
        
        #### Jak poprawić TTE?
        
        1. **Trening progowy (SST/Tempo)**: 2-3 sesje tygodniowo
        2. **Długie jazdy Z2**: Buduj bazę aerobową
        3. **Poprawa ekonomii**: Praca nad kadencją i techniką
        """)


def _render_power_distribution_chart(df_plot: pd.DataFrame, result: TTEResult) -> None:
    """Render power distribution chart with TTE range highlighted."""
    fig = go.Figure()
    
    # Power trace
    fig.add_trace(go.Scatter(
        x=df_plot['time_min'],
        y=df_plot['watts'],
        name='Moc',
        line=dict(color='#1f77b4', width=1),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='Moc: %{y:.0f} W<extra></extra>'
    ))
    
    # Target range shading
    fig.add_hrect(
        y0=result.target_power_min,
        y1=result.target_power_max,
        fillcolor="rgba(0, 255, 0, 0.1)",
        line=dict(color="green", width=1, dash="dash"),
        annotation_text=f"Zakres TTE ({result.target_pct}% ± {result.tolerance_pct}%)",
        annotation_position="top left"
    )
    
    # Layout
    fig.update_layout(
        template="plotly_dark",
        title="Rozkład Mocy z Zakresem TTE",
        hovermode="x unified",
        xaxis=dict(
            title="Czas [min]",
            tickformat=".0f",
            hoverformat=".0f"
        ),
        yaxis=dict(
            title="Moc [W]",
            tickformat=".0f"
        ),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1, x=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
