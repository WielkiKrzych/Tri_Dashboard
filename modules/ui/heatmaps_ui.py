"""
Power Zone Heatmap UI Module.

Renders the heatmap tab with controls for:
- FTP input
- Resolution toggle (hourly / weekday_hourly)
- Date range filter
- Plotly heatmap display
- Export buttons
"""
import streamlit as st
import pandas as pd
from typing import Optional

from modules.heatmaps import (
    power_zone_heatmap,
    plot_power_zone_heatmap,
    export_heatmap_json,
    get_power_zones,
    DEFAULT_ZONE_EDGES,
)


def render_heatmaps_tab(
    df_plot: pd.DataFrame,
    ftp_default: int = 250,
    power_col: str = "watts"
) -> None:
    """Render the Power Zone Heatmap tab.
    
    Args:
        df_plot: DataFrame with power data
        ftp_default: Default FTP value from sidebar/session
        power_col: Name of power column
    """
    st.header("🔥 Heatmapa Stref Mocy")
    
    if power_col not in df_plot.columns:
        st.warning("Brak danych mocy do analizy heatmapy.")
        return
    
    # ===== CONTROLS =====
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ftp = st.number_input(
            "FTP [W]",
            min_value=50,
            max_value=500,
            value=ftp_default,
            step=5,
            help="Functional Threshold Power. Strefy będą obliczone jako % FTP."
        )
    
    with col2:
        resolution = st.radio(
            "Rozdzielczość",
            options=["hourly", "weekday_hourly"],
            format_func=lambda x: "Godzinowa (0-23)" if x == "hourly" else "Dzień × Godzina",
            horizontal=True,
            help="Wybierz jak grupować dane: tylko godzina lub dzień tygodnia × godzina."
        )
    
    with col3:
        # Date filter (if timestamp available)
        if 'timestamp' in df_plot.columns:
            df_plot['_ts_temp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
            min_date = df_plot['_ts_temp'].min().date()
            max_date = df_plot['_ts_temp'].max().date()
            
            date_range = st.date_input(
                "Zakres dat",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df_plot['_ts_temp'].dt.date >= start_date) & (df_plot['_ts_temp'].dt.date <= end_date)
                df_filtered = df_plot[mask].copy()
            else:
                df_filtered = df_plot.copy()
            
            df_filtered = df_filtered.drop(columns=['_ts_temp'], errors='ignore')
        else:
            df_filtered = df_plot.copy()
            st.caption("Brak timestampów - używam syntetycznych godzin.")
    
    # ===== ZONE PREVIEW =====
    with st.expander("📊 Podgląd stref mocy"):
        zones = get_power_zones(ftp)
        zone_data = [
            {
                "Strefa": z.name,
                "% FTP": f"{z.lower_pct*100:.0f}-{z.upper_pct*100:.0f}%",
                "Moc [W]": f"{z.lower_watts:.0f}-{z.upper_watts:.0f}"
            }
            for z in zones
        ]
        st.table(pd.DataFrame(zone_data))
    
    # ===== COMPUTE HEATMAP =====
    try:
        pivot, metadata = power_zone_heatmap(
            df_filtered,
            ftp=ftp,
            resolution=resolution,
            power_col=power_col
        )
        
        if metadata["total_seconds"] == 0:
            st.warning("Brak danych mocy w wybranym zakresie.")
            return
        
        # ===== DISPLAY HEATMAP =====
        fig = plot_power_zone_heatmap(
            pivot,
            resolution=resolution,
            title=f"Heatmapa Stref Mocy (FTP={ftp}W)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== STATS =====
        st.subheader("📈 Statystyki")
        
        total_min = metadata["total_seconds"] / 60
        st.metric("Łączny czas analizy", f"{total_min:.0f} min")
        
        # Zone distribution
        zone_cols = st.columns(len(metadata["zone_distribution"]))
        for i, (zone, seconds) in enumerate(metadata["zone_distribution"].items()):
            with zone_cols[i]:
                pct = (seconds / metadata["total_seconds"] * 100) if metadata["total_seconds"] > 0 else 0
                st.metric(zone.split()[0], f"{seconds/60:.0f}m", f"{pct:.0f}%")
        
        # ===== EXPORT =====
        with st.expander("📥 Eksportuj dane"):
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if st.button("💾 JSON", key="export_heatmap_json"):
                    import tempfile
                    import os
                    json_path = os.path.join(tempfile.gettempdir(), "heatmap_data.json")
                    export_heatmap_json(pivot, metadata, json_path)
                    with open(json_path, 'r') as f:
                        st.download_button(
                            "📥 Pobierz JSON",
                            f.read(),
                            file_name="heatmap_data.json",
                            mime="application/json"
                        )
            
            with col_exp2:
                st.info("Użyj przycisku aparatu w prawym górnym rogu wykresu aby zapisać PNG.")
    
    except Exception as e:
        st.error(f"Błąd podczas generowania heatmapy: {e}")
