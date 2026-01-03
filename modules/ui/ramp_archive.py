"""
Ramp Test Archive UI.

Displays a list of executed Ramp Tests from the CSV index.
Allows downloading reports in HTML format.
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

from modules.reporting.persistence import load_ramp_test_report
from modules.reporting.html_generator import generate_html_report

def render_ramp_archive():
    """Render the Ramp Test Archive view."""
    st.header("🗄️ Archiwum Raportów Ramp Test")
    
    # 1. Locate Index CSV
    base_dir = "reports/ramp_tests"
    index_path = os.path.join(base_dir, "index.csv")
    
    if not os.path.exists(index_path):
        st.info("Brak wykonanych testów. Przeprowadź analizę, aby zobaczyć raporty tutaj.")
        return
        
    # 2. Load Index
    try:
        df = pd.read_csv(index_path)
    except Exception as e:
        st.error(f"Błąd odczytu indeksu raportów: {e}")
        return
        
    if df.empty:
        st.info("Brak wykonanych testów.")
        return
        
    # 3. Display Table
    st.markdown("### Historia Analiz")
    
    # Sort by date descending
    if 'test_date' in df.columns:
        df['test_date'] = pd.to_datetime(df['test_date'])
        df = df.sort_values(by='test_date', ascending=False)
    
    # Select columns to display
    display_cols = ['test_date', 'session_id', 'athlete_id', 'method_version']
    # Filter only existing columns
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Interactive dataframe
    selection = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    # 4. Handle Selection
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        record = df.iloc[idx]
        
        st.divider()
        st.subheader(f"Szczegóły Testu: {record['test_date'].date()}")
        
        json_path = record.get('json_path')
        
        if not json_path or not os.path.exists(json_path):
            st.error(f"Plik raportu nie istnieje: {json_path}")
            return
            
        # Load canonical JSON
        try:
            report_data = load_ramp_test_report(json_path)
        except Exception as e:
            st.error(f"Błąd odczytu raportu: {e}")
            return
            
        # Display basic JSON info (optional preview)
        meta = report_data.get('metadata', {})
        col1, col2 = st.columns(2)
        with col1:
             st.write(f"**ID Sesji:** `{meta.get('session_id')}`")
             st.write(f"**Wersja metody:** `{meta.get('method_version')}`")
        with col2:
             st.write(f"**Timestamp:** `{meta.get('analysis_timestamp')}`")
             st.write(f"**Notatka:** {meta.get('notes', '-')}")

        # 5. Download buttons
        st.markdown("#### 📥 Pobierz raport")
        
        btn_col1, btn_col2 = st.columns(2)
        
        # PDF Download (existing file only, no generation)
        with btn_col1:
            pdf_path = record.get('pdf_path', '')
            
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                
                st.download_button(
                    label="📕 Pobierz PDF",
                    data=pdf_data,
                    file_name=f"raport_ramp_{record['test_date'].date()}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            else:
                st.info("PDF niedostępny. Wygeneruj najpierw raport.")
        
        # HTML Generation on demand
        with btn_col2:
            if st.button("📄 Wygeneruj HTML"):
                with st.spinner("Generowanie..."):
                    html_content = generate_html_report(report_data)
                    
                    st.download_button(
                        label="⬇️ Pobierz HTML",
                        data=html_content,
                        file_name=f"raport_ramp_{record['test_date'].date()}.html",
                        mime="text/html"
                    )
        
        # Preview (in expander)
        with st.expander("📋 Podgląd danych JSON"):
            st.json(report_data)

