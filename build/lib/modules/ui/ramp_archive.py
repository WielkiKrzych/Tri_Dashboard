"""
Ramp Test Archive UI.

Displays a list of executed Ramp Tests from the CSV index.
Allows downloading reports in PDF format.
"""
import streamlit as st
import pandas as pd
import os

from modules.reporting.persistence import load_ramp_test_report

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
    
    # Ensure pdf_path is string
    if 'pdf_path' not in df.columns:
        df['pdf_path'] = ""
    else:
        df['pdf_path'] = df['pdf_path'].fillna("").astype(str)
        
    # Add visual indicator for PDF availability (single source of truth: pdf_path + file exists)
    def _check_pdf_exists(path: str) -> str:
        """Check if PDF file exists at given path."""
        if not path or not path.strip() or path.lower() == 'nan':
            return "❌"
        exists = os.path.exists(path)
        return "✅" if exists else "❌"
    
    df['PDF'] = df['pdf_path'].apply(_check_pdf_exists)
    
    # Select columns to display
    display_cols = ['test_date', 'session_id', 'athlete_id', 'PDF', 'method_version']
    # Filter only existing columns (PDF is synthetic, so it exists)
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Interactive dataframe
    selection = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    st.caption("💡 Kliknij na wiersz, aby zobaczyć szczegóły i opcje PDF")
    
    # 4. Handle Selection
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        # Map back to original dataframe row using index if needed, 
        # but since we filtered/sorted "df" in place (mostly), we need to be careful.
        # Streamlit 1.35+ returns integer index relative to the displayed data?
        # Actually, st.dataframe returns keys relative to the original index if not hidden?
        # Let's rely on the fact that we sorted 'df' and passed it directly.
        # BUT: we modified 'df' above with fillna/columns.
        
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
        
        # Extract session_id early for use in widget keys
        session_id = record.get('session_id', 'unknown')
        
        col1, col2 = st.columns(2)
        with col1:
             st.write(f"**ID Sesji:** `{meta.get('session_id')}`")
             st.write(f"**Wersja metody:** `{meta.get('method_version')}`")
        with col2:
             st.write(f"**Timestamp:** `{meta.get('analysis_timestamp')}`")
             st.write(f"**Notatka:** {meta.get('notes', '-')}")

        # ===================================================================
        # METRYKA DOKUMENTU EDITOR (editable fields for PDF title page)
        # ===================================================================
        st.divider()
        st.markdown("### ✏️ Edycja Metryki Dokumentu")
        st.caption("Te wartości pojawią się na stronie tytułowej PDF")
        
        meta_col1, meta_col2 = st.columns(2)
        
        with meta_col1:
            # Data Testu - editable date
            default_test_date = record['test_date'].date() if pd.notna(record.get('test_date')) else None
            edited_test_date = st.date_input(
                "📅 Data Testu",
                value=default_test_date,
                key=f"edit_test_date_{session_id}"
            )
            
            # Imię i Nazwisko Osoby Badanej
            subject_name = st.text_input(
                "👤 Imię i Nazwisko Osoby Badanej",
                value=st.session_state.get(f"subject_name_{session_id}", ""),
                placeholder="np. Jan Kowalski",
                key=f"subject_name_{session_id}"
            )
        
        with meta_col2:
            # Wiek / Wzrost / Waga
            subject_anthropometry = st.text_input(
                "📊 Wiek / Wzrost / Waga",
                value=st.session_state.get(f"subject_anthropometry_{session_id}", ""),
                placeholder="np. 35 lat / 178 cm / 72 kg",
                key=f"subject_anthropometry_{session_id}"
            )
            
            # Read-only display
            st.text_input(
                "🔒 ID Sesji (auto)",
                value=meta.get('session_id', '')[:8],
                disabled=True
            )
        
        st.divider()

        # 5. Download buttons (PDF only)
        raw_path = record.get('pdf_path', '')
        pdf_path = str(raw_path).strip() if raw_path and str(raw_path).lower() != 'nan' else ""
        pdf_exists = os.path.exists(pdf_path) if pdf_path else False
        
        if pdf_path and pdf_exists:
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    st.download_button(
                        label="📕 Pobierz PDF",
                        data=pdf_data,
                        file_name=f"raport_ramp_{record['test_date'].date()}.pdf",
                        mime="application/pdf",
                        type="primary",
                        key=f"pdf_dl_{session_id}"
                    )
                
                with btn_col2:
                    # Force regeneration with CURRENT MANUAL VALUES from session_state
                    if st.button("⚡ Generuj z wartościami manualnymi", key=f"regen_manual_{session_id}", type="secondary"):
                        from modules.reporting.persistence import generate_ramp_test_pdf
                        
                        # Get CPET result for Upper Aerobic range
                        cpet_result = st.session_state.get('cpet_vt_result', {})
                        
                        # Collect manual overrides from session_state
                        manual_overrides = {
                            # VT1/VT2 from Manual Thresholds tab
                            "manual_vt1_watts": st.session_state.get("manual_vt1_watts", 0),
                            "manual_vt2_watts": st.session_state.get("manual_vt2_watts", 0),
                            "vt1_hr": st.session_state.get("vt1_hr", 0),
                            "vt2_hr": st.session_state.get("vt2_hr", 0),
                            "vt1_ve": st.session_state.get("vt1_ve", 0),
                            "vt2_ve": st.session_state.get("vt2_ve", 0),
                            "vt1_br": st.session_state.get("vt1_br", 0),
                            "vt2_br": st.session_state.get("vt2_br", 0),
                            # SmO2 from Manual SmO2 tab
                            "smo2_lt1_m": st.session_state.get("smo2_lt1_m", 0),
                            "smo2_lt2_m": st.session_state.get("smo2_lt2_m", 0),
                            # CP from Sidebar
                            "cp_input": st.session_state.get("cp_input", 0),
                            # CCI Breakpoint from Intervals tab
                            "cci_breakpoint_manual": st.session_state.get("cci_breakpoint_manual", 0),
                            # VE Breakpoint from Manual Thresholds tab
                            "ve_breakpoint_manual": st.session_state.get("ve_breakpoint_manual", 0),
                            # Reoxy Half-Time from SmO2 Manual tab
                            "reoxy_halftime_manual": st.session_state.get("reoxy_halftime_manual", 0),
                            # ======= TEST PROTOCOL fields from Manual Thresholds tab =======
                            "test_start_power": st.session_state.get("test_start_power", 0),
                            "test_end_power": st.session_state.get("test_end_power", 0),
                            "test_duration": st.session_state.get("test_duration", ""),
                            # ======= CPET 4-point data for Upper Aerobic range =======
                            "vt1_onset_watts": cpet_result.get('vt1_onset_watts') or cpet_result.get('vt1_watts'),
                            "rcp_onset_watts": cpet_result.get('rcp_onset_watts') or cpet_result.get('vt2_watts'),
                            # ======= Metryka Dokumentu fields =======
                            "test_date_override": str(edited_test_date) if edited_test_date else None,
                            "subject_name": subject_name if subject_name else "",
                            "subject_anthropometry": subject_anthropometry if subject_anthropometry else "",
                        }
                        
                        with st.spinner("Generowanie PDF z wartościami manualnymi..."):
                            generate_ramp_test_pdf(session_id, manual_overrides=manual_overrides)
                            st.success("✅ PDF wygenerowany z wartościami manualnymi!")
                            st.rerun()
            except Exception as e:
                st.error(f"Błąd PDF: {e}")
        else:
            if st.button("📕 Wygeneruj PDF", key=f"gen_pdf_{session_id}", type="primary"):
                from modules.reporting.persistence import generate_ramp_test_pdf
                with st.spinner("Generowanie..."):
                    generate_ramp_test_pdf(session_id)
                    st.rerun()
