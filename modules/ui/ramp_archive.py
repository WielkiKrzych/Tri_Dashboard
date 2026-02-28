"""
Ramp Test Archive UI.

Single place for saving JSON reports and generating PDF with manual overrides.
"""
import logging
import streamlit as st
import pandas as pd
import os

from modules.reporting.persistence import (
    load_ramp_test_report,
    generate_ramp_test_pdf,
    save_ramp_test_report,
)
from modules.manual_overrides import get_manual_overrides, to_dict

logger = logging.getLogger(__name__)


def _count_manual_values(overrides: dict) -> int:
    """Count how many manual override values are set (non-None, non-zero)."""
    manual_keys = [
        "manual_vt1_watts",
        "manual_vt2_watts",
        "smo2_lt1_m",
        "smo2_lt2_m",
        "cp_input",
    ]
    return sum(
        1 for k in manual_keys if overrides.get(k) and overrides[k] > 0
    )


def _build_overrides(edited_test_date, subject_name: str, subject_anthropometry: str) -> dict:
    """
    Build the full manual_overrides dict for PDF generation.

    Uses to_dict(get_manual_overrides()) as SINGLE SOURCE OF TRUTH,
    then merges document metadata and CPET data on top.
    """
    overrides = to_dict(get_manual_overrides())

    # CPET 4-point data for Upper Aerobic range
    cpet_result = st.session_state.get("cpet_vt_result", {})
    if cpet_result:
        overrides["vt1_onset_watts"] = (
            cpet_result.get("vt1_onset_watts") or cpet_result.get("vt1_watts")
        )
        overrides["rcp_onset_watts"] = (
            cpet_result.get("rcp_onset_watts") or cpet_result.get("vt2_watts")
        )

    # Document metadata (title page)
    overrides["test_date_override"] = str(edited_test_date) if edited_test_date else None
    overrides["subject_name"] = subject_name or ""
    overrides["subject_anthropometry"] = subject_anthropometry or ""

    return overrides


def _render_save_json_section() -> None:
    """
    Render the JSON save section when pending pipeline result exists.

    This replaces the old render_report_section from vent_thresholds_report.py.
    """
    pending_result = st.session_state.get("pending_pipeline_result")
    if pending_result is None:
        return

    st.markdown("---")
    st.subheader("💾 Zapisz Raport JSON")
    st.info(
        "💡 Analiza zakończona. Zapisz raport JSON, a następnie wygeneruj PDF poniżej."
    )

    overrides = to_dict(get_manual_overrides())
    manual_count = _count_manual_values(overrides)

    can_save = True
    if manual_count == 0:
        st.warning(
            "⚠️ **Brak wartości manualnych!** Raport zostanie zapisany wyłącznie "
            "z wartościami automatycznymi (algorytm)."
        )
        bypass_warning = st.checkbox(
            "Zapisz raport mimo braku wartości manualnych",
            key="bypass_manual_warning",
            help="Wartości manualne mają wyższy poziom zaufania. "
            "Dodaj je w zakładkach Vent - Progi Manuals / SmO2 - Progi Manuals.",
        )
        if not bypass_warning:
            can_save = False
            st.caption(
                "Aby dodać wartości manualne, przejdź do **Vent - Progi Manuals** "
                "lub **SmO2 - Progi Manuals**."
            )
    else:
        st.success(
            f"✅ Wykryto {manual_count} wartości manualnych - zostaną użyte w raporcie"
        )

    if not can_save:
        return

    if not st.button(
        "💾 ZAPISZ RAPORT JSON", type="primary", use_container_width=True
    ):
        return

    st.session_state["report_generation_requested"] = True
    with st.spinner("Zapisywanie raportu..."):
        try:
            source_df = st.session_state.get("pending_source_df")
            file_name = st.session_state.get("pending_uploaded_file_name", "unknown")

            session_type = st.session_state.get("session_type")
            ramp_classification = st.session_state.get("ramp_classification")
            ramp_confidence = (
                ramp_classification.confidence if ramp_classification else 1.0
            )

            manual_overrides = to_dict(get_manual_overrides())

            save_result = save_ramp_test_report(
                pending_result,
                notes=f"User-triggered save from UI. File: {file_name}",
                session_type=session_type,
                ramp_confidence=ramp_confidence,
                source_file=file_name,
                source_df=source_df,
                manual_overrides=manual_overrides,
            )

            saved_path = save_result.get("path", "")

            if save_result.get("gated"):
                reason = save_result.get("reason", "unknown")
                st.error(f"❌ Raport NIE zapisany: {reason}")
            elif save_result.get("deduplicated"):
                st.warning("⚠️ Raport dla tego pliku już istnieje")
            elif saved_path:
                st.success("✅ Raport JSON zapisany pomyślnie!")
                st.info(f"📁 JSON: `{saved_path}`")
                st.balloons()
                st.session_state.pop("pending_pipeline_result", None)
                st.rerun()
            else:
                st.error("❌ Nieznany błąd zapisu")

        except Exception as e:
            st.error(f"❌ Błąd zapisu raportu: {e}")
            logger.warning("Report save failed: %s", e)


def render_ramp_archive():
    """Render the Ramp Test Archive view."""
    st.header("🗄️ Ramp Archive")

    # ===================================================================
    # SAVE JSON SECTION (visible when pending pipeline result exists)
    # ===================================================================
    _render_save_json_section()

    # ===================================================================
    # ARCHIVE TABLE
    # ===================================================================
    base_dir = "reports/ramp_tests"
    index_path = os.path.join(base_dir, "index.csv")

    if not os.path.exists(index_path):
        st.info("Brak zapisanych raportów. Wgraj plik i przeprowadź analizę.")
        return

    try:
        df = pd.read_csv(index_path)
    except Exception as e:
        st.error(f"Błąd odczytu indeksu raportów: {e}")
        return

    if df.empty:
        st.info("Brak zapisanych raportów.")
        return

    st.markdown("### Historia Analiz")

    if "test_date" in df.columns:
        df["test_date"] = pd.to_datetime(df["test_date"])
        df = df.sort_values(by="test_date", ascending=False)

    if "pdf_path" not in df.columns:
        df["pdf_path"] = ""
    else:
        df["pdf_path"] = df["pdf_path"].fillna("").astype(str)

    def _check_pdf_exists(path: str) -> str:
        if not path or not path.strip() or path.lower() == "nan":
            return "❌"
        return "✅" if os.path.exists(path) else "❌"

    df["PDF"] = df["pdf_path"].apply(_check_pdf_exists)

    display_cols = ["test_date", "session_id", "athlete_id", "PDF", "method_version"]
    display_cols = [c for c in display_cols if c in df.columns]

    selection = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
    )

    st.caption("💡 Kliknij na wiersz, aby zobaczyć szczegóły i wygenerować PDF")

    # ===================================================================
    # SELECTED REPORT DETAILS
    # ===================================================================
    if not (selection and selection.selection.rows):
        return

    idx = selection.selection.rows[0]
    record = df.iloc[idx]

    st.divider()
    st.subheader(f"Szczegóły Testu: {record['test_date'].date()}")

    json_path = record.get("json_path")

    if not json_path or not os.path.exists(json_path):
        st.error(f"Plik raportu nie istnieje: {json_path}")
        return

    try:
        report_data = load_ramp_test_report(json_path)
    except Exception as e:
        st.error(f"Błąd odczytu raportu: {e}")
        return

    meta = report_data.get("metadata", {})
    session_id = record.get("session_id", "unknown")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**ID Sesji:** `{meta.get('session_id')}`")
        st.write(f"**Wersja metody:** `{meta.get('method_version')}`")
    with col2:
        st.write(f"**Timestamp:** `{meta.get('analysis_timestamp')}`")
        st.write(f"**Notatka:** {meta.get('notes', '-')}")

    # ===================================================================
    # METRYKA DOKUMENTU EDITOR
    # ===================================================================
    st.divider()
    st.markdown("### ✏️ Edycja Metryki Dokumentu")
    st.caption("Te wartości pojawią się na stronie tytułowej PDF")

    meta_col1, meta_col2 = st.columns(2)

    with meta_col1:
        default_test_date = (
            record["test_date"].date() if pd.notna(record.get("test_date")) else None
        )
        edited_test_date = st.date_input(
            "📅 Data Testu",
            value=default_test_date,
            key=f"edit_test_date_{session_id}",
        )

        subject_name = st.text_input(
            "👤 Imię i Nazwisko Osoby Badanej",
            value=st.session_state.get(f"subject_name_{session_id}", ""),
            placeholder="np. Jan Kowalski",
            key=f"subject_name_{session_id}",
        )

    with meta_col2:
        subject_anthropometry = st.text_input(
            "📊 Wiek / Wzrost / Waga",
            value=st.session_state.get(f"subject_anthropometry_{session_id}", ""),
            placeholder="np. 35 lat / 178 cm / 72 kg",
            key=f"subject_anthropometry_{session_id}",
        )

        st.text_input(
            "🔒 ID Sesji (auto)",
            value=meta.get("session_id", "")[:8],
            disabled=True,
        )

    # ===================================================================
    # MANUAL OVERRIDES SUMMARY
    # ===================================================================
    st.divider()
    overrides = to_dict(get_manual_overrides())
    manual_count = _count_manual_values(overrides)

    if manual_count > 0:
        st.success(
            f"✅ Wykryto {manual_count} wartości manualnych - zostaną użyte w PDF"
        )
    else:
        st.warning(
            "⚠️ Brak wartości manualnych. PDF zostanie wygenerowany wyłącznie "
            "z wartościami automatycznymi (algorytm). "
            "Aby dodać wartości manualne, przejdź do **Vent - Progi Manuals** "
            "lub **SmO2 - Progi Manuals**."
        )

    # ===================================================================
    # PDF GENERATION BUTTONS
    # ===================================================================
    raw_path = record.get("pdf_path", "")
    pdf_path = str(raw_path).strip() if raw_path and str(raw_path).lower() != "nan" else ""
    pdf_exists = os.path.exists(pdf_path) if pdf_path else False

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if pdf_exists:
            try:
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label="📕 Pobierz PDF",
                    data=pdf_data,
                    file_name=f"raport_ramp_{record['test_date'].date()}.pdf",
                    mime="application/pdf",
                    type="primary",
                    key=f"pdf_dl_{session_id}",
                )
            except Exception as e:
                st.error(f"Błąd odczytu PDF: {e}")

    with btn_col2:
        button_label = (
            "⚡ Generuj PDF z wartościami manualnymi"
            if pdf_exists
            else "📕 Wygeneruj PDF"
        )
        if st.button(button_label, key=f"regen_manual_{session_id}", type="primary"):
            full_overrides = _build_overrides(
                edited_test_date, subject_name, subject_anthropometry
            )
            with st.spinner("Generowanie PDF..."):
                try:
                    generate_ramp_test_pdf(session_id, manual_overrides=full_overrides)
                    st.success("✅ PDF wygenerowany pomyślnie!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Błąd generowania PDF: {e}")
