"""
Vent Thresholds — report generation button and save validation.
"""
import logging
import streamlit as st
from modules.reporting.persistence import save_ramp_test_report
from modules.manual_overrides import get_manual_overrides, to_dict

logger = logging.getLogger(__name__)


def render_report_section() -> None:
    """
    Render the report generation section.

    Reads pending_pipeline_result and related keys from st.session_state.
    Shows validation warnings, manual-override count, and GENERUJ RAPORT button.
    """
    st.markdown("---")
    st.subheader("📄 Generowanie Raportu")
    st.info(
        "💡 Raport NIE generuje się automatycznie. Kliknij przycisk poniżej, aby wygenerować i zapisać raport."
    )

    pending_result = st.session_state.get("pending_pipeline_result")

    if pending_result is None:
        st.warning(
            "⚠️ Brak danych do wygenerowania raportu. Najpierw wgraj plik i poczekaj na analizę."
        )
        return

    manual_overrides = to_dict(get_manual_overrides())

    manual_keys = [
        "manual_vt1_watts",
        "manual_vt2_watts",
        "smo2_lt1_m",
        "smo2_lt2_m",
        "cp_input",
    ]
    manual_values_set = sum(
        1 for k in manual_keys if manual_overrides.get(k) and manual_overrides[k] > 0
    )

    can_generate = True
    if manual_values_set == 0:
        st.warning(
            "⚠️ **Brak wartości manualnych!** Raport zostanie wygenerowany wyłącznie z wartościami automatycznymi (algorytm)."
        )
        bypass_warning = st.checkbox(
            "✅ Generuj raport mimo braku wartości manualnych",
            key="bypass_manual_warning",
            help="Zaznacz, aby wygenerować raport z wartościami automatycznymi. Wartości manualne mają wyższy poziom zaufania.",
        )
        if not bypass_warning:
            can_generate = False
            st.caption(
                "Aby dodać wartości manualne, przejdź do zakładki **Vent - Progi Manuals** lub **SmO2 - Progi Manuals**."
            )
    else:
        st.success(
            f"✅ Wykryto {manual_values_set} wartości manualnych - zostaną użyte w raporcie"
        )

    if not can_generate:
        return

    if not st.button("📄 GENERUJ RAPORT", type="primary", use_container_width=True):
        return

    st.session_state["report_generation_requested"] = True
    with st.spinner("Generowanie raportu..."):
        try:
            source_df = st.session_state.get("pending_source_df")
            file_name = st.session_state.get("pending_uploaded_file_name", "unknown")
            cp = st.session_state.get("pending_cp_input", 0)

            session_type = st.session_state.get("session_type")
            ramp_classification = st.session_state.get("ramp_classification")
            ramp_confidence = ramp_classification.confidence if ramp_classification else 1.0

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
            pdf_path = save_result.get("pdf_path", "")

            if save_result.get("gated"):
                reason = save_result.get("reason", "unknown")
                st.error(f"❌ Raport NIE zapisany: {reason}")
            elif save_result.get("deduplicated"):
                st.warning("⚠️ Raport dla tego pliku już istnieje")
            elif saved_path:
                st.success("✅ Raport wygenerowany pomyślnie!")
                st.info(f"📁 JSON: `{saved_path}`")
                if pdf_path:
                    st.info(f"📄 PDF: `{pdf_path}`")
                st.balloons()
                st.session_state.pop("pending_pipeline_result", None)
            else:
                st.error("❌ Nieznany błąd zapisu")

        except Exception as e:
            st.error(f"❌ Błąd generowania raportu: {e}")
            logger.warning("Report generation failed: %s", e)
