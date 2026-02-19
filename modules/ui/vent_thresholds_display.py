"""
Vent Thresholds — 4-threshold cards, metabolic zones table, analysis notes, theory section.
"""
import streamlit as st
import pandas as pd


def render_threshold_cards(cpet_result: dict, target_df: pd.DataFrame, cp_input: float) -> None:
    """
    Render method badge, 4 threshold cards (VT1_onset, VT1_steady, RCP_onset, RCP_steady),
    metabolic zones table, and analysis notes.
    """
    has_gas = cpet_result.get("has_gas_exchange", False)

    if has_gas:
        st.success("✅ **Tryb CPET**: Analiza VE/VO2 i VE/VCO2")
    else:
        st.info(
            "ℹ️ **Tryb VE-only**: 4-punktowa analiza CPET (VT1_onset, VT1_steady, RCP_onset, RCP_steady)"
        )

    def get_metric_with_fallback(result_key, power_val, col_map):
        val = cpet_result.get(result_key)
        if val is None and power_val:
            mask = (target_df["watts"] >= power_val - 10) & (target_df["watts"] <= power_val + 10)
            if mask.any():
                try:
                    for src_col in col_map:
                        if src_col in target_df.columns:
                            v = target_df.loc[mask, src_col].mean()
                            if pd.notna(v) and v > 0:
                                return (
                                    int(v)
                                    if "hr" in result_key or "br" in result_key
                                    else round(v, 1)
                                )
                except Exception:
                    pass
        return val

    # Row 1: VT1_onset and VT1_steady
    col1, col2 = st.columns(2)

    with col1:
        vt1_onset_w = cpet_result.get("vt1_onset_watts") or cpet_result.get("vt1_watts")
        vt1_hr = get_metric_with_fallback("vt1_hr", vt1_onset_w, ["hr"])
        vt1_ve = get_metric_with_fallback("vt1_ve", vt1_onset_w, ["tymeventilation"])
        vt1_br = get_metric_with_fallback("vt1_br", vt1_onset_w, ["tymebreathrate"])

        if vt1_onset_w:
            hr_line = (
                f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt1_hr)} bpm</p>'
                if vt1_hr
                else ""
            )
            ve_line = (
                f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt1_ve} L/min</p>'
                if vt1_ve
                else ""
            )
            br_line = (
                f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt1_br)} oddech/min</p>'
                if (vt1_br and vt1_br > 0)
                else ""
            )
            st.markdown(
                f"""
            <div style="padding:12px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h4 style="margin:0; color: #ffa15a; font-size:0.9em;">VT1_onset</h4>
                <p style="margin:0; color:#888; font-size:0.75em;">GET / LT1 Onset</p>
                <h2 style="margin:5px 0; font-size:2em;">{int(vt1_onset_w)} W</h2>
                {hr_line}{ve_line}{br_line}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(vt1_onset_w / cp_input) * 100:.0f}% CP")
        else:
            st.warning("VT1_onset: Nie wykryto")

    with col2:
        vt1_steady_w = cpet_result.get("vt1_steady_watts")
        vt1_steady_hr = get_metric_with_fallback("vt1_steady_hr", vt1_steady_w, ["hr"])
        vt1_steady_ve = cpet_result.get("vt1_steady_ve")
        vt1_steady_br = cpet_result.get("vt1_steady_br")
        is_interpolated = cpet_result.get("vt1_steady_is_interpolated", False)

        if vt1_steady_w:
            hr_line = (
                f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt1_steady_hr)} bpm</p>'
                if vt1_steady_hr
                else ""
            )
            ve_line = (
                f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt1_steady_ve} L/min</p>'
                if vt1_steady_ve
                else ""
            )
            br_line = (
                f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt1_steady_br)} oddech/min</p>'
                if (vt1_steady_br and vt1_steady_br > 0)
                else ""
            )
            if is_interpolated:
                st.markdown(
                    f"""
                <div style="padding:12px; border-radius:8px; border:2px dashed #888; background-color: #1a1a1a;">
                    <h4 style="margin:0; color: #888; font-size:0.9em;">VT1_steady (Interpolated)</h4>
                    <p style="margin:0; color:#666; font-size:0.7em;">⚠️ No physiological plateau detected</p>
                    <h2 style="margin:5px 0; font-size:2em; color:#aaa;">{int(vt1_steady_w)} W</h2>
                    {hr_line}{ve_line}{br_line}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style="padding:12px; border-radius:8px; border:2px solid #00cc96; background-color: #222;">
                    <h4 style="margin:0; color: #00cc96; font-size:0.9em;">VT1_steady</h4>
                    <p style="margin:0; color:#888; font-size:0.75em;">LT1 Steady (Upper Aerobic Ceiling) ✓plateau</p>
                    <h2 style="margin:5px 0; font-size:2em;">{int(vt1_steady_w)} W</h2>
                    {hr_line}{ve_line}{br_line}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            if cp_input > 0:
                st.caption(f"~{(vt1_steady_w / cp_input) * 100:.0f}% CP")
        else:
            st.warning("VT1_steady: Nie wykryto")

    # Row 2: RCP_onset and RCP_steady
    col3, col4 = st.columns(2)

    with col3:
        rcp_onset_w = cpet_result.get("rcp_onset_watts") or cpet_result.get("vt2_watts")
        vt2_hr = get_metric_with_fallback("vt2_hr", rcp_onset_w, ["hr"])
        vt2_ve = cpet_result.get("vt2_ve")
        vt2_br = get_metric_with_fallback("vt2_br", rcp_onset_w, ["tymebreathrate"])

        if rcp_onset_w:
            hr_line = (
                f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt2_hr)} bpm</p>'
                if vt2_hr
                else ""
            )
            ve_line = (
                f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt2_ve} L/min</p>'
                if vt2_ve
                else ""
            )
            br_line = (
                f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt2_br)} oddech/min</p>'
                if (vt2_br and vt2_br > 0)
                else ""
            )
            st.markdown(
                f"""
            <div style="padding:12px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h4 style="margin:0; color: #ef553b; font-size:0.9em;">RCP_onset</h4>
                <p style="margin:0; color:#888; font-size:0.75em;">VT2 / LT2 Onset (Respiratory Compensation Point)</p>
                <h2 style="margin:5px 0; font-size:2em;">{int(rcp_onset_w)} W</h2>
                {hr_line}{ve_line}{br_line}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(rcp_onset_w / cp_input) * 100:.0f}% CP")
        else:
            st.warning("RCP_onset: Nie wykryto")

    with col4:
        rcp_steady_w = cpet_result.get("rcp_steady_watts")
        rcp_steady_hr = cpet_result.get("rcp_steady_hr")
        rcp_steady_ve = cpet_result.get("rcp_steady_ve")
        rcp_steady_br = cpet_result.get("rcp_steady_br")

        if rcp_steady_w:
            hr_line = (
                f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(rcp_steady_hr)} bpm</p>'
                if rcp_steady_hr
                else ""
            )
            ve_line = (
                f'<p style="margin:0; color:#aaa;"><b>VE:</b> {rcp_steady_ve} L/min</p>'
                if rcp_steady_ve
                else ""
            )
            br_line = (
                f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(rcp_steady_br)} oddech/min</p>'
                if (rcp_steady_br and rcp_steady_br > 0)
                else ""
            )
            st.markdown(
                f"""
            <div style="padding:12px; border-radius:8px; border:2px solid #ab63fa; background-color: #222;">
                <h4 style="margin:0; color: #ab63fa; font-size:0.9em;">RCP_steady</h4>
                <p style="margin:0; color:#888; font-size:0.75em;">Full RCP (Severe Domain Entry)</p>
                <h2 style="margin:5px 0; font-size:2em;">{int(rcp_steady_w)} W</h2>
                {hr_line}{ve_line}{br_line}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(rcp_steady_w / cp_input) * 100:.0f}% CP")
        else:
            st.info("RCP_steady: Nie wykryto (za mało danych)")

    # Metabolic zones table
    zones = cpet_result.get("metabolic_zones", [])
    if zones and len(zones) >= 4:
        st.markdown("---")
        st.subheader("🎯 Strefy Metaboliczne")

        zone_data = []
        for z in zones:
            hr_range = ""
            if z.get("hr_min") and z.get("hr_max"):
                hr_range = f"{z['hr_min']} - {z['hr_max']}"
            elif z.get("hr_max"):
                hr_range = f"< {z['hr_max']}"
            elif z.get("hr_min"):
                hr_range = f"> {z['hr_min']}"

            zone_name = z["name"]
            if z.get("is_interpolated"):
                zone_name += " ⚠️"

            zone_data.append(
                {
                    "Strefa": f"Z{z['zone']}",
                    "Nazwa": zone_name,
                    "Moc (W)": f"{z['power_min']} - {z['power_max']}",
                    "HR (bpm)": hr_range,
                    "Trening": z["training"],
                    "Domena": z.get("domain", ""),
                }
            )

            if z.get("subzones") and isinstance(z["subzones"], list):
                for sz in z["subzones"]:
                    zone_data.append(
                        {
                            "Strefa": "",
                            "Nazwa": f"  └ {sz['name']}",
                            "Moc (W)": f"{sz['power_min']} - {sz['power_max']}",
                            "HR (bpm)": "",
                            "Trening": "",
                            "Domena": "",
                        }
                    )

        st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)

        interp = cpet_result.get("no_steady_state_interpretation")
        if interp:
            st.warning(f"📋 **Interpretacja:** {interp}")

    # Analysis notes
    analysis_notes = cpet_result.get("analysis_notes", [])
    if analysis_notes:
        with st.expander("📋 Notatki z analizy CPET", expanded=False):
            for note in analysis_notes:
                if note.startswith("⚠️"):
                    st.warning(note)
                else:
                    st.info(note)


def render_theory_section() -> None:
    """Render the collapsible theory/help section."""
    with st.expander("🫁 TEORIA: Progi Wentylacyjne (VT1 / VT2)", expanded=False):
        st.markdown("""
        ## Co to są progi wentylacyjne?

        **Progi wentylacyjne** to punkty, w których wentylacja (VE) zaczyna rosnąć nieliniowo względem mocy.

        | Próg | Inna nazwa | Fizjologia | % VO2max |
        |------|-----------|------------|----------|
        | **VT1** | Próg tlenowy, LT1 | Początek akumulacji mleczanu | ~50-60% |
        | **VT2** | Próg beztlenowy, LT2, OBLA | Maksymalny laktat steady-state | ~75-85% |

        ---

        ## Jak działa detekcja?

        System stosuje:
        1. **Sliding Window Analysis**: Skanuje okno po oknie, żeby znaleźć przejścia w nachyleniu (slope)
        2. **Breakpoint Detection**: Szuka punktów załamania krzywej VE vs Power
        3. **Sensitivity Analysis**: Uruchamia algorytm kilkukrotnie z różnymi parametrami

        ---

        ## Zastosowanie progów

        | Strefa | Zakres | Cel treningowy |
        |--------|--------|----------------|
        | **Z1 (Recovery)** | < VT1 | Regeneracja, rozgrzewka |
        | **Z2 (Endurance)** | VT1 - środek | Baza tlenowa |
        | **Z3 (Tempo)** | środek - VT2 | Sweet Spot |
        | **Z4 (Threshold)** | VT2 ± 5% | FTP, próg |
        | **Z5+ (VO2max)** | > VT2 | Interwały, moc szczytowa |

        ---

        ## Reliability Score (Niezawodność)

        * **HIGH**: Wynik jest stabilny niezależnie od wygładzania
        * **MEDIUM**: Wynik zależy nieco od parametrów
        * **LOW**: Duża zmienność (>15W różnicy) - sugeruje "szumiący" sygnał

        ---

        ## Wymagania testu

        ⚠️ **Dla wiarygodnych wyników potrzebujesz:**
        - Test stopniowany (Ramp Test) z liniowym wzrostem mocy
        - Minimum 10-15 minut narastającego obciążenia
        - Czysty sygnał wentylacji (stabilny sensor)
        - Brak przerw i wahań mocy
        """)
