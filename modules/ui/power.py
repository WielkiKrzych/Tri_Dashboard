import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from modules.config import Config
from modules.plots import apply_chart_style

def render_power_tab(df_plot, df_plot_resampled, cp_input, w_prime_input):
    st.subheader("Wykres Mocy i W'")
    fig_pw = go.Figure()
    fig_pw.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['watts_smooth'], name="Moc", fill='tozeroy', line=dict(color=Config.COLOR_POWER, width=1), hovertemplate="Moc: %{y:.0f} W<extra></extra>"))
    fig_pw.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['w_prime_balance'], name="W' Bal", yaxis="y2", line=dict(color=Config.COLOR_HR, width=2), hovertemplate="W' Bal: %{y:.0f} J<extra></extra>"))
    fig_pw.update_layout(
        template="plotly_dark", 
        title="Zarządzanie Energią (Moc vs W')", 
        hovermode="x unified", 
        xaxis=dict(
            title="Czas [min]",
            tickformat=".0f",
            hoverformat=".0f"
        ),
        yaxis=dict(title="Moc [W]"), 
        yaxis2=dict(title="W' Balance [J]", overlaying="y", side="right", showgrid=False)
    )
    st.plotly_chart(fig_pw, use_container_width=True)
    
    st.info("""
    **💡 Interpretacja: Energia Beztlenowa (W' Balance)**

    Ten wykres pokazuje, ile "zapałek" masz jeszcze w pudełku.

    * **Czerwona Linia (W' Bal):** Poziom energii beztlenowej w Dżulach [J].
    * **Moc Krytyczna (CP):** To Twoja granica tlenowa (jak FTP, ale fizjologicznie precyzyjniejsza).

    **Jak to działa?**
    * **Moc < CP (Strefa Tlenowa):** Nie spalasz W'. Jeśli jechałeś mocno wcześniej, bateria się ładuje (czerwona linia rośnie).
    * **Moc > CP (Strefa Beztlenowa):** Zaczynasz "palić zapałki". Czerwona linia spada. Im mocniej depczesz, tym szybciej spada.
    * **W' = 0 J (Wyczerpanie):** "Odcina prąd". Nie jesteś w stanie utrzymać mocy powyżej CP ani sekundy dłużej. Musisz zwolnić, żeby zregenerować.

    **Scenariusze:**
    1.  **Interwały:** W' powinno spadać w trakcie powtórzenia (wysiłek) i rosnąć w przerwie (regeneracja). Jeśli nie wraca do 100% przed kolejnym startem, kumulujesz zmęczenie.
    2.  **Finisz:** Idealnie rozegrany wyścig to taki, gdzie W' spada do zera dokładnie na linii mety. Jeśli zostało Ci 10kJ, mogłeś finiszować mocniej. Jeśli spadło do zera 500m przed metą - przeszarżowałeś.
    3.  **Błędne CP:** Jeśli podczas spokojnej jazdy W' ciągle spada, Twoje CP jest ustawione za wysoko. Jeśli finiszujesz "w trupa", a W' pokazuje wciąż 50% - Twoje CP lub W' są niedoszacowane.
    """)

    st.subheader("Czas w Strefach Mocy (Time in Zones)")
    if 'watts' in df_plot.columns:
        bins = [0, 0.55*cp_input, 0.75*cp_input, 0.90*cp_input, 1.05*cp_input, 1.20*cp_input, 10000]
        labels = ['Z1: Regeneracja', 'Z2: Wytrzymałość', 'Z3: Tempo', 'Z4: Próg', 'Z5: VO2Max', 'Z6: Beztlenowa']
        colors = ['#A0A0A0', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
        df_z = df_plot.copy()
        df_z['Zone'] = pd.cut(df_z['watts'], bins=bins, labels=labels, right=False)
        pcts = (df_z['Zone'].value_counts().sort_index() / len(df_z) * 100).round(1)
        fig_z = px.bar(x=pcts.values, y=labels, orientation='h', text=pcts.apply(lambda x: f"{x}%"), color=labels, color_discrete_sequence=colors)
        fig_z.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(apply_chart_style(fig_z), use_container_width=True)

        st.info("""
        **💡 Interpretacja Treningowa:**
        * **Polaryzacja:** Dobry plan często ma dużo Z1/Z2 (baza) i trochę Z5/Z6 (bodziec), a mało "śmieciowych kilometrów" w Z3. Strefa Z3 to "szara strefa", która męczy, ale nie daje dużych korzyści adaptacyjnych, jednakże zużywa dużo glikogenu. Mimo tego, w triathlonie Z3 ma swoje miejsce (jazda na czas) i warto ją stosować taktycznie.
        * **Długie Wyścigi (Triathlon):** Większość czasu powinna być w Z2, z akcentami w Z4 (próg mleczanowy) i Z5 (VO2Max) dla poprawy wydolności. Spędzanie czasu w Z3 powinno być ograniczone ale taktyczne (np. jazda na czas).
        * **Sprinty i Criterium:** Więcej czasu w Z4/Z5/Z6, ale z odpowiednią regeneracją w Z1. Dużo interwałów wysokiej intensywności. Ważne jest, aby nie zaniedbywać Z2 dla budowy bazy tlenowej.
        * **Regeneracja:** Z1 to strefa regeneracyjna, idealna na dni odpoczynku lub bardzo lekkie sesje. Może pomóc w usuwaniu metabolitów i poprawie krążenia bez dodatkowego stresu. "Nie trenować" to też trening.
        * **Adaptacje Fizjologiczne:**
        * **Z1 (Szary):** Regeneracja i krążenie.
        * **Z2 (Zielony):** Kluczowe dla budowania mitochondriów i spalania tłuszczu. Podstawa wytrzymałości.
        * **Z3 (Żółty):** Mieszana strefa, poprawia ekonomię jazdy i tolerancję na wysiłek, ale może prowadzić do zmęczenia bez odpowiedniej regeneracji.
        * **Z4/Z5 (Pomarańczowy/Czerwony):** Budują tolerancję na mleczan i VO2Max, ale wymagają długiej regeneracji. Nie powinny dominować w planie treningowym.
        """)

        st.markdown("### 📚 Kompendium Fizjologii Stref (Deep Dive)")
        with st.expander("🟩 Z1/Z2: Fundament Tlenowy (< 75% CP)", expanded=True):
            st.markdown("""
            * **Metabolizm:** Dominacja Wolnych Kwasów Tłuszczowych (WKT). RER ~0.7-0.85. Oszczędność glikogenu.
            * **Fizjologia:**
                * Biogeneza mitochondriów (więcej "pieców" energetycznych).
                * Angiogeneza (tworzenie nowych naczyń włosowatych).
                * Wzrost aktywności enzymów oksydacyjnych.
            * **Biomechanika:** Rekrutacja głównie włókien wolnokurczliwych (Typ I).
            * **SmO2:** Stabilne, wysokie wartości (Równowaga Podaż=Popyt).
            * **Oddech (VT):** Poniżej VT1. Pełna konwersacja.
            * **Typowy Czas:** 1.5h - 6h+.
            """)

        with st.expander("🟨 Z3: Tempo / Sweet Spot (76-90% CP)"):
            st.markdown("""
            * **Metabolizm:** Miks węglowodanów i tłuszczów (RER ~0.85-0.95). Zaczyna się znaczne zużycie glikogenu.
            * **Fizjologia:** "Strefa Szara". Bodziec tlenowy, ale już z narastającym zmęczeniem.
            * **Zastosowanie:** Trening specyficzny pod 70.3 / Ironman (długie utrzymanie mocy).
            * **SmO2:** Stabilne, ale niższe niż w Z2. Możliwy powolny trend spadkowy.
            * **Oddech (VT):** Okolice VT1. Głęboki, rytmiczny oddech.
            * **Typowy Czas:** 45 min - 2.5h.
            """)

        with st.expander("🟧 Z4: Próg Mleczanowy (91-105% CP)"):
            st.markdown("""
            * **Metabolizm:** Dominacja glikogenu (RER ~1.0). Produkcja mleczanu równa się jego utylizacji (MLSS).
            * **Fizjologia:** Poprawa tolerancji na kwasicę. Zwiększenie magazynów glikogenu.
            * **Biomechanika:** Rekrutacja włókien pośrednich (Typ IIa).
            * **SmO2:** Granica równowagi. Utrzymuje się na stałym, niskim poziomie.
            * **Oddech (VT):** Pomiędzy VT1 a VT2. Oddech mocny, utrudniona mowa.
            * **Typowy Czas:** Interwały 8-30 min (łącznie do 60-90 min w sesji).
            """)

        with st.expander("🟥 Z5/Z6: VO2Max i Beztlenowa (> 106% CP)"):
            st.markdown("""
            * **Metabolizm:** Wyłącznie glikogen + Fosfokreatyna (PCr). RER > 1.1.
            * **Fizjologia:** Maksymalny pobór tlenu (pułap tlenowy). Szybkie narastanie długu tlenowego.
            * **Biomechanika:** Pełna rekrutacja wszystkich włókien (Typ IIx). Duży moment siły.
            * **SmO2:** Gwałtowny spadek (Desaturacja).
            * **Oddech (VT):** Powyżej VT2 (RCP). Hiperwentylacja.
            * **Typowy Czas:** Z5: 3-8 min. Z6: < 2 min.
            """)
    
    st.divider()
    st.subheader("🔥 Symulator 'Spalania Zapałek' (W' Attack)")
    st.markdown("Sprawdź, jak konkretny atak wpłynie na Twoje rezerwy energii.")

    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        sim_watts = st.slider("Moc Ataku [W]", min_value=int(cp_input), max_value=int(cp_input*2.5), value=int(cp_input*1.2), step=10)
        sim_dur = st.slider("Czas Trwania [sek]", min_value=10, max_value=300, value=60, step=10)

        if sim_watts > cp_input:
            w_burned = (sim_watts - cp_input) * sim_dur
            w_rem = w_prime_input - w_burned
            w_rem_pct = (w_rem / w_prime_input) * 100
        else:
            w_burned = 0; w_rem = w_prime_input; w_rem_pct = 100
        if w_rem < 0: w_rem = 0; w_rem_pct = 0
        st.markdown(f"**Spalone:** {w_burned:.0f} J\n**Pozostało:** {w_rem:.0f} J ({w_rem_pct:.1f}%)")
    with c_sim2:
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = w_rem,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pozostałe W'"},
            gauge = {
                'axis': {'range': [0, w_prime_input], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, w_prime_input*0.25], 'color': "red"},
                    {'range': [w_prime_input*0.25, w_prime_input*0.5], 'color': "orange"},
                    {'range': [w_prime_input*0.5, w_prime_input], 'color': "green"}],
            }
        ))
        st.plotly_chart(apply_chart_style(fig_g), use_container_width=True)
    
    if w_rem_pct == 0:
        st.error("💀 **TOTAL FAILURE!** Ten atak wyczerpie Cię całkowicie. Nie dojedziesz.")
    elif w_rem_pct < 25:
        st.warning("⚠️ **KRYTYCZNIE:** Bardzo ryzykowny atak. Zostaniesz na oparach.")
    else:
        st.success("✅ **BEZPIECZNIE:** Masz zapas na taki ruch.")
