"""
Ventilation tab — theory/help section explaining VE, BR, VT concepts.
"""
import streamlit as st


def render_vent_theory() -> None:
    """Render the collapsible theory section for ventilation analysis."""
    with st.expander("🫁 TEORIA: Interpretacja Wentylacji", expanded=False):
        st.markdown("""
        ## Co oznacza Wentylacja (VE)?

        **VE (Minute Ventilation)** to objętość powietrza wdychanego/wydychanego na minutę.
        Mierzona przez sensory oddechowe np. **CORE, Tyme Wear, Garmin HRM-Pro (estymacja)**.

        | Parametr | Opis | Jednostka |
        |----------|------|-----------|
        | **VE** | Wentylacja minutowa | L/min |
        | **BR / RR** | Częstość oddechów | oddechy/min |
        | **VT** | Objętość oddechowa (VE/BR) | L |

        ---

        ## Strefy VE i ich znaczenie

        | VE (L/min) | Interpretacja | Typ wysiłku |
        |------------|---------------|-------------|
        | **20-40** | Spokojny oddech | Recovery, rozgrzewka |
        | **40-80** | Umiarkowany wysiłek | Tempo, Sweet Spot |
        | **80-120** | Intensywny wysiłek | Threshold, VO2max |
        | **> 120** | Maksymalny wysiłek | Sprint, test wyczerpania |

        ---

        ## Trend VE (Slope) - Co oznacza nachylenie?

        | Trend | Wartość | Interpretacja |
        |-------|---------|---------------|
        | 🟢 **Stabilny** | ~ 0 | Steady state, VE odpowiada obciążeniu |
        | 🟡 **Łagodny wzrost** | 0.01-0.05 | Normalna adaptacja do wysiłku |
        | 🔴 **Gwałtowny wzrost** | > 0.05 | Możliwy próg wentylacyjny (VT1/VT2) |

        ---

        ## BR (Breathing Rate) - Częstość oddechów

        **BR** odzwierciedla strategię oddechową:

        - **⬆️ Wzrost BR przy stałej VE**: Płytszy oddech, możliwe zmęczenie przepony
        - **⬇️ Spadek BR przy stałej VE**: Głębszy oddech, lepsza efektywność
        - **➡️ Stabilny BR**: Optymalna strategia oddechowa

        ### Praktyczny przykład:
        - **VE=100, BR=30**: Objętość oddechowa = 3.3L (głęboki oddech)
        - **VE=100, BR=50**: Objętość oddechowa = 2.0L (płytki oddech - nieefektywne!)

        ---

        ## Zastosowania Treningowe VE

        ### 1️⃣ Detekcja Progów (VT1, VT2)
        - **VT1 (Próg tlenowy)**: Pierwszy nieliniowy skok VE względem mocy
        - **VT2 (Próg beztlenowy)**: Drugi, gwałtowniejszy skok VE
        - 🔗 Użyj zakładki **"Ventilation - Progi"** do automatycznej detekcji

        ### 2️⃣ Kontrola Intensywności
        - Jeśli VE rośnie szybciej niż moc → zbliżasz się do progu
        - Stabilna VE przy stałej mocy → jesteś w strefie tlenowej

        ### 3️⃣ Efektywność Oddechowa
        - Optymalna częstość BR: 20-40 oddechów/min
        - Powyżej 50/min: możliwe zmęczenie, stres, lub panika

        ### 4️⃣ Detekcja Zmęczenia
        - **BR rośnie przy spadku VE**: Zmęczenie przepony
        - **VE fluktuuje chaotycznie**: Możliwe odwodnienie lub hipoglikemia

        ---

        ## Korelacja VE vs Moc

        Wykres scatter pokazuje zależność między mocą a wentylacją:

        - **Liniowa zależność**: Normalna odpowiedź fizjologiczna
        - **Punkt załamania**: Próg wentylacyjny (VT)
        - **Stroma krzywa**: Niska wydolność, szybkie zadyszenie

        ### Kolor punktów (czas):
        - **Wczesne punkty (ciemne)**: Początek treningu
        - **Późne punkty (jasne)**: Koniec treningu, kumulacja zmęczenia

        ---

        ## Limitacje Pomiaru VE

        ⚠️ **Czynniki wpływające na dokładność:**
        - Pozycja sensora na klatce piersiowej
        - Oddychanie ustami vs nosem
        - Warunki atmosferyczne (wysokość, wilgotność)
        - Intensywność mowy podczas jazdy

        💡 **Wskazówka**: Dla dokładnej detekcji progów wykonaj Test Stopniowany (Ramp Test)!
        """)
