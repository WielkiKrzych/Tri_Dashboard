import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Macro Trends Analyst", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #00cc96 !important; font-family: 'Arial', sans-serif; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #ffffff; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #a0a0a0; }
    </style>
""", unsafe_allow_html=True)

# --- FUNKCJE ---
@st.cache_data
def load_and_filter_data(file):
    try:
        # 1. Wczytanie
        df = pd.read_csv(file)
        
        # 2. Standaryzacja nazw kolumn
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # 3. Filtracja (Tylko Rower)
        # Szukamy kolumny 'type' lub 'activity_type'
        type_col = 'type' if 'type' in df.columns else None
        
        if type_col:
            # Zostawiamy Ride i VirtualRide
            df = df[df[type_col].isin(['Ride', 'VirtualRide'])].copy()
        
        # 4. Konwersja daty
        date_col = 'start_date_local' if 'start_date_local' in df.columns else 'start_date'
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            df['date'] = df[date_col] # Alias dla wygody
            df['year_week'] = df[date_col].dt.to_period('W').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {e}")
        return pd.DataFrame()

def calculate_trends(df):
    # Obliczamy proste EF jeśli nie ma (Moc / HR)
    if 'icu_efficiency' not in df.columns and 'icu_average_watts' in df.columns and 'average_heartrate' in df.columns:
        df['icu_efficiency'] = df['icu_average_watts'] / df['average_heartrate']
        # Czyścimy nieskończoności i zera
        df['icu_efficiency'] = df['icu_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Obliczamy Formę (TSB) jeśli mamy Fitness i Fatigue
    if 'icu_fitness' in df.columns and 'icu_fatigue' in df.columns:
        df['tsb_calc'] = df['icu_fitness'] - df['icu_fatigue']
        
    return df

# --- UI APLIKACJI ---
st.title("📈 Macro Trends: Analiza Sezonowa")
st.markdown("To narzędzie analizuje **długoterminowe trendy** Twojej wydolności na podstawie historii aktywności.")

# Sidebar
with st.sidebar:
    st.header("📂 Dane")
    uploaded_file = st.file_uploader("Wgraj 'Activities.csv' (z Intervals/Strava)", type=['csv'])
    
    st.divider()
    st.info("""
    **Jak to czytać?**
    * **Efficiency Factor (EF):** Ile watów generujesz z jednego uderzenia serca (W/bpm).
    * **Wzrost EF:** Twoja "pojemność silnika" rośnie.
    * **Spadek EF:** Zmęczenie lub roztrenowanie.
    """)

if uploaded_file is not None:
    df_raw = load_and_filter_data(uploaded_file)
    
    if df_raw.empty:
        st.warning("Plik pusty lub brak aktywności typu 'Ride'.")
        st.stop()
        
    df = calculate_trends(df_raw)
    
    # Zakres dat
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    st.sidebar.write(f"📅 **Zakres:** {min_date} -> {max_date}")
    st.sidebar.write(f"🚴 **Liczba treningów:** {len(df)}")

    # --- ZAKŁADKI ---
    tab_eff, tab_pmc, tab_ftp, tab_vol = st.tabs(["🚀 Efektywność (EF)", "📊 Fitness (PMC)", "⚡ FTP History", "📅 Objętość"])

    # 1. EFFICIENCY FACTOR (EF)
    with tab_eff:
        st.header("Efficiency Factor (W/HR) Trend")
        
        # Filtrujemy szumy (np. jazdy < 20 min lub bez mocy)
        mask_ef = (df['moving_time'] > 1200) & (df['icu_efficiency'] > 0.5) & (df['icu_efficiency'] < 3.0)
        df_ef = df[mask_ef].copy()
        
        # Rolling average (np. 14 dni) żeby wygładzić wykres
        df_ef['ef_smooth'] = df_ef['icu_efficiency'].rolling(window=10, center=True).mean()
        
        if not df_ef.empty:
            curr_ef = df_ef['ef_smooth'].iloc[-1]
            prev_ef = df_ef['ef_smooth'].iloc[-15] if len(df_ef) > 15 else curr_ef
            delta_ef = (curr_ef - prev_ef) / prev_ef * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Aktualne EF (Średnia 10 sesji)", f"{curr_ef:.2f}", f"{delta_ef:.1f}% vs 2 tyg temu")
            c2.metric("Max EF w sezonie", f"{df_ef['icu_efficiency'].max():.2f}")
            c3.metric("Baza treningów", f"{len(df_ef)}")

            fig_ef = go.Figure()
            
            # Punkty (Poszczególne treningi)
            fig_ef.add_trace(go.Scatter(
                x=df_ef['date'], y=df_ef['icu_efficiency'],
                mode='markers', name='Sesja',
                marker=dict(color='rgba(255, 255, 255, 0.3)', size=4),
                hoverinfo='x+y'
            ))
            
            # Linia Trendu
            fig_ef.add_trace(go.Scatter(
                x=df_ef['date'], y=df_ef['ef_smooth'],
                mode='lines', name='Trend EF (Wygładzony)',
                line=dict(color='#00cc96', width=3)
            ))
            
            fig_ef.update_layout(
                template="plotly_dark",
                title="Rozwój Bazy Tlenowej (EF)",
                yaxis_title="EF [W/bpm]",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_ef, use_container_width=True)
            
            st.info("💡 **Tip:** Rosnąca zielona linia to najlepszy dowód na to, że trening tlenowy działa. Oznacza, że generujesz więcej mocy przy tym samym tętnie.")
        else:
            st.warning("Brak wystarczających danych mocy/tętna do obliczenia EF.")

    # 2. PMC (FITNESS / FATIGUE)
    with tab_pmc:
        st.header("Performance Management Chart (CTL/ATL)")
        
        if 'icu_fitness' in df.columns:
            fig_pmc = go.Figure()
            
            # CTL (Fitness) - Niebieski obszar
            fig_pmc.add_trace(go.Scatter(
                x=df['date'], y=df['icu_fitness'],
                mode='lines', name='Fitness (CTL)',
                fill='tozeroy',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # ATL (Fatigue) - Różowa linia
            if 'icu_fatigue' in df.columns:
                fig_pmc.add_trace(go.Scatter(
                    x=df['date'], y=df['icu_fatigue'],
                    mode='lines', name='Fatigue (ATL)',
                    line=dict(color='#e377c2', width=1.5, dash='dot')
                ))
                
            # Form (TSB) - Słupki na dole? Może osobny wykres, ale wrzućmy jako linię dla czytelności
            if 'tsb_calc' in df.columns:
                fig_pmc.add_trace(go.Scatter(
                    x=df['date'], y=df['tsb_calc'],
                    mode='lines', name='Forma (TSB)',
                    yaxis='y2',
                    line=dict(color='#ff7f0e', width=1)
                ))

            fig_pmc.update_layout(
                template="plotly_dark",
                title="Obciążenie Treningowe",
                yaxis=dict(title="CTL / ATL"),
                yaxis2=dict(title="Forma (TSB)", overlaying='y', side='right', range=[-50, 50], showgrid=False),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0),
                height=500
            )
            st.plotly_chart(fig_pmc, use_container_width=True)
        else:
            st.warning("Brak kolumny 'icu_fitness' (CTL). Upewnij się, że plik ma dane o obciążeniu.")

    # 3. FTP HISTORY
    with tab_ftp:
        st.header("Historia eFTP / FTP")
        
        col_ftp = 'icu_ftp' if 'icu_ftp' in df.columns else ('icu_eftp' if 'icu_eftp' in df.columns else None)
        
        if col_ftp:
            fig_ftp = px.line(df, x='date', y=col_ftp, title="Zmiany FTP w czasie", markers=True)
            fig_ftp.update_traces(line_color='#ab63fa', marker=dict(size=4))
            fig_ftp.update_layout(template="plotly_dark", yaxis_title="FTP [W]")
            st.plotly_chart(fig_ftp, use_container_width=True)
        else:
            st.warning("Brak danych FTP w pliku.")

    # 4. VOLUME
    with tab_vol:
        st.header("Objętość Tygodniowa")
        
        # Agregacja tygodniowa
        if 'moving_time' in df.columns:
            df['hours'] = df['moving_time'] / 3600
            weekly_vol = df.groupby('year_week')['hours'].sum().reset_index()
            # year_week ma format "YYYY-MM-DD/YYYY-MM-DD" z to_period('W'), bierzemy początek tygodnia
            weekly_vol['week_start'] = pd.to_datetime(weekly_vol['year_week'].str.split('/').str[0]) 
            
            fig_vol = px.bar(weekly_vol, x='week_start', y='hours', title="Godziny w tygodniu")
            fig_vol.update_traces(marker_color='#00cc96')
            fig_vol.update_layout(template="plotly_dark", yaxis_title="Godziny [h]", bargap=0.1)
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.warning("Brak danych o czasie trwania.")

else:
    # Ekran startowy (Placeholder)
    st.markdown("""
    ### 👈 Wgraj plik w pasku bocznym
    Pobierz **Activities.csv** ze strony Intervals.icu (Settings -> Download All Data) lub podobnej.
    """)