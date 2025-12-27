from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import zipfile
import io
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime
from .plots import add_stats_to_legend

def generate_docx_report(metrics, df_plot, df_plot_resampled, uploaded_file, cp_input, 
                        vt1_watts, vt2_watts, rider_weight, vt1_vent, vt2_vent, w_prime_input):
    """
    Generuje raport .docx z rozszerzonymi KPI, W', VO2max i czystymi notatkami.
    """
    
    doc = Document()
    
    # --- OBLICZENIA POBIERANE Z METRICS (Code Centralization) ---
    # Wszystkie kluczowe obliczenia powinny być wykonane w app.py lub modules/calculations.py
    # i przekazane w słowniku metrics.
    
    np_val = metrics.get('np', 0)
    total_work_kj = metrics.get('work_kj', 0)
    avg_pp = metrics.get('avg_pp', 0)
    max_hsi = metrics.get('max_hsi', 0)
    max_core = metrics.get('max_core', 0)
    avg_core = metrics.get('avg_core', 0)
    avg_rmssd = metrics.get('avg_rmssd', 0)
    carbs_total = metrics.get('carbs_total', 0)
    vo2_max_est = metrics.get('vo2_max_est', 0)

    # Uzupełnianie brakujących prostych statystyk (Fallback)
    if np_val == 0 and 'watts' in df_plot.columns:
         rolling_30s = df_plot['watts'].rolling(window=30, min_periods=1).mean()
         np_val = np.power(np.mean(np.power(rolling_30s, 4)), 0.25)

    if total_work_kj == 0 and 'watts' in df_plot.columns:
        total_work_kj = df_plot['watts'].sum() / 1000

    if carbs_total == 0 and 'watts' in df_plot.columns:
        energy_kcal_sec = (df_plot['watts'] / 0.22) / 4184.0
        carbs_burned_sec = (energy_kcal_sec * 1.0) / 4.0 
        carbs_total = carbs_burned_sec.sum()

    if max_core == 0 and 'core_temperature' in df_plot.columns:
        max_core = df_plot['core_temperature'].max()
        avg_core = df_plot['core_temperature'].mean()

    if max_hsi == 0 and 'hsi' in df_plot.columns:
        max_hsi = df_plot['hsi'].max()
    
    # --- KONIEC OBLICZEŃ ---

    # STYLE
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10)
    
    # NAGŁÓWEK
    title = doc.add_heading('Pro Athlete Dashboard - Raport Treningowy', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    date_para = doc.add_paragraph(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.runs[0].font.color.rgb = RGBColor(100, 100, 100)
    
    source_para = doc.add_paragraph(f"Plik: {uploaded_file.name if uploaded_file else 'Brak'}")
    source_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph() 
    
    # SEKCJA 1: PODSUMOWANIE KPI (ROZSZERZONE)
    doc.add_heading('1. Podsumowanie KPI', level=1)
    
    # Tabela z dużą ilością danych (18 wierszy)
    kpi_data = [
        ("Średnia Moc", f"{metrics.get('avg_watts', 0):.0f} W"),
        ("Normalized Power (NP)", f"{np_val:.0f} W"),
        ("Praca Całkowita", f"{total_work_kj:.0f} kJ"),
        ("Średnie Tętno", f"{metrics.get('avg_hr', 0):.0f} bpm"),
        ("Średnia Kadencja", f"{metrics.get('avg_cadence', 0):.0f} rpm"),
        ("Średnia Wentylacja (VE)", f"{metrics.get('avg_vent', 0):.1f} L/min"),
        ("Średnie Oddechy (RR)", f"{metrics.get('avg_rr', 0):.1f} /min"),
        ("Średnie SmO2", f"{df_plot['smo2'].mean() if 'smo2' in df_plot.columns else 0:.1f}%"),
        ("Min SmO2", f"{df_plot['smo2'].min() if 'smo2' in df_plot.columns else 0:.1f}%"),
        ("Max SmO2", f"{df_plot['smo2'].max() if 'smo2' in df_plot.columns else 0:.1f}%"),
        ("Power/HR", f"{metrics.get('power_hr', 0):.2f}"),
        ("Efficiency Factor (EF)", f"{metrics.get('ef_factor', 0):.2f}"),
        ("Średnie Pulse Power", f"{avg_pp:.2f} W/bpm"),
        ("Średnie RMSSD", f"{avg_rmssd:.1f} ms"),
        ("Max HSI (Indeks Ciepła)", f"{max_hsi:.1f}"),
        ("Max Temperatura Ciała", f"{max_core:.2f} °C"),
        ("Średnia Temperatura Ciała", f"{avg_core:.2f} °C"),
        ("Spalone Węglowodany", f"{carbs_total:.0f} g")
    ]

    table = doc.add_table(rows=len(kpi_data)+1, cols=2)
    table.style = 'Light Grid Accent 1'
    
    # Nagłówki tabeli
    table.rows[0].cells[0].text = 'Metryka'
    table.rows[0].cells[1].text = 'Wartość'
    
    # Wypełnianie danymi
    for i, (label, val) in enumerate(kpi_data):
        row_cells = table.rows[i+1].cells
        row_cells[0].text = label
        row_cells[1].text = val
    
    doc.add_paragraph()
    
    # SEKCJA 2: PROGI TRENINGOWE (DODANO W' i VO2max)
    doc.add_heading('2. Progi i Parametry Fizjologiczne', level=1)
    
    p = doc.add_paragraph()
    p.add_run(f"VT1 (Próg Tlenowy): ").bold = True
    p.add_run(f"{vt1_watts} W @ {vt1_vent} L/min\n")
    
    p.add_run(f"VT2 (Próg Beztlenowy): ").bold = True
    p.add_run(f"{vt2_watts} W @ {vt2_vent} L/min\n")
    
    p.add_run(f"CP (Moc Krytyczna): ").bold = True
    p.add_run(f"{cp_input} W\n")
    
    p.add_run(f"W' (Pojemność Beztlenowa): ").bold = True
    p.add_run(f"{w_prime_input} J\n")
    
    p.add_run(f"Szacunkowe VO2max (MMP 5'): ").bold = True
    p.add_run(f"{vo2_max_est:.1f} ml/kg/min\n")
    
    p.add_run(f"Waga zawodnika: ").bold = True
    p.add_run(f"{rider_weight} kg")
    
    doc.add_paragraph()
    
    # SEKCJA 3: STREFY MOCY (POPRAWKA Z6)
    doc.add_heading('3. Czas w Strefach Mocy', level=1)
    
    if 'watts' in df_plot.columns:
        # Definicja stref
        bins = [0, 0.55*cp_input, 0.75*cp_input, 0.90*cp_input, 1.05*cp_input, 1.20*cp_input, 10000]
        labels = ['Z1 Recovery', 'Z2 Endurance', 'Z3 Tempo', 'Z4 Threshold', 'Z5 VO2Max', 'Z6 Anaerobic']
        
        # Obliczenia
        dfz = df_plot.copy()
        dfz['Zone'] = pd.cut(dfz['watts'], bins=bins, labels=labels, right=False)
        
        table2 = doc.add_table(rows=7, cols=3)
        table2.style = 'Light Grid Accent 1'
        
        table2.rows[0].cells[0].text = 'Strefa'
        table2.rows[0].cells[1].text = 'Zakres'
        table2.rows[0].cells[2].text = 'Czas'
        
        for i, (label, low, high) in enumerate(zip(labels, bins[:-1], bins[1:])):
            count = len(dfz[dfz['Zone'] == label])
            time_min = count / 60
            
            # POPRAWKA: Wyświetlanie 2000 W zamiast 10000 W dla estetyki
            display_high = 2000 if high == 10000 else int(high)
            display_low = int(low)
            
            table2.rows[i+1].cells[0].text = label
            table2.rows[i+1].cells[1].text = f"{display_low}-{display_high} W"
            table2.rows[i+1].cells[2].text = f"{time_min:.1f} min"
    
    doc.add_paragraph()
    
    # SEKCJA 4: NOTATKI (WYCZYSZCZONE)
    doc.add_heading('4. Notatki Trenera / Zawodnika', level=1)
    
    # Pusty paragraf jako przestrzeń edytowalna w Pages/Word
    note_p = doc.add_paragraph("[Miejsce na Twoje notatki...]")
    note_p.runs[0].font.italic = True
    note_p.runs[0].font.color.rgb = RGBColor(150, 150, 150)
    
    # Dodajemy kilka pustych linii, żeby wizualnie zrobić miejsce, ale bez "___"
    for _ in range(5):
        doc.add_paragraph("")
    
    # STOPKA
    doc.add_paragraph("---")
    footer = doc.add_paragraph("Raport wygenerowany przez Pro Athlete Dashboard | Streamlit App")
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.color.rgb = RGBColor(128, 128, 128)
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    return doc

def export_all_charts_as_png(df_plot, df_plot_resampled, cp_input, vt1_watts, vt2watts,
                            metrics, rider_weight, uploaded_file, vent_start_sec, vent_end_sec, smo2_start_sec, smo2_end_sec):
    """
    Export wykresów PNG z pełną legendą statystyczną (Ghost Traces).
    """
    
    zip_buffer = BytesIO()
    
    # Konfiguracja wizualna
    layout_args = dict(
        template='plotly_dark',
        height=600,
        width=1200,
        font=dict(family="Inter", size=14),
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(font=dict(size=12))
    )

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # --- 1. POWER ---
        if 'watts_smooth' in df_plot_resampled.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['watts_smooth'],
                                   name='Power', fill='tozeroy', line=dict(color='#00cc96', width=1.5)))
            
            # Statystyki
            avg_p = df_plot_resampled['watts_smooth'].mean()
            max_p = df_plot_resampled['watts_smooth'].max()
            norm_p = np.power(np.mean(np.power(df_plot_resampled['watts_smooth'], 4)), 0.25)
            
            legend_stats = [
                f"⚡ Avg: {avg_p:.0f} W",
                f"🔥 Max: {max_p:.0f} W",
                f"📈 NP (est): {norm_p:.0f} W",
                f"⚖️ W/kg: {avg_p/rider_weight:.2f}"
            ]
            add_stats_to_legend(fig, legend_stats)
            
            fig.update_layout(title='1. Power Profile (W)', xaxis_title='Time (min)', yaxis_title='Power (W)', **layout_args)
            zipf.writestr('01_Power.png', fig.to_image(format='png', width=1200, height=600))
        
        # --- 2. HR ---
        if 'heartrate_smooth' in df_plot_resampled.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['heartrate_smooth'],
                                   name='HR', fill='tozeroy', line=dict(color='#ef553b', width=1.5)))
            
            # Statystyki
            avg_hr = df_plot_resampled['heartrate_smooth'].mean()
            max_hr = df_plot_resampled['heartrate_smooth'].max()
            min_hr = df_plot_resampled[df_plot_resampled['heartrate_smooth'] > 40]['heartrate_smooth'].min()
            
            legend_stats = [
                f"❤️ Avg: {avg_hr:.0f} bpm",
                f"🔥 Max: {max_hr:.0f} bpm",
                f"💤 Min: {min_hr:.0f} bpm"
            ]
            add_stats_to_legend(fig, legend_stats)

            fig.update_layout(title='2. Heart Rate (bpm)', xaxis_title='Time (min)', yaxis_title='HR (bpm)', **layout_args)
            zipf.writestr('02_HeartRate.png', fig.to_image(format='png', width=1200, height=600))

        # --- 3. SmO2 ---
        if 'smo2_smooth' in df_plot_resampled.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['smo2_smooth'],
                                   name='SmO2', line=dict(color='#ab63fa', width=2)))
            
            # Statystyki
            avg_smo2 = df_plot_resampled['smo2_smooth'].mean()
            min_smo2 = df_plot_resampled['smo2_smooth'].min()
            max_smo2 = df_plot_resampled['smo2_smooth'].max()
            
            legend_stats = [
                f"📊 Avg: {avg_smo2:.1f}%",
                f"🔻 Min: {min_smo2:.1f}%",
                f"🔺 Max: {max_smo2:.1f}%"
            ]
            add_stats_to_legend(fig, legend_stats)

            fig.update_layout(title='3. Muscle Oxygenation (SmO2)', xaxis_title='Time (min)', yaxis_title='SmO2 (%)', 
                            yaxis=dict(range=[0, 100]), **layout_args)
            zipf.writestr('03_SmO2.png', fig.to_image(format='png', width=1200, height=600))

        # --- 4. VE + RR (Dual Axis) ---
        if 'tymeventilation_smooth' in df_plot_resampled.columns:
            fig = go.Figure()
            # VE
            fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['tymeventilation_smooth'],
                                   name='VE', line=dict(color='#ffa15a', width=2)))
            
            legend_stats = []
            avg_ve = df_plot_resampled['tymeventilation_smooth'].mean()
            max_ve = df_plot_resampled['tymeventilation_smooth'].max()
            legend_stats.append(f"🫁 Avg VE: {avg_ve:.1f} L/min")
            legend_stats.append(f"🔥 Max VE: {max_ve:.1f} L/min")

            # RR (Prawa oś)
            if 'tymebreathrate_smooth' in df_plot_resampled.columns:
                fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['tymebreathrate_smooth'],
                                       name='RR', line=dict(color='#19d3f3', width=2, dash='dot'), yaxis='y2'))
                avg_rr = df_plot_resampled['tymebreathrate_smooth'].mean()
                legend_stats.append(f"💨 Avg RR: {avg_rr:.1f} /min")
            
            add_stats_to_legend(fig, legend_stats)

            fig.update_layout(title='4. Ventilation (VE) & Respiratory Rate (RR)', 
                            xaxis_title='Time (min)', yaxis=dict(title='VE (L/min)'),
                            yaxis2=dict(title='RR (bpm)', overlaying='y', side='right'), **layout_args)
            zipf.writestr('04_Ventilation_RR.png', fig.to_image(format='png', width=1200, height=600))

        # --- 5. PULSE POWER ---
        if 'watts_smooth' in df_plot_resampled.columns and 'heartrate_smooth' in df_plot_resampled.columns:
            mask = (df_plot_resampled['watts_smooth'] > 50) & (df_plot_resampled['heartrate_smooth'] > 90)
            df_pp = df_plot_resampled[mask].copy()
            if not df_pp.empty:
                df_pp['pp'] = df_pp['watts_smooth'] / df_pp['heartrate_smooth']
                df_pp['pp_smooth'] = df_pp['pp'].rolling(window=30, center=True).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_pp['time_min'], y=df_pp['pp_smooth'],
                                       name='Pulse Power', line=dict(color='#FFD700', width=2)))
                
                # Tu był błąd! Teraz używamy 'stats' biblioteki (bo lista nazywa się 'legend_stats')
                slope, intercept, _, _, _ = stats.linregress(df_pp['time_min'], df_pp['pp'])
                trend = intercept + slope * df_pp['time_min']
                fig.add_trace(go.Scatter(x=df_pp['time_min'], y=trend, name='Trend', line=dict(color='white', dash='dash')))

                # Statystyki
                avg_eff = df_pp['pp'].mean()
                total_drift = slope * (df_pp['time_min'].iloc[-1] - df_pp['time_min'].iloc[0])
                drift_pct = (total_drift / intercept) * 100 if intercept != 0 else 0
                
                legend_stats = [
                    f"🔋 Avg EF: {avg_eff:.2f} W/bpm",
                    f"📉 Drift: {drift_pct:.1f}%"
                ]
                add_stats_to_legend(fig, legend_stats)

                fig.update_layout(title='5. Pulse Power (Watts / Heart Beat)', xaxis_title='Time (min)', 
                                yaxis_title='Efficiency (W/bpm)', **layout_args)
                zipf.writestr('05_PulsePower.png', fig.to_image(format='png', width=1200, height=600))

        # --- 6. HRV TIME (Alpha-1) --- (Note: DFA calculations would need to be re-run or passed in, omitting for brevity of migration unless critical)
        # Assuming df_plot has what we need or we skip detailed HRV plots in PNG batch for now if they depend on local function calls not present.
        # But wait, app.py calls calculate_dynamic_dfa. We can pass it or re-calculate.
        # Ideally, we pass the dataframe that has alpha1 already if computed.
        # For now, I will omit complex recalculations inside report generation and rely on what's passed or simplify.
        # Actually, let's keep it simple for now and rely on passed `df_plot` having necessary columns if possible, but DFA is calculated on fly in app.
        
        # --- 7. POINCARE PLOT --- (Omitting for simple migration, can be added if requested)

        # --- 8. TORQUE vs SmO2 ---
        if 'torque' in df_plot.columns and 'smo2' in df_plot.columns:
            df_bins = df_plot.copy()
            df_bins['Torque_Bin'] = (df_bins['torque'] // 2 * 2).astype(int)
            bin_stats = df_bins.groupby('Torque_Bin')['smo2'].agg(['mean', 'std', 'count']).reset_index()
            bin_stats = bin_stats[bin_stats['count'] > 10]
            
            if not bin_stats.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bin_stats['Torque_Bin'], y=bin_stats['mean']+bin_stats['std'],
                                       mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=bin_stats['Torque_Bin'], y=bin_stats['mean']-bin_stats['std'],
                                       mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,75,75,0.2)', showlegend=False))
                fig.add_trace(go.Scatter(x=bin_stats['Torque_Bin'], y=bin_stats['mean'],
                                       mode='lines+markers', name='Mean SmO2', line=dict(color='#FF4B4B', width=3)))
                
                # Statystyki
                max_t_idx = bin_stats['Torque_Bin'].idxmax()
                max_t = bin_stats.loc[max_t_idx, 'Torque_Bin']
                smo2_at_max = bin_stats.loc[max_t_idx, 'mean']
                
                legend_stats = [
                    f"💪 Max Torque: {max_t:.0f} Nm",
                    f"🩸 SmO2 @ Max: {smo2_at_max:.1f}%"
                ]
                add_stats_to_legend(fig, legend_stats)

                fig.update_layout(title='8. Mechanical Impact: Torque vs SmO2', xaxis_title='Torque (Nm)', 
                                yaxis_title='SmO2 (%)', **layout_args)
                zipf.writestr('08_Torque_SmO2.png', fig.to_image(format='png', width=1200, height=600))

        # --- 9. SmO2 ANALYSIS (FULL + SELECTION + STATS) ---
        s_sec = smo2_start_sec
        e_sec = smo2_end_sec
        
        if 'smo2_smooth' in df_plot_resampled.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['smo2_smooth'],
                name='SmO2 (Full)', line=dict(color='#FF4B4B', width=1.5)))
            
            if 'watts' in df_plot_resampled.columns:
                 fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['watts_smooth'],
                     name='Power', line=dict(color='#1f77b4', width=1), opacity=0.3, yaxis='y2'))

            if s_sec is not None and e_sec is not None:
                s_min = s_sec / 60.0
                e_min = e_sec / 60.0
                fig.add_vrect(x0=s_min, x1=e_min, fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="ANALYSIS")
                
                mask = (df_plot['time'] >= s_sec) & (df_plot['time'] <= e_sec)
                df_sel = df_plot.loc[mask]
                
                if not df_sel.empty:
                    duration = e_sec - s_sec
                    avg_w_sel = df_sel['watts_smooth'].mean() if 'watts_smooth' in df_sel else 0
                    avg_s_sel = df_sel['smo2_smooth'].mean()
                    min_s_sel = df_sel['smo2_smooth'].min()
                    max_s_sel = df_sel['smo2_smooth'].max()
                    
                    slope, intercept, _, _, _ = stats.linregress(df_sel['time'], df_sel['smo2_smooth'])
                    
                    x_trend_min = df_sel['time'] / 60.0
                    y_trend = intercept + slope * df_sel['time']
                    fig.add_trace(go.Scatter(x=x_trend_min, y=y_trend, name='Trend', line=dict(color='yellow', dash='solid', width=3)))

                    m_dur, s_dur = divmod(int(duration), 60)
                    legend_stats = [
                        f"⏱️ Time: {m_dur:02d}:{s_dur:02d}",
                        f"⚡ Avg W: {avg_w_sel:.0f} W",
                        f"📉 Slope: {slope:.4f} %/s",
                        f"📊 Avg SmO2: {avg_s_sel:.1f}%",
                        f"🔻 Min: {min_s_sel:.1f}%",
                        f"🔺 Max: {max_s_sel:.1f}%"
                    ]
                    add_stats_to_legend(fig, legend_stats)

            fig.update_layout(title='9. SmO2 Kinetics Analysis', xaxis_title='Time (min)', yaxis=dict(title='SmO2 (%)'),
                yaxis2=dict(title='Power (W)', overlaying='y', side='right', showgrid=False), **layout_args)
            zipf.writestr('09_SmO2_Analysis.png', fig.to_image(format='png', width=1200, height=600))

        # --- 10. VENT ANALYSIS (FULL + SELECTION + STATS) ---
        s_v_sec = vent_start_sec
        e_v_sec = vent_end_sec
        
        if 'tymeventilation_smooth' in df_plot_resampled.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['tymeventilation_smooth'],
                name='VE (Full)', line=dict(color='#ffa15a', width=1.5)))
            
            if 'watts' in df_plot_resampled.columns:
                 fig.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['watts_smooth'],
                     name='Power', line=dict(color='#1f77b4', width=1), opacity=0.3, yaxis='y2'))
            
            if s_v_sec is not None and e_v_sec is not None:
                s_v_min = s_v_sec / 60.0
                e_v_min = e_v_sec / 60.0
                fig.add_vrect(x0=s_v_min, x1=e_v_min, fillcolor="orange", opacity=0.15, layer="below", line_width=0, annotation_text="ANALYSIS")
                
                mask_v = (df_plot['time'] >= s_v_sec) & (df_plot['time'] <= e_v_sec)
                df_v = df_plot.loc[mask_v]
                
                if not df_v.empty:
                    duration_v = e_v_sec - s_v_sec
                    avg_w_v = df_v['watts_smooth'].mean() if 'watts_smooth' in df_v else 0
                    avg_ve = df_v['tymeventilation_smooth'].mean()
                    min_ve = df_v['tymeventilation_smooth'].min()
                    max_ve = df_v['tymeventilation_smooth'].max()
                    
                    slope_v, intercept_v, _, _, _ = stats.linregress(df_v['time'], df_v['tymeventilation_smooth'])
                    
                    x_trend_v_min = df_v['time'] / 60.0
                    y_trend_v = intercept_v + slope_v * df_v['time']
                    fig.add_trace(go.Scatter(x=x_trend_v_min, y=y_trend_v, name='Trend', line=dict(color='white', dash='solid', width=3)))

                    m_dur_v, s_dur_v = divmod(int(duration_v), 60)
                    legend_stats = [
                        f"⏱️ Time: {m_dur_v:02d}:{s_dur_v:02d}",
                        f"⚡ Avg W: {avg_w_v:.0f} W",
                        f"📈 Slope: {slope_v:.4f} L/s",
                        f"🫁 Avg VE: {avg_ve:.1f} L/min",
                        f"🔻 Min: {min_ve:.1f}",
                        f"🔺 Max: {max_ve:.1f}"
                    ]
                    add_stats_to_legend(fig, legend_stats)

            fig.update_layout(title='10. Ventilation Threshold Analysis', xaxis_title='Time (min)', yaxis=dict(title='VE (L/min)'),
                yaxis2=dict(title='Power (W)', overlaying='y', side='right', showgrid=False), **layout_args)
            zipf.writestr('10_Vent_Analysis.png', fig.to_image(format='png', width=1200, height=600))

        # README
        readme = f"""RAPORT WYKRESÓW Z PEŁNĄ ANALIZĄ
Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Plik: {uploaded_file.name}

Wszystkie wykresy zawierają statystyki w legendzie (Avg, Min, Max, Trends).
Wykresy 9 i 10 zawierają analizę odcinków wybranych w aplikacji.
"""
        zipf.writestr('00_README.txt', readme)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
