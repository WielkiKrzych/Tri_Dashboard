"""
DOCX Builder Module.

Generates editable Word documents for Ramp Test reports.
Mirrors the structure of the PDF report.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from .pdf.builder import map_ramp_json_to_pdf_data

logger = logging.getLogger("Tri_Dashboard.DOCXBuilder")

def build_ramp_docx(
    report_data: Dict[str, Any],
    figure_paths: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None
) -> Optional[str]:
    """Build complete multi-page Ramp Test DOCX report (mirrors PDF)."""
    if not HAS_DOCX:
        logger.error("python-docx library not installed. Cannot generate DOCX.")
        return None

    try:
        figure_paths = figure_paths or {}
        doc = Document()
        
        # 1. Map Data
        data = map_ramp_json_to_pdf_data(report_data)
        meta = data['metadata']
        thresholds = data['thresholds']
        cp = data['cp_model']
        smo2 = data['smo2']
        smo2_manual = data['smo2_manual']
        conf = data['confidence']
        kpi = data['kpi']

        # === PAGE 1: OKŁADKA / PODSUMOWANIE ===
        doc.add_heading("Raport z Testu Ramp", 0)
        
        p = doc.add_paragraph()
        p.add_run(f"Data: {meta['test_date']} | ID: {meta['session_id'][:8]} | v{meta['method_version']}").bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_heading("Kluczowe Wyniki", 1)
        
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Parametr'
        hdr_cells[1].text = 'Wartość'
        
        def add_row(k, v):
            row = table.add_row().cells
            row[0].text = k
            row[1].text = str(v)

        add_row("VT1 (Próg tlenowy)", f"{thresholds['vt1_watts']} W")
        add_row("VT2 (Próg beztlenowy)", f"{thresholds['vt2_watts']} W")
        add_row("Critical Power (CP)", f"{cp['cp_watts']} W")
        
        # Pmax
        pmax = meta.get("pmax_watts", "brak danych")
        add_row("Moc Maksymalna (Pmax)", f"{pmax} W" if pmax != "brak danych" else pmax)
        
        # Weight-adjusted
        weight = float(meta['athlete_weight_kg']) if meta['athlete_weight_kg'] and str(meta['athlete_weight_kg']).replace('.','',1).isdigit() else 0
        if weight > 0 and cp['cp_watts'].isdigit():
            rel_cp = float(cp['cp_watts']) / weight
            add_row("Relatywne CP", f"{rel_cp:.2f} W/kg")

        # KPI
        add_row("Efficiency Factor", kpi.get("ef", "brak danych"))
        add_row("Pa:Hr (Decoupling)", f"{kpi.get('pa_hr', '---')}%")
        
        # Chart: Ramp Profile
        if "ramp_profile" in figure_paths:
            doc.add_heading("Profil Przebiegu Testu", 2)
            doc.add_picture(figure_paths["ramp_profile"], width=Inches(6))

        doc.add_page_break()

        # === PAGE 2: SZCZEGÓŁY PROGÓW ===
        doc.add_heading("Szczegóły Progów VT1 / VT2", 1)
        
        table_vt = doc.add_table(rows=1, cols=4)
        table_vt.style = 'Table Grid'
        hdr = table_vt.rows[0].cells
        hdr[0].text = "Próg"
        hdr[1].text = "Moc [W]"
        hdr[2].text = "HR [bpm]"
        hdr[3].text = "VE [L/min]"
        
        for name, watts, hr, ve in [
            ("VT1", thresholds['vt1_watts'], thresholds['vt1_hr'], thresholds['vt1_ve']),
            ("VT2", thresholds['vt2_watts'], thresholds['vt2_hr'], thresholds['vt2_ve'])
        ]:
            cells = table_vt.add_row().cells
            cells[0].text = name
            cells[1].text = str(watts)
            cells[2].text = str(hr)
            cells[3].text = str(ve)

        _add_education_block(doc,
            "Dlaczego to ma znaczenie? (VT1 / VT2)",
            "Progi wentylacyjne to Twoje najważniejsze drogowskazy w planowaniu obciążeń. "
            "VT1 wyznacza granicę komfortu tlenowego i „przepalania” tłuszczy – to tu budujesz bazę na długie godziny. "
            "VT2 to Twój „szklany sufit” – powyżej niego kwas narasta szybciej niż organizm go utylizuje, "
            "co wymaga długiej regeneracji. Znajomość tych punktów pozwala unikać „strefy zgubnej” między progami, "
            "gdzie zmęczenie jest duże, a adaptacje nieoptymalne. Jako trener używam ich, by każda Twoja minuta "
            "na rowerze miała konkretny cel fizjologiczny. Dzięki temu nie trenujesz po prostu „ciężko”, "
            "ale trenujesz mądrze i precyzyjnie."
        )

        if "ve_profile" in figure_paths:
             doc.add_picture(figure_paths["ve_profile"], width=Inches(6))
             
        doc.add_page_break()
        
        # === PAGE 3: SmO2 ===
        doc.add_heading("Analiza Saturacji Mięśniowej (SmO₂)", 1)
        
        doc.add_paragraph(
            "SmO₂ (Saturacja Mięśniowa) dostarcza informacji o balansie między podażą a zapotrzebowaniem na tlen "
            "bezpośrednio w pracującym mięśniu. Jest to sygnał lokalny."
        )

        table_smo2 = doc.add_table(rows=1, cols=3)
        table_smo2.style = 'Table Grid'
        h = table_smo2.rows[0].cells
        h[0].text = "Punkt"
        h[1].text = "Moc [W]"
        h[2].text = "HR [bpm]"
        
        r1 = table_smo2.add_row().cells
        r1[0].text = "SmO₂ Drop Point (LT1)"
        r1[1].text = str(smo2_manual.get('lt1_watts'))
        r1[2].text = str(smo2_manual.get('lt1_hr'))
        
        r2 = table_smo2.add_row().cells
        r2[0].text = "SmO₂ Steady Limit (LT2)"
        r2[1].text = str(smo2_manual.get('lt2_watts'))
        r2[2].text = str(smo2_manual.get('lt2_hr'))

        _add_education_block(doc,
            "Dlaczego to ma znaczenie? (SmO₂ LT1 / LT2)",
            "Saturacja mięśniowa pokazuje prawda bezpośrednio z Twoich nóg, reagując bez opóźnień typowych dla tętna. "
            "LT1 to moment, gdy zapotrzebowanie na tlen zaczyna przeważać nad dostawą – sygnał początku realnej pracy. "
            "LT2 to punkt, w którym system traci kontrolę nad bilansem tlenowym i wchodzi w głęboką desaturację. "
            "Monitorując te trendy, wykrywamy czy ograniczeniem jest Twoje serce, czy naczynia krwionośne w nogach. "
            "Jeśli progi mięśniowe występują przed wentylacyjnymi, wiemy że musimy popracować nad kapilaryzacją. "
            "To narzędzie pozwala nam doprecyzować Twoje strefy z dokładnością do kilku watów."
        )

        if "smo2_power" in figure_paths:
             doc.add_picture(figure_paths["smo2_power"], width=Inches(6))

        doc.add_page_break()

        # === PAGE 4: PDC & CP ===
        if "pdc_curve" in figure_paths:
            doc.add_heading("Krzywa Power-Duration (PDC) i CP", 1)
            doc.add_picture(figure_paths["pdc_curve"], width=Inches(6))
            
            doc.add_paragraph(f"Critical Power (CP): {cp['cp_watts']} W")
            doc.add_paragraph(f"W' (Pojemność Beztlenowa): {cp['w_prime_kj']} kJ")

            _add_education_block(doc,
                "Dlaczego to ma znaczenie? (CP / W')",
                "Model CP/W' to Twoja cyfrowa bateria, która mówi na co Cię stać w decydującym momencie wyścigu. "
                "Critical Power (CP) to Twoja najwyższa moc „długodystansowa”, utrzymywana bez wyczerpania rezerw. "
                "W' to Twój „bak paliwa” na ataki, krótkie podjazdy i sprinty powyżej mocy progowej. "
                "Każdy skok powyżej CP kosztuje konkretną ilość dżuli, a regeneracja następuje dopiero poniżej tego progu. "
                "Rozumienie tego balansu pozwala decydować, czy odpowiedzieć na atak, czy czekać na swoją szansę. "
                "To serce Twojej strategii, które mówi nam, jak optymalnie zarządzać Twoimi siłami."
            )
            
            doc.add_page_break()

        # === PAGE 5: BIOMECHANIKA ===
        if any(k in figure_paths for k in ["biomech_summary", "biomech_torque_smo2"]):
            doc.add_heading("Analiza Biomechaniczna", 1)
            doc.add_paragraph(
                "Analiza biomechaniczna skupia się na sposobie generowania mocy. "
                "Balans między kadencją a momentem obrotowym pozwala zidentyfikować optymalny styl jazdy."
            )
            
            if "biomech_summary" in figure_paths:
                doc.add_picture(figure_paths["biomech_summary"], width=Inches(6))
                
            if "biomech_torque_smo2" in figure_paths:
                doc.add_picture(figure_paths["biomech_torque_smo2"], width=Inches(6))
                doc.add_paragraph(
                    "Interpretacja: Spadek saturacji (SmO₂) przy wysokich momentach obrotowych może świadczyć o okluzji mechanicznej."
                )
            
            doc.add_page_break()

        # === PAGE 6: MODEL METABOLICZNY ===
        if any(k in figure_paths for k in ["vlamax_balance", "limiters_radar"]):
            doc.add_heading("Model Metaboliczny (INSCYD-style)", 1)
            
            if "vlamax_balance" in figure_paths:
                doc.add_picture(figure_paths["vlamax_balance"], width=Inches(6))
                
            if "limiters_radar" in figure_paths:
                doc.add_picture(figure_paths["limiters_radar"], width=Inches(6))
                
            doc.add_paragraph(
                "Twój profil metaboliczny wynika z relacji między VO₂max (zdolność tlenowa) "
                "a VLaMax (zdolność glikolityczna). Limiter wskazuje priorytety treningowe."
            )
            
            doc.add_page_break()

        # === PAGE 7: DRYF I KPI ===
        if any(k in figure_paths for k in ["drift_heatmap_hr", "drift_heatmap_smo2"]) or kpi.get("ef") != "brak danych":
            doc.add_heading("Dryf Fizjologiczny i Wskaźniki KPI", 1)
            
            if "drift_heatmap_hr" in figure_paths:
                doc.add_heading("Mapa Dryfu (HR vs Power)", 2)
                doc.add_picture(figure_paths["drift_heatmap_hr"], width=Inches(6))
                
            if "drift_heatmap_smo2" in figure_paths:
                doc.add_heading("Mapa Oksydacji (SmO2 vs Power)", 2)
                doc.add_picture(figure_paths["drift_heatmap_smo2"], width=Inches(6))
                
            doc.add_heading("KPI - Key Performance Indicators", 2)
            kpi_table = doc.add_table(rows=1, cols=3)
            kpi_table.style = 'Table Grid'
            kh = kpi_table.rows[0].cells
            kh[0].text = "Metryka"
            kh[1].text = "Wartość"
            kh[2].text = "Interpretacja"
            
            def add_kpi(m, v, i):
                c = kpi_table.add_row().cells
                c[0].text = m
                c[1].text = str(v)
                c[2].text = i

            add_kpi("Efficiency Factor", kpi.get("ef", "---"), "Moc na uderzenie serca")
            add_kpi("Pa:Hr (Decoupling)", f"{kpi.get('pa_hr', '---')}%", "Stabilność krążenia")
            add_kpi("% SmO2 Drift", f"{kpi.get('smo2_drift', '---')}%", "Zmęczenie lokalne")
            add_kpi("VO2max Estimate", f"{kpi.get('vo2max_est', '---')} ml/kg", "Szacowany pułap")

            _add_education_block(doc,
                "Dlaczego to ma znaczenie? (Cardiac Drift)",
                "Dryf tętna to sygnał ostrzegawczy Twojego układu chłodzenia, którego nie wolno ignorować. "
                "Jeśli przy stałej mocy tętno systematycznie rośnie, serce musi pracować ciężej, "
                "by przetłoczyć krew nie tylko do mięśni, ale i do skóry w celu ochłodzenia organizmu. "
                "Oznacza to spadek efektywności (EF) i nieproporcjonalnie wysoki koszt energetyczny ruchu. "
                "Śledząc ten parametr, wiemy kiedy warto zainwestować w trening w cieple lub poprawić picie. "
                "To klucz do utrzymania stabilnego tempa w drugiej połowie długodystansowych startów."
            )

            doc.add_page_break()

        # === PAGE 8: INTERPRETACJA ===
        doc.add_heading("Interpretacja i Wnioski", 1)
        
        doc.add_paragraph(f"Ogólna pewność wyniku: {conf.get('overall_confidence', 0)*100:.0f}% ({conf.get('confidence_level')})")
        
        if conf.get('warnings'):
            doc.add_heading("Ostrzeżenia i Zastrzeżenia", 2)
            for w in conf['warnings']:
                doc.add_paragraph(f"⚠️ {w}")
                
        if conf.get('notes'):
            doc.add_heading("Notatki z Analizy", 2)
            for n in conf['notes']:
                doc.add_paragraph(f"ℹ️ {n}")
                
        doc.add_page_break()

        # === PAGE 9: TERMOREGULACJA ===
        if any(k in figure_paths for k in ["thermal_hsi", "thermal_efficiency"]):
            doc.add_heading("Analiza Termoregulacji", 1)
            
            if "thermal_hsi" in figure_paths:
                doc.add_heading("Heat Strain Index (HSI)", 2)
                doc.add_picture(figure_paths["thermal_hsi"], width=Inches(6))
                
            if "thermal_efficiency" in figure_paths:
                doc.add_heading("Cardiovascular Efficiency (Cardiac Drift)", 2)
                doc.add_picture(figure_paths["thermal_efficiency"], width=Inches(6))
                
            doc.add_page_break()

        # === PAGE 10: STREFY TRENINGOWE ===
        doc.add_heading("Sugerowane Strefy Treningowe", 1)
        
        vt1_str = str(thresholds.get("vt1_watts", "0"))
        vt2_str = str(thresholds.get("vt2_watts", "0"))
        
        if vt1_str.isdigit() and vt2_str.isdigit() and int(vt1_str) > 0 and int(vt2_str) > 0:
            vt1 = float(vt1_str)
            vt2 = float(vt2_str)
            
            z1_max = int(vt1 * 0.8)
            z2_min = z1_max
            z2_max = int(vt1)
            z3_min = z2_max
            z3_max = int(vt2)
            z4_min = z3_max
            z4_max = int(vt2 * 1.05)
            z5_min = z4_max
            
            zones_data = [
                ["Strefa", "Zakres [W]", "Opis", "Cel treningowy"],
                ["Z1 Recovery", f"< {z1_max}", "Bardzo łatwy", "Regeneracja"],
                ["Z2 Endurance", f"{z2_min}–{z2_max}", "Komfortowy", "Baza tlenowa"],
                ["Z3 Tempo", f"{z3_min}–{z3_max}", "Umiarkowany", "Próg"],
                ["Z4 Threshold", f"{z4_min}–{z4_max}", "Ciężki", "Wytrzymałość"],
                ["Z5 VO₂max", f"> {z5_min}", "Maksymalny", "Kapacytacja"],
            ]
            
            z_table = doc.add_table(rows=1, cols=4)
            z_table.style = 'Table Grid'
            zh = z_table.rows[0].cells
            for i, h_text in enumerate(zones_data[0]):
                zh[i].text = h_text
                
            for z_row in zones_data[1:]:
                cells = z_table.add_row().cells
                for i, val in enumerate(z_row):
                    cells[i].text = val
        else:
            doc.add_paragraph("Brak danych do wyznaczenia stref (wymagane VT1 i VT2).")
            
        doc.add_page_break()

        # === PAGE 11: TEORIA ===
        doc.add_heading("Teoria i Metodologia", 1)
        doc.add_paragraph(
            "Prawidłowa interpretacja wyniku wymaga zrozumienia podstaw fizjologii sportu. "
            "Report wykorzystuje model trzystrefowy oparty na progach wentylacyjnych."
        )
        
        doc.add_heading("VO2max vs VLaMax", 2)
        doc.add_paragraph(
            "VO2max to pułap tlenowy, VLaMax to maksymalne tempo produkcji mleczanu. "
            "To współzależność tych dwóch parametrów determinuje Twoją moc progową (FTP/CP)."
        )

        doc.add_heading("Hierarchia Sygnałów i Protokół", 2)
        doc.add_paragraph(
            "Nie wszystkie dane są równe. W naszej metodologii najważniejsza jest Wentylacja (VE), "
            "ponieważ najdokładniej odzwierciedla stan metaboliczny całego ciała. HR i SmO₂ to sygnały wspierające. "
            "Długość kroku (np. 1-2 minuty) jest krytyczna, by sygnały zdążyły się ustabilizować. "
            "Zrozumienie opóźnień (HR reaguje najwolniej, SmO₂ najszybciej) pozwala na precyzyjną detekcję progów."
        )
        
        doc.add_page_break()
        
        # === PAGE 12: OGRANICZENIA ===
        doc.add_heading("Ograniczenia Interpretacji", 1)
        
        limitations = [
            ("1. To nie jest badanie medyczne.", 
             "Wyniki są szacunkami algorytmicznymi, nie pomiarami laboratoryjnymi."),
            ("2. Dokładność zależy od jakości danych.", 
             "Niepoprawna kalibracja czujników może wpłynąć na wyniki."),
            ("3. Progi są przybliżeniami.", 
             "VT1/VT2 mogą się różnić od wyników spirometrycznych."),
            ("4. Wyniki są jednorazowe.", 
             "Wydolność zmienia się w czasie – powtarzaj testy co 6-8 tygodni."),
        ]
        
        for lt, ld in limitations:
            p = doc.add_paragraph()
            p.add_run(lt).bold = True
            doc.add_paragraph(ld)

        # Save
        if output_path:
            doc.save(output_path)
            return output_path
            
    except Exception as e:
        logger.error(f"DOCX Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    return None

def _add_education_block(doc, title: str, content: str):
    """Helper to add a styled education block to DOCX."""
    doc.add_paragraph("Część edukacyjna – do zrozumienia wyników", style='Caption')
    
    # Simple distinguished block using indentation and italics
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.right_indent = Inches(0.3)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    
    run_title = p.add_run(f"{title}\n")
    run_title.bold = True
    run_title.font.size = Pt(11)
    
    run_content = p.add_run(content)
    run_content.italic = True
    run_content.font.color.rgb = RGBColor(80, 80, 80)
