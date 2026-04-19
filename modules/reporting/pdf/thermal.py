"""
Thermal analysis page builder for PDF reports.

Extracted from layout.py for module separation.
"""

import logging
from typing import Any, Dict, List

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer, Table

logger = logging.getLogger("Tri_Dashboard.PDFThermal")


# ==============================================================================
# LAZY IMPORT HELPER
# ==============================================================================


def _build_chart(chart_path: str, title: str, styles: Dict, max_height_mm: int = 90) -> List:
    """Delegate to layout._build_chart via lazy import to avoid circular dependency."""
    from .layout import _build_chart as _layout_build_chart

    return _layout_build_chart(chart_path, title, styles, max_height_mm)


# ============================================================================
# PAGE 6: TERMOREGULACJA
# ============================================================================


def build_page_thermal(
    thermo_data: Dict[str, Any], figure_paths: Dict[str, str], styles: Dict
) -> List:
    """Build Page 6: Thermal Analysis - Enhanced with metrics and recommendations."""

    from reportlab.platypus import TableStyle

    elements = []

    elements.append(Paragraph("<font size='14'>4.3 TERMOREGULACJA</font>", styles["center"]))
    elements.append(
        Paragraph(
            "<font size='10' color='#7F8C8D'>Dynamika temperatury, tolerancja cieplna, rekomendacje</font>",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 6 * mm))

    elements.append(
        Paragraph(
            "Ciepło jest cichym zabójcą wydajności. Wzrost temperatury głębokiej (Core Temp) "
            "powoduje przekierowanie krwi do skóry (chłodzenie), co zabiera tlen pracującym mięśniom.",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 4 * mm))

    # Chart 1: Core Temp vs HSI
    if figure_paths and "thermal_hsi" in figure_paths:
        elements.extend(
            _build_chart(
                figure_paths["thermal_hsi"], "Temp. Głęboka vs Indeks Zmęczenia (HSI)", styles
            )
        )
        elements.append(Spacer(1, 6 * mm))

    # === KEY NUMBERS TABLE (Thermoregulation) ===
    # thermo_data is now passed directly
    metrics = thermo_data.get("metrics", {})
    classification = thermo_data.get("classification", {})

    # Check if we have actual data (not empty/default)
    has_thermo_data = bool(metrics) and metrics.get("max_core_temp", 0) > 35

    # Get values with proper fallbacks
    if has_thermo_data:
        max_temp = metrics.get("max_core_temp", 0)
        delta_10min = metrics.get("delta_per_10min", 0)
        time_38_0 = metrics.get("time_to_38_0_min")
        time_38_5 = metrics.get("time_to_38_5_min")
        peak_hsi = metrics.get("peak_hsi", 0)

        max_temp_str = f"{max_temp:.1f} C"
        delta_str = f"{delta_10min:.2f} C"
        peak_hsi_str = f"{peak_hsi:.1f}"
    else:
        max_temp_str = "---"
        delta_str = "---"
        time_38_0 = None
        time_38_5 = None
        peak_hsi_str = "---"

    # Classification
    tolerance = classification.get("heat_tolerance", "unknown") if has_thermo_data else "unknown"
    tolerance_color = classification.get("color", "#808080") if has_thermo_data else "#808080"
    tolerance_label = {"good": "DOBRA", "moderate": "ŚREDNIA", "poor": "SŁABA"}.get(
        tolerance, "BRAK DANYCH"
    )

    elements.append(Paragraph("KLUCZOWE LICZBY", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))

    key_data = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Max temp. głęboka", max_temp_str, "Szczytowa temperatura głęboka"],
        ["Delta temp. / 10 min", delta_str, f"Tolerancja: {tolerance_label}"],
        ["Czas do 38.0°C", f"{time_38_0:.0f} min" if time_38_0 else "---", "Próg ostrzegawczy"],
        ["Czas do 38.5°C", f"{time_38_5:.0f} min" if time_38_5 else "---", "Próg krytyczny"],
        ["Peak HSI", peak_hsi_str, "Indeks obciążenia cieplnego"],
    ]

    table = Table(key_data, colWidths=[45 * mm, 35 * mm, 85 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1F77B4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "DejaVuSans-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#555555")),
                ("ROWHEIGHT", (0, 0), (-1, -1), 10 * mm),
                # Delta row gets classification color
                ("BACKGROUND", (0, 2), (-1, 2), HexColor(tolerance_color)),
                ("TEXTCOLOR", (0, 2), (-1, 2), HexColor("#FFFFFF")),
                # Other rows light
                ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#333333")),
                ("BACKGROUND", (0, 3), (-1, 3), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 3), (-1, 3), HexColor("#333333")),
                ("BACKGROUND", (0, 4), (-1, 4), HexColor("#e8e8e8")),
                ("TEXTCOLOR", (0, 4), (-1, 4), HexColor("#333333")),
                ("BACKGROUND", (0, 5), (-1, 5), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 5), (-1, 5), HexColor("#333333")),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 6 * mm))

    # === CLASSIFICATION VERDICT ===
    if tolerance == "poor":
        verdict_text = (
            "<b>SŁABA TOLERANCJA CIEPLNA</b><br/>"
            "Tempo narastania temperatury przekracza próg bezpieczny. "
            "Redystrybucja krwi do skóry konkuruje z dostawą O₂ do mięśni. "
            "Ryzyko przegrzania jest wysokie."
        )
        verdict_color = "#E74C3C"
    elif tolerance == "moderate":
        verdict_text = (
            "<b>ŚREDNIA TOLERANCJA</b><br/>"
            "Układ chłodzenia radzi sobie, ale istnieje margines do poprawy. "
            "Adaptacja cieplna nie jest pełna."
        )
        verdict_color = "#F39C12"
    else:
        verdict_text = (
            "<b>DOBRA TOLERANCJA</b><br/>"
            "Tempo narastania temperatury mieści się w normie. "
            "Układ termoregulacji skutecznie balansuje między chłodzeniem a perfuzją."
        )
        verdict_color = "#27AE60"

    white_style = ParagraphStyle(
        "thermo_verdict", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9
    )
    verdict_box = Table([[Paragraph(verdict_text, white_style)]], colWidths=[165 * mm])
    verdict_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(verdict_color)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(verdict_box)
    elements.append(Spacer(1, 6 * mm))

    # === HR/EF CONNECTION ===
    elements.append(PageBreak())
    elements.append(Paragraph("<b>POŁĄCZENIE Z DRYFEM HR I EF</b>", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))
    elements.append(
        Paragraph(
            "<b>Mechanizm fizjologiczny:</b> Wysoka temperatura wymusza redystrybucję krwi do skóry "
            "w celu chłodzenia. Serce musi pompować większą objętość krwi, by utrzymać zarówno chłodzenie, "
            "jak i dostawę O₂ do mięśni. Efekt: wzrost HR przy stałej mocy (cardiac drift), "
            "spadek Efficiency Factor (EF). To jest kardynalny syndrom przegrzania.",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 2 * mm))
    elements.append(
        Paragraph(
            "<b>Konsekwencje dla wydolności:</b> Każdy 1°C wzrostu temperatury rdzenia powoduje "
            "wzrost HR o 8-10 bpm oraz spadek VO₂max o 1.5-2%. Przy temperaturze >39°C następuje "
            "ochronne ograniczenie rekrutacji jednostek motorycznych przez OUN.",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 2 * mm))
    elements.append(
        Paragraph(
            "<b>Adaptacja cieplna:</b> Po 10-14 dniach treningu w cieple (60-90min @ Z2, temp >28°C) "
            "obserwuje się: wcześniejsze pocenie, niższy HR przy tej samej mocy, zmniejszony drift, "
            "oraz poprawę tolerancji cieplnej o 15-20%.",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 4 * mm))

    # === TRAINING RECOMMENDATIONS ===
    elements.append(Paragraph("<b>REKOMENDACJE</b>", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))

    if tolerance == "poor":
        recommendations = [
            "PRIORYTET: Trening w cieple (heat acclimation) - 10-14 dni, 60-90min @ Z2",
            "Pre-cooling: kamizelka lodowa przed startem",
            "Nawodnienie: 500-800ml/h + elektrolity",
            "Unikaj zawodów >28°C do czasu adaptacji",
        ]
    elif tolerance == "moderate":
        recommendations = [
            "Rozważ 5-7 dni treningu w cieple przed ważnymi zawodami",
            "Chłodzenie zewnętrzne: woda na głowę co 15-20 min",
            "Kontroluj wagę przed/po treningu (max -2%)",
        ]
    else:
        recommendations = [
            "Adaptacja wystarczająca - możesz startować w cieple",
            "Utrzymuj nawodnienie 500-750ml/h + elektrolity (Na+ 500-700mg/L)",
            "Kontynuuj okresowy trening w cieple (1x/tyg)",
        ]

    white_rec_style = ParagraphStyle(
        "rec_white", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9
    )
    rec_data = [[Paragraph(f"* {rec}", white_rec_style)] for rec in recommendations]
    rec_table = Table(rec_data, colWidths=[165 * mm])
    rec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#16213e")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(rec_table)

    # ========================================================================
    # CARDIAC DRIFT ANALYSIS - PRO LAYOUT
    # ========================================================================
    # Use KeepTogether to prevent orphan heading
    drift_header = [
        Spacer(1, 6 * mm),
        Paragraph("Analiza Dryfu Efektywności (Cardiac Drift)", styles["title"]),
        Paragraph(
            "<font size='10' color='#7F8C8D'>Dynamika EF, klasyfikacja dryfu, implikacje treningowe</font>",
            styles["body"],
        ),
        Spacer(1, 4 * mm),
        Paragraph("KLUCZOWE SYGNAŁY", styles["heading"]),
    ]
    elements.append(KeepTogether(drift_header))

    # Get drift data from thermo_data (will be populated by persistence.py)
    drift_data = thermo_data.get("cardiac_drift", {})
    drift_metrics = drift_data.get("metrics", {})
    drift_signals = drift_data.get("key_signals", {})
    drift_class = drift_data.get("classification", {})
    drift_interp = drift_data.get("interpretation", {})

    # Check if we have drift data
    has_drift_data = bool(drift_metrics) and drift_metrics.get("ef_start", 0) > 0

    if has_drift_data:
        ef_start = drift_metrics.get("ef_start", 0)
        ef_end = drift_metrics.get("ef_end", 0)
        delta_pct = drift_metrics.get("delta_ef_pct", 0)
        ef_slope = drift_metrics.get("ef_vs_temp_slope")
        hsi_peak = drift_signals.get("hsi_peak", 0)
        smo2_drift = drift_signals.get("smo2_drift_pct", 0)

        drift_level = drift_class.get("drift_level", "unknown")
        drift_type = drift_class.get("drift_type", "unknown")
        drift_color = drift_class.get("color", "#808080")

        # Status helpers
        def get_delta_status(d):
            if abs(d) < 5:
                return ("STABILNY", "#27AE60")
            elif abs(d) < 10:
                return ("UMIARKOWANY", "#F39C12")
            else:
                return ("WYSOKI", "#E74C3C")

        def get_hsi_status(h):
            if h < 5:
                return ("NISKI", "#27AE60")
            elif h < 8:
                return ("OSTRZEŻENIE", "#F39C12")
            else:
                return ("KRYTYCZNY", "#E74C3C")

        def get_smo2_status(s):
            if abs(s) < 5:
                return ("STABILNY", "#27AE60")
            else:
                return ("DRYF OBWODOWY", "#F39C12")

        delta_status, delta_color = get_delta_status(delta_pct)
        hsi_status, hsi_color = get_hsi_status(hsi_peak)
        smo2_status, smo2_color = get_smo2_status(smo2_drift)
    else:
        ef_start = 0
        ef_end = 0
        delta_pct = 0
        ef_slope = None
        hsi_peak = 0
        smo2_drift = 0
        drift_level = "unknown"
        drift_type = "unknown"
        drift_color = "#808080"
        delta_status, delta_color = ("BRAK", "#808080")
        hsi_status, hsi_color = ("BRAK", "#808080")
        smo2_status, smo2_color = ("BRAK", "#808080")

    # === KEY SIGNALS BOX ===
    elements.append(Spacer(1, 2 * mm))

    key_signals_data = [
        ["Sygnał", "Wartość", "Status"],
        ["EF Start", f"{ef_start:.2f} W/bpm" if ef_start > 0 else "---", "BAZOWY"],
        [
            "EF End",
            f"{ef_end:.2f} W/bpm" if ef_end > 0 else "---",
            f"{delta_pct:+.1f}%" if has_drift_data else "---",
        ],
        [
            "dEF / dC",
            f"{ef_slope:.3f} W/bpm/C" if ef_slope else "---",
            drift_type.upper() if has_drift_data else "---",
        ],
        ["HSI Szczyt", f"{hsi_peak:.1f}" if hsi_peak > 0 else "---", hsi_status],
        ["SmO2 Dryf", f"{smo2_drift:+.1f}%" if has_drift_data else "---", smo2_status],
    ]

    signals_table = Table(key_signals_data, colWidths=[45 * mm, 50 * mm, 70 * mm])
    signals_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "DejaVuSans-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#555555")),
                ("ROWHEIGHT", (0, 0), (-1, -1), 9 * mm),
                # Row backgrounds
                ("BACKGROUND", (0, 1), (1, 1), HexColor("#e8f5e9")),
                ("TEXTCOLOR", (0, 1), (1, 1), HexColor("#333333")),
                ("BACKGROUND", (2, 1), (2, 1), HexColor("#3498DB")),
                ("TEXTCOLOR", (2, 1), (2, 1), HexColor("#FFFFFF")),
                # Delta row - color by status
                ("BACKGROUND", (0, 2), (1, 2), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 2), (1, 2), HexColor("#333333")),
                ("BACKGROUND", (2, 2), (2, 2), HexColor(delta_color)),
                ("TEXTCOLOR", (2, 2), (2, 2), HexColor("#FFFFFF")),
                # EF slope row
                ("BACKGROUND", (0, 3), (1, 3), HexColor("#e8e8e8")),
                ("TEXTCOLOR", (0, 3), (1, 3), HexColor("#333333")),
                ("BACKGROUND", (2, 3), (2, 3), HexColor(drift_color)),
                ("TEXTCOLOR", (2, 3), (2, 3), HexColor("#FFFFFF")),
                # HSI row
                ("BACKGROUND", (0, 4), (1, 4), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 4), (1, 4), HexColor("#333333")),
                ("BACKGROUND", (2, 4), (2, 4), HexColor(hsi_color)),
                ("TEXTCOLOR", (2, 4), (2, 4), HexColor("#FFFFFF")),
                # SmO2 row
                ("BACKGROUND", (0, 5), (1, 5), HexColor("#e8e8e8")),
                ("TEXTCOLOR", (0, 5), (1, 5), HexColor("#333333")),
                ("BACKGROUND", (2, 5), (2, 5), HexColor(smo2_color)),
                ("TEXTCOLOR", (2, 5), (2, 5), HexColor("#FFFFFF")),
            ]
        )
    )
    elements.append(signals_table)
    elements.append(Spacer(1, 6 * mm))

    # === CLASSIFICATION VERDICT BOX ===
    # INTEGRATION: Cardiac drift interpretation now considers thermal verdict
    # NOTE: This is the MECHANISM diagnosis box. The RACE SIMULATION box follows separately.
    if has_drift_data:
        base_mechanism = drift_interp.get("mechanism", "")
        delta_pct_abs = abs(delta_pct)

        # Integrate thermal verdict with cardiac drift interpretation
        if tolerance == "poor" and delta_pct_abs > 5:
            verdict_text = (
                "<b>DIAGNOZA MECHANIZMU: ZMĘCZENIE CENTRALNE</b><br/>"
                f"Słaba tolerancja cieplna + dryf EF {delta_pct:+.1f}% → "
                "Redystrybucja krwi do skóry ogranicza perfuzję mięśniową. "
                "Priorytet: adaptacja cieplna przed intensyfikacją treningu."
            )
            drift_color = "#E74C3C"
        elif tolerance == "moderate" and delta_pct_abs > 7:
            verdict_text = (
                "<b>DIAGNOZA MECHANIZMU: STRES TERMICZNY</b><br/>"
                f"Umiarkowana tolerancja cieplna + dryf EF {delta_pct:+.1f}% → "
                "Obciążenie termiczne przyspiesza dryf sercowy. "
                "Zalecana adaptacja cieplna 5-10 dni przed zawodami."
            )
            drift_color = "#F39C12"
        else:
            verdict_text = "<b>DIAGNOZA MECHANIZMU:</b> " + (
                f"{base_mechanism}"
                if base_mechanism
                else "Stabilność EF w akceptowalnym zakresie — normalny dryf sercowy."
            )
    else:
        verdict_text = (
            "<b>DIAGNOZA MECHANIZMU:</b> Brak danych dryfu — analiza wymaga danych EF (power/HR)."
        )
        drift_color = "#808080"

    verdict_style = ParagraphStyle(
        "verdict_white", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9
    )
    verdict_box = Table([[Paragraph(verdict_text, verdict_style)]], colWidths=[165 * mm])
    verdict_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(drift_color)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(verdict_box)
    elements.append(Spacer(1, 6 * mm))

    # === RACE CONSEQUENCE SIMULATION ===
    # Get core temp from thermo_data
    thermo_metrics = thermo_data.get("metrics", {})
    core_temp_peak = thermo_metrics.get("max_core_temp", 0)

    # Calculate expected power loss after 60 min
    # Threshold: dEF/dC exists AND core_temp > 38
    ef_slope_val = ef_slope if ef_slope else 0

    if abs(ef_slope_val) > 0.01 and core_temp_peak > 38.0:
        # Assume conservative: 1% EF loss ≈ 0.7% power loss
        # If we have delta_pct (change over test), extrapolate to 60 min
        # Use absolute delta_pct as base (already computed)
        power_loss_pct = abs(delta_pct) * 0.7

        # Clamp to reasonable range (2-15%)
        power_loss_pct = max(2.0, min(15.0, power_loss_pct))

        simulation_text = (
            f"<b>SYMULACJA KONSEKWENCJI WYŚCIGOWYCH:</b> Przy obecnym koszcie termicznym (temp. rd. {core_temp_peak:.1f}°C, "
            f"dryf EF {delta_pct:+.1f}%), oczekiwany spadek efektywnej mocy wynosi ~{power_loss_pct:.0f}% "
            "po 60 min w gorących warunkach."
        )
        sim_color = "#E74C3C" if power_loss_pct > 8 else "#F39C12"
    else:
        simulation_text = (
            "<b>SYMULACJA KONSEKWENCJI WYŚCIGOWYCH:</b> Obciążenie termiczne w akceptowalnych granicach. "
            "Brak przewidywanego znaczącego spadku mocy dla 60 min wysokości w bieżących warunkach."
        )
        sim_color = "#27AE60"

    sim_style = ParagraphStyle(
        "sim_text", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9
    )
    sim_box = Table([[Paragraph(simulation_text, sim_style)]], colWidths=[165 * mm])
    sim_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(sim_color)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(sim_box)
    elements.append(Spacer(1, 6 * mm))

    # Chart: EF vs Time/Temp
    if figure_paths and "thermal_efficiency" in figure_paths:
        elements.extend(
            _build_chart(
                figure_paths["thermal_efficiency"], "Efektywność vs Czas/Temperatura", styles
            )
        )
        elements.append(Spacer(1, 4 * mm))

    # === TRAINING IMPLICATIONS BOX ===
    # UWAGA: Usunięto PageBreak aby zapobiec pustej stronie
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("IMPLIKACJE TRENINGOWE", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))

    # Add extended introduction
    elements.append(
        Paragraph(
            "<b>Zastosowanie w praktyce:</b> Poniższe zalecenia treningowe wynikają bezpośrednio "
            "z analizy dryfu sercowego i termicznego. Dostosuj intensywność i strategię "
            "nawodnienia do warunków środowiskowych.",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 2 * mm))

    if has_drift_data:
        implications = drift_interp.get("training_implications", [])[:5]
    else:
        implications = [
            "Brak danych do wygenerowania rekomendacji",
            "Upewnij się, że plik źródłowy zawiera kolumny Power i HR",
        ]

    impl_style = ParagraphStyle(
        "impl_white", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=8
    )
    impl_data = [[Paragraph(f"→ {impl}", impl_style)] for impl in implications]
    impl_table = Table(impl_data, colWidths=[165 * mm])
    impl_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#16213e")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    elements.append(impl_table)

    return elements
