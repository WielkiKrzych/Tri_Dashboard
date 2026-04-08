"""
SmO2 analysis page builder for PDF reports.

Extracted from layout.py for module separation.
"""

from reportlab.platypus import Paragraph, Spacer, Table, KeepTogether, TableStyle
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
import logging
from typing import Dict, List

from .styles import COLORS
from .cards import build_metric_card

logger = logging.getLogger("Tri_Dashboard.PDFSmO2")


# ==============================================================================
# SIGNAL QUALITY HELPERS (duplicated from layout.py — lightweight, no circular dep)
# ==============================================================================


def _signal_quality_label(confidence: float) -> str:
    """Convert numeric confidence to professional signal quality label."""
    if confidence >= 0.85:
        return "bardzo dobra"
    elif confidence >= 0.7:
        return "dobra"
    elif confidence >= 0.5:
        return "wystarczająca"
    else:
        return "podstawowa"


def get_confidence_prefix(confidence: float) -> str:
    """Return empty prefix - interpretation text is self-sufficient."""
    return ""


def get_confidence_suffix(confidence: float) -> str:
    """Get methodology note based on signal quality level."""
    return ""


# ==============================================================================
# LAZY IMPORT HELPER
# ==============================================================================


def _build_chart(chart_path: str, title: str, styles: Dict, max_height_mm: int = 90) -> List:
    """Delegate to layout._build_chart via lazy import to avoid circular dependency."""
    from .layout import _build_chart as _layout_build_chart

    return _layout_build_chart(chart_path, title, styles, max_height_mm)


# ==============================================================================
# MAIN PAGE BUILDER
# ==============================================================================


def build_page_smo2(smo2_data, smo2_manual, figure_paths, styles):
    """Build SmO2 analysis page - PREMIUM MUSCLE OXYGENATION DIAGNOSTIC."""

    elements = []
    smo2_advanced = smo2_data.get("advanced_metrics", {})

    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(
        Paragraph("<font size='14'>3.3 OKSYGENACJA MIĘŚNIOWA (SmO₂)</font>", styles["center"])
    )
    elements.append(
        Paragraph(
            "<font size='10' color='#7F8C8D'>Kliniczna analiza dostawy i wykorzystania tlenu</font>",
            styles["center"],
        )
    )
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 1. METRIC CARDS ROW
    # ==========================================================================

    slope = smo2_advanced.get("slope_per_100w", 0) if smo2_advanced else 0
    halftime = smo2_advanced.get("halftime_reoxy_sec") if smo2_advanced else None
    coupling = smo2_advanced.get("hr_coupling_r", 0) if smo2_advanced else 0
    data_quality = smo2_advanced.get("data_quality", "unknown") if smo2_advanced else "unknown"

    def _smo2_card(title, value, unit, interpretation, color):
        return build_metric_card(
            title,
            value,
            unit,
            color,
            interpretation=interpretation,
            col_width_mm=55,
            value_font_size=16,
            top_padding=8,
            bottom_padding=8,
            styles=styles,
        )

    slope_color = "#E74C3C" if slope < -10 else ("#F39C12" if slope < -5 else "#2ECC71")
    slope_interp = (
        "Szybka (>10%/100W)"
        if slope < -10
        else ("Umiarkowana (5-10%)" if slope < -5 else "Łagodna (<5%)")
    )
    card1 = _smo2_card("DESATURACJA", f"{slope:.1f}", "%/100W", slope_interp, slope_color)

    if halftime:
        ht_color = "#2ECC71" if halftime < 20 else ("#F39C12" if halftime <= 45 else "#E74C3C")
        ht_interp = (
            "Szybki (<20s)"
            if halftime < 20
            else ("Typowy po rampie (20-45s)" if halftime <= 45 else f"Wolny (>45s)")
        )
        card2 = _smo2_card(
            "REOKSYGENACJA", f"{halftime:.0f}", "sekund (po rampie)", ht_interp, ht_color
        )
    else:
        card2 = _smo2_card("REOKSYGENACJA", "---", "sekund", "Brak danych", "#7F8C8D")

    coup_color = (
        "#3498DB" if abs(coupling) > 0.6 else ("#F39C12" if abs(coupling) > 0.3 else "#2ECC71")
    )
    coup_interp = (
        "Silna (centralna)"
        if abs(coupling) > 0.6
        else ("Umiarkowana" if abs(coupling) > 0.3 else "Słaba (lokalna)")
    )
    card3 = _smo2_card("KORELACJA HR", f"{coupling:.2f}", "r-Pearson", coup_interp, coup_color)

    cards_row = Table([[card1, card2, card3]], colWidths=[58 * mm, 58 * mm, 58 * mm])
    cards_row.setStyle(
        TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")])
    )
    elements.append(cards_row)
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 2. OXYGEN DELIVERY MECHANISM PANEL
    # ==========================================================================

    limiter_type = smo2_advanced.get("limiter_type", "unknown") if smo2_advanced else "unknown"
    limiter_conf = smo2_advanced.get("limiter_confidence", 0) if smo2_advanced else 0
    interpretation_adv = smo2_advanced.get("interpretation", "") if smo2_advanced else ""

    mechanism_colors = {
        "local": "#3498DB",
        "central": "#E74C3C",
        "metabolic": "#F39C12",
        "unknown": "#7F8C8D",
    }
    mechanism_names = {
        "local": "OBWODOWY",
        "central": "CENTRALNY",
        "metabolic": "MIESZANY",
        "unknown": "NIEOKREŚLONY",
    }
    mechanism_icons = {"local": "💪", "central": "❤️", "metabolic": "🔥", "unknown": "❓"}

    mech_color = HexColor(mechanism_colors.get(limiter_type, "#7F8C8D"))
    mech_name = mechanism_names.get(limiter_type, "UNDEFINED")
    mech_icon = mechanism_icons.get(limiter_type, "❓")

    elements.append(Paragraph("<b>DOMINUJĄCY MECHANIZM DOSTAWY TLENU</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    verdict_content = [
        Paragraph(f"<font color='white'><b>{mech_icon} {mech_name}</b></font>", styles["center"]),
        Paragraph(
            f"<font size='10' color='white'>jakość sygnału: {_signal_quality_label(limiter_conf)}</font>",
            styles["center"],
        ),
    ]
    verdict_table = Table([[verdict_content]], colWidths=[170 * mm])
    verdict_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), mech_color),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    elements.append(verdict_table)
    elements.append(Spacer(1, 3 * mm))

    if interpretation_adv:
        for line in interpretation_adv.split("\n")[:2]:
            line_with_confidence = (
                get_confidence_prefix(limiter_conf) + line + get_confidence_suffix(limiter_conf)
            )
            elements.append(Paragraph(line_with_confidence, styles["body"]))
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 3. SmO2 THRESHOLDS (compact cards)
    # ==========================================================================

    lt1 = smo2_manual.get("lt1_watts", "---")
    lt2 = smo2_manual.get("lt2_watts", "---")
    lt1_hr = smo2_manual.get("lt1_hr", "---")
    lt2_hr = smo2_manual.get("lt2_hr", "---")
    lt1_smo2_pct = smo2_manual.get("lt1_smo2", "---")
    lt2_smo2_pct = smo2_manual.get("lt2_smo2", "---")

    def fmt(val):
        if val in ("brak danych", None, "---"):
            return "---"
        try:
            return f"{float(val):.0f}"
        except (ValueError, TypeError):
            return str(val)

    def fmt_pct(val):
        if val in ("brak danych", None, "---", 0, "0", "0.0"):
            return ""
        try:
            return f"SmO₂: {float(val):.1f}%"
        except (ValueError, TypeError):
            return ""

    elements.append(Paragraph("<b>PROGI OKSYGENACJI MIĘŚNIOWEJ</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    lt1_smo2_line = fmt_pct(lt1_smo2_pct)
    lt2_smo2_line = fmt_pct(lt2_smo2_pct)

    lt1_card = [
        Paragraph(
            "<font size='9' color='#7F8C8D'>SmO₂-T1 (powiązany z LT1)</font>", styles["center"]
        ),
        Paragraph(f"<font size='14'><b>{fmt(lt1)} W</b></font>", styles["center"]),
        Paragraph(f"<font size='9'>@ {fmt(lt1_hr)} bpm</font>", styles["center"]),
    ]
    if lt1_smo2_line:
        lt1_card.append(
            Paragraph(f"<font size='9' color='#1ABC9C'>{lt1_smo2_line}</font>", styles["center"])
        )

    lt2_card = [
        Paragraph(
            "<font size='9' color='#7F8C8D'>SmO₂-T2 (powiązany z LT2)</font>", styles["center"]
        ),
        Paragraph(f"<font size='14'><b>{fmt(lt2)} W</b></font>", styles["center"]),
        Paragraph(f"<font size='9'>@ {fmt(lt2_hr)} bpm</font>", styles["center"]),
    ]
    if lt2_smo2_line:
        lt2_card.append(
            Paragraph(f"<font size='9' color='#E74C3C'>{lt2_smo2_line}</font>", styles["center"])
        )

    lt1_table = Table([[lt1_card]], colWidths=[85 * mm])
    lt1_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#E8F6F3")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor("#1ABC9C")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    lt2_table = Table([[lt2_card]], colWidths=[85 * mm])
    lt2_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FDEDEC")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor("#E74C3C")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    thresh_row = Table([[lt1_table, lt2_table]], colWidths=[88 * mm, 88 * mm])
    elements.append(thresh_row)
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 4. TRAINING DECISION CARDS
    # ==========================================================================

    recommendations = smo2_advanced.get("recommendations", []) if smo2_advanced else []
    if recommendations:
        elements.append(
            Paragraph("<b>DECYZJE TRENINGOWE NA PODSTAWIE KINETYKI O₂</b>", styles["subheading"])
        )
        elements.append(Spacer(1, 3 * mm))

        expected = {
            "local": ["Wzrost bazowego SmO₂ o 2-4%", "Szybsza reoksygenacja", "Zmniejszenie slope"],
            "central": [
                "Wyższe SmO₂ przy tym samym HR",
                "Lepsza korelacja",
                "Stabilniejsza saturacja",
            ],
            "metabolic": ["Późniejszy drop point", "Mniejszy slope", "Lepsza tolerancja kwasu"],
        }
        exp_list = expected.get(
            limiter_type, ["Poprawa ogólna", "Stabilniejsza saturacja", "Lepszy klirens"]
        )

        for i, rec in enumerate(recommendations[:5]):
            exp_resp = exp_list[i] if i < len(exp_list) else "Poprawa wydolności"
            card_content = [
                Paragraph(f"<font size='10'><b>{i + 1}. {rec}</b></font>", styles["body"]),
                Paragraph(
                    f"<font size='8' color='#27AE60'>Spodziewany efekt: {exp_resp}</font>",
                    styles["body"],
                ),
            ]
            card_table = Table([[card_content]], colWidths=[170 * mm])
            card_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), COLORS["background"]),
                        ("BOX", (0, 0), (-1, -1), 0.5, COLORS["border"]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            elements.append(card_table)
            elements.append(Spacer(1, 2 * mm))

    # Chart
    if figure_paths and "smo2_power" in figure_paths:
        elements.append(Spacer(1, 4 * mm))
        elements.extend(_build_chart(figure_paths["smo2_power"], "SmO₂ vs Power Profile", styles))

    # Data quality
    quality_color = (
        "#2ECC71" if data_quality == "good" else ("#F39C12" if data_quality == "low" else "#7F8C8D")
    )
    quality_label = (
        "Wysoka"
        if data_quality == "good"
        else ("Niska" if data_quality == "low" else "Brak danych")
    )
    elements.append(Spacer(1, 4 * mm))
    elements.append(
        Paragraph(
            f"<font size='8' color='#7F8C8D'>Jakość danych: </font><font size='8' color='{quality_color}'><b>{quality_label}</b></font>",
            styles["body"],
        )
    )

    # ==========================================================================
    # 5. REFERENCE BENCHMARK TABLE (MINI-BENCHMARK)
    # ==========================================================================
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("<b>WZORZEC PORÓWNAWCZY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    # Interpret metrics for benchmark (Vasquez Bonilla 2023: -5 to -15%/100W trained cyclists)
    slope_interp_full = (
        "Szybka desaturacja (ref: -5 do -15%/100W wytren.)"
        if slope < -10
        else (
            "Umiarkowana (typowa dla wytrenowanych)"
            if slope < -5
            else "Łagodna — dobra zdolność oksydacyjna"
        )
    )
    if halftime:
        # Arnold et al. 2024: HRT vastus lat trained = 8-17s (submaks), post-ramp slower
        ht_interp_full = (
            "Szybki (post-ramp <20s = wysoka kapil.)"
            if halftime < 20
            else (
                "Typowy po teście maks. (20-45s)"
                if halftime <= 45
                else "Wolny — sugeruje niską gęstość kapil."
            )
        )
    else:
        ht_interp_full = "Brak danych"
    coup_interp_full = (
        "Silna dominacja serca (centralny)"
        if abs(coupling) > 0.6
        else ("Zrównoważona" if abs(coupling) > 0.3 else "Dominacja obwodowa (lokalna)")
    )

    bench_data = [
        ["Metryka", "Twoja wartość", "Interpretacja kliniczna"],
        ["Desaturacja SmO₂", f"{slope:.1f} %/100W", slope_interp_full],
        ["Czas reoksygenacji", f"{halftime:.0f} s" if halftime else "---", ht_interp_full],
        ["Korelacja HR-SmO₂", f"{coupling:.2f}", coup_interp_full],
    ]

    bench_table = Table(bench_data, colWidths=[40 * mm, 40 * mm, 85 * mm])
    bench_table.setStyle(
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
                ("ROWHEIGHT", (0, 0), (-1, -1), 9 * mm),
                ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#333333")),
                ("BACKGROUND", (0, 2), (-1, 2), HexColor("#e8e8e8")),
                ("TEXTCOLOR", (0, 2), (-1, 2), HexColor("#333333")),
                ("BACKGROUND", (0, 3), (-1, 3), HexColor("#f5f5f5")),
                ("TEXTCOLOR", (0, 3), (-1, 3), HexColor("#333333")),
            ]
        )
    )
    elements.append(bench_table)
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 6. CONCLUSIVE STATEMENT (Links SmO2 with Biomechanics)
    # ==========================================================================

    # Generate conclusive statement based on limiter type
    if limiter_type == "central":
        conclusion = (
            "<b>WNIOSEK:</b> Poprawa VO2max da realny wzrost mocy tylko jeśli utrzymasz "
            "niską okluzję mechaniczną. Priorytet: treningi Z2/Z3 + interwały <95% HR max."
        )
        conclusion_color = "#E74C3C"
    elif limiter_type == "local":
        conclusion = (
            "<b>WNIOSEK:</b> Perfuzja mięśniowa jest limitująca - poprawa siły lub kadencji "
            "może zredukować okluzję i zwolnić desaturację. Priorytet: Strength Endurance."
        )
        conclusion_color = "#3498DB"
    else:
        conclusion = (
            "<b>WNIOSEK:</b> Balans między dostawą a zużyciem O₂ jest dobry. "
            "Kontynuuj zróżnicowany trening, monitorując SmO₂ w sesjach tempo."
        )
        conclusion_color = "#27AE60"

    conclusion_style = ParagraphStyle(
        "conclusion", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9
    )
    conclusion_box = Table([[Paragraph(conclusion, conclusion_style)]], colWidths=[165 * mm])
    conclusion_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(conclusion_color)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(conclusion_box)
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 7. CROSS-VALIDATION VT vs SmO2 + THb CONTEXT
    # ==========================================================================

    # Cross-validation note
    vt1_cross = smo2_manual.get("lt1_watts", "---")
    vt2_cross = smo2_manual.get("lt2_watts", "---")

    cross_note_parts = []
    cross_note_parts.append(
        "<b>WALIDACJA KRZYŻOWA VT vs SmO₂:</b> "
        "Progi wentylacyjne (VT) i progi oksygenacji mięśniowej (SmO₂-T) "
        "to <i>różne sygnały fizjologiczne</i> (Feldmann et al. 2022, Perrey & Ferrari 2024). "
    )

    try:
        vt1_f = float(smo2_manual.get("lt1_watts", 0) or 0)
        vt2_f = float(smo2_manual.get("lt2_watts", 0) or 0)
        vt1_ref = (
            float(smo2_data.get("thresholds", {}).get("vt1_watts", 0) or 0)
            if isinstance(smo2_data.get("thresholds"), dict)
            else 0
        )
        vt2_ref = (
            float(smo2_data.get("thresholds", {}).get("vt2_watts", 0) or 0)
            if isinstance(smo2_data.get("thresholds"), dict)
            else 0
        )

        if vt1_f > 0 and vt1_ref > 0:
            div1 = abs(vt1_f - vt1_ref) / vt1_ref * 100
            cross_note_parts.append(
                f"Rozbieżność T1 vs VT1: {div1:.0f}% "
                f"({'dobra zgodność' if div1 < 10 else 'umiarkowana — interpretować ostrożnie' if div1 < 15 else 'KRYTYCZNA — wymagana weryfikacja'}). "
            )
    except (ValueError, TypeError):
        pass

    cross_note_parts.append(
        "Zgodność ICC SmO₂-T1 vs VT1 = 0.53 (umiarkowana), T2 vs VT2 = 0.80 (dobra) wg Perrey & Ferrari 2024."
    )

    cross_thb_block = [
        Paragraph("".join(cross_note_parts), styles["small"]),
        Spacer(1, 2 * mm),
        Paragraph(
            "<b>Nota THb:</b> THb (Total Hemoglobin) odzwierciedla objętość krwi w mięśniu "
            "(Alvares et al. 2020). Rosnący THb = wazodylatacja; spadający = okluzja naczyniowa. "
            "THb pomaga odróżnić zwiększoną ekstrakcję O₂ od zmniejszonego przepływu.",
            styles["small"],
        ),
    ]
    elements.append(KeepTogether(cross_thb_block))

    return elements
