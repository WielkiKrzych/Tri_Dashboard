"""
Biomechanics & Occlusion Physiology page builder for PDF reports.

Extracted from layout.py for module separation.
"""

import logging
from typing import Any, Dict, List, Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

from .styles import FONT_FAMILY, FONT_FAMILY_BOLD

logger = logging.getLogger("Tri_Dashboard.PDFBiomech")


# ==============================================================================
# LAZY IMPORT HELPER
# ==============================================================================


def _build_chart(chart_path: str, title: str, styles: Dict, max_height_mm: int = 90) -> List:
    """Delegate to layout._build_chart via lazy import to avoid circular dependency."""
    from .layout import _build_chart as _layout_build_chart

    return _layout_build_chart(chart_path, title, styles, max_height_mm)


# ============================================================================
# PAGE: BIOMECHANIKA
# ============================================================================


def build_page_biomech(
    figure_paths: Dict[str, str], styles: Dict, biomech_data: Optional[Dict[str, Any]] = None
) -> List:
    """Build Biomechanics & Occlusion Physiology page - INSCYD/WKO Quality."""

    from reportlab.lib.styles import ParagraphStyle

    elements = []

    elements.append(Paragraph("<font size='14'>3.4 BIOMECHANIKA</font>", styles["center"]))
    elements.append(
        Paragraph(
            "<font size='10' color='#7F8C8D'>Fizjologia okluzji i relacja siła–tlen</font>",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 6 * mm))

    # Chart 1: Torque vs Cadence
    if figure_paths and "biomech_summary" in figure_paths:
        elements.extend(
            _build_chart(figure_paths["biomech_summary"], "Moment Obrotowy vs Kadencja", styles)
        )
        elements.append(Spacer(1, 4 * mm))

    # Chart 2: Torque vs SmO2 (Occlusion)
    if figure_paths and "biomech_torque_smo2" in figure_paths:
        elements.extend(
            _build_chart(
                figure_paths["biomech_torque_smo2"], "Fizjologia Okluzji (Siła vs Tlen)", styles
            )
        )
        elements.append(Spacer(1, 6 * mm))

    # === KEY NUMBERS BOX ===
    if biomech_data and "metrics" in biomech_data:
        metrics = biomech_data["metrics"]
        classification = biomech_data.get("classification", {})

        elements.append(PageBreak())
        elements.append(Paragraph("KLUCZOWE LICZBY", styles["heading"]))
        elements.append(Spacer(1, 2 * mm))

        # Determine colors based on classification
        level = classification.get("level", "unknown")
        level_color = classification.get("color", "#808080")
        level_label = {"low": "NISKA", "moderate": "UMIARKOWANA", "high": "WYSOKA"}.get(
            level, "---"
        )

        # Build metrics table
        occlusion_idx = metrics.get("occlusion_index", 0)
        slope = metrics.get("regression_slope", 0)
        r2 = metrics.get("regression_r2", 0)
        smo2_base = metrics.get("smo2_baseline", 0)
        torque_base = metrics.get("torque_at_baseline", 0)
        torque_10 = metrics.get("torque_at_minus_10")
        torque_20 = metrics.get("torque_at_minus_20")

        key_data = [
            ["Metryka", "Wartość", "Interpretacja"],
            ["INDEKS OKLUZJI", f"{occlusion_idx:.3f}", f"Okluzja: {level_label}"],
            ["Nachylenie SmO2/Torque", f"{slope:.4f} %/Nm", f"R² = {r2:.2f}"],
            ["SmO2 bazowe", f"{smo2_base:.1f} %", f"@ {torque_base:.0f} Nm"],
            [
                "Torque @ SmO2 -10%",
                f"{torque_10:.0f} Nm" if torque_10 else "---",
                "Próg umiarkowanej okluzji",
            ],
            [
                "Torque @ SmO2 -20%",
                f"{torque_20:.0f} Nm" if torque_20 else "---",
                "Próg istotnej okluzji",
            ],
        ]

        table = Table(key_data, colWidths=[50 * mm, 40 * mm, 75 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ("BACKGROUND", (0, 1), (-1, 1), HexColor(level_color)),  # OCCLUSION INDEX row
                    ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#FFFFFF")),
                    ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),  # ALL cells
                    ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "DejaVuSans-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 0), (1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#555555")),
                    ("ROWHEIGHT", (0, 0), (-1, -1), 12 * mm),
                    # Light backgrounds for data rows (except OCCLUSION INDEX row)
                    ("BACKGROUND", (0, 2), (-1, 2), HexColor("#f5f5f5")),
                    ("TEXTCOLOR", (0, 2), (-1, 2), HexColor("#333333")),
                    ("BACKGROUND", (0, 3), (-1, 3), HexColor("#e8e8e8")),
                    ("TEXTCOLOR", (0, 3), (-1, 3), HexColor("#333333")),
                    ("BACKGROUND", (0, 4), (-1, 4), HexColor("#f5f5f5")),
                    ("TEXTCOLOR", (0, 4), (-1, 4), HexColor("#333333")),
                    ("BACKGROUND", (0, 5), (-1, 5), HexColor("#e8e8e8")),
                    ("TEXTCOLOR", (0, 5), (-1, 5), HexColor("#333333")),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 6 * mm))

        # === SAFE TORQUE WINDOW ===
        elements.append(Paragraph("<b>BEZPIECZNE OKNO MOMENTU OBROTOWEGO</b>", styles["heading"]))
        elements.append(Spacer(1, 2 * mm))

        # Define zones
        safe_max = torque_10 if torque_10 else torque_base
        warning_min = torque_10 if torque_10 else 0
        warning_max = torque_20 if torque_20 else 0
        critical_min = torque_20 if torque_20 else 0

        # Build zone table
        zone_rows = [
            ["Strefa", "Zakres Momentu", "Status"],
            [
                "BEZPIECZNA",
                f"< {safe_max:.0f} Nm" if safe_max else "---",
                "Pełna dostawa tlenowa utrzymana",
            ],
            [
                "OSTRZEŻENIE",
                f"{warning_min:.0f}–{warning_max:.0f} Nm" if warning_min and warning_max else "---",
                "Częściowa okluzja, SmO2 spada",
            ],
            [
                "KRYTYCZNA",
                f"> {critical_min:.0f} Nm" if critical_min else "---",
                "Silna okluzja, szybka desaturacja",
            ],
        ]

        zone_table = Table(zone_rows, colWidths=[35 * mm, 50 * mm, 80 * mm])
        zone_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2C3E50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ("FONTNAME", (0, 0), (-1, -1), FONT_FAMILY),
                    ("FONTNAME", (0, 0), (-1, 0), FONT_FAMILY_BOLD),
                    ("FONTNAME", (0, 1), (0, -1), FONT_FAMILY_BOLD),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#555555")),
                    ("ROWHEIGHT", (0, 0), (-1, -1), 10 * mm),
                    # Zone colors
                    ("BACKGROUND", (0, 1), (0, 1), HexColor("#27AE60")),  # SAFE
                    ("TEXTCOLOR", (0, 1), (0, 1), HexColor("#FFFFFF")),
                    ("BACKGROUND", (0, 2), (0, 2), HexColor("#F39C12")),  # WARNING
                    ("TEXTCOLOR", (0, 2), (0, 2), HexColor("#FFFFFF")),
                    ("BACKGROUND", (0, 3), (0, 3), HexColor("#E74C3C")),  # CRITICAL
                    ("TEXTCOLOR", (0, 3), (0, 3), HexColor("#FFFFFF")),
                    # Data cells
                    ("BACKGROUND", (1, 1), (-1, -1), HexColor("#f8f9fa")),
                    ("TEXTCOLOR", (1, 1), (-1, -1), HexColor("#333333")),
                ]
            )
        )
        elements.append(zone_table)
        elements.append(Spacer(1, 3 * mm))

        # Collapse verdict
        collapse_point = torque_20 if torque_20 else (torque_10 if torque_10 else 0)
        if collapse_point:
            collapse_text = (
                f"Efektywna dostawa tlenowa załamuje się powyżej {collapse_point:.0f} Nm."
            )
        else:
            collapse_text = "Progi momentu obrotowego niedostępne dla oceny okluzji."
        elements.append(Paragraph(collapse_text, styles["body"]))
        elements.append(Spacer(1, 6 * mm))

        # === VERDICT BOX ===
        elements.append(Paragraph("WERDYKT", styles["heading"]))
        elements.append(Spacer(1, 2 * mm))

        # Verdict based on classification
        if level == "high":
            verdict_text = (
                "<b>WYSOKA OKLUZJA MECHANICZNA</b><br/>"
                "Przy wysokich momentach obrotowych naczynia mięśniowe są mechanicznie "
                "kompresowane, ograniczając perfuzję mimo dostępnego VO₂ systemowego. "
                "Styl siłowy (niska kadencja) prowadzi do przedwczesnej hipoksji lokalnej."
            )
            verdict_color = "#E74C3C"
        elif level == "moderate":
            verdict_text = (
                "<b>UMIARKOWANA OKLUZJA</b><br/>"
                "Spadek SmO₂ jest proporcjonalny do wzrostu momentu obrotowego. "
                "Mięśnie wykazują pewną tolerancję na siły, ale istnieje wyraźna granica "
                "powyżej której desaturacja przyspiesza."
            )
            verdict_color = "#F39C12"
        else:
            verdict_text = (
                "<b>NISKA OKLUZJA</b><br/>"
                "Kapilaryzacja mięśniowa jest wystarczająca, aby utrzymać perfuzję "
                "nawet przy wysokich momentach obrotowych. Możliwość efektywnej pracy siłowej."
            )
            verdict_color = "#27AE60"

        verdict_box = Table(
            [
                [
                    Paragraph(
                        verdict_text,
                        ParagraphStyle(
                            "verdict",
                            parent=styles["body"],
                            textColor=HexColor("#FFFFFF"),
                            fontSize=9,
                        ),
                    )
                ]
            ],
            colWidths=[165 * mm],
        )
        verdict_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor(verdict_color)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("BOX", (0, 0), (-1, -1), 1, HexColor(verdict_color)),
                ]
            )
        )
        elements.append(verdict_box)
        elements.append(Spacer(1, 6 * mm))

        # === MECHANISM DESCRIPTION ===
        interpretation = biomech_data.get("interpretation", {})
        mechanism = interpretation.get("mechanism", "")
        riding_style = interpretation.get("riding_style", "")

        if mechanism:
            elements.append(Paragraph("<b>MECHANIZM FIZJOLOGICZNY</b>", styles["heading"]))
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph(mechanism, styles["body"]))
            elements.append(Spacer(1, 4 * mm))

        if riding_style:
            elements.append(Paragraph("<b>WPŁYW NA STYL JAZDY</b>", styles["heading"]))
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph(riding_style, styles["body"]))
            elements.append(Spacer(1, 2 * mm))
            # Cadence context for occlusion interpretation (Hammer et al. 2021)
            avg_cadence = metrics.get("avg_cadence", 0)
            if avg_cadence and avg_cadence > 0:
                if avg_cadence < 75:
                    cad_note = (
                        f"<b>Kontekst kadencji ({avg_cadence:.0f} RPM):</b> Niska kadencja zwiększa "
                        "moment obrotowy na obrót i wydłuża fazę skurczu izometrycznego, "
                        "nasilając mechaniczną okluzję naczyń (Hammer et al. 2021). "
                        "Rozważ podniesienie kadencji do 85-95 RPM w strefach Z3-Z4, "
                        "aby zmniejszyć szczytowe ciśnienie śródmięśniowe."
                    )
                elif avg_cadence > 100:
                    cad_note = (
                        f"<b>Kontekst kadencji ({avg_cadence:.0f} RPM):</b> Wysoka kadencja "
                        "minimalizuje okluzję mechaniczną, ale zwiększa koszt metaboliczny "
                        "koordynacji nerwowo-mięśniowej. Optymalny zakres: 85-95 RPM."
                    )
                else:
                    cad_note = (
                        f"<b>Kontekst kadencji ({avg_cadence:.0f} RPM):</b> Kadencja w optymalnym "
                        "zakresie, równoważącym okluzję mechaniczną z kosztem metabolicznym."
                    )
                elements.append(Paragraph(cad_note, styles["body"]))
            elements.append(Spacer(1, 4 * mm))

        # === TRAINING RECOMMENDATIONS ===
        recommendations = interpretation.get("recommendations", [])
        if recommendations:
            elements.append(Paragraph("<b>ZALECENIA TRENINGOWE</b>", styles["heading"]))
            elements.append(Spacer(1, 2 * mm))

            # Use white text style for visibility on dark background
            white_style = ParagraphStyle(
                "white_body", parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9
            )
            rec_data = [[Paragraph(f"• {rec}", white_style)] for rec in recommendations]
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
    else:
        # Fallback for legacy data without biomech analysis
        elements.append(
            Paragraph(
                "<b>Interpretacja:</b> Spadek saturacji (SmO₂) przy wysokich momentach obrotowych "
                "może świadczyć o okluzji mechanicznej lub niskiej efektywności układu krążenia "
                "w warunkach wysokiego napięcia mięśniowego.",
                styles["body"],
            )
        )

    return elements
