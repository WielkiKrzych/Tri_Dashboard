from reportlab.platypus import Paragraph, Spacer, Table, Image, PageBreak, KeepTogether, TableStyle
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
import os
import logging
from typing import Dict, Any, List, Optional

from .styles import (
    COLORS,
    PAGE_WIDTH,
    MARGIN,
    get_table_style,
    FONT_FAMILY,
    FONT_FAMILY_BOLD,
    FONT_FAMILY_ITALIC,
)
from .cards import build_metric_card
from ...calculations.version import RAMP_METHOD_VERSION

logger = logging.getLogger("Tri_Dashboard.PDFMetabolic")


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


def build_page_metabolic_engine(metabolic_data: Dict[str, Any], styles: Dict) -> List:
    """Build Metabolic Engine & Training Strategy page - PREMIUM."""

    elements = []

    profile = metabolic_data.get("profile", {})
    block = metabolic_data.get("training_block", {})

    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph("<font size='14'>2.4 SILNIK METABOLICZNY</font>", styles["center"]))
    elements.append(
        Paragraph(
            "<font size='10' color='#7F8C8D'>Profil metaboliczny i strategia 6-8 tygodni</font>",
            styles["center"],
        )
    )
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 1. METABOLIC PROFILE PANEL
    # ==========================================================================

    vo2max = profile.get("vo2max", 0)
    vlamax = profile.get("vlamax", 0)
    cp = profile.get("cp_watts") or 0
    ratio = profile.get("vo2max_vlamax_ratio", 0)
    phenotype = profile.get("phenotype", "unknown")
    vo2max_source = profile.get("vo2max_source", "unknown")
    data_quality = profile.get("data_quality", "unknown")

    elements.append(Paragraph("<b>PROFIL METABOLICZNY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    # VO2max card - handle n/a and show source
    if vo2max and vo2max > 0:
        vo2_color = "#2ECC71" if vo2max >= 60 else ("#F39C12" if vo2max >= 50 else "#E74C3C")
        vo2_val = f"{vo2max:.0f}"
        vo2_source_label = {
            "acsm_5min": "ACSM",
            "acsm_cp": "~CP",
            "metrics_direct": "test",
            "ramp_test_peak": "test",
            "intervals_api": "API",
        }.get(vo2max_source, "")
    else:
        vo2_color = "#7F8C8D"
        vo2_val = "n/a"
        vo2_source_label = "brak danych"

    card1 = build_metric_card(
        "VO₂max",
        vo2_val,
        "ml/kg/min",
        vo2_color,
        subtitle=vo2_source_label,
        interp_font_size=7,
        styles=styles,
    )

    vla_color = "#2ECC71" if vlamax < 0.4 else ("#F39C12" if vlamax < 0.6 else "#E74C3C")
    card2 = build_metric_card(
        "VLaMax",
        f"{vlamax:.2f}",
        "mmol/L/s",
        vla_color,
        subtitle="model metaboliczny",
        interp_font_size=7,
        styles=styles,
    )
    card3 = build_metric_card(
        "CP / FTP",
        f"{cp:.0f}",
        "W",
        "#3498DB",
        subtitle="obliczone",
        interp_font_size=7,
        styles=styles,
    )

    # Ratio card - show n/a if insufficient data
    if ratio and ratio > 0 and vo2max > 0:
        ratio_color = "#2ECC71" if ratio > 130 else ("#F39C12" if ratio > 90 else "#E74C3C")
        ratio_val = f"{ratio:.0f}"
    else:
        ratio_color = "#7F8C8D"
        ratio_val = "n/a"
    card4 = build_metric_card(
        "VO₂/VLa RATIO",
        ratio_val,
        "(ratio)",
        ratio_color,
        subtitle="analiza wieloczynnikowa",
        interp_font_size=7,
        styles=styles,
    )

    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),  # Align cards to top for consistent height
            ]
        )
    )
    elements.append(cards_row)
    elements.append(Spacer(1, 3 * mm))

    # Phenotype badge
    phenotype_colors = {
        "diesel": "#27AE60",
        "allrounder": "#3498DB",
        "puncher": "#F39C12",
        "sprinter": "#E74C3C",
        "climber": "#8E44AD",
    }
    phenotype_names = {
        "diesel": "DIESEL (Aerobic Dominant)",
        "allrounder": "ALLROUNDER",
        "puncher": "PUNCHER",
        "sprinter": "SPRINTER (Glycolytic)",
        "climber": "CLIMBER (Low VLaMax)",
    }

    phenotype_badge = Paragraph(
        f"<font color='white'><b>FENOTYP: {phenotype_names.get(phenotype, phenotype.upper())}</b></font>",
        styles["center"],
    )
    phenotype_table = Table([[phenotype_badge]], colWidths=[170 * mm])
    phenotype_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    HexColor(phenotype_colors.get(phenotype, "#7F8C8D")),
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(phenotype_table)
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 2. LIMITER DIAGNOSIS
    # ==========================================================================

    limiter = profile.get("limiter", "unknown")
    limiter_conf = profile.get("limiter_confidence", 0)

    elements.append(Paragraph("<b>DIAGNOZA OGRANICZEŃ</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    limiter_colors = {
        "aerobic": "#E74C3C",
        "glycolytic": "#F39C12",
        "mixed": "#3498DB",
        "unknown": "#7F8C8D",
    }
    limiter_names = {
        "aerobic": "WYDOLNOŚĆ TLENOWA",
        "glycolytic": "DOMINACJA GLIKOLITYCZNA",
        "mixed": "MIESZANY / ZBALANSOWANY",
        "unknown": "NIEOKREŚLONY",
    }

    limiter_content = [
        Paragraph(
            f"<font color='white'><b>{limiter_names.get(limiter, 'NIEOKREŚLONY')}</b></font>",
            styles["center"],
        ),
        Paragraph(
            f"<font size='10' color='white'>jakość sygnału: {_signal_quality_label(limiter_conf)}</font>",
            styles["center"],
        ),
    ]
    limiter_table = Table([[limiter_content]], colWidths=[170 * mm])
    limiter_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(limiter_colors.get(limiter, "#7F8C8D"))),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(limiter_table)
    elements.append(Spacer(1, 4 * mm))

    # ==========================================================================
    # 3. STRATEGY VERDICT
    # ==========================================================================

    target = profile.get("adaptation_target", "unknown")
    strategy_interp = profile.get("strategy_interpretation", "")

    elements.append(Paragraph("<b>GŁÓWNY CEL ADAPTACYJNY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    target_colors = {
        "increase_vo2max": "#E74C3C",
        "lower_vlamax": "#27AE60",
        "maintain_balance": "#3498DB",
    }
    target_names = {
        "increase_vo2max": "↑ ZWIĘKSZ VO₂max",
        "lower_vlamax": "↓ OBNIŻ VLaMax",
        "maintain_balance": "↔ UTRZYMAJ BALANS",
    }

    target_badge = Paragraph(
        f"<font color='white'><b>{target_names.get(target, target.upper())}</b></font>",
        styles["center"],
    )
    target_table = Table([[target_badge]], colWidths=[170 * mm])
    target_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(target_colors.get(target, "#7F8C8D"))),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(target_table)
    elements.append(Spacer(1, 3 * mm))

    if strategy_interp:
        # Skip first line which duplicates target name in badge above
        lines = strategy_interp.split("\n")
        for line in lines[1:4]:  # Start from line 2, take up to 3 lines
            elements.append(Paragraph(line, styles["body"]))
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 4. TRAINING BLOCK
    # ==========================================================================

    sessions = block.get("sessions", [])
    block_name = block.get("name", "Training Block")
    weeks = block.get("duration_weeks", 6)

    elements.append(Paragraph(f"<b>BLOK TRENINGOWY: {block_name}</b>", styles["subheading"]))
    elements.append(
        Paragraph(
            f"<font size='9' color='#7F8C8D'>{weeks} tygodni | {block.get('primary_focus', '')}</font>",
            styles["body"],
        )
    )
    elements.append(Spacer(1, 3 * mm))

    for i, session in enumerate(sessions[:5]):
        name = session.get("name", "---")
        power = session.get("power_range", "---")
        duration = session.get("duration", "---")
        goal = session.get("adaptation_goal", "---")
        freq = session.get("frequency", "---")
        failure = session.get("failure_criteria", "---")
        exp_smo2 = session.get("expected_smo2", "---")
        exp_hr = session.get("expected_hr", "---")

        session_content = [
            Paragraph(
                f"<font size='10'><b>{i + 1}. {name}</b></font> <font size='9' color='#7F8C8D'>({freq})</font>",
                styles["body"],
            ),
            Paragraph(
                f"<font size='9'><b>Moc:</b> {power} | <b>Czas:</b> {duration}</font>",
                styles["body"],
            ),
            Paragraph(f"<font size='8'><b>Cel:</b> {goal}</font>", styles["body"]),
            Paragraph(
                f"<font size='8' color='#27AE60'>Spodziewany efekt: SmO₂ {exp_smo2}, HR {exp_hr}</font>",
                styles["body"],
            ),
            Paragraph(
                f"<font size='8' color='#E74C3C'>⚠ Sygnał ostrz.: {failure}</font>", styles["body"]
            ),
        ]

        session_table = Table([[session_content]], colWidths=[170 * mm])
        session_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLORS["background"]),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLORS["border"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        elements.append(session_table)
        elements.append(Spacer(1, 2 * mm))

    # ==========================================================================
    # 5. KPI MONITORING
    # ==========================================================================

    kpi_progress = block.get("kpi_progress", [])
    kpi_regress = block.get("kpi_regress", [])

    if kpi_progress or kpi_regress:
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph("<b>KPI MONITORING</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))

        # Progress KPIs
        if kpi_progress:
            elements.append(
                Paragraph("<font color='#27AE60'><b>✓ Sygnały postępu:</b></font>", styles["body"])
            )
            for kpi in kpi_progress[:3]:
                elements.append(Paragraph(f"<font size='9'>• {kpi}</font>", styles["body"]))

        # Regress KPIs
        if kpi_regress:
            elements.append(Spacer(1, 4 * mm))  # Extra spacing before regress section
            elements.append(
                Paragraph(
                    "<font color='#E74C3C'><b>✗ Sygnały regresu / overreaching:</b></font>",
                    styles["body"],
                )
            )
            for kpi in kpi_regress[:3]:
                elements.append(Paragraph(f"<font size='9'>• {kpi}</font>", styles["body"]))

    return elements
