"""
Shared card builders for PDF report metric cards.

Extracted from layout.py to eliminate duplicated nested helper functions.
Each page builder (SmO2, Cardiovascular, Ventilation, Metabolic) previously
defined its own local ``build_card`` / ``build_metric_card`` with minor
variations in font size, padding, and column width.

This module provides a single ``build_metric_card`` with parameters that
cover all four variants.
"""

from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from typing import Dict


def build_metric_card(
    title: str,
    value: str,
    unit: str,
    color: str,
    *,
    interpretation: str = "",
    subtitle: str = "",
    col_width_mm: float = 42.0,
    value_font_size: int = 14,
    interp_font_size: int = 8,
    top_padding: int = 6,
    bottom_padding: int = 6,
    styles: Dict = None,
) -> Table:
    """Build a styled metric card for PDF report pages.

    Parameters
    ----------
    title:
        Card header (small gray text).
    value:
        Main metric value displayed prominently.
    unit:
        Unit label below the value.
    color:
        Hex color string for the border and value text (e.g. ``"#2ECC71"``).
    interpretation:
        Optional interpretation line below the unit.  When provided *and*
        ``subtitle`` is empty, a Spacer is inserted before this line
        (matching the original SmO2 / Cardiovascular layout).
    subtitle:
        Optional smaller gray subtitle (used by the Metabolic page).
        Mutually exclusive with *interpretation* in practice.
    col_width_mm:
        Card column width in millimetres.  Defaults to 42 mm (the most
        common width); the SmO2 page uses 55 mm.
    value_font_size:
        Font size for the main value.  Most pages use 14; SmO2 and
        Cardiovascular use 16.
    interp_font_size:
        Font size for the *interpretation* or *subtitle* line.
    top_padding:
        Cell top padding in points.
    bottom_padding:
        Cell bottom padding in points.
    styles:
        PDF styles dict (must contain a ``"center"`` ParagraphStyle).

    Returns
    -------
    Table
        A ReportLab Table styled as a metric card.
    """
    if styles is None:
        raise ValueError("styles dict is required (pass from page builder)")

    card_content = [
        Paragraph(
            f"<font size='8' color='#7F8C8D'>{title}</font>",
            styles["center"],
        ),
        Paragraph(
            f"<font size='{value_font_size}' color='{color}'><b>{value}</b></font>",
            styles["center"],
        ),
        Paragraph(f"<font size='9'>{unit}</font>", styles["center"]),
    ]

    if subtitle:
        card_content.append(
            Paragraph(
                f"<font size='{interp_font_size}' color='#95A5A6'>{subtitle}</font>",
                styles["center"],
            )
        )
    elif interpretation:
        card_content.append(Spacer(1, 1 * mm))
        card_content.append(
            Paragraph(
                f"<font size='{interp_font_size}'>{interpretation}</font>",
                styles["center"],
            )
        )

    card_table = Table([[card_content]], colWidths=[col_width_mm * mm])
    card_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F8F9FA")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor(color)),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), top_padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), bottom_padding),
            ]
        )
    )
    return card_table
