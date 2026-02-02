"""
Summary Page PDF Generator.

Generates a multi-page PDF from the Summary tab with each chart and values on separate pages.
"""

import io
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    PageBreak,
    Spacer,
    Paragraph,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd
import numpy as np
import os

# Set matplotlib backend to Agg (no display required)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Register fonts with Polish character support
def _register_polish_fonts():
    """Register DejaVuSans fonts for Polish character support."""
    try:
        import matplotlib

        font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(font_dir, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(
            TTFont("DejaVuSans-Bold", os.path.join(font_dir, "DejaVuSans-Bold.ttf"))
        )
        return "DejaVuSans", "DejaVuSans-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def generate_summary_pdf(
    df_plot: pd.DataFrame,
    metrics: Dict[str, Any],
    cp_input: int,
    w_prime_input: int,
    rider_weight: float,
    vt1_watts: int,
    vt2_watts: int,
    lt1_watts: int,
    lt2_watts: int,
    threshold_result: Any,
    smo2_result: Any,
    uploaded_file_name: str,
) -> bytes:
    """
    Generate PDF from Summary tab with each section on separate page.

    Each page contains:
    - Title and section number
    - Chart (if applicable)
    - All values and statistics

    Returns:
        PDF bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # Register Polish fonts
    font_normal, font_bold = _register_polish_fonts()

    # Prepare styles with Polish font support
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontName=font_bold,
        fontSize=24,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=30,
        alignment=1,  # Center
    )
    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=18,
        textColor=colors.HexColor("#2ca02c"),
        spaceAfter=20,
    )
    value_style = ParagraphStyle(
        "ValueStyle", parent=styles["Normal"], fontName=font_normal, fontSize=12, spaceAfter=10
    )

    story = []

    # Page 0: Title Page
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("📊 Podsumowanie Treningu", title_style))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"Plik: {uploaded_file_name}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"CP: {cp_input} W | W': {w_prime_input} J", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"Waga: {rider_weight} kg", styles["Normal"]))
    story.append(PageBreak())

    # Page 1: Training Overview (Wykres 1)
    story.append(Paragraph("1️⃣ Przebieg Treningu", section_style))
    story.append(Spacer(1, 0.5 * cm))

    # Add training chart
    if "watts" in df_plot.columns:
        chart_bytes = _create_training_chart_matplotlib(df_plot)
        if chart_bytes:
            story.append(RLImage(io.BytesIO(chart_bytes), width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.5 * cm))

    # Add basic metrics
    if metrics:
        metrics_data = [
            ["Parametr", "Wartość"],
            ["Średnia moc", f"{metrics.get('avg_power', 0):.0f} W"],
            ["Maksymalna moc", f"{metrics.get('max_power', 0):.0f} W"],
            ["Średnie HR", f"{metrics.get('avg_hr', 0):.0f} bpm"],
            ["Maksymalne HR", f"{metrics.get('max_hr', 0):.0f} bpm"],
            ["Średnie VE", f"{metrics.get('avg_ve', 0):.1f} L/min"],
            ["Średnie BR", f"{metrics.get('avg_br', 0):.0f} /min"],
        ]
        metrics_table = Table(metrics_data, colWidths=[8 * cm, 8 * cm])
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                ]
            )
        )
        story.append(metrics_table)

    story.append(PageBreak())

    # Page 2: Ventilation VE and BR (Wykres 2)
    story.append(Paragraph("2️⃣ Wentylacja (VE) i Oddechy (BR)", section_style))
    story.append(Spacer(1, 0.5 * cm))

    if "tymeventilation" in df_plot.columns:
        chart_bytes = _create_ve_br_chart_matplotlib(df_plot)
        if chart_bytes:
            story.append(RLImage(io.BytesIO(chart_bytes), width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.5 * cm))

        # VE Statistics
        ve_min = df_plot["tymeventilation"].min()
        ve_max = df_plot["tymeventilation"].max()
        ve_mean = df_plot["tymeventilation"].mean()

        story.append(Paragraph("<b>🫁 Statystyki VE (Wentylacja):</b>", value_style))
        story.append(
            Paragraph(
                f"Min: {ve_min:.1f} L/min | Max: {ve_max:.1f} L/min | Śr: {ve_mean:.1f} L/min",
                value_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        # BR Statistics
        if "tymebreathrate" in df_plot.columns:
            br_min = df_plot["tymebreathrate"].min()
            br_max = df_plot["tymebreathrate"].max()
            br_mean = df_plot["tymebreathrate"].mean()

            story.append(Paragraph("<b>🌬️ Statystyki BR (Oddechy):</b>", value_style))
            story.append(
                Paragraph(
                    f"Min: {br_min:.0f} /min | Max: {br_max:.0f} /min | Śr: {br_mean:.0f} /min",
                    value_style,
                )
            )

    story.append(PageBreak())

    # Page 3: CP Model (Wykres 3)
    story.append(Paragraph("3️⃣ Model Matematyczny CP", section_style))
    story.append(Spacer(1, 0.5 * cm))

    # CP Model values
    cp_data = [
        ["Parametr", "Wartość"],
        ["CP (Critical Power)", f"{cp_input} W"],
        ["W' (W Prime)", f"{w_prime_input} J"],
        ["W' (kJ)", f"{w_prime_input / 1000:.1f} kJ"],
    ]

    if "watts" in df_plot.columns:
        max_power = df_plot["watts"].max()
        cp_data.append(["Maksymalna moc", f"{max_power:.0f} W"])
        cp_data.append(["% CP", f"{(max_power / cp_input * 100):.1f}%" if cp_input > 0 else "N/A"])

    cp_table = Table(cp_data, colWidths=[8 * cm, 8 * cm])
    cp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9467bd")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(cp_table)

    story.append(PageBreak())

    # Page 4: SmO2 vs THb (Wykres 4)
    story.append(Paragraph("4️⃣ SmO2 vs THb w czasie", section_style))
    story.append(Spacer(1, 0.5 * cm))

    if "smo2" in df_plot.columns:
        chart_bytes = _create_smo2_thb_chart_matplotlib(df_plot)
        if chart_bytes:
            story.append(RLImage(io.BytesIO(chart_bytes), width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.5 * cm))

        # SmO2 Statistics
        smo2_min = df_plot["smo2"].min()
        smo2_max = df_plot["smo2"].max()
        smo2_mean = df_plot["smo2"].mean()

        story.append(Paragraph("<b>🩸 SmO2 Statystyki:</b>", value_style))
        story.append(
            Paragraph(
                f"Min: {smo2_min:.1f}% | Max: {smo2_max:.1f}% | Śr: {smo2_mean:.1f}%", value_style
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        # THb Statistics
        if "thb" in df_plot.columns:
            thb_min = df_plot["thb"].min()
            thb_max = df_plot["thb"].max()
            thb_mean = df_plot["thb"].mean()

            story.append(Paragraph("<b>💉 THb Statystyki:</b>", value_style))
            story.append(
                Paragraph(
                    f"Min: {thb_min:.2f} g/dL | Max: {thb_max:.2f} g/dL | Śr: {thb_mean:.2f} g/dL",
                    value_style,
                )
            )

    story.append(PageBreak())

    # Page 5: Ventilatory Thresholds VT1/VT2 (Wykres 5)
    story.append(Paragraph("5️⃣ Progi Wentylacyjne (VT1/VT2)", section_style))
    story.append(Spacer(1, 0.5 * cm))

    # VT1 and VT2 data
    vt_data = [
        ["Próg", "Moc", "HR", "VE", "BR", "TV"],
    ]

    # Get VT1 values
    vt1_hr = getattr(threshold_result, "vt1_hr", 0) or 0
    vt1_ve = getattr(threshold_result, "vt1_ve", 0) or 0
    vt1_br = getattr(threshold_result, "vt1_br", 0) or 0
    vt1_tv = (vt1_ve / vt1_br * 1000) if vt1_br > 0 else 0

    vt_data.append(
        [
            "VT1 (Próg Tlenowy)",
            f"{vt1_watts} W",
            f"{vt1_hr:.0f} bpm" if vt1_hr else "-",
            f"{vt1_ve:.1f} L/min" if vt1_ve else "-",
            f"{vt1_br:.0f} /min" if vt1_br else "-",
            f"{vt1_tv:.0f} mL" if vt1_tv else "-",
        ]
    )

    # Get VT2 values
    vt2_hr = getattr(threshold_result, "vt2_hr", 0) or 0
    vt2_ve = getattr(threshold_result, "vt2_ve", 0) or 0
    vt2_br = getattr(threshold_result, "vt2_br", 0) or 0
    vt2_tv = (vt2_ve / vt2_br * 1000) if vt2_br > 0 else 0

    vt_data.append(
        [
            "VT2 (Próg Beztlenowy)",
            f"{vt2_watts} W",
            f"{vt2_hr:.0f} bpm" if vt2_hr else "-",
            f"{vt2_ve:.1f} L/min" if vt2_ve else "-",
            f"{vt2_br:.0f} /min" if vt2_br else "-",
            f"{vt2_tv:.0f} mL" if vt2_tv else "-",
        ]
    )

    vt_table = Table(vt_data, colWidths=[4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    vt_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ffa15a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ffe4cc")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#ffcccc")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(vt_table)

    if cp_input > 0:
        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                f"VT1: ~{(vt1_watts / cp_input) * 100:.0f}% CP | VT2: ~{(vt2_watts / cp_input) * 100:.0f}% CP",
                value_style,
            )
        )

    story.append(PageBreak())

    # Page 6: SmO2 Thresholds LT1/LT2 (Wykres 6)
    story.append(Paragraph("6️⃣ Progi SmO2 (LT1/LT2)", section_style))
    story.append(Spacer(1, 0.5 * cm))

    # LT1 and LT2 data
    lt_data = [
        ["Próg", "Moc", "HR", "SmO2"],
    ]

    # Get LT1 values from smo2_result
    if smo2_result:
        lt1_hr = getattr(smo2_result, "t1_hr", 0) or 0
        lt1_smo2 = getattr(smo2_result, "t1_smo2", 0) or 0
        lt2_hr = getattr(smo2_result, "t2_onset_hr", 0) or 0
        lt2_smo2 = getattr(smo2_result, "t2_onset_smo2", 0) or 0
    else:
        lt1_hr = lt1_smo2 = lt2_hr = lt2_smo2 = 0

    lt_data.append(
        [
            "LT1 (SteadyState)",
            f"{lt1_watts} W",
            f"{lt1_hr:.0f} bpm" if lt1_hr else "-",
            f"{lt1_smo2:.1f}%" if lt1_smo2 else "-",
        ]
    )

    lt_data.append(
        [
            "LT2 (RCP)",
            f"{lt2_watts} W" if lt2_watts > 0 else "Nie wykryto",
            f"{lt2_hr:.0f} bpm" if lt2_hr else "-",
            f"{lt2_smo2:.1f}%" if lt2_smo2 else "-",
        ]
    )

    lt_table = Table(lt_data, colWidths=[5 * cm, 5 * cm, 5 * cm, 5 * cm])
    lt_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2ca02c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ccffcc")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#ffcccc")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(lt_table)

    if cp_input > 0 and lt1_watts > 0:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"LT1: ~{(lt1_watts / cp_input) * 100:.0f}% CP", value_style))

    story.append(PageBreak())

    # Page 7: TDI Analysis
    story.append(Paragraph("7️⃣ Threshold Discordance Index (TDI)", section_style))
    story.append(Spacer(1, 0.5 * cm))

    if vt1_watts > 0 and lt1_watts > 0:
        tdi = abs(vt1_watts - lt1_watts) / ((vt1_watts + lt1_watts) / 2) * 100
        story.append(Paragraph(f"<b>TDI (VT1 vs LT1):</b> {tdi:.1f}%", value_style))

        if tdi < 10:
            story.append(Paragraph("✅ Wysoka zgodność progów", value_style))
        elif tdi < 20:
            story.append(Paragraph("⚠️ Umiarkowana zgodność progów", value_style))
        else:
            story.append(Paragraph("❌ Niska zgodność progów - wymaga weryfikacji", value_style))

    story.append(PageBreak())

    # Page 8: VO2max Estimation
    story.append(Paragraph("8️⃣ Estymacja VO2max z Niepewnością (CI95%)", section_style))
    story.append(Spacer(1, 0.5 * cm))

    # Calculate VO2max
    if "watts" in df_plot.columns and rider_weight > 0:
        mmp_5min = df_plot["watts"].rolling(300, min_periods=1).mean().max()
        vo2max = 16.61 + 8.87 * (mmp_5min / rider_weight)

        story.append(Paragraph(f"<b>Estymowane VO2max:</b> {vo2max:.1f} ml/kg/min", value_style))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"MMP5: {mmp_5min:.0f} W", value_style))
        story.append(Paragraph(f"Waga: {rider_weight:.1f} kg", value_style))
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph("<i>Wzór: VO2max = 16.61 + 8.87 × (MMP5 / Waga)</i>", styles["Italic"])
        )

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def _create_training_chart_matplotlib(df_plot: pd.DataFrame) -> Optional[bytes]:
    """Create training overview chart using matplotlib."""
    try:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

        # Power
        if "watts" in df_plot.columns:
            ax1.plot(time_x, df_plot["watts"], color="#1f77b4", linewidth=1.5, label="Moc (W)")
            ax1.set_xlabel("Czas (s)")
            ax1.set_ylabel("Moc (W)", color="#1f77b4")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")

        # HR
        if "hr" in df_plot.columns:
            ax2 = ax1.twinx()
            ax2.plot(time_x, df_plot["hr"], color="#d62728", linewidth=1.5, label="HR (bpm)")
            ax2.set_ylabel("HR (bpm)", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")

        plt.title("Przebieg Treningu")
        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue()
    except Exception as e:
        print(f"Error creating training chart: {e}")
        return None


def _create_ve_br_chart_matplotlib(df_plot: pd.DataFrame) -> Optional[bytes]:
    """Create VE/BR chart using matplotlib."""
    try:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

        # VE
        if "tymeventilation" in df_plot.columns:
            ve_smooth = df_plot["tymeventilation"].rolling(10, center=True).mean()
            ax1.plot(time_x, ve_smooth, color="#ffa15a", linewidth=2, label="VE (L/min)")
            ax1.set_xlabel("Czas (s)")
            ax1.set_ylabel("VE (L/min)", color="#ffa15a")
            ax1.tick_params(axis="y", labelcolor="#ffa15a")

        # BR
        if "tymebreathrate" in df_plot.columns:
            ax2 = ax1.twinx()
            br_smooth = df_plot["tymebreathrate"].rolling(10, center=True).mean()
            ax2.plot(time_x, br_smooth, color="#00cc96", linewidth=2, label="BR (/min)")
            ax2.set_ylabel("BR (/min)", color="#00cc96")
            ax2.tick_params(axis="y", labelcolor="#00cc96")

        plt.title("Wentylacja (VE) i Oddechy (BR)")
        plt.tight_layout()

        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue()
    except Exception as e:
        print(f"Error creating VE/BR chart: {e}")
        return None


def _create_smo2_thb_chart_matplotlib(df_plot: pd.DataFrame) -> Optional[bytes]:
    """Create SmO2/THb chart using matplotlib."""
    try:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

        # SmO2
        if "smo2" in df_plot.columns:
            smo2_smooth = df_plot["smo2"].rolling(5, center=True).mean()
            ax1.plot(time_x, smo2_smooth, color="#2ca02c", linewidth=2, label="SmO2 (%)")
            ax1.set_xlabel("Czas (s)")
            ax1.set_ylabel("SmO2 (%)", color="#2ca02c")
            ax1.tick_params(axis="y", labelcolor="#2ca02c")

        # THb
        if "thb" in df_plot.columns:
            ax2 = ax1.twinx()
            thb_smooth = df_plot["thb"].rolling(5, center=True).mean()
            ax2.plot(time_x, thb_smooth, color="#9467bd", linewidth=2, label="THb (g/dL)")
            ax2.set_ylabel("THb (g/dL)", color="#9467bd")
            ax2.tick_params(axis="y", labelcolor="#9467bd")

        plt.title("SmO2 vs THb w czasie")
        plt.tight_layout()

        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue()
    except Exception as e:
        print(f"Error creating SmO2/THb chart: {e}")
        return None
