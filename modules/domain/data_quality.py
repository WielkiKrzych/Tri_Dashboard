"""
Data quality / confidence bands for physiological analysis results.

This is the canonical mapping from a numeric confidence score (0.0–1.0)
to a qualitative band that UI tabs render consistently. The contract is
defined in docs/DOMAIN_MODEL.md §4.

Usage
-----
    from modules.domain.data_quality import DataQuality, quality_band

    band = quality_band(vt1_confidence)          # DataQuality enum
    label = band.label_pl                        # "Wysoka" / "Warunkowa" / ...
    color = band.color                           # "#2ecc71" / "#f39c12" / ...

    if band.is_trustworthy:
        show_value(vt1_watts)
    else:
        show_caveat(vt1_watts, band.short_caveat_pl)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class _BandMeta:
    """Presentation metadata for a data-quality band."""

    label_pl: str
    label_en: str
    color: str  # CSS-safe hex
    short_caveat_pl: str
    is_trustworthy: bool


class DataQuality(Enum):
    """Qualitative band for a detected threshold / metric.

    Thresholds (see DOMAIN_MODEL.md §4):
        confidence >= 0.80        → HIGH
        0.60 <= confidence < 0.80 → CONDITIONAL
        0.40 <= confidence < 0.60 → LOW
        confidence < 0.40         → INVALID
    """

    HIGH = _BandMeta(
        label_pl="Wysoka jakość",
        label_en="High",
        color="#2ecc71",
        short_caveat_pl="Wynik wiarygodny.",
        is_trustworthy=True,
    )
    CONDITIONAL = _BandMeta(
        label_pl="Warunkowa",
        label_en="Conditional",
        color="#f39c12",
        short_caveat_pl="Wynik do użycia z zastrzeżeniem (krótki protokół / szum sygnału).",
        is_trustworthy=True,
    )
    LOW = _BandMeta(
        label_pl="Niska",
        label_en="Low",
        color="#e67e22",
        short_caveat_pl="Wynik orientacyjny — nie podejmuj decyzji treningowych wyłącznie na jego podstawie.",
        is_trustworthy=False,
    )
    INVALID = _BandMeta(
        label_pl="Niewiarygodna",
        label_en="Invalid",
        color="#e74c3c",
        short_caveat_pl="Wynik niewiarygodny — sprawdź jakość danych.",
        is_trustworthy=False,
    )

    # ---- Presentation passthroughs (so callers don't touch ._value_) ----

    @property
    def label_pl(self) -> str:
        return self.value.label_pl

    @property
    def label_en(self) -> str:
        return self.value.label_en

    @property
    def color(self) -> str:
        return self.value.color

    @property
    def short_caveat_pl(self) -> str:
        return self.value.short_caveat_pl

    @property
    def is_trustworthy(self) -> bool:
        return self.value.is_trustworthy


# Band thresholds — keep these in sync with docs/DOMAIN_MODEL.md §4.
_HIGH_MIN = 0.80
_CONDITIONAL_MIN = 0.60
_LOW_MIN = 0.40


def quality_band(confidence: Optional[float]) -> DataQuality:
    """Map a confidence score (0.0–1.0) to a DataQuality band.

    A ``None`` score is treated as INVALID (no measurement available).
    Values outside [0, 1] are clipped before comparison.
    """
    if confidence is None:
        return DataQuality.INVALID
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return DataQuality.INVALID
    if c != c:  # NaN
        return DataQuality.INVALID
    c = max(0.0, min(1.0, c))
    if c >= _HIGH_MIN:
        return DataQuality.HIGH
    if c >= _CONDITIONAL_MIN:
        return DataQuality.CONDITIONAL
    if c >= _LOW_MIN:
        return DataQuality.LOW
    return DataQuality.INVALID


def format_band_badge_html(
    band: DataQuality, prefix: Optional[str] = None
) -> str:
    """Render a small inline HTML badge for a quality band.

    Suitable for ``st.markdown(..., unsafe_allow_html=True)``. The caller is
    responsible for the surrounding layout. HTML is generated from a fixed
    template with no user-controlled strings other than ``prefix`` (which is
    HTML-escaped).
    """
    import html

    safe_prefix = html.escape(prefix) + " " if prefix else ""
    return (
        f'<span style="background:{band.color}22;'
        f'border:1px solid {band.color};'
        f'color:{band.color};'
        f'padding:2px 8px;border-radius:4px;font-size:0.85em;font-weight:600;">'
        f"{safe_prefix}{band.label_pl}</span>"
    )
