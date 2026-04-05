"""
Cross-check between independent threshold sources.

Physiological thresholds are estimated from multiple signals (ventilation,
SmO₂, HR, gas exchange). They should roughly coincide in a well-executed
ramp, but systematic deltas or wide spread indicate either signal-quality
issues or genuine inter-signal disagreement that the user must be aware of.

This module is the canonical cross-check surface consumed by the Summary
tab. It does NOT average sources silently — conflicts are surfaced, not
hidden. Agreement bands come from docs/DOMAIN_MODEL.md §6.

Usage
-----
    from modules.domain.threshold_crosscheck import crosscheck_threshold

    verdict = crosscheck_threshold(
        "VT2",
        sources={
            "ventilation": 285.0,
            "smo2": 275.0,
            "hr": 292.0,
        },
    )
    if verdict.is_conflict:
        ui.warning(verdict.message_pl)
    else:
        ui.info(verdict.message_pl)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AgreementLevel(Enum):
    """Qualitative verdict on source agreement.

    Thresholds apply to the spread (max−min) across all non-None sources,
    per docs/DOMAIN_MODEL.md §6.

        spread <= 10 W  → STRONG
        spread <= 20 W  → MODERATE
        spread <= 30 W  → WEAK
        spread >  30 W  → CONFLICT
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONFLICT = "conflict"
    INSUFFICIENT = "insufficient"  # fewer than 2 sources available

    @property
    def is_conflict(self) -> bool:
        return self is AgreementLevel.CONFLICT

    @property
    def is_trustworthy(self) -> bool:
        return self in {AgreementLevel.STRONG, AgreementLevel.MODERATE}


# Band thresholds — keep in sync with docs/DOMAIN_MODEL.md §6.
_STRONG_MAX_W = 10.0
_MODERATE_MAX_W = 20.0
_WEAK_MAX_W = 30.0


@dataclass(frozen=True)
class CrosscheckVerdict:
    """Result of a pairwise / multi-source threshold cross-check."""

    metric: str  # e.g. "VT1", "VT2", "LT1"
    sources: Dict[str, float]  # source_name -> watts (only non-None kept)
    level: AgreementLevel
    spread_watts: Optional[float]  # max − min across sources (None if <2)
    consensus_watts: Optional[float]  # median of sources (None if empty)
    deltas: List[Tuple[str, str, float]] = field(default_factory=list)
    # list of (source_a, source_b, |delta_watts|), all unordered pairs

    @property
    def is_conflict(self) -> bool:
        return self.level.is_conflict

    @property
    def is_trustworthy(self) -> bool:
        return self.level.is_trustworthy

    @property
    def message_pl(self) -> str:
        """Polish interpretation message for UI display."""
        n_src = len(self.sources)
        if self.level is AgreementLevel.INSUFFICIENT:
            return (
                f"{self.metric}: za mało źródeł do cross-checku "
                f"({n_src} dostępne, wymagane ≥ 2)."
            )
        assert self.spread_watts is not None
        spread = f"Δ = {self.spread_watts:.0f} W"
        consensus = (
            f", konsensus ≈ {self.consensus_watts:.0f} W"
            if self.consensus_watts is not None
            else ""
        )
        if self.level is AgreementLevel.STRONG:
            return (
                f"✅ {self.metric}: progi spójne ({n_src} źródeł, {spread}){consensus}."
            )
        if self.level is AgreementLevel.MODERATE:
            return (
                f"ℹ️ {self.metric}: zgodność umiarkowana ({n_src} źródeł, {spread})"
                f"{consensus}."
            )
        if self.level is AgreementLevel.WEAK:
            return (
                f"⚠️ {self.metric}: słaba zgodność ({n_src} źródeł, {spread}) — "
                f"sprawdź jakość sygnałów{consensus}."
            )
        # CONFLICT
        pair_strs = ", ".join(
            f"{a} vs {b}: {d:.0f} W" for a, b, d in self.deltas
        )
        return (
            f"❌ {self.metric}: KONFLIKT źródeł ({spread}). "
            f"Różnice: {pair_strs}. Nie uśredniaj — zbadaj przyczynę."
        )


def crosscheck_threshold(
    metric: str,
    sources: Dict[str, Optional[float]],
) -> CrosscheckVerdict:
    """Cross-check a single threshold (e.g. VT2) across independent sources.

    Args:
        metric: human-readable metric name (used in messages, e.g. "VT2").
        sources: mapping of source_name → threshold in watts. ``None`` values
            are filtered out. At least 2 non-None sources are required.

    Returns:
        A ``CrosscheckVerdict`` with agreement level, spread, consensus
        (median), and pairwise deltas.
    """
    non_null = {
        k: float(v) for k, v in sources.items() if v is not None and not _is_nan(v)
    }
    if len(non_null) < 2:
        return CrosscheckVerdict(
            metric=metric,
            sources=non_null,
            level=AgreementLevel.INSUFFICIENT,
            spread_watts=None,
            consensus_watts=_median(list(non_null.values())) if non_null else None,
        )

    values = list(non_null.values())
    spread = max(values) - min(values)
    consensus = _median(values)

    deltas = _pairwise_deltas(non_null)

    level = _classify_spread(spread)

    return CrosscheckVerdict(
        metric=metric,
        sources=non_null,
        level=level,
        spread_watts=spread,
        consensus_watts=consensus,
        deltas=deltas,
    )


def _classify_spread(spread_watts: float) -> AgreementLevel:
    if spread_watts <= _STRONG_MAX_W:
        return AgreementLevel.STRONG
    if spread_watts <= _MODERATE_MAX_W:
        return AgreementLevel.MODERATE
    if spread_watts <= _WEAK_MAX_W:
        return AgreementLevel.WEAK
    return AgreementLevel.CONFLICT


def _pairwise_deltas(sources: Dict[str, float]) -> List[Tuple[str, str, float]]:
    items = sorted(sources.items())
    out: List[Tuple[str, str, float]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_name, a_val = items[i]
            b_name, b_val = items[j]
            out.append((a_name, b_name, abs(a_val - b_val)))
    # Sort largest delta first — most informative when surfacing conflicts.
    out.sort(key=lambda triple: triple[2], reverse=True)
    return out


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def _is_nan(x: float) -> bool:
    try:
        return x != x
    except TypeError:
        return False
