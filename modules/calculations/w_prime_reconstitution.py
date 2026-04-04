"""
W' Reconstitution Map Module.

Analyzes W' (anaerobic work capacity) depletion and reconstitution patterns
during a training session. Provides a visual "map" showing when W' was
depleted, how fast it recovered, and how much remained at key moments.

Built on the Skiba (2015) mono-exponential and Caen (2021) bi-exponential
models already implemented in modules.calculations.w_prime.

References:
    - Skiba PF et al. (2015). "Validation of a novel intermittent W' model."
    - Caen et al. (2021). "Bi-exponential W' reconstitution." EJAP.
    - Chorley (2022). "Practical application of the W' balance model." IJSPP.
    - Welburn et al. (2025). "W' reconstitution modelling." EJAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .common import ensure_pandas
from .w_prime import calculate_w_prime_fast, calculate_w_prime_biexp


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReconstitutionEvent:
    """A single W' depletion-reconstitution cycle."""
    depletion_start_s: float
    depletion_end_s: float
    min_w_prime_j: float
    min_w_prime_pct: float
    recovery_duration_s: float
    recovery_rate_j_per_s: float
    recovery_pct: float
    intensity_during_depletion: float  # avg watts
    intensity_during_recovery: float   # avg watts


@dataclass(frozen=True)
class ReconstitutionSummary:
    """Session-level W' reconstitution summary."""
    total_depletions: int
    deepest_depletion_j: float
    deepest_depletion_pct: float
    fastest_recovery_rate_j_s: float
    avg_recovery_rate_j_s: float
    time_below_20pct_s: int
    time_below_20pct_pct: float
    events: list[ReconstitutionEvent]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_w_prime_reconstitution_map(
    df: pd.DataFrame,
    cp: float,
    w_prime_cap: float,
    model: str = "biexp",
    sport: int = 0,
) -> tuple[pd.DataFrame, ReconstitutionSummary]:
    """
    Compute W' balance and extract reconstitution events.

    Args:
        df: DataFrame with 'watts' and 'time' (or index as seconds)
        cp: Critical Power [W]
        w_prime_cap: W' capacity [J]
        model: "skiba" (mono-exponential) or "biexp" (bi-exponential)
        sport: 0=cycling, 1=running, 2=swimming (for biexp model)

    Returns:
        (df_with_wbal, summary)
        df_with_wbal has added columns: w_prime_balance, w_prime_pct,
        is_depleted, is_recovering
    """
    df = ensure_pandas(df)
    if df is None or "watts" not in df.columns:
        return df, _empty_summary()

    watts = df["watts"].fillna(0).values.astype(np.float64)

    # Time column
    if "time" in df.columns:
        time_arr = df["time"].values.astype(np.float64)
    else:
        time_arr = np.arange(len(watts), dtype=np.float64)

    # Ensure time is monotonically increasing
    if len(time_arr) > 1:
        time_arr = np.maximum.accumulate(time_arr)

    # Calculate W' balance
    if model == "biexp":
        w_bal = calculate_w_prime_biexp(watts, time_arr, cp, w_prime_cap, sport)
    else:
        w_bal = calculate_w_prime_fast(watts, time_arr, cp, w_prime_cap)

    # Add to DataFrame
    result = df.copy()
    result["w_prime_balance"] = w_bal
    result["w_prime_pct"] = (w_bal / w_prime_cap * 100) if w_prime_cap > 0 else 0
    result["is_depleted"] = w_bal < (w_prime_cap * 0.20)  # Below 20%
    result["is_recovering"] = np.concatenate([[False], w_bal[1:] > w_bal[:-1]])

    # Extract reconstitution events
    events = _extract_reconstitution_events(result, cp, w_prime_cap)
    summary = _build_summary(events, result, w_prime_cap)

    return result, summary


def _extract_reconstitution_events(
    df: pd.DataFrame,
    cp: float,
    w_prime_cap: float,
) -> list[ReconstitutionEvent]:
    """
    Identify W' depletion-reconstitution cycles.

    A cycle is defined as:
    1. Depletion phase: W' drops below 50% of capacity
    2. Recovery phase: W' rises back above 50%
    """
    events = []
    threshold_pct = 0.50
    threshold_j = w_prime_cap * threshold_pct

    wbal = df["w_prime_balance"].values
    watts = df["watts"].fillna(0).values
    time_arr = df.get("time", pd.Series(range(len(df)))).values

    in_depletion = False
    dep_start = 0
    dep_min_wbal = w_prime_cap
    dep_min_idx = 0

    for i in range(len(wbal)):
        if not in_depletion and wbal[i] < threshold_j:
            # Start of depletion
            in_depletion = True
            dep_start = i
            dep_min_wbal = wbal[i]
            dep_min_idx = i
        elif in_depletion:
            if wbal[i] < dep_min_wbal:
                dep_min_wbal = wbal[i]
                dep_min_idx = i

            if wbal[i] >= threshold_j:
                # End of depletion cycle
                dep_end = dep_min_idx
                rec_end = i

                # Average intensity during depletion
                dep_watts = watts[dep_start:dep_end + 1]
                avg_dep_intensity = float(np.mean(dep_watts)) if len(dep_watts) > 0 else 0

                # Average intensity during recovery
                rec_watts = watts[dep_min_idx:rec_end + 1]
                avg_rec_intensity = float(np.mean(rec_watts)) if len(rec_watts) > 0 else 0

                # Recovery metrics
                dep_duration = float(time_arr[dep_end] - time_arr[dep_start]) if dep_end > dep_start else 1.0
                rec_duration = float(time_arr[rec_end] - time_arr[dep_min_idx]) if rec_end > dep_min_idx else 1.0
                recovered_j = wbal[rec_end] - dep_min_wbal
                recovery_rate = recovered_j / rec_duration if rec_duration > 0 else 0
                recovery_pct = ((wbal[rec_end] - dep_min_wbal) / (w_prime_cap - dep_min_wbal) * 100) if dep_min_wbal < w_prime_cap else 100

                events.append(ReconstitutionEvent(
                    depletion_start_s=float(time_arr[dep_start]),
                    depletion_end_s=float(time_arr[dep_end]),
                    min_w_prime_j=round(dep_min_wbal, 0),
                    min_w_prime_pct=round(dep_min_wbal / w_prime_cap * 100, 1) if w_prime_cap > 0 else 0,
                    recovery_duration_s=round(rec_duration, 1),
                    recovery_rate_j_per_s=round(recovery_rate, 2),
                    recovery_pct=round(min(recovery_pct, 100), 1),
                    intensity_during_depletion=round(avg_dep_intensity, 0),
                    intensity_during_recovery=round(avg_rec_intensity, 0),
                ))
                in_depletion = False

    return events


def _build_summary(
    events: list[ReconstitutionEvent],
    df: pd.DataFrame,
    w_prime_cap: float,
) -> ReconstitutionSummary:
    """Build session-level summary from events."""
    if not events:
        return _empty_summary()

    deepest = min(events, key=lambda e: e.min_w_prime_j)
    recovery_rates = [e.recovery_rate_j_per_s for e in events if e.recovery_rate_j_per_s > 0]
    avg_recovery = float(np.mean(recovery_rates)) if recovery_rates else 0.0
    fastest = max(recovery_rates) if recovery_rates else 0.0

    # Time below 20%
    if "is_depleted" in df.columns:
        time_below_20 = int(df["is_depleted"].sum())
        time_below_20_pct = round(time_below_20 / len(df) * 100, 1) if len(df) > 0 else 0
    else:
        time_below_20 = 0
        time_below_20_pct = 0.0

    return ReconstitutionSummary(
        total_depletions=len(events),
        deepest_depletion_j=deepest.min_w_prime_j,
        deepest_depletion_pct=deepest.min_w_prime_pct,
        fastest_recovery_rate_j_s=round(fastest, 2),
        avg_recovery_rate_j_s=round(avg_recovery, 2),
        time_below_20pct_s=time_below_20,
        time_below_20pct_pct=time_below_20_pct,
        events=events,
    )


def _empty_summary() -> ReconstitutionSummary:
    return ReconstitutionSummary(
        total_depletions=0,
        deepest_depletion_j=0,
        deepest_depletion_pct=0,
        fastest_recovery_rate_j_s=0,
        avg_recovery_rate_j_s=0,
        time_below_20pct_s=0,
        time_below_20pct_pct=0,
        events=[],
    )


# ---------------------------------------------------------------------------
# Reconstitution table (for display)
# ---------------------------------------------------------------------------

def build_reconstitution_table(
    events: list[ReconstitutionEvent],
) -> pd.DataFrame:
    """Format reconstitution events as a display-ready DataFrame."""
    if not events:
        return pd.DataFrame()

    rows = []
    for i, ev in enumerate(events, 1):
        rows.append({
            "Cykl": i,
            "Start wyczerpania [s]": f"{ev.depletion_start_s:.0f}",
            "Koniec wyczerpania [s]": f"{ev.depletion_end_s:.0f}",
            "Min W' [J]": f"{ev.min_w_prime_j:.0f}",
            "Min W' [%]": f"{ev.min_w_prime_pct:.1f}",
            "Czas regeneracji [s]": f"{ev.recovery_duration_s:.0f}",
            "Tempo regeneracji [J/s]": f"{ev.recovery_rate_j_per_s:.2f}",
            "Odzyskano [%]": f"{ev.recovery_pct:.1f}",
            "Śr. moc (wyczerpanie) [W]": f"{ev.intensity_during_depletion:.0f}",
            "Śr. moc (regeneracja) [W]": f"{ev.intensity_during_recovery:.0f}",
        })

    return pd.DataFrame(rows)


def get_reconstitution_interpretation(summary: ReconstitutionSummary) -> str:
    """
    Generate Polish interpretation of W' reconstitution profile.
    """
    if summary.total_depletions == 0:
        return "🟢 Brak znaczących wyczerpań W'. Trening w strefie tlenowej."

    parts = []

    # Depth assessment
    if summary.deepest_depletion_pct < 10:
        parts.append(
            f"🔴 Głębokie wyczerpanie W' do {summary.deepest_depletion_pct:.1f}% "
            f"({summary.deepest_depletion_j:.0f} J). Ryzyko nagłego spadku mocy."
        )
    elif summary.deepest_depletion_pct < 30:
        parts.append(
            f"🟠 Umiarkowane wyczerpanie W' do {summary.deepest_depletion_pct:.1f}%. "
            "Zarządzaj intensywnością, aby uniknąć całkowitego wyczerpania."
        )
    else:
        parts.append(
            f"🟢 Płytkie wyczerpanie W' do {summary.deepest_depletion_pct:.1f}%. "
            "Dobra rezerwa beztlenowa utrzymywana przez cały trening."
        )

    # Recovery rate assessment
    if summary.avg_recovery_rate_j_s > 30:
        parts.append(
            f"⚡ Szybka regeneracja W' ({summary.avg_recovery_rate_j_s:.1f} J/s). "
            "Wskazuje na wysoką zdolność oksydacyjną i dobrą kondycję mitochondrialną."
        )
    elif summary.avg_recovery_rate_j_s > 15:
        parts.append(
            f"👍 Umiarkowana regeneracja W' ({summary.avg_recovery_rate_j_s:.1f} J/s). "
            "Typowa dla wytrenowanych amatorów."
        )
    else:
        parts.append(
            f"🐢 Wolna regeneracja W' ({summary.avg_recovery_rate_j_s:.1f} J/s). "
            "Warto pracować nad podstawą tlenową i zdolnością do regeneracji między wysiłkami."
        )

    # Time below 20%
    if summary.time_below_20pct_pct > 15:
        parts.append(
            f"⚠️ {summary.time_below_20pct_pct:.1f}% czasu poniżej 20% W'. "
            "Znaczna część treningu w strefie krytycznej — ryzyko przetrenowania."
        )

    return "\n\n".join(parts)
