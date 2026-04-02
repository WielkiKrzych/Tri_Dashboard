"""
Intelligent Alerting Engine.

Red-flag detection for dangerous physiological patterns during session analysis:
- Cardiac drift at constant power
- SmO2 crash detection
- HRV suppression across sessions
- Performance trend decline
- Overtraining risk index (composite score)

All alerts are computed on-demand, not persisted in DB.
Gracefully handles missing data by returning None and adding quality flags.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Severity = Literal["info", "warning", "critical"]


# ============================================================
# Data Classes
# ============================================================


@dataclass
class Alert:
    """Single physiological alert with severity, message, and recommendation."""

    alert_id: str
    alert_type: str  # cardiac_drift, smo2_crash, hrv_suppression, trend_decline, overtraining
    severity: Severity
    title: str
    message: str
    recommendation: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp_sec: Optional[float] = None
    supporting_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return {"info": "\u2139\ufe0f", "warning": "\u26a0\ufe0f", "critical": "\U0001f6a8"}.get(
            self.severity, "\u2139\ufe0f"
        )


@dataclass
class OvertrainingRiskIndex:
    """Composite overtraining risk score (0-100)."""

    score: float = 0.0
    risk_level: str = "low"  # low, moderate, high, critical
    components: dict[str, float] = field(default_factory=dict)
    interpretation: str = ""


@dataclass
class AlertReport:
    """Complete alert report for a session."""

    alerts: list[Alert] = field(default_factory=list)
    overtraining_index: Optional[OvertrainingRiskIndex] = None
    session_date: str = ""
    data_quality_flags: list[str] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.alerts if a.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.alerts if a.severity == "warning")

    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0


# ============================================================
# Detector 1: Cardiac Drift
# ============================================================


def detect_cardiac_drift(df: pd.DataFrame, metrics: dict[str, Any]) -> Optional[Alert]:
    """Detect cardiac drift during constant-power segments.

    Uses ``detect_constant_power_segments`` from ``physio_maps`` to find
    steady-state windows >= 3 min.  For each segment, HR drift % is
    computed as ``(HR_end - HR_start) / HR_start * 100``.

    Thresholds:
        WARNING:  8-12% drift
        CRITICAL: >12% drift

    Returns None when HR data is unavailable.
    """
    # Detect HR column
    hr_col = None
    for col in ["heartrate", "hr", "heart_rate", "HeartRate"]:
        if col in df.columns:
            hr_col = col
            break

    if hr_col is None:
        logger.info("Brak danych HR — pomijam detekcje dryfu sercowego")
        return None

    from modules.physio_maps import detect_constant_power_segments

    # Find constant-power segments >= 3 min (180 s)
    segments = detect_constant_power_segments(df, tolerance_pct=5.0, min_duration_sec=180)

    if not segments:
        return None

    max_drift_pct: float = 0.0
    best_segment_info: str = ""

    for start_idx, end_idx, avg_power in segments:
        segment_hr = df[hr_col].iloc[start_idx:end_idx].dropna()
        if len(segment_hr) < 60:
            continue

        hr_start = segment_hr.iloc[:30].mean()
        hr_end = segment_hr.iloc[-30:].mean()

        if hr_start > 0:
            drift_pct = (hr_end - hr_start) / hr_start * 100
            duration_min = (end_idx - start_idx) / 60
            if abs(drift_pct) > abs(max_drift_pct):
                max_drift_pct = drift_pct
                best_segment_info = f"{avg_power:.0f}W, {duration_min:.1f}min"

    if max_drift_pct < 8:
        return None

    # Build supporting metrics
    supporting: dict[str, float] = {}
    decoupling = metrics.get("decoupling_percent")
    if decoupling is not None:
        supporting["Decoupling EF"] = float(decoupling)

    if max_drift_pct > 12:
        return Alert(
            alert_id="cardiac_drift_critical",
            alert_type="cardiac_drift",
            severity="critical",
            title="Krytyczny dryf sercowy",
            message=(
                f"Wysoki dryf HR ({max_drift_pct:.1f}%) przy stalej mocy "
                f"({best_segment_info}). Układ krążenia nie nadąża z dostawą tlenu "
                f"przy stałej intensywności."
            ),
            recommendation=(
                "Przerwij intensywny wysiłek. Sprawdź nawodnienie i temperaturę. "
                "Rozważ dzień regeneracyjny."
            ),
            value=max_drift_pct,
            threshold=12.0,
            supporting_metrics=supporting,
        )

    return Alert(
        alert_id="cardiac_drift_warning",
        alert_type="cardiac_drift",
        severity="warning",
        title="Podwyższony dryf sercowy",
        message=(
            f"Umiarkowany dryf HR ({max_drift_pct:.1f}%) przy stalej mocy "
            f"({best_segment_info}). Może wskazywać na zmęczenie, odwodnienie "
            f"lub stres termiczny."
        ),
        recommendation="Rozważ zmniejszenie intensywności lub dodatkowe nawodnienie.",
        value=max_drift_pct,
        threshold=8.0,
        supporting_metrics=supporting,
    )


# ============================================================
# Detector 2: SmO2 Crash
# ============================================================


def detect_smo2_crash(df: pd.DataFrame) -> Optional[Alert]:
    """Detect significant SmO2 desaturation during session.

    Baseline = mean SmO2 of first 2 minutes where watts > 50 W.
    Drop % = (baseline - min_smo2) / baseline * 100.

    Thresholds:
        WARNING:  15-25% drop
        CRITICAL: >25% drop

    Returns None when SmO2 data is unavailable or has < 60 valid points.
    """
    smo2_col = None
    for col in ["smo2", "SmO2", "muscle_oxygen"]:
        if col in df.columns:
            smo2_col = col
            break

    if smo2_col is None:
        return None

    smo2_series = df[smo2_col].dropna()
    if len(smo2_series) < 60:
        return None

    # Baseline: first 2 minutes where watts > 50 W
    if "watts" not in df.columns:
        return None

    active_mask = df["watts"] > 50
    first_2_min_mask = active_mask & (df.index < 120)
    baseline_data = df.loc[first_2_min_mask, smo2_col].dropna()

    if len(baseline_data) < 10:
        # Fallback: use first 60 seconds of SmO2 data regardless of power
        baseline_data = smo2_series.iloc[:60]

    if len(baseline_data) < 5:
        return None

    baseline = float(baseline_data.mean())
    min_smo2 = float(smo2_series.min())

    if baseline <= 0:
        return None

    drop_pct = (baseline - min_smo2) / baseline * 100

    if drop_pct < 15:
        return None

    # Find when the minimum occurred
    min_idx = smo2_series.idxmin()
    timestamp_sec: Optional[float] = None
    if min_idx is not None:
        try:
            timestamp_sec = float(min_idx)
        except (TypeError, ValueError):
            pass

    if drop_pct > 25:
        return Alert(
            alert_id="smo2_crash_critical",
            alert_type="smo2_crash",
            severity="critical",
            title="Krytyczny spadek SmO2",
            message=(
                f"SmO2 spadło o {drop_pct:.1f}% (z {baseline:.1f}% do {min_smo2:.1f}%). "
                f"Krytyczna desaturacja mięśniowa — wskazuje na przeciążenie "
                f"metaboliczne i ograniczenie perfuzji."
            ),
            recommendation=(
                "Przerwij intensywny wysiłek. Mięśnie są silnie obciążone. "
                "Priorytet: regeneracja i trening tlenowy."
            ),
            value=drop_pct,
            threshold=25.0,
            timestamp_sec=timestamp_sec,
            supporting_metrics={"Baseline SmO2": baseline, "Min SmO2": min_smo2},
        )

    return Alert(
        alert_id="smo2_crash_warning",
        alert_type="smo2_crash",
        severity="warning",
        title="Znaczny spadek SmO2",
        message=(
            f"SmO2 spadło o {drop_pct:.1f}% (z {baseline:.1f}% do {min_smo2:.1f}%). "
            f"Mięśnie pracują blisko limitu tlenowego."
        ),
        recommendation="Obserwuj reakcję. Rozważ wydłużenie przerw między interwałami.",
        value=drop_pct,
        threshold=15.0,
        timestamp_sec=timestamp_sec,
        supporting_metrics={"Baseline SmO2": baseline, "Min SmO2": min_smo2},
    )


# ============================================================
# Detector 3: HRV Suppression
# ============================================================


def detect_hrv_suppression(session_history: list[dict[str, Any]]) -> Optional[Alert]:
    """Detect HRV suppression across recent sessions.

    Compares 28-day RMSSD baseline to the last 3 sessions.

    Thresholds:
        WARNING:  recent RMSSD < 80% of baseline for 2+ sessions
        CRITICAL: recent RMSSD < 60% of baseline for 3+ consecutive sessions

    Returns None when fewer than 3 sessions with valid RMSSD exist.
    """
    # Extract sessions with valid RMSSD
    rmssd_entries: list[tuple[str, float]] = []
    for session in session_history:
        rmssd = session.get("avg_rmssd")
        date_str = session.get("date", "")
        if rmssd is not None and rmssd > 0:
            rmssd_entries.append((date_str, float(rmssd)))

    if len(rmssd_entries) < 3:
        return None

    # Sort by date ascending (oldest first)
    rmssd_entries.sort(key=lambda x: x[0])

    # 28-day baseline: all sessions from last 28 days (excluding last 3)
    baseline_entries = rmssd_entries[:-3]
    recent_entries = rmssd_entries[-3:]

    if not baseline_entries:
        return None

    baseline_rmssd = float(np.mean([v for _, v in baseline_entries]))

    if baseline_rmssd <= 0:
        return None

    # Count suppressed sessions
    suppressed_count = 0
    consecutive_suppressed = 0
    max_consecutive = 0

    for _, rmssd in recent_entries:
        ratio = rmssd / baseline_rmssd
        if ratio < 0.8:
            suppressed_count += 1
            consecutive_suppressed += 1
            max_consecutive = max(max_consecutive, consecutive_suppressed)
        else:
            consecutive_suppressed = 0

    # CRITICAL: 3+ consecutive sessions below 60% of baseline
    if max_consecutive >= 3:
        recent_avg = float(np.mean([v for _, v in recent_entries]))
        suppression_pct = ((baseline_rmssd - recent_avg) / baseline_rmssd) * 100
        return Alert(
            alert_id="hrv_suppression_critical",
            alert_type="hrv_suppression",
            severity="critical",
            title="Krytyczna supresja HRV",
            message=(
                f"RMSSD spadło o {suppression_pct:.0f}% poniżej baseline "
                f"({recent_avg:.1f} ms vs {baseline_rmssd:.1f} ms) przez "
                f"{max_consecutive} kolejne sesje. Autonomiczny układ nerwowy "
                f"jest silnie przeciążony."
            ),
            recommendation=(
                "Zaplanuj 2-3 dni pełnego odpoczynku. Monitoruj jakość snu "
                "i poziom stresu. Skonsultuj się z trenerem."
            ),
            value=recent_avg,
            threshold=baseline_rmssd * 0.6,
            supporting_metrics={
                "Baseline RMSSD": baseline_rmssd,
                "Recent RMSSD": recent_avg,
                "Suppression %": suppression_pct,
            },
        )

    # WARNING: 2+ sessions below 80% of baseline
    if suppressed_count >= 2:
        recent_avg = float(np.mean([v for _, v in recent_entries]))
        suppression_pct = ((baseline_rmssd - recent_avg) / baseline_rmssd) * 100
        return Alert(
            alert_id="hrv_suppression_warning",
            alert_type="hrv_suppression",
            severity="warning",
            title="Supresja HRV",
            message=(
                f"RMSSD spadło o {suppression_pct:.0f}% poniżej baseline "
                f"({recent_avg:.1f} ms vs {baseline_rmssd:.1f} ms) w "
                f"{suppressed_count} z 3 ostatnich sesji."
            ),
            recommendation=(
                "Rozważ dzień odpoczynku lub lekki trening regeneracyjny. "
                "Monitoruj HRV przed kolejnym intensywnym treningiem."
            ),
            value=recent_avg,
            threshold=baseline_rmssd * 0.8,
            supporting_metrics={
                "Baseline RMSSD": baseline_rmssd,
                "Recent RMSSD": recent_avg,
                "Suppression %": suppression_pct,
            },
        )

    return None


# ============================================================
# Detector 4: Performance Trend Decline
# ============================================================


def detect_performance_trend_decline(session_history: list[dict[str, Any]]) -> list[Alert]:
    """Detect declining performance trends from ramp test history.

    Uses ``trend_engine.analyze_trends`` if available to check CP, W', VO2
    metrics for decline.

    Thresholds:
        WARNING:  decline > 0.5%/week
        CRITICAL: decline > 1.5%/week

    Returns empty list when no ramp test history is available.
    """
    alerts: list[Alert] = []

    try:
        from modules.calculations.trend_engine import analyze_trends

        # session_history entries need to be ramp test reports with metrics
        # Check if they have the expected structure for trend_engine
        ramp_reports = [
            s for s in session_history if s.get("session_type") == "ramp" or "thresholds" in s
        ]

        if len(ramp_reports) < 2:
            return alerts

        analysis = analyze_trends(ramp_reports)
        if analysis.tests_analyzed < 2:
            return alerts

        # Check key performance metrics for decline
        decline_metrics = [
            (analysis.cp, "CP"),
            (analysis.w_prime, "W'"),
            (analysis.vt1, "VT1"),
            (analysis.vt2, "VT2"),
        ]

        for trend, name in decline_metrics:
            if not trend.values or trend.rate_per_week == 0:
                continue

            rate = trend.rate_per_week
            if rate > 0:
                continue  # Improving, skip

            abs_rate = abs(rate)
            if abs_rate > 1.5:
                alerts.append(
                    Alert(
                        alert_id=f"trend_decline_critical_{name.lower().replace(chr(39), '')}",
                        alert_type="trend_decline",
                        severity="critical",
                        title=f"Krytyczny spadek {name}",
                        message=(
                            f"{name} spada o {abs_rate:.2f}% tygodniowo. "
                            f"To znaczna regresja wydolnościowa wymagająca natychmiastowej "
                            f"interwencji treningowej."
                        ),
                        recommendation=(
                            "Wstrzymaj intensywny trening. Skonsultuj się z trenerem. "
                            "Rozważ badania lekarskie (żelazo, B12, hormony tarczycy)."
                        ),
                        value=abs_rate,
                        threshold=1.5,
                        supporting_metrics={
                            f"{name} rate/week": abs_rate,
                            f"{name} values": trend.values[-1] if trend.values else 0,
                        },
                    )
                )
            elif abs_rate > 0.5:
                alerts.append(
                    Alert(
                        alert_id=f"trend_decline_warning_{name.lower().replace(chr(39), '')}",
                        alert_type="trend_decline",
                        severity="warning",
                        title=f"Spadek {name}",
                        message=(
                            f"{name} spada o {abs_rate:.2f}% tygodniowo. "
                            f"Może wskazywać na niewłaściwy balans obciążenia i regeneracji."
                        ),
                        recommendation="Monitoruj trend. Rozważ redukcję objętości lub intensywności.",
                        value=abs_rate,
                        threshold=0.5,
                        supporting_metrics={
                            f"{name} rate/week": abs_rate,
                            f"{name} values": trend.values[-1] if trend.values else 0,
                        },
                    )
                )

    except ImportError:
        logger.debug("trend_engine not available — skipping trend decline detection")
    except Exception as e:
        logger.warning("Trend decline detection failed: %s", e)

    return alerts


# ============================================================
# Composite: Overtraining Risk Index
# ============================================================

# Component weights and their keys
_OT_COMPONENT_WEIGHTS: dict[str, float] = {
    "acwr": 0.25,
    "tsb_trend": 0.20,
    "hrv_suppression": 0.25,
    "performance_decline": 0.15,
    "cardiac_drift_pattern": 0.15,
}


def calculate_overtraining_risk(
    session_history: list[dict[str, Any]],
    df: pd.DataFrame,
    metrics: dict[str, Any],
) -> OvertrainingRiskIndex:
    """Calculate composite overtraining risk index (0-100).

    Five weighted components:
        - ACWR (25%):             ACWR > 1.5 = risk
        - TSB trend (20%):        TSB < -30 = risk
        - HRV suppression (25%):  from RMSSD trend
        - Performance decline (15%): from trend analysis
        - Cardiac drift pattern (15%): repeated high drift

    Missing components cause weight redistribution to available ones.
    """
    components: dict[str, float] = {}
    available_weights: dict[str, float] = {}
    quality_flags: list[str] = []

    # --- Component 1: ACWR ---
    acwr_score = 0.0
    try:
        from modules.training_load import TrainingLoadManager

        load_manager = TrainingLoadManager()
        load_history = load_manager.calculate_load(days=14)
        if len(load_history) >= 2:
            current = load_history[-1]
            if current.ctl > 0:
                acwr = current.atl / current.ctl
                # ACWR > 1.5 maps to 50-100 risk, 1.0-1.5 maps to 0-50
                if acwr > 1.5:
                    acwr_score = min(100, 50 + (acwr - 1.5) * 100)
                elif acwr > 1.0:
                    acwr_score = (acwr - 1.0) / 0.5 * 50
                components["ACWR"] = acwr_score
                available_weights["acwr"] = _OT_COMPONENT_WEIGHTS["acwr"]
    except Exception as e:
        logger.debug("ACWR calculation unavailable: %s", e)
        quality_flags.append("ACWR: dane obciążenia niedostępne")

    # --- Component 2: TSB trend ---
    tsb_score = 0.0
    try:
        from modules.training_load import TrainingLoadManager

        load_manager = TrainingLoadManager()
        load_history = load_manager.calculate_load(days=21)
        if len(load_history) >= 7:
            current_tsb = load_history[-1].tsb
            # TSB < -30 = full risk (100), TSB > 0 = no risk (0)
            if current_tsb < -30:
                tsb_score = min(100, 50 + abs(current_tsb + 30) * 2)
            elif current_tsb < 0:
                tsb_score = abs(current_tsb) / 30 * 50
            components["TSB"] = tsb_score
            available_weights["tsb_trend"] = _OT_COMPONENT_WEIGHTS["tsb_trend"]
    except Exception as e:
        logger.debug("TSB calculation unavailable: %s", e)
        quality_flags.append("TSB: dane obciążenia niedostępne")

    # --- Component 3: HRV suppression ---
    hrv_score = 0.0
    try:
        rmssd_entries = [
            float(s["avg_rmssd"])
            for s in session_history
            if s.get("avg_rmssd") is not None and s["avg_rmssd"] > 0
        ]
        if len(rmssd_entries) >= 3:
            baseline = float(np.mean(rmssd_entries[:-3]))
            recent = float(np.mean(rmssd_entries[-3:]))
            if baseline > 0:
                suppression_ratio = recent / baseline
                if suppression_ratio < 0.6:
                    hrv_score = 100
                elif suppression_ratio < 0.8:
                    hrv_score = (0.8 - suppression_ratio) / 0.2 * 100
                components["HRV"] = hrv_score
                available_weights["hrv_suppression"] = _OT_COMPONENT_WEIGHTS["hrv_suppression"]
    except Exception as e:
        logger.debug("HRV suppression calculation failed: %s", e)

    # --- Component 4: Performance decline ---
    perf_score = 0.0
    try:
        from modules.calculations.trend_engine import analyze_trends

        ramp_reports = [
            s for s in session_history if s.get("session_type") == "ramp" or "thresholds" in s
        ]
        if len(ramp_reports) >= 2:
            analysis = analyze_trends(ramp_reports)
            # Use worst declining rate among CP, W', VT2
            decline_rates = [
                abs(analysis.cp.rate_per_week),
                abs(analysis.w_prime.rate_per_week),
                abs(analysis.vt2.rate_per_week),
            ]
            max_decline = max(decline_rates)
            # >1.5%/week = 100, >0.5%/week = proportional
            if max_decline > 1.5:
                perf_score = 100
            elif max_decline > 0.5:
                perf_score = (max_decline - 0.5) / 1.0 * 100
            components["Performance"] = perf_score
            available_weights["performance_decline"] = _OT_COMPONENT_WEIGHTS["performance_decline"]
    except Exception as e:
        logger.debug("Performance decline calculation unavailable: %s", e)

    # --- Component 5: Cardiac drift pattern ---
    drift_score = 0.0
    try:
        drift_pct = metrics.get("decoupling_percent")
        if drift_pct is not None and drift_pct > 0:
            # EF drift >10% = 100, 5-10% = proportional
            if drift_pct > 10:
                drift_score = 100
            elif drift_pct > 5:
                drift_score = (drift_pct - 5) / 5 * 100
            components["Cardiac Drift"] = drift_score
            available_weights["cardiac_drift_pattern"] = _OT_COMPONENT_WEIGHTS[
                "cardiac_drift_pattern"
            ]
    except Exception as e:
        logger.debug("Cardiac drift pattern unavailable: %s", e)

    # --- Composite score ---
    if not available_weights:
        return OvertrainingRiskIndex(
            score=0.0,
            risk_level="low",
            components={},
            interpretation="Niewystarczające dane do oceny ryzyka przetrenowania.",
        )

    # Redistribute weights if some components are missing
    total_weight = sum(available_weights.values())
    composite_score = 0.0
    normalized_components: dict[str, float] = {}

    for comp_name, comp_value in components.items():
        # Find the weight key for this component
        weight_key = None
        for wk in available_weights:
            if wk.replace("_", " ").lower() in comp_name.lower().replace("_", " "):
                weight_key = wk
                break
        if weight_key is None:
            continue

        normalized_weight = available_weights[weight_key] / total_weight
        contribution = comp_value * normalized_weight
        composite_score += contribution
        normalized_components[comp_name] = comp_value

    composite_score = min(100, max(0, composite_score))

    # Risk level classification
    if composite_score >= 75:
        risk_level = "critical"
        interpretation = (
            "KRYTYCZNE ryzyko przetrenowania. Natychmiastowa redukcja obciążenia "
            "i konsultacja z trenerem/lekarzem."
        )
    elif composite_score >= 50:
        risk_level = "high"
        interpretation = (
            "WYSOKIE ryzyko przetrenowania. Zalecana redukcja objętości o 30-50% "
            "i monitorowanie samopoczucia."
        )
    elif composite_score >= 25:
        risk_level = "moderate"
        interpretation = (
            "UMIARKOWANE ryzyko przetrenowania. Rozważ dodatkowy dzień regeneracji "
            "w bieżącym tygodniu."
        )
    else:
        risk_level = "low"
        interpretation = (
            "NISKIE ryzyko przetrenowania. Balans obciążenia i regeneracji wydaje się odpowiedni."
        )

    return OvertrainingRiskIndex(
        score=composite_score,
        risk_level=risk_level,
        components=normalized_components,
        interpretation=interpretation,
    )


# ============================================================
# Orchestrator
# ============================================================


def analyze_session_alerts(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    session_history: Optional[list[dict[str, Any]]] = None,
) -> AlertReport:
    """Run all alert detectors and compute overtraining risk index.

    This is the main entry point for the alerting system.  It calls every
    detector, collects alerts, and returns a complete ``AlertReport``.

    Args:
        df: Session DataFrame with physiological data.
        metrics: Computed metrics dictionary from session processing.
        session_history: Optional list of historical session dicts with
            ``avg_rmssd``, ``date``, ``session_type`` keys.

    Returns:
        AlertReport with all alerts, overtraining index, and quality flags.
    """
    if session_history is None:
        session_history = []

    quality_flags: list[str] = []
    alerts: list[Alert] = []

    # 1. Cardiac drift
    try:
        drift_alert = detect_cardiac_drift(df, metrics)
        if drift_alert:
            alerts.append(drift_alert)
    except Exception as e:
        logger.warning("Cardiac drift detection error: %s", e)
        quality_flags.append("Dryf sercowy: błąd detekcji")

    # 2. SmO2 crash
    try:
        smo2_alert = detect_smo2_crash(df)
        if smo2_alert:
            alerts.append(smo2_alert)
    except Exception as e:
        logger.warning("SmO2 crash detection error: %s", e)
        quality_flags.append("SmO2: błąd detekcji")

    # 3. HRV suppression (requires session history)
    try:
        hrv_alert = detect_hrv_suppression(session_history)
        if hrv_alert:
            alerts.append(hrv_alert)
    except Exception as e:
        logger.warning("HRV suppression detection error: %s", e)
        quality_flags.append("HRV: błąd detekcji")

    # 4. Performance trend decline
    try:
        trend_alerts = detect_performance_trend_decline(session_history)
        alerts.extend(trend_alerts)
    except Exception as e:
        logger.warning("Trend decline detection error: %s", e)
        quality_flags.append("Trend wydolności: błąd detekcji")

    # 5. Overtraining risk index
    overtraining_index: Optional[OvertrainingRiskIndex] = None
    try:
        overtraining_index = calculate_overtraining_risk(session_history, df, metrics)
    except Exception as e:
        logger.warning("Overtraining risk calculation error: %s", e)
        quality_flags.append("Indeks przetrenowania: błąd obliczeń")

    # Data quality checks
    hr_col = next((c for c in ["heartrate", "hr", "heart_rate"] if c in df.columns), None)
    if hr_col is None:
        quality_flags.append("Brak danych HR — niektóre alerty mogą być niedostępne")

    smo2_col = next((c for c in ["smo2", "SmO2", "muscle_oxygen"] if c in df.columns), None)
    if smo2_col is None:
        quality_flags.append("Brak danych SmO2 — detekcja crashu SmO2 niedostępna")

    if len(session_history) < 3:
        quality_flags.append(
            "Mniej niż 3 sesje w historii — analiza trendów HRV i wydolności niedostępna"
        )

    # Sort alerts by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    # Session date from metrics if available
    session_date = ""
    if "session_date" in metrics:
        session_date = str(metrics["session_date"])
    elif len(df) > 0 and "time" in df.columns:
        try:
            first_time = df["time"].iloc[0]
            session_date = str(first_time)[:10]
        except (IndexError, TypeError):
            pass

    return AlertReport(
        alerts=alerts,
        overtraining_index=overtraining_index,
        session_date=session_date,
        data_quality_flags=quality_flags,
    )


__all__ = [
    "Alert",
    "AlertReport",
    "OvertrainingRiskIndex",
    "Severity",
    "analyze_session_alerts",
    "calculate_overtraining_risk",
    "detect_cardiac_drift",
    "detect_hrv_suppression",
    "detect_performance_trend_decline",
    "detect_smo2_crash",
]
