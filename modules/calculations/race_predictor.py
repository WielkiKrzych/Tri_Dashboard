"""
Race-Day Power Predictor Module.

Predicts sustainable power output for race distances/durations based on
CP/W' model, power-duration curve, and environmental adjustments.

References:
    - Skiba et al. (2012, 2015). W' balance model.
    - Chorley (2022). Practical application of W' in racing.
    - Jones et al. (2021). Physiological basis of endurance performance.
    - Petridou & Nikolaidis (2023). Pacing strategies in endurance events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .common import ensure_pandas


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RacePrediction:
    """Single race prediction result."""
    distance_km: float
    discipline: str          # "cycling" | "triathlon" | "running"
    predicted_duration_min: float
    avg_power_w: float
    power_w_per_kg: float
    tss: float
    np_w: float
    if_value: float
    confidence: float        # 0-1
    assumptions: str


# ---------------------------------------------------------------------------
# Core prediction engine
# ---------------------------------------------------------------------------

def predict_race_power(
    cp: float,
    w_prime: float,
    weight_kg: float,
    duration_min: float,
    discipline: str = "cycling",
    course_type: str = "flat",
    wind_speed_kmh: float = 0.0,
    temperature_c: float = 20.0,
    elevation_gain_m: float = 0.0,
    race_format: str = "time_trial",
) -> RacePrediction:
    """
    Predict sustainable average power for a given race duration.

    Uses a 3-parameter critical power model with environmental corrections:
        P(d) = CP + W' / t   (hyperbolic model)

    For durations > critical duration, power decays below CP due to
    substrate depletion and neuromuscular fatigue (per Jones 2021).

    Args:
        cp: Critical Power [W]
        w_prime: W' anaerobic capacity [J]
        weight_kg: Rider weight [kg]
        duration_min: Target race duration [min]
        discipline: Sport type
        course_type: "flat", "rolling", "hilly", "mountain"
        wind_speed_kmh: Average headwind (positive = headwind)
        temperature_c: Ambient temperature
        elevation_gain_m: Total elevation gain
        race_format: "time_trial", "mass_start", "circuit"

    Returns:
        RacePrediction with power, TSS, IF, and confidence.
    """
    duration_sec = duration_min * 60.0

    # --- Base power from CP/W' model ---
    # Hyperbolic: P = CP + W'/t
    base_power = cp + w_prime / duration_sec

    # --- Long-duration decay (substrate depletion) ---
    # For events > 60 min, power decays ~2-3% per hour below CP
    # (Jones et al. 2021, Scott et al. 2023)
    if duration_min > 60:
        decay_hours = (duration_min - 60) / 60.0
        decay_rate = 0.025  # 2.5% per hour beyond 60 min
        long_duration_factor = 1.0 - decay_rate * decay_hours
        long_duration_factor = max(long_duration_factor, 0.80)  # Floor at 80%
    else:
        long_duration_factor = 1.0

    # --- Discipline adjustment ---
    discipline_factor = {
        "cycling": 1.0,
        "triathlon": 0.92,   # ~8% lower due to run conservation
        "running": 0.85,
    }.get(discipline, 1.0)

    # --- Course type adjustment ---
    course_factor = {
        "flat": 1.0,
        "rolling": 0.97,
        "hilly": 0.93,
        "mountain": 0.88,
    }.get(course_type, 1.0)

    # --- Wind adjustment ---
    # Headwind increases power demand; tailwind decreases it
    # Approximate: +1% power per 5 km/h headwind
    wind_factor = 1.0 - (wind_speed_kmh / 500.0)

    # --- Temperature adjustment ---
    # Optimal ~15-18°C. Deviation reduces sustainable power.
    # (Périard et al. 2021)
    temp_optimal = 16.0
    temp_delta = abs(temperature_c - temp_optimal)
    temp_factor = 1.0 - (temp_delta * 0.008)  # ~0.8% per °C deviation
    temp_factor = max(temp_factor, 0.85)

    # --- Elevation adjustment ---
    # Climbing requires higher power per km but lower avg power overall
    # due to slower speeds and recovery on descents
    if duration_min > 0:
        climb_rate = elevation_gain_m / duration_min  # m/min
        if climb_rate > 15:  # >15 m/min = very hilly
            elevation_factor = 0.90 + (15.0 / climb_rate) * 0.10
        else:
            elevation_factor = 1.0
    else:
        elevation_factor = 1.0

    # --- Race format adjustment ---
    # Mass-start races have surges and drafting → lower average power
    # but higher variability (NP > AP)
    format_factor = {
        "time_trial": 1.0,
        "circuit": 0.95,
        "mass_start": 0.90,
    }.get(race_format, 1.0)

    # --- Combine all factors ---
    predicted_power = (
        base_power
        * long_duration_factor
        * discipline_factor
        * course_factor
        * wind_factor
        * temp_factor
        * elevation_factor
        * format_factor
    )

    # Sanity: power should not exceed CP + W'/60s (max sprint)
    max_possible = cp + w_prime / 60.0
    predicted_power = min(predicted_power, max_possible)
    predicted_power = max(predicted_power, cp * 0.50)  # Floor at 50% CP

    # --- Derived metrics ---
    power_per_kg = predicted_power / weight_kg if weight_kg > 0 else 0.0

    # Normalized Power approximation (NP ≈ AP for TT, higher for mass-start)
    np_multiplier = {
        "time_trial": 1.02,
        "circuit": 1.08,
        "mass_start": 1.15,
    }.get(race_format, 1.02)
    np_w = predicted_power * np_multiplier

    # Intensity Factor
    if cp > 0:
        if_value = np_w / cp
    else:
        if_value = 0.0

    # TSS = (duration_hr × NP × IF × 3600) / (CP × 3600) × 100
    # Simplified: TSS = duration_hr × IF² × 100
    duration_hr = duration_min / 60.0
    tss = duration_hr * if_value * if_value * 100.0

    # --- Confidence scoring ---
    confidence = _calculate_prediction_confidence(
        duration_min=duration_min,
        cp=cp,
        w_prime=w_prime,
        discipline=discipline,
        course_type=course_type,
    )

    # --- Assumptions string ---
    assumptions = _build_assumptions(
        discipline=discipline,
        course_type=course_type,
        wind_speed_kmh=wind_speed_kmh,
        temperature_c=temperature_c,
        elevation_gain_m=elevation_gain_m,
        race_format=race_format,
        long_duration_factor=long_duration_factor,
        discipline_factor=discipline_factor,
        course_factor=course_factor,
        wind_factor=wind_factor,
        temp_factor=temp_factor,
        elevation_factor=elevation_factor,
        format_factor=format_factor,
    )

    return RacePrediction(
        distance_km=0.0,  # Not applicable without speed model
        discipline=discipline,
        predicted_duration_min=round(duration_min, 1),
        avg_power_w=round(predicted_power, 0),
        power_w_per_kg=round(power_per_kg, 2),
        tss=round(tss, 0),
        np_w=round(np_w, 0),
        if_value=round(if_value, 2),
        confidence=round(confidence, 2),
        assumptions=assumptions,
    )


def predict_race_duration(
    cp: float,
    w_prime: float,
    weight_kg: float,
    target_power_w: float,
    discipline: str = "cycling",
    course_type: str = "flat",
    wind_speed_kmh: float = 0.0,
    temperature_c: float = 20.0,
    elevation_gain_m: float = 0.0,
    race_format: str = "time_trial",
) -> Optional[RacePrediction]:
    """
    Predict how long an athlete can sustain a given target power.

    Inverse of predict_race_power — solves for duration given power.

    Returns None if target power is unsustainable (above max possible).
    """
    max_possible = cp + w_prime / 60.0
    if target_power_w > max_possible:
        return None

    # Binary search for duration
    lo, hi = 1.0, 600.0  # 1 min to 10 hours
    for _ in range(50):  # Converges quickly
        mid = (lo + hi) / 2.0
        pred = predict_race_power(
            cp=cp,
            w_prime=w_prime,
            weight_kg=weight_kg,
            duration_min=mid,
            discipline=discipline,
            course_type=course_type,
            wind_speed_kmh=wind_speed_kmh,
            temperature_c=temperature_c,
            elevation_gain_m=elevation_gain_m,
            race_format=race_format,
        )
        if pred.avg_power_w > target_power_w:
            lo = mid
        else:
            hi = mid

    return predict_race_power(
        cp=cp,
        w_prime=w_prime,
        weight_kg=weight_kg,
        duration_min=round((lo + hi) / 2.0, 1),
        discipline=discipline,
        course_type=course_type,
        wind_speed_kmh=wind_speed_kmh,
        temperature_c=temperature_c,
        elevation_gain_m=elevation_gain_m,
        race_format=race_format,
    )


def generate_race_predictions_table(
    cp: float,
    w_prime: float,
    weight_kg: float,
    discipline: str = "cycling",
    course_type: str = "flat",
    wind_speed_kmh: float = 0.0,
    temperature_c: float = 20.0,
    elevation_gain_m: float = 0.0,
    race_format: str = "time_trial",
) -> pd.DataFrame:
    """
    Generate a table of race predictions for common distances/durations.

    Returns DataFrame with columns:
        duration_min, discipline, avg_power, power_per_kg, tss, if_value, confidence
    """
    durations = [
        (5, "Prolog / TT krótki"),
        (10, "ITT 10km"),
        (20, "ITT 20km"),
        (30, "ITT 30km"),
        (40, "ITT 40km"),
        (60, "ITT 1h"),
        (90, "Kryterium / Circuit"),
        (120, "Wyścig masowy 2h"),
        (180, "Pół-IRONMAN bike"),
        (240, "IRONMAN bike"),
        (300, "Ultradystans 5h"),
        (360, "Ultradystans 6h"),
    ]

    rows = []
    for dur_min, label in durations:
        pred = predict_race_power(
            cp=cp,
            w_prime=w_prime,
            weight_kg=weight_kg,
            duration_min=dur_min,
            discipline=discipline,
            course_type=course_type,
            wind_speed_kmh=wind_speed_kmh,
            temperature_c=temperature_c,
            elevation_gain_m=elevation_gain_m,
            race_format=race_format,
        )
        rows.append({
            "Dystans": label,
            "Czas [min]": pred.predicted_duration_min,
            "Śr. Moc [W]": pred.avg_power_w,
            "Moc [W/kg]": pred.power_w_per_kg,
            "NP [W]": pred.np_w,
            "IF": pred.if_value,
            "TSS": pred.tss,
            "Pewność [%]": pred.confidence * 100,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_prediction_confidence(
    duration_min: float,
    cp: float,
    w_prime: float,
    discipline: str,
    course_type: str,
) -> float:
    """
    Estimate confidence in the prediction (0-1).

    Higher confidence for:
    - Durations within the validated CP/W' range (2-60 min)
    - Cycling discipline (most data available)
    - Flat courses (simpler model)
    """
    base = 0.70

    # Duration confidence: best in 5-120 min range
    if 5 <= duration_min <= 120:
        base += 0.15
    elif 2 <= duration_min <= 240:
        base += 0.05
    else:
        base -= 0.10

    # Discipline confidence
    if discipline == "cycling":
        base += 0.10
    elif discipline == "triathlon":
        base += 0.05
    else:
        base -= 0.05

    # Course complexity
    if course_type == "flat":
        base += 0.05
    elif course_type == "mountain":
        base -= 0.10

    # Clamp
    return max(0.30, min(0.95, base))


def _build_assumptions(**kwargs) -> str:
    """Build a human-readable assumptions string."""
    lines = []

    factors = [
        ("Dyscyplina", kwargs.get("discipline"), kwargs.get("discipline_factor")),
        ("Typ trasy", kwargs.get("course_type"), kwargs.get("course_factor")),
        ("Wiatr", f"{kwargs.get('wind_speed_kmh', 0):.0f} km/h", kwargs.get("wind_factor")),
        ("Temperatura", f"{kwargs.get('temperature_c', 20):.0f}°C", kwargs.get("temp_factor")),
        ("Przewyższenie", f"{kwargs.get('elevation_gain_m', 0):.0f} m", kwargs.get("elevation_factor")),
        ("Format", kwargs.get("race_format"), kwargs.get("format_factor")),
    ]

    for label, value, factor in factors:
        if factor is not None and factor != 1.0:
            pct = (factor - 1.0) * 100
            sign = "+" if pct > 0 else ""
            lines.append(f"  • {label}: {value} ({sign}{pct:.1f}%)")

    if not lines:
        return "Warunki standardowe — brak korekt środowiskowych."

    return "\n".join(["Korekty zastosowane:"] + lines)


def get_pacing_recommendations(
    predicted_power: float,
    cp: float,
    duration_min: float,
    race_format: str = "time_trial",
) -> list[str]:
    """
    Generate pacing strategy recommendations for race day.

    Based on Petridou & Nikolaidis (2023) and Muehlbauer & Schindler (2022).
    """
    recs = []
    intensity = predicted_power / cp if cp > 0 else 1.0

    if race_format == "time_trial":
        recs.append(
            "🚴 Strategia TT: Utrzymuj równomierne tempo przez cały dystans. "
            "Unikaj startu >105% docelowej mocy — koszt energetyczny nieodwracalny."
        )
        if duration_min > 30:
            recs.append(
                "⏱️ Dla wysiłków >30min: Pierwsze 5min na 95% docelowej mocy, "
                "potem stopniowo do 100%. Zapobiega wczesnemu zużyciu W'."
            )
    elif race_format == "mass_start":
        recs.append(
            "🏁 Strategia masowa: Oczekuj 15-25% wyższego NP niż AP "
            "przez zrywy i drafting. Planuj wysiłki na kluczowe momenty."
        )
        recs.append(
            "📍 Pozycjonowanie: Utrzymuj czołową 1/3 peletonu, "
            "aby uniknąć efektu 'harmonijki' (dodatkowe 5-10% wysiłku)."
        )
    elif race_format == "circuit":
        recs.append(
            "🔄 Strategia obwodowa: Identifikuj kluczowe sekcje (podjazdy, "
            "zakręty) i oszczędzaj energię na prostych przez drafting."
        )

    if intensity > 1.05:
        recs.append(
            "⚠️ Intensywność >105% CP: Wymaga pełnego W' przed startem. "
            "Unikaj ciężkich treningów 48h przed zawodami."
        )
    elif intensity > 0.95:
        recs.append(
            "🎯 Intensywność 95-105% CP: Kluczowy jest pacing. "
            "Każde przekroczenie o 5% skraca czas do wyczerpania o ~15%."
        )
    else:
        recs.append(
            "✅ Intensywność <95% CP: W strefie komfortu. "
            "Skup się na nawodnieniu, odżywianiu i technice."
        )

    return recs
