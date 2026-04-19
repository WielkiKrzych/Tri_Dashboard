"""Sleep and Recovery integration calculations.

Computes sleep scores, composite recovery indices, and parses
sleep data from Garmin/Oura CSV exports.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class SleepData:
    """Single night sleep data."""

    date: str
    total_sleep_hours: float
    deep_sleep_hours: float
    rem_sleep_hours: float
    sleep_efficiency_pct: float
    hr_during_sleep: Optional[float]
    hrv_during_sleep: Optional[float]
    wake_count: int


@dataclass
class SleepScore:
    """Sleep quality score 0-100."""

    date: str
    score: float
    level: str
    interpretation: str


@dataclass
class CompositeRecovery:
    """Combined recovery assessment from sleep + HRV + training load."""

    date: str
    sleep_score: float
    hrv_readiness: float
    training_load_status: str
    composite_score: float
    recommendation: str


def calculate_sleep_score(sleep_data: SleepData) -> SleepScore:
    """Calculate 0-100 sleep score from sleep data.

    Weighting: duration (50%) + efficiency (30%) + deep/REM ratio (20%).
    Optimal: 7-9h, >85% efficiency, >20% deep, >20% REM.
    """
    duration = sleep_data.total_sleep_hours
    efficiency = sleep_data.sleep_efficiency_pct
    deep_pct = (sleep_data.deep_sleep_hours / duration * 100) if duration > 0 else 0
    rem_pct = (sleep_data.rem_sleep_hours / duration * 100) if duration > 0 else 0

    # Duration score (0-100)
    if 7 <= duration <= 9:
        duration_score = 100
    elif 6 <= duration < 7:
        duration_score = 60 + (duration - 6) * 40
    elif 9 < duration <= 10:
        duration_score = 100 - (duration - 9) * 20
    elif 5 <= duration < 6:
        duration_score = 30 + (duration - 5) * 30
    else:
        duration_score = max(0, 30 - abs(duration - 7) * 10)

    # Efficiency score (0-100)
    if efficiency >= 90:
        eff_score = 100
    elif efficiency >= 85:
        eff_score = 80 + (efficiency - 85) * 4
    elif efficiency >= 75:
        eff_score = 50 + (efficiency - 75) * 3
    else:
        eff_score = max(0, efficiency * 0.6)

    # Deep/REM score (0-100)
    deep_score = min(100, deep_pct * 4) if deep_pct >= 15 else deep_pct * 4 * 0.6
    rem_score = min(100, rem_pct * 4) if rem_pct >= 15 else rem_pct * 4 * 0.6
    composition_score = (deep_score + rem_score) / 2

    # Weighted total
    total = duration_score * 0.5 + eff_score * 0.3 + composition_score * 0.2
    total = round(np.clip(total, 0, 100), 1)

    level, interp = _sleep_level(total, duration, efficiency, deep_pct)

    return SleepScore(
        date=sleep_data.date,
        score=total,
        level=level,
        interpretation=interp,
    )


def _sleep_level(score: float, duration: float, eff: float, deep_pct: float) -> tuple:
    if score >= 85:
        return "🟢 Doskonały", "Sen optymalny — organizm w pełni odnowiony."
    elif score >= 70:
        return "🟡 Dobry", "Sen dobry, ale jest miejsce na poprawę."
    elif score >= 50:
        notes = []
        if duration < 7:
            notes.append("zbyt krótki sen")
        if eff < 85:
            notes.append("niska efektywność")
        if deep_pct < 15:
            notes.append("mało snu głębokiego")
        detail = ", ".join(notes) if notes else "obniżona jakość"
        return "🟠 Przeciętny", f"Sen przeciętny: {detail}."
    return "🔴 Słaby", "Sen niewystarczający — rozważ poprawę higieny snu."


def calculate_composite_recovery(
    sleep_score: float,
    hrv_readiness_score: float,
    current_tsb: float,
) -> CompositeRecovery:
    """Calculate composite recovery from sleep, HRV, and training load.

    Weighted: sleep 30% + HRV 40% + training load 30%.
    """
    # Normalize TSB to 0-100 scale
    if current_tsb > 25:
        tsb_score = 100
    elif current_tsb > 5:
        tsb_score = 70 + (current_tsb - 5) / 20 * 30
    elif current_tsb > -10:
        tsb_score = 40 + (current_tsb + 10) / 15 * 30
    elif current_tsb > -30:
        tsb_score = 10 + (current_tsb + 30) / 20 * 30
    else:
        tsb_score = max(0, 10 + (current_tsb + 30))

    composite = sleep_score * 0.3 + hrv_readiness_score * 0.4 + tsb_score * 0.3
    composite = round(np.clip(composite, 0, 100), 1)

    if current_tsb > 25:
        load_status = "🟢 Świeży"
    elif current_tsb > -10:
        load_status = "🟠 Optymalne obciążenie"
    else:
        load_status = "🔴 Zmęczony"

    rec = _recovery_recommendation(composite, sleep_score, hrv_readiness_score, tsb_score)

    today = datetime.now().strftime("%Y-%m-%d")
    return CompositeRecovery(
        date=today,
        sleep_score=round(sleep_score, 1),
        hrv_readiness=round(hrv_readiness_score, 1),
        training_load_status=load_status,
        composite_score=composite,
        recommendation=rec,
    )


def _recovery_recommendation(composite: float, sleep: float, hrv: float, tsb: float) -> str:
    if composite >= 80:
        return "✅ Pełna gotowość. Możesz wykonać trening o wysokiej intensywności."
    elif composite >= 60:
        return "🟡 Umiarkowana gotowość. Trening w strefie tempa, unikaj overreaching."
    elif composite >= 40:
        weakest = min(
            ("sen", sleep),
            ("HRV", hrv),
            ("obciążenie", tsb),
            key=lambda x: x[1],
        )
        return f"🟠 Obniżona gotowość. Najsłabszy czynnik: {weakest[0]}. Zalecana regeneracja."
    return "⛔ Niska gotowość. Całkowity odpoczynek. Skonsultuj się z trenerem."


def parse_garmin_sleep_csv(csv_path: str) -> List[SleepData]:
    """Parse Garmin Connect sleep CSV export.

    Expected columns: calendarDate, sleepTimeSeconds, deepSleepSeconds,
    remSleepSeconds, sleepEfficiencyPercentage, averageHeartRate,
    averageRespirationRate, awakeSleepSeconds.
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    records: List[SleepData] = []
    for _, row in df.iterrows():
        try:
            total_h = row.get("sleepTimeSeconds", 0) / 3600
            deep_h = row.get("deepSleepSeconds", 0) / 3600
            rem_h = row.get("remSleepSeconds", 0) / 3600
            eff = row.get("sleepEfficiencyPercentage", 0)
            hr = row.get("averageHeartRate")
            wake = (
                int(row.get("awakeSleepSeconds", 0) / 60 / 5) if row.get("awakeSleepSeconds") else 0
            )

            records.append(
                SleepData(
                    date=str(row.get("calendarDate", "")),
                    total_sleep_hours=round(total_h, 2),
                    deep_sleep_hours=round(deep_h, 2),
                    rem_sleep_hours=round(rem_h, 2),
                    sleep_efficiency_pct=round(eff, 1),
                    hr_during_sleep=float(hr) if pd.notna(hr) else None,
                    hrv_during_sleep=None,
                    wake_count=wake,
                )
            )
        except (ValueError, TypeError, ZeroDivisionError):
            continue

    return records


def parse_oura_sleep_csv(csv_path: str) -> List[SleepData]:
    """Parse Oura Ring sleep CSV export.

    Expected columns: day, total_sleep, deep_sleep, rem_sleep,
    sleep_efficiency, average_hr, average_hrv, awake_time.
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    records: List[SleepData] = []
    for _, row in df.iterrows():
        try:
            total_h = row.get("total_sleep", 0)
            if isinstance(total_h, str):
                parts = total_h.split(":")
                total_h = int(parts[0]) + int(parts[1]) / 60

            deep_h = row.get("deep_sleep", 0) or 0
            rem_h = row.get("rem_sleep", 0) or 0
            eff = row.get("sleep_efficiency", 0) or 0
            hr = row.get("average_hr")
            hrv = row.get("average_hrv")
            awake = row.get("awake_time", 0) or 0

            records.append(
                SleepData(
                    date=str(row.get("day", "")),
                    total_sleep_hours=round(float(total_h), 2),
                    deep_sleep_hours=round(float(deep_h), 2),
                    rem_sleep_hours=round(float(rem_h), 2),
                    sleep_efficiency_pct=round(float(eff), 1),
                    hr_during_sleep=float(hr) if pd.notna(hr) else None,
                    hrv_during_sleep=float(hrv) if pd.notna(hrv) else None,
                    wake_count=int(awake / 5) if awake else 0,
                )
            )
        except (ValueError, TypeError, ZeroDivisionError):
            continue

    return records
