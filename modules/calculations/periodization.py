"""Periodization Planner calculations.

Creates training periodization plans with Base/Build/Peak/Race blocks,
TSS targets, and validation.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np


@dataclass
class TrainingBlock:
    """Single training block in periodization plan."""

    start_date: str
    end_date: str
    block_type: str
    target_tss_low: float
    target_tss_high: float
    focus: str


@dataclass
class PeriodizationPlan:
    """Complete periodization plan."""

    start_date: str
    race_date: str
    blocks: List[TrainingBlock] = field(default_factory=list)
    total_weeks: int = 12
    peak_week: int = 0


def create_periodization_plan(
    race_date: datetime,
    current_ctl: float,
    current_atl: float,
    total_weeks: int = 12,
) -> PeriodizationPlan:
    """Create a periodization plan leading to race_date.

    Standard periodization:
    - Base (60% of time): CTL × 0.8-1.0 TSS/day
    - Build (25% of time): CTL × 1.0-1.3 TSS/day
    - Peak (10% of time): CTL × 0.6-0.8 TSS/day (taper)
    - Race (5% of time): CTL × 0.3-0.5 TSS/day (race week)
    """
    ctl = max(current_ctl, 30.0)

    start_date = race_date - timedelta(weeks=total_weeks)

    base_weeks = max(1, int(total_weeks * 0.60))
    build_weeks = max(1, int(total_weeks * 0.25))
    peak_weeks = max(1, int(total_weeks * 0.10))
    race_weeks = max(1, total_weeks - base_weeks - build_weeks - peak_weeks)

    blocks: List[TrainingBlock] = []
    current = start_date

    block_configs = [
        ("Base", base_weeks, 0.8, 1.0, "Budowanie bazy tlenowej, długie treningi Z1-Z2"),
        ("Build", build_weeks, 1.0, 1.3, "Intensyfikacja, interwały, treningi progowe"),
        ("Peak", peak_weeks, 0.6, 0.8, "Taper — redukcja objętości, utrzymanie intensywności"),
        ("Race", race_weeks, 0.3, 0.5, "Tydzień startowy — aktywacja i regeneracja"),
    ]

    peak_week_num = base_weeks + build_weeks + 1

    for btype, weeks, tss_lo_pct, tss_hi_pct, focus in block_configs:
        end = current + timedelta(weeks=weeks) - timedelta(days=1)
        blocks.append(
            TrainingBlock(
                start_date=current.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                block_type=btype,
                target_tss_low=round(ctl * tss_lo_pct, 0),
                target_tss_high=round(ctl * tss_hi_pct, 0),
                focus=focus,
            )
        )
        current = end + timedelta(days=1)

    return PeriodizationPlan(
        start_date=start_date.strftime("%Y-%m-%d"),
        race_date=race_date.strftime("%Y-%m-%d"),
        blocks=blocks,
        total_weeks=total_weeks,
        peak_week=peak_week_num,
    )


def validate_plan(plan: PeriodizationPlan) -> List[str]:
    """Validate plan and return warnings for overambitious schedules."""
    warnings: List[str] = []

    for block in plan.blocks:
        if block.target_tss_high > 200:
            warnings.append(
                f"⚠️ {block.block_type}: wysoki TSS ({block.target_tss_high:.0f}) — "
                "ryzyko przeuczenia. Rozważ zmniejszenie."
            )

        try:
            start = datetime.strptime(block.start_date, "%Y-%m-%d")
            end = datetime.strptime(block.end_date, "%Y-%m-%d")
            days = (end - start).days + 1
            if days < 7:
                warnings.append(f"⚠️ {block.block_type}: tylko {days} dni — zbyt krótki blok.")
        except ValueError:
            pass

    if plan.total_weeks > 20:
        warnings.append(
            "⚠️ Plany powyżej 20 tygodni są trudne do realizacji — rozważ podział na mezocykle."
        )

    if plan.total_weeks < 8:
        warnings.append("⚠️ Poniżej 8 tygodni — ograniczona możliwość budowania formy.")

    return warnings


def get_weekly_schedule(block: TrainingBlock) -> List[dict]:
    """Generate 7-day TSS distribution template for a block.

    Distribution follows typical training patterns:
    - Mon: easy/recovery
    - Tue: quality workout
    - Wed: easy/moderate
    - Thu: quality workout
    - Fri: easy/recovery
    - Sat: long/endurance
    - Sun: rest
    """
    tss_mid = (block.target_tss_low + block.target_tss_high) / 2

    distributions = {
        "Base": [0.6, 1.1, 0.7, 1.1, 0.6, 1.4, 0.0],
        "Build": [0.5, 1.3, 0.6, 1.3, 0.5, 1.5, 0.0],
        "Peak": [0.5, 1.2, 0.4, 1.1, 0.3, 0.8, 0.0],
        "Race": [0.4, 0.8, 0.3, 0.6, 0.2, 0.0, 0.0],
    }

    dist = distributions.get(block.block_type, distributions["Base"])
    days_pl = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Ndz"]

    schedule = []
    for i, (day_name, factor) in enumerate(zip(days_pl, dist)):
        tss = round(tss_mid * factor, 0)
        if factor == 0:
            intensity = "Odpoczynek"
        elif factor >= 1.3:
            intensity = "Wysoka"
        elif factor >= 0.9:
            intensity = "Umiarkowana"
        else:
            intensity = "Niska"

        schedule.append(
            {
                "day": day_name,
                "tss": tss,
                "intensity": intensity,
            }
        )

    return schedule
