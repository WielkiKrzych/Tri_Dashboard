"""
Fueling/Hydration engine — event nutrition planning.

Models carbohydrate burn, sweat rate, sodium loss, and glycogen balance
to produce a structured hourly fueling plan for endurance events.

References:
    Jeukendrup AE (2014). Carbohydrate intake during exercise. Sports Med.
    Sawka MN et al. (2007). Exercise and fluid replacement. ACSM position stand.
    Périard JD et al. (2021). Exercise under heat stress. Sports Med.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FuelingStep:
    time_min: float
    action: str
    carb_g: float
    fluid_ml: float
    sodium_mg: float
    glycogen_balance: float
    notes: str


@dataclass
class FuelingPlan:
    duration_hours: float
    target_power: float
    weight: float
    carb_intake_g_h: float
    sodium_mg_h: float
    fluid_ml_h: float
    glycogen_stores_g: float
    plan_steps: List[FuelingStep] = field(default_factory=list)


def estimate_carb_burn_rate(power_w: float, weight_kg: float, intensity_pct_ftp: float) -> float:
    """Estimate carbohydrate oxidation rate [g/h]."""
    energy_kcal_h = power_w * 3.6
    cho_fraction = (
        0.15
        if intensity_pct_ftp < 0.55
        else 0.30 + (intensity_pct_ftp - 0.55) * 1.5
        if intensity_pct_ftp < 0.75
        else 0.60 + (intensity_pct_ftp - 0.75) * 1.2
        if intensity_pct_ftp < 1.00
        else min(0.90 + (intensity_pct_ftp - 1.0) * 0.5, 1.0)
    )
    return energy_kcal_h * cho_fraction / 4.0


def estimate_sweat_rate(
    intensity_pct: float,
    weight_kg: float,
    temp_c: Optional[float] = None,
    humidity: Optional[float] = None,
) -> float:
    """Estimate sweat rate [L/h] based on intensity and conditions."""
    base_rate = 0.5 + intensity_pct * 0.8
    heat_factor = 1.0
    if temp_c is not None and temp_c > 20:
        heat_factor += (temp_c - 20) * 0.03
    if humidity is not None and humidity > 60:
        heat_factor += (humidity - 60) * 0.005
    return min(base_rate * heat_factor, 2.5)


def calculate_fueling_plan(
    duration_hours: float,
    target_power_w: float,
    weight_kg: float,
    intensity_pct_ftp: float,
    heat: bool = False,
    altitude: bool = False,
) -> FuelingPlan:
    """Generate a complete hourly fueling plan."""
    carb_burn = estimate_carb_burn_rate(target_power_w, weight_kg, intensity_pct_ftp)

    temp_c = 35.0 if heat else None
    sweat_rate = estimate_sweat_rate(intensity_pct_ftp, weight_kg, temp_c)
    fluid_ml_h = sweat_rate * 1000 * 0.8
    sodium_mg_h = sweat_rate * 800

    glycogen = 450.0
    glycogen_balance = glycogen

    recommended_carb = min(carb_burn, 90.0)
    altitude_carb_bonus = 10.0 if altitude else 0.0
    carb_intake = min(recommended_carb + altitude_carb_bonus, 120.0)

    steps: List[FuelingStep] = []
    num_steps = max(1, int(duration_hours * 2))
    step_duration_min = (duration_hours * 60) / num_steps

    for i in range(num_steps):
        t_min = (i + 1) * step_duration_min
        consumed = carb_intake * step_duration_min / 60.0
        burned = carb_burn * step_duration_min / 60.0
        glycogen_balance = glycogen_balance - burned + consumed
        glycogen_balance = max(glycogen_balance, 0.0)

        fluid_step = fluid_ml_h * step_duration_min / 60.0
        sodium_step = sodium_mg_h * step_duration_min / 60.0

        if glycogen_balance < 100:
            action = "🚨 URGENT: Zwiększ spożycie CHO!"
            notes = f"Zapas glikogenu krytyczny: {glycogen_balance:.0f}g"
        elif glycogen_balance < 200:
            action = "⚠️ Żel + izotonik"
            notes = f"Glikogen spada: {glycogen_balance:.0f}g"
        else:
            action = "✅ Pij i jedz wg planu"
            notes = f"Glikogen OK: {glycogen_balance:.0f}g"

        steps.append(
            FuelingStep(
                time_min=t_min,
                action=action,
                carb_g=consumed,
                fluid_ml=fluid_step,
                sodium_mg=sodium_step,
                glycogen_balance=glycogen_balance,
                notes=notes,
            )
        )

    return FuelingPlan(
        duration_hours=duration_hours,
        target_power=target_power_w,
        weight=weight_kg,
        carb_intake_g_h=carb_intake,
        sodium_mg_h=sodium_mg_h,
        fluid_ml_h=fluid_ml_h,
        glycogen_stores_g=glycogen,
        plan_steps=steps,
    )
