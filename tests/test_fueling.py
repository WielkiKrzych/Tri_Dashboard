"""Tests for fueling plan calculations."""

import numpy as np
import pytest

from modules.calculations.fueling import (
    FuelingPlan,
    FuelingStep,
    calculate_fueling_plan,
    estimate_sweat_rate,
    estimate_carb_burn_rate,
)


class TestEstimateCarbBurnRate:
    def test_low_intensity(self):
        rate = estimate_carb_burn_rate(150, 75, 0.50)
        assert 20 <= rate <= 80

    def test_high_intensity(self):
        rate = estimate_carb_burn_rate(300, 75, 1.05)
        assert rate > 60

    def test_very_high_intensity(self):
        rate = estimate_carb_burn_rate(350, 75, 1.20)
        assert rate > 80

    def test_zero_power(self):
        rate = estimate_carb_burn_rate(0, 75, 0.5)
        assert rate >= 0

    def test_increases_with_intensity(self):
        low = estimate_carb_burn_rate(200, 75, 0.50)
        mid = estimate_carb_burn_rate(200, 75, 0.75)
        high = estimate_carb_burn_rate(200, 75, 1.00)
        assert low < mid < high


class TestEstimateSweatRate:
    def test_default_range(self):
        rate = estimate_sweat_rate(0.7, 75)
        assert 0.3 <= rate <= 2.5

    def test_hot_weather_increases(self):
        normal = estimate_sweat_rate(0.7, 75)
        hot = estimate_sweat_rate(0.7, 75, temp_c=35)
        assert hot > normal

    def test_humidity_increases(self):
        normal = estimate_sweat_rate(0.7, 75)
        humid = estimate_sweat_rate(0.7, 75, humidity=85)
        assert humid > normal

    def test_capped_at_max(self):
        rate = estimate_sweat_rate(1.0, 75, temp_c=45, humidity=95)
        assert rate <= 2.5


class TestCalculateFuelingPlan:
    def test_plan_has_steps(self):
        plan = calculate_fueling_plan(2.0, 200, 75, 0.75)
        assert len(plan.plan_steps) > 0

    def test_duration_matches(self):
        plan = calculate_fueling_plan(3.0, 200, 75, 0.75)
        last_step = plan.plan_steps[-1]
        assert last_step.time_min <= 3.0 * 60 + 1

    def test_glycogen_balance_never_negative(self):
        plan = calculate_fueling_plan(4.0, 200, 75, 0.85)
        for step in plan.plan_steps:
            assert step.glycogen_balance >= -10

    def test_plan_metadata(self):
        plan = calculate_fueling_plan(2.0, 200, 75, 0.75)
        assert plan.duration_hours == 2.0
        assert plan.target_power == 200
        assert plan.weight == 75
        assert plan.carb_intake_g_h > 0
        assert plan.fluid_ml_h > 0

    def test_heat_mode(self):
        normal = calculate_fueling_plan(2.0, 200, 75, 0.75)
        hot = calculate_fueling_plan(2.0, 200, 75, 0.75, heat=True)
        assert hot.fluid_ml_h >= normal.fluid_ml_h

    def test_altitude_bonus(self):
        normal = calculate_fueling_plan(2.0, 200, 75, 0.75)
        alt = calculate_fueling_plan(2.0, 200, 75, 0.75, altitude=True)
        assert alt.carb_intake_g_h >= normal.carb_intake_g_h

    def test_step_has_all_fields(self):
        plan = calculate_fueling_plan(2.0, 200, 75, 0.75)
        step = plan.plan_steps[0]
        assert step.time_min > 0
        assert step.carb_g >= 0
        assert step.fluid_ml >= 0
        assert step.sodium_mg >= 0
