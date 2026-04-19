"""Tests for periodization plan calculations."""

import pytest

from datetime import datetime, timedelta

from modules.calculations.periodization import (
    TrainingBlock,
    PeriodizationPlan,
    create_periodization_plan,
    validate_plan,
    get_weekly_schedule,
)


@pytest.fixture
def standard_plan():
    race = datetime.now() + timedelta(weeks=12)
    return create_periodization_plan(race, current_ctl=60.0, current_atl=40.0, total_weeks=12)


@pytest.fixture
def short_plan():
    race = datetime.now() + timedelta(weeks=6)
    return create_periodization_plan(race, current_ctl=50.0, current_atl=30.0, total_weeks=6)


class TestCreatePeriodizationPlan:
    def test_12_week_plan(self, standard_plan):
        assert standard_plan.total_weeks == 12
        assert len(standard_plan.blocks) >= 3

    def test_has_base_build_peak_race(self, standard_plan):
        block_types = [b.block_type for b in standard_plan.blocks]
        assert "Base" in block_types
        assert "Build" in block_types
        assert "Peak" in block_types

    def test_dates_consistent(self, standard_plan):
        for i in range(len(standard_plan.blocks) - 1):
            current_end = standard_plan.blocks[i].end_date
            next_start = standard_plan.blocks[i + 1].start_date
            assert current_end < next_start

    def test_race_date_matches(self):
        race = datetime(2026, 9, 15)
        plan = create_periodization_plan(race, 60, 40, 12)
        assert plan.race_date == "2026-09-15"

    def test_tss_targets_positive(self, standard_plan):
        for block in standard_plan.blocks:
            assert block.target_tss_low > 0
            assert block.target_tss_high > block.target_tss_low

    def test_base_lower_than_build(self, standard_plan):
        base = next(b for b in standard_plan.blocks if b.block_type == "Base")
        build = next(b for b in standard_plan.blocks if b.block_type == "Build")
        assert base.target_tss_high <= build.target_tss_high


class TestValidatePlan:
    def test_reasonable_plan_no_warnings(self, standard_plan):
        warnings = validate_plan(standard_plan)
        critical = [w for w in warnings if "wysoki TSS" in w]
        assert len(critical) == 0

    def test_too_short_plan_warns(self):
        race = datetime.now() + timedelta(weeks=5)
        plan = create_periodization_plan(race, 50, 30, 5)
        warnings = validate_plan(plan)
        assert any("Poniżej 8" in w for w in warnings)

    def test_very_long_plan_warns(self):
        race = datetime.now() + timedelta(weeks=24)
        plan = create_periodization_plan(race, 50, 30, 24)
        warnings = validate_plan(plan)
        assert any("20 tygodni" in w for w in warnings)


class TestGetWeeklySchedule:
    def test_seven_days(self, standard_plan):
        block = standard_plan.blocks[0]
        schedule = get_weekly_schedule(block)
        assert len(schedule) == 7

    def test_has_day_names(self, standard_plan):
        block = standard_plan.blocks[0]
        schedule = get_weekly_schedule(block)
        assert schedule[0]["day"] == "Pon"
        assert schedule[-1]["day"] == "Ndz"

    def test_sunday_is_rest(self, standard_plan):
        block = standard_plan.blocks[0]
        schedule = get_weekly_schedule(block)
        assert schedule[-1]["tss"] == 0
        assert schedule[-1]["intensity"] == "Odpoczynek"

    def test_tss_values_reasonable(self, standard_plan):
        block = standard_plan.blocks[0]
        schedule = get_weekly_schedule(block)
        for day in schedule:
            assert day["tss"] >= 0

    def test_all_blocks_have_schedule(self, standard_plan):
        for block in standard_plan.blocks:
            schedule = get_weekly_schedule(block)
            assert len(schedule) == 7
