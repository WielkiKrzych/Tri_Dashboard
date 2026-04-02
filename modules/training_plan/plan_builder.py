"""
Core plan building logic.

Distributes TSS across days, applies periodization phases,
and assembles multi-week TrainingPlan objects.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .models import (
    PlannedDay,
    PlannedWeek,
    PlannedWorkout,
    TrainingPlan,
    WorkoutType,
    PeriodizationPhase,
)
from .periodization import PERIODIZATION_TEMPLATES, suggest_baseline_tss


# ---------------------------------------------------------------------------
# Default weekly template (Mon–Sun)
# ---------------------------------------------------------------------------

_DEFAULT_WEEKLY_TEMPLATE: Dict[int, WorkoutType] = {
    0: WorkoutType.ENDURANCE,  # Mon
    1: WorkoutType.RECOVERY,  # Tue
    2: WorkoutType.TEMPO,  # Wed
    3: WorkoutType.REST,  # Thu
    4: WorkoutType.THRESHOLD,  # Fri
    5: WorkoutType.ENDURANCE,  # Sat
    6: WorkoutType.REST,  # Sun
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_DESCRIPTIONS: Dict[WorkoutType, str] = {
    WorkoutType.REST: "Całkowity odpoczynek",
    WorkoutType.RECOVERY: "Lekka aktywność regeneracyjna",
    WorkoutType.ENDURANCE: "Trening wytrzymałościowy w strefie Z2",
    WorkoutType.TEMPO: "Praca w strefie tempa (Z3)",
    WorkoutType.THRESHOLD: "Interwały progi (Z4/Z5)",
    WorkoutType.VO2MAX: "Interwały VO2max (Z5+)",
    WorkoutType.ANAEROBIC: "Sprinty i praca beztlenowa",
    WorkoutType.STRENGTH: "Trening siłowy uzupełniający",
    WorkoutType.CROSS_TRAIN: "Alternatywna aktywność (pływanie, bieganie)",
}


def _default_description(wt: WorkoutType) -> str:
    """Return a Polish description for a workout type."""
    return _DESCRIPTIONS.get(wt, "Trening")


def _estimate_duration(tss: float, wt: WorkoutType) -> int:
    """Rough duration in minutes from TSS and workout type.

    Uses an assumed average intensity factor per workout type.
    """
    if wt == WorkoutType.REST:
        return 0
    if wt == WorkoutType.RECOVERY:
        return max(20, int(tss / 15))
    if wt == WorkoutType.STRENGTH:
        return max(30, int(tss / 10))
    if wt == WorkoutType.CROSS_TRAIN:
        return max(30, int(tss / 12))
    # Endurance sports — assume IF ~0.75 for endurance, higher for intensity
    intensity_factor = wt.default_tss_factor
    if intensity_factor <= 0:
        return 0
    # TSS = (IF^2) * hours * 100  →  hours = TSS / (IF^2 * 100)
    assumed_if = 0.65 + 0.15 * intensity_factor  # 0.65–0.95 range
    hours = tss / (assumed_if**2 * 100)
    return max(20, int(hours * 60))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_week(
    week_number: int,
    start_date: date,
    phase: PeriodizationPhase,
    weekly_tss_target: float,
    template: Optional[Dict[int, WorkoutType]] = None,
) -> PlannedWeek:
    """Build a single training week with TSS distributed across training days.

    Args:
        week_number: 1-based week index in the plan.
        start_date: Monday of this week.
        phase: Current periodization phase.
        weekly_tss_target: Target total TSS for this week.
        template: Day-of-week → WorkoutType mapping (0=Mon, 6=Sun).
                  Defaults to _DEFAULT_WEEKLY_TEMPLATE.

    Returns:
        A PlannedWeek with 7 PlannedDay entries.
    """
    tmpl = template or _DEFAULT_WEEKLY_TEMPLATE

    # Identify training days and their relative TSS weights
    training_days_info: list[tuple[int, WorkoutType]] = []
    for dow in range(7):
        wt = tmpl.get(dow, WorkoutType.REST)
        if wt != WorkoutType.REST:
            training_days_info.append((dow, wt))

    # Calculate proportional TSS per training day
    total_weight = sum(wt.default_tss_factor for _, wt in training_days_info)
    days: list[PlannedDay] = []

    training_dow_set = {dow for dow, _ in training_days_info}

    for dow in range(7):
        day_date = start_date + timedelta(days=dow)

        if dow not in training_dow_set:
            days.append(PlannedDay(date=day_date, day_of_week=dow, is_rest=True))
            continue

        wt = tmpl[dow]
        weight_ratio = wt.default_tss_factor / total_weight if total_weight > 0 else 0

        # Base TSS for this day, then modulate by phase intensity
        day_tss = weekly_tss_target * weight_ratio * phase.intensity_factor
        day_tss = round(day_tss, 1)

        duration = _estimate_duration(day_tss, wt)
        description = _default_description(wt)

        workout = PlannedWorkout(
            workout_type=wt,
            tss_target=day_tss,
            duration_min=duration,
            description=description,
        )
        days.append(PlannedDay(date=day_date, day_of_week=dow, workout=workout, is_rest=False))

    return PlannedWeek(
        week_number=week_number,
        start_date=start_date,
        phase=phase,
        weekly_tss_target=weekly_tss_target,
        days=tuple(days),
    )


def build_plan(
    name: str = "Plan treningowy",
    start_date: Optional[date] = None,
    total_weeks: int = 13,
    weekly_tss_baseline: Optional[float] = None,
    phase_distribution: Optional[List[Tuple[PeriodizationPhase, int]]] = None,
    template: Optional[Dict[int, WorkoutType]] = None,
    athlete_id: str = "default",
) -> TrainingPlan:
    """Build a complete multi-week training plan with periodization.

    Args:
        name: Plan name.
        start_date: Plan start date (defaults to next Monday).
        total_weeks: Number of weeks in the plan (1-24).
        weekly_tss_baseline: Starting weekly TSS. Auto-suggested if None.
        phase_distribution: List of (phase, weeks) tuples.
                            Defaults to "standard" template, truncated/extended to match total_weeks.
        template: Day-of-week → WorkoutType mapping.
        athlete_id: Athlete identifier.

    Returns:
        A fully-built TrainingPlan.
    """
    if start_date is None:
        # Start on next Monday
        today = date.today()
        days_until_monday = (7 - today.weekday()) % 7
        start_date = today + timedelta(days=days_until_monday)

    if weekly_tss_baseline is None:
        weekly_tss_baseline = suggest_baseline_tss()

    total_weeks = max(1, min(24, total_weeks))

    # Resolve phase distribution
    if phase_distribution is None:
        phase_distribution = _resolve_distribution(total_weeks)

    # Build weeks with progressive ramp (+4% per week within plan)
    weeks: list[PlannedWeek] = []
    week_idx = 0
    weekly_ramp = 1.04  # +4% per week

    for phase, phase_weeks in phase_distribution:
        phase_start_tss = weekly_tss_baseline * phase.weekly_volume_factor
        if weeks:
            # Continue from where previous phase ended
            phase_start_tss = weeks[-1].weekly_tss_target * phase.weekly_volume_factor
            phase_start_tss = max(phase_start_tss, weekly_tss_baseline * 0.5)

        for w in range(phase_weeks):
            if week_idx >= total_weeks:
                break
            week_num = week_idx + 1
            week_start = start_date + timedelta(weeks=week_idx)

            # Ramp TSS within the phase
            week_tss = phase_start_tss * (weekly_ramp**w)
            week_tss = round(week_tss, 1)

            week = build_week(
                week_number=week_num,
                start_date=week_start,
                phase=phase,
                weekly_tss_target=week_tss,
                template=template,
            )
            weeks.append(week)
            week_idx += 1

    return TrainingPlan(
        name=name,
        start_date=start_date,
        weeks=weeks,
        athlete_id=athlete_id,
        weekly_tss_baseline=weekly_tss_baseline,
    )


def apply_template_to_plan(
    plan: TrainingPlan,
    template: Dict[int, WorkoutType],
) -> TrainingPlan:
    """Rebuild all weeks of an existing plan with a new weekly template.

    The plan's phase distribution, TSS targets, and metadata are preserved.
    Only the day-level workout types change.

    Args:
        plan: Existing TrainingPlan.
        template: New day-of-week → WorkoutType mapping.

    Returns:
        A new TrainingPlan with rebuilt weeks.
    """
    new_weeks: list[PlannedWeek] = []
    for week in plan.weeks:
        new_week = build_week(
            week_number=week.week_number,
            start_date=week.start_date,
            phase=week.phase,
            weekly_tss_target=week.weekly_tss_target,
            template=template,
        )
        new_weeks.append(new_week)

    plan.weeks = new_weeks
    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_distribution(total_weeks: int) -> List[Tuple[PeriodizationPhase, int]]:
    """Resolve a phase distribution that fits exactly total_weeks.

    Uses the 'standard' template and extends/truncates the last phase.
    """
    dist = list(PERIODIZATION_TEMPLATES["standard"])
    remaining = total_weeks

    result: list[tuple[PeriodizationPhase, int]] = []
    for i, (phase, weeks) in enumerate(dist):
        if i == len(dist) - 1:
            # Last phase absorbs remaining weeks
            result.append((phase, max(1, remaining)))
        else:
            allocated = min(weeks, remaining)
            if allocated <= 0:
                break
            result.append((phase, allocated))
            remaining -= allocated

    return result
