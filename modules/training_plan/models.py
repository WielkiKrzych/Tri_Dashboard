"""
Data models for the Training Plan Builder.

Defines enums, frozen dataclasses for workouts, days, weeks, and full plans.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkoutType(Enum):
    """Types of training sessions with emoji labels and TSS weight factors."""

    REST = "Odpoczynek"
    RECOVERY = "Regeneracja"
    ENDURANCE = "Wytrzymałość"
    TEMPO = "Tempo"
    THRESHOLD = "Próg"
    VO2MAX = "VO2max"
    ANAEROBIC = "Anaerobowe"
    STRENGTH = "Siła"
    CROSS_TRAIN = "Cross-training"

    @property
    def emoji(self) -> str:
        return {
            WorkoutType.REST: "😴",
            WorkoutType.RECOVERY: "🧘",
            WorkoutType.ENDURANCE: "🚴",
            WorkoutType.TEMPO: "🔥",
            WorkoutType.THRESHOLD: "⚡",
            WorkoutType.VO2MAX: "🫀",
            WorkoutType.ANAEROBIC: "💨",
            WorkoutType.STRENGTH: "🏋️",
            WorkoutType.CROSS_TRAIN: "🏊",
        }[self]

    @property
    def default_tss_factor(self) -> float:
        """Relative TSS contribution (1.0 = standard session)."""
        return {
            WorkoutType.REST: 0.0,
            WorkoutType.RECOVERY: 0.3,
            WorkoutType.ENDURANCE: 0.8,
            WorkoutType.TEMPO: 1.0,
            WorkoutType.THRESHOLD: 1.2,
            WorkoutType.VO2MAX: 1.4,
            WorkoutType.ANAEROBIC: 1.3,
            WorkoutType.STRENGTH: 0.5,
            WorkoutType.CROSS_TRAIN: 0.6,
        }[self]


class PeriodizationPhase(Enum):
    """Macrocycle periodization phases with volume and intensity factors."""

    BASE = "Baza"
    BUILD = "Budowa"
    PEAK = "Szczyt"
    TAPER = "Tapering"
    RECOVERY = "Regeneracja"

    @property
    def emoji(self) -> str:
        return {
            PeriodizationPhase.BASE: "🧱",
            PeriodizationPhase.BUILD: "🏗️",
            PeriodizationPhase.PEAK: "🏔️",
            PeriodizationPhase.TAPER: "📉",
            PeriodizationPhase.RECOVERY: "🧘",
        }[self]

    @property
    def weekly_volume_factor(self) -> float:
        return {
            PeriodizationPhase.BASE: 0.85,
            PeriodizationPhase.BUILD: 1.0,
            PeriodizationPhase.PEAK: 1.10,
            PeriodizationPhase.TAPER: 0.70,
            PeriodizationPhase.RECOVERY: 0.50,
        }[self]

    @property
    def intensity_factor(self) -> float:
        return {
            PeriodizationPhase.BASE: 0.70,
            PeriodizationPhase.BUILD: 0.85,
            PeriodizationPhase.PEAK: 1.0,
            PeriodizationPhase.TAPER: 0.80,
            PeriodizationPhase.RECOVERY: 0.40,
        }[self]


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedWorkout:
    """A single planned training session."""

    workout_type: WorkoutType
    tss_target: float
    duration_min: int
    description: str = ""
    notes: str = ""


@dataclass(frozen=True)
class PlannedDay:
    """One day within a training week."""

    date: date
    day_of_week: int  # 0=Mon .. 6=Sun
    workout: Optional[PlannedWorkout] = None
    is_rest: bool = False

    @property
    def tss(self) -> float:
        if self.workout is None:
            return 0.0
        return self.workout.tss_target


@dataclass(frozen=True)
class PlannedWeek:
    """A full training week (7 days)."""

    week_number: int
    start_date: date
    phase: PeriodizationPhase
    weekly_tss_target: float
    days: tuple[PlannedDay, ...] = field(default_factory=tuple)

    @property
    def total_tss(self) -> float:
        return sum(d.tss for d in self.days)

    @property
    def training_days(self) -> int:
        return sum(1 for d in self.days if not d.is_rest and d.workout is not None)


@dataclass
class TrainingPlan:
    """A multi-week training plan."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "Plan treningowy"
    start_date: date = field(default_factory=date.today)
    weeks: list[PlannedWeek] = field(default_factory=list)
    athlete_id: str = "default"
    weekly_tss_baseline: float = 300.0

    @property
    def total_weeks(self) -> int:
        return len(self.weeks)

    @property
    def end_date(self) -> date:
        if not self.weeks:
            return self.start_date
        last_week = self.weeks[-1]
        return last_week.start_date + timedelta(days=6)
