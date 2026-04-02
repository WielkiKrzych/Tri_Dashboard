"""SQLite persistence for training plans."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from typing import Optional

from modules.db.base import BaseStore

from .models import (
    PeriodizationPhase,
    PlannedDay,
    PlannedWeek,
    PlannedWorkout,
    TrainingPlan,
    WorkoutType,
)

logger = logging.getLogger(__name__)

_WORKOUT_TYPE_MAP = {e.value: e for e in WorkoutType}
_PHASE_MAP = {e.value: e for e in PeriodizationPhase}


class TrainingPlanStore(BaseStore):
    """CRUD store for TrainingPlan objects backed by SQLite."""

    table_name = "training_plans"
    _schema_sql = """
        CREATE TABLE IF NOT EXISTS training_plans (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            start_date    TEXT NOT NULL,
            athlete_id    TEXT NOT NULL DEFAULT 'default',
            weekly_tss_baseline REAL NOT NULL DEFAULT 300.0,
            weeks_json    TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        )
    """

    def save_plan(self, plan: TrainingPlan) -> None:
        now = datetime.now().isoformat()
        weeks_json = _serialize_plan(plan)
        self.upsert(
            """
            INSERT INTO training_plans (id, name, start_date, athlete_id,
                weekly_tss_baseline, weeks_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                start_date = excluded.start_date,
                athlete_id = excluded.athlete_id,
                weekly_tss_baseline = excluded.weekly_tss_baseline,
                weeks_json = excluded.weeks_json,
                updated_at = excluded.updated_at
            """,
            (
                plan.id,
                plan.name,
                plan.start_date.isoformat(),
                plan.athlete_id,
                plan.weekly_tss_baseline,
                weeks_json,
                now,
                now,
            ),
        )

    def get_plan(self, plan_id: str) -> Optional[TrainingPlan]:
        row = self.query_one("SELECT * FROM training_plans WHERE id = ?", (plan_id,))
        if row is None:
            return None
        return _deserialize_row(row)

    def list_plans(self, athlete_id: str = "default") -> list[TrainingPlan]:
        rows = self.query(
            "SELECT * FROM training_plans WHERE athlete_id = ? ORDER BY created_at DESC",
            (athlete_id,),
        )
        return [_deserialize_row(r) for r in rows]

    def delete_plan(self, plan_id: str) -> bool:
        return self.delete("id", plan_id)


def _serialize_plan(plan: TrainingPlan) -> str:
    def _day_to_dict(d: PlannedDay) -> dict:
        result = {
            "date": d.date.isoformat(),
            "day_of_week": d.day_of_week,
            "is_rest": d.is_rest,
        }
        if d.workout is not None:
            result["workout"] = {
                "workout_type": d.workout.workout_type.value,
                "tss_target": d.workout.tss_target,
                "duration_min": d.workout.duration_min,
                "description": d.workout.description,
                "notes": d.workout.notes,
            }
        return result

    weeks_data = []
    for w in plan.weeks:
        weeks_data.append(
            {
                "week_number": w.week_number,
                "start_date": w.start_date.isoformat(),
                "phase": w.phase.value,
                "weekly_tss_target": w.weekly_tss_target,
                "days": [_day_to_dict(d) for d in w.days],
            }
        )

    return json.dumps(
        {
            "weeks": weeks_data,
        },
        ensure_ascii=False,
    )


def _deserialize_row(row: sqlite3.Row) -> TrainingPlan:
    plan_id = row["id"]
    name = row["name"]
    start_date_str = row["start_date"]
    athlete_id = row["athlete_id"]
    tss_baseline = row["weekly_tss_baseline"]
    weeks_json = row["weeks_json"]

    data = json.loads(weeks_json)
    weeks: list[PlannedWeek] = []

    for wd in data.get("weeks", []):
        days: list[PlannedDay] = []
        for dd in wd.get("days", []):
            workout = None
            if "workout" in dd:
                wk = dd["workout"]
                workout = PlannedWorkout(
                    workout_type=_WORKOUT_TYPE_MAP[wk["workout_type"]],
                    tss_target=wk["tss_target"],
                    duration_min=wk["duration_min"],
                    description=wk.get("description", ""),
                    notes=wk.get("notes", ""),
                )
            days.append(
                PlannedDay(
                    date=date.fromisoformat(dd["date"]),
                    day_of_week=dd["day_of_week"],
                    workout=workout,
                    is_rest=dd.get("is_rest", workout is None),
                )
            )

        weeks.append(
            PlannedWeek(
                week_number=wd["week_number"],
                start_date=date.fromisoformat(wd["start_date"]),
                phase=_PHASE_MAP[wd["phase"]],
                weekly_tss_target=wd["weekly_tss_target"],
                days=tuple(days),
            )
        )

    return TrainingPlan(
        id=plan_id,
        name=name,
        start_date=date.fromisoformat(start_date_str),
        weeks=weeks,
        athlete_id=athlete_id,
        weekly_tss_baseline=tss_baseline,
    )
