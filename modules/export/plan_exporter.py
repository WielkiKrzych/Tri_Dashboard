"""CSV export for training plans."""

from __future__ import annotations

import csv
import io

from modules.training_plan.models import TrainingPlan


def export_plan_csv(plan: TrainingPlan) -> str:
    """Export full daily plan as CSV string.

    Columns: Tydzień, Faza, Data, Dzień, Typ treningu, TSS, Czas (min), Opis
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(
        ["Tydzień", "Faza", "Data", "Dzień", "Typ treningu", "TSS", "Czas (min)", "Opis"]
    )

    day_names = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Ndz"]

    for week in plan.weeks:
        for day in week.days:
            if day.is_rest or day.workout is None:
                writer.writerow(
                    [
                        week.week_number,
                        f"{week.phase.emoji} {week.phase.value}",
                        day.date.isoformat(),
                        day_names[day.day_of_week],
                        "Odpoczynek",
                        0,
                        0,
                        "",
                    ]
                )
            else:
                w = day.workout
                writer.writerow(
                    [
                        week.week_number,
                        f"{week.phase.emoji} {week.phase.value}",
                        day.date.isoformat(),
                        day_names[day.day_of_week],
                        f"{w.workout_type.emoji} {w.workout_type.value}",
                        round(w.tss_target, 1),
                        w.duration_min,
                        w.description,
                    ]
                )

    return buf.getvalue()


def export_plan_weekly_summary(plan: TrainingPlan) -> str:
    """Export weekly summary CSV string.

    Columns: Tydzień, Faza, Data rozpoczęcia, TSS cel, TSS rzeczywisty, Dni treningowe
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(
        [
            "Tydzień",
            "Faza",
            "Data rozpoczęcia",
            "TSS cel",
            "TSS suma",
            "Dni treningowe",
        ]
    )

    for week in plan.weeks:
        writer.writerow(
            [
                week.week_number,
                f"{week.phase.emoji} {week.phase.value}",
                week.start_date.isoformat(),
                round(week.weekly_tss_target, 1),
                round(week.total_tss, 1),
                week.training_days,
            ]
        )

    return buf.getvalue()
