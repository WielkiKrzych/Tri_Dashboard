"""Population reference data for cycling performance benchmarks."""

# Reference percentile tables for males (age 20-59)
# Each tuple: (percentile, value)
# Values are in W/kg for power metrics, mL/kg/min for VO2max, J/kg for W'

MALE_CP_WKG: dict[str, list[tuple[float, float]]] = {
    "20-29": [(5, 2.8), (10, 3.2), (25, 3.8), (50, 4.5), (75, 5.2), (90, 5.8), (95, 6.3)],
    "30-39": [(5, 2.6), (10, 3.0), (25, 3.5), (50, 4.2), (75, 4.9), (90, 5.5), (95, 6.0)],
    "40-49": [(5, 2.4), (10, 2.8), (25, 3.2), (50, 3.9), (75, 4.5), (90, 5.1), (95, 5.6)],
    "50-59": [(5, 2.2), (10, 2.5), (25, 3.0), (50, 3.5), (75, 4.1), (90, 4.6), (95, 5.1)],
}

FEMALE_CP_WKG: dict[str, list[tuple[float, float]]] = {
    "20-29": [(5, 2.2), (10, 2.6), (25, 3.1), (50, 3.7), (75, 4.3), (90, 4.8), (95, 5.3)],
    "30-39": [(5, 2.1), (10, 2.4), (25, 2.9), (50, 3.5), (75, 4.0), (90, 4.5), (95, 5.0)],
    "40-49": [(5, 1.9), (10, 2.2), (25, 2.7), (50, 3.2), (75, 3.7), (90, 4.2), (95, 4.7)],
    "50-59": [(5, 1.7), (10, 2.0), (25, 2.4), (50, 2.9), (75, 3.4), (90, 3.8), (95, 4.3)],
}

MALE_VO2MAX: dict[str, list[tuple[float, float]]] = {
    "20-29": [(5, 33), (10, 37), (25, 42), (50, 48), (75, 54), (90, 59), (95, 63)],
    "30-39": [(5, 31), (10, 35), (25, 40), (50, 45), (75, 51), (90, 56), (95, 60)],
    "40-49": [(5, 29), (10, 33), (25, 37), (50, 42), (75, 48), (90, 53), (95, 57)],
    "50-59": [(5, 26), (10, 30), (25, 34), (50, 39), (75, 44), (90, 49), (95, 53)],
}

FEMALE_VO2MAX: dict[str, list[tuple[float, float]]] = {
    "20-29": [(5, 28), (10, 32), (25, 37), (50, 42), (75, 48), (90, 53), (95, 57)],
    "30-39": [(5, 26), (10, 30), (25, 34), (50, 39), (75, 44), (90, 49), (95, 53)],
    "40-49": [(5, 24), (10, 28), (25, 32), (50, 37), (75, 42), (90, 47), (95, 51)],
    "50-59": [(5, 22), (10, 25), (25, 29), (50, 34), (75, 39), (90, 43), (95, 47)],
}

MALE_WPRIME_JKG: dict[str, list[tuple[float, float]]] = {
    "20-29": [(5, 120), (10, 150), (25, 190), (50, 240), (75, 290), (90, 340), (95, 370)],
    "30-39": [(5, 110), (10, 140), (25, 180), (50, 225), (75, 275), (90, 320), (95, 350)],
    "40-49": [(5, 100), (10, 130), (25, 165), (50, 210), (75, 255), (90, 300), (95, 330)],
    "50-59": [(5, 90), (10, 115), (25, 150), (50, 190), (75, 235), (90, 275), (95, 305)],
}

FEMALE_WPRIME_JKG: dict[str, list[tuple[float, float]]] = {
    "20-29": [(5, 95), (10, 120), (25, 155), (50, 200), (75, 245), (90, 285), (95, 315)],
    "30-39": [(5, 85), (10, 110), (25, 145), (50, 185), (75, 230), (90, 270), (95, 300)],
    "40-49": [(5, 75), (10, 100), (25, 130), (50, 170), (75, 210), (90, 250), (95, 280)],
    "50-59": [(5, 65), (10, 85), (25, 115), (50, 150), (75, 190), (90, 225), (95, 255)],
}


def get_age_bracket(age: int) -> str:
    """Map age to bracket string.

    Args:
        age: Athlete age in years.

    Returns:
        Age bracket string like '20-29'.
    """
    if age < 20:
        return "20-29"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "50-59"


def get_reference_table(metric: str, gender: str, age: int) -> list[tuple[float, float]]:
    """Get the reference percentile table for a metric/gender/age.

    Args:
        metric: 'cp_wkg', 'vo2max', or 'wprime_jkg'.
        gender: 'M' or 'F'.
        age: Athlete age in years.

    Returns:
        List of (percentile, value) tuples.
    """
    bracket = get_age_bracket(age)
    gender_key = "M" if gender.upper() in ("M", "MALE", "TRUE") else "F"

    tables = {
        "cp_wkg": MALE_CP_WKG if gender_key == "M" else FEMALE_CP_WKG,
        "vo2max": MALE_VO2MAX if gender_key == "M" else FEMALE_VO2MAX,
        "wprime_jkg": MALE_WPRIME_JKG if gender_key == "M" else FEMALE_WPRIME_JKG,
    }

    table = tables.get(metric, {})
    return table.get(bracket, table.get("30-39", []))


def interpolate_percentile(value: float, ref_table: list[tuple[float, float]]) -> float:
    """Linear interpolation to find percentile for a value.

    Args:
        value: Athlete's metric value.
        ref_table: List of (percentile, value) tuples, sorted by value.

    Returns:
        Estimated percentile (0-99).
    """
    if not ref_table:
        return 50.0

    sorted_table = sorted(ref_table, key=lambda x: x[1])

    if value <= sorted_table[0][1]:
        return max(1.0, sorted_table[0][0])
    if value >= sorted_table[-1][1]:
        return min(99.0, sorted_table[-1][0])

    for i in range(len(sorted_table) - 1):
        p1, v1 = sorted_table[i]
        p2, v2 = sorted_table[i + 1]
        if v1 <= value <= v2:
            t = (value - v1) / (v2 - v1) if v2 != v1 else 0.5
            return p1 + t * (p2 - p1)

    return 50.0


def classify_percentile(percentile: float) -> str:
    """Classify percentile into descriptive category.

    Args:
        percentile: Percentile value (0-100).

    Returns:
        Polish-language category string.
    """
    if percentile >= 95:
        return "Elita"
    elif percentile >= 90:
        return "Wybitny"
    elif percentile >= 75:
        return "Bardzo dobry"
    elif percentile >= 50:
        return "Powyżej średniej"
    elif percentile >= 25:
        return "Poniżej średniej"
    else:
        return "Wymaga poprawy"
