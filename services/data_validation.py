"""
Data Validation Service

Handles DataFrame validation logic for uploaded training files.
"""

import pandas as pd

from modules.config import Config
import logging
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that DataFrame has minimum required structure and valid data.

    Checks for:
    - Non-empty DataFrame
    - Required columns (e.g., 'time')
    - At least one data column (watts, heartrate, etc.)
    - Minimum number of records
    - Data integrity (timestamps monotonic, reasonable ranges)

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    warnings: list[str] = []

    # 1. Basic Structure
    if df is None or df.empty:
        return False, "Plik jest pusty lub nie udało się go wczytać."

    cols = df.columns

    # 2. Required Columns
    for req in Config.VALIDATION_REQUIRED_COLS:
        if req not in cols:
            return False, f"Brak wymaganej kolumny: '{req}'"

    # 3. Data Presence
    if not any(col in cols for col in Config.VALIDATION_DATA_COLS):
        return (
            False,
            f"Brak wymaganych kolumn danych. Oczekiwane przynajmniej jedna z: {Config.VALIDATION_DATA_COLS}",
        )

    # 4. Length Check
    if len(df) < Config.MIN_DF_LENGTH:
        return False, f"Za mało danych ({len(df)} rekordów). Minimum: {Config.MIN_DF_LENGTH}."

    # 5. Type & Integrity Checks

    # Time monotonicity
    if "time" in cols:
        if not pd.api.types.is_numeric_dtype(df["time"]):
            return False, "Kolumna 'time' musi być liczbowa."
        if df["time"].isnull().all():
            return False, "Kolumna 'time' zawiera same wartości puste (NaN)."

        # Check for monotonicity — time must be non-decreasing
        if not df["time"].is_monotonic_increasing:
            n_violations = (df["time"].diff() < 0).sum()
            if n_violations > len(df) * 0.05:
                return (
                    False,
                    f"Kolumna 'time' nie jest monotonicznie rosnąca "
                    f"({n_violations} naruszeń). Sprawdź poprawność pliku.",
                )

        # H11: Temporal bounds validation
        if pd.api.types.is_numeric_dtype(df["time"]):
            time_vals = df["time"].dropna()
            if len(time_vals) > 10:
                # Check sample rate (expect ~1Hz ±50%)
                time_diffs = time_vals.diff().dropna()
                median_dt = time_diffs.median()
                if median_dt > 0:
                    if median_dt > 3.0:
                        warnings.append(
                            f"Low sample rate: median interval {median_dt:.1f}s "
                            "(expected ~1s). Data may be downsampled."
                        )
                    # Check for gaps >60s
                    large_gaps = time_diffs[time_diffs > 60]
                    if len(large_gaps) > 0:
                        warnings.append(
                            f"Detected {len(large_gaps)} time gaps >60s. "
                            "File may contain merged sessions or pauses."
                        )

    # Type Validation - Convert or reject non-numeric data
    validation_failures = []

    # Watts validation — strict, no silent coercion (C5)
    if "watts" in cols:
        col_watts = "watts"
        if not pd.api.types.is_numeric_dtype(df[col_watts]):
            df = df.copy()
            # Handle EU locale decimal comma (M17)
            if df[col_watts].dtype == object:
                # Try comma→dot conversion first
                try:
                    df[col_watts] = df[col_watts].astype(str).str.replace(",", ".", regex=False)
                    df[col_watts] = pd.to_numeric(df[col_watts], errors="coerce")
                except (ValueError, TypeError):
                    pass

            if not pd.api.types.is_numeric_dtype(df[col_watts]):
                df[col_watts] = pd.to_numeric(df[col_watts], errors="coerce")

            # Strict: reject if >10% of values became NaN after conversion
            nan_pct = df[col_watts].isna().sum() / len(df) * 100
            if nan_pct > 10:
                return False, (
                    f"Power data quality too low: {nan_pct:.0f}% non-numeric values. "
                    "Check file format and decimal separator (dot vs comma)."
                )

            if df[col_watts].isna().all():
                return False, "All power values are non-numeric after conversion."

        max_w = df["watts"].max()
        if max_w > Config.VALIDATION_MAX_WATTS:
            validation_failures.append(
                f"Moc maksymalna ({max_w:.0f} W) przekracza limit ({Config.VALIDATION_MAX_WATTS} W). Sprawdź jednostki."
            )

    if "heartrate" in cols:
        if not pd.api.types.is_numeric_dtype(df["heartrate"]):
            try:
                df = df.copy()  # Prevent mutation of original
                df["heartrate"] = pd.to_numeric(df["heartrate"], errors="coerce")
                if df["heartrate"].isna().all():
                    return False, "Kolumna 'heartrate' zawiera nieprawidłowe dane (nie-liczbowe)."
            except (ValueError, TypeError) as e:
                logger.warning("Heartrate conversion failed: %s", e)
                return False, "Kolumna 'heartrate' zawiera nieprawidłowe dane (nie-liczbowe)."
        max_hr = df["heartrate"].max()
        if max_hr > Config.VALIDATION_MAX_HR:
            validation_failures.append(
                f"Tętno maksymalne ({max_hr:.0f} bpm) przekracza limit ({Config.VALIDATION_MAX_HR} bpm)."
            )

    if "cadence" in cols:
        if not pd.api.types.is_numeric_dtype(df["cadence"]):
            try:
                df = df.copy()  # Prevent mutation of original
                df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce")
                if df["cadence"].isna().all():
                    return False, "Kolumna 'cadence' zawiera nieprawidłowe dane (nie-liczbowe)."
            except (ValueError, TypeError) as e:
                logger.warning("Cadence conversion failed: %s", e)
                return False, "Kolumna 'cadence' zawiera nieprawidłowe dane (nie-liczbowe)."
        max_cad = df["cadence"].max()
        if max_cad > Config.VALIDATION_MAX_CADENCE:
            validation_failures.append(
                f"Kadencja ({max_cad:.0f} rpm) przekracza limit ({Config.VALIDATION_MAX_CADENCE} rpm)."
            )

    if validation_failures:
        return False, "Błędy walidacji danych:\n" + "\n".join(validation_failures)

    # H12: Cross-signal biomechanical envelope checks
    has_hr = "heartrate" in cols
    if "watts" in cols and has_hr:
        hr_col = next((c for c in ["heartrate", "hr", "heart_rate"] if c in cols), None)
        if hr_col:
            df_active = df[(df["watts"] > 200) & (df[hr_col] > 0)].copy()
            if len(df_active) > 30:
                avg_hr_high_power = df_active[hr_col].mean()
                avg_power = df_active["watts"].mean()
                # If high power but very low HR: likely power meter malfunction
                if avg_power > 300 and avg_hr_high_power < 100:
                    warnings.append(
                        f"Suspect data: avg power {avg_power:.0f}W with avg HR "
                        f"{avg_hr_high_power:.0f}bpm. Possible power meter malfunction."
                    )

    return True, ""
