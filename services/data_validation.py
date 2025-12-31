"""
Data Validation Service

Handles DataFrame validation logic for uploaded training files.
"""

import pandas as pd
from typing import Tuple


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that DataFrame has minimum required structure.
    
    Checks for:
    - Non-empty DataFrame
    - At least one data column (watts, heartrate, cadence, smo2, power)
    - Minimum number of records (10)
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "Plik jest pusty lub nie udało się go wczytać."
    
    # Check for at least one data column
    data_cols = ['watts', 'heartrate', 'cadence', 'smo2', 'power']
    has_data = any(col in df.columns for col in data_cols)
    
    if not has_data:
        return False, f"Brak wymaganych kolumn danych. Oczekiwane: {data_cols}"
    
    if len(df) < 10:
        return False, f"Za mało danych ({len(df)} rekordów). Minimum: 10."
    
    return True, ""
