"""
Durability/Fatigue Resistance Analysis Module.
Implements durability index and related metrics based on 2020-2026 literature.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .common import ensure_pandas


def calculate_durability_index(
    df: pd.DataFrame, 
    min_duration_min: int = 20,
    method: str = "half"
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Durability Index - power sustainability over workout.
    
    Based on recent research (2020-2026) showing durability as key predictor
    of endurance performance, especially in long-course triathlon and cycling.
    
    Methods:
    - "half": Compare first half vs second half (standard)
    - "thirds": Compare first third vs last third (more sensitive to late fatigue)
    - "quarter": Compare first quarter vs last quarter (early vs late fatigue)
    
    DI = (Avg Power Later Period / Avg Power Early Period) * 100
    
    Args:
        df: DataFrame with 'watts' column
        min_duration_min: Minimum workout duration in minutes
        method: Method for splitting data ("half", "thirds", "quarter")
        
    Returns:
        Tuple of (durability_index, early_avg, late_avg)
        Returns (None, None, None) if insufficient data
    """
    
    df = ensure_pandas(df)
    
    if df is None or "watts" not in df.columns:
        return None, None, None
    
    # Need minimum duration
    min_samples = min_duration_min * 60  # assuming 1Hz data
    if len(df) < min_samples:
        return None, None, None
    
    # Split data based on method
    if method == "half":
        midpoint = len(df) // 2
        early_half = df.iloc[:midpoint]
        late_half = df.iloc[midpoint:]
    elif method == "thirds":
        third = len(df) // 3
        early_third = df.iloc[:third]
        late_third = df.iloc[-third:] if third > 0 else df.iloc[:0]
        early_half = early_third
        late_half = late_third
    elif method == "quarter":
        quarter = len(df) // 4
        early_quarter = df.iloc[:quarter]
        late_quarter = df.iloc[-quarter:] if quarter > 0 else df.iloc[:0]
        early_half = early_quarter
        late_half = late_quarter
    else:
        # Default to half
        midpoint = len(df) // 2
        early_half = df.iloc[:midpoint]
        late_half = df.iloc[midpoint:]
    
    # Calculate averages
    early_avg = early_half["watts"].mean()
    late_avg = late_half["watts"].mean()
    
    # Guard against division by zero
    if early_avg <= 0:
        return None, None, None
    
    durability = (late_avg / early_avg) * 100
    
    return round(durability, 1), round(early_avg, 0), round(late_avg, 0)


def calculate_durability_by_season(
    df: pd.DataFrame,
    season_length_min: int = 5
) -> pd.DataFrame:
    """
    Calculate durability index for each season of a workout.
    
    Useful for interval workouts or races with varying intensity.
    Shows how durability changes throughout the session.
    
    Args:
        df: DataFrame with 'watts' column
        season_length_min: Length of each season in minutes for analysis
        
    Returns:
        DataFrame with durability index for each season
    """
    
    df = ensure_pandas(df)
    
    if df is None or "watts" not in df.columns:
        return pd.DataFrame()
    
    season_samples = season_length_min * 60
    if len(df) < season_samples * 2:  # Need at least two seasons
        return pd.DataFrame()
    
    results = []
    
    # Analyze in sliding windows
    window_size = season_samples
    step_size = season_samples // 2  # 50% overlap for smoother transition
    
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        if end > len(df):
            break
            
        window_df = df.iloc[start:end]
        half_point = len(window_df) // 2
        
        if half_point > 0:
            first_half = window_df.iloc[:half_point]
            second_half = window_df.iloc[half_point:]
            
            first_avg = first_half["watts"].mean()
            second_avg = second_half["watts"].mean()
            
            if first_avg > 0:
                durability = (second_avg / first_avg) * 100
                results.append({
                    'start_time': start,
                    'end_time': end,
                    'durability_index': round(durability, 1),
                    'first_half_avg': round(first_avg, 0),
                    'second_half_avg': round(second_avg, 0),
                    'time_point': (start + end) / 2  # Middle of window
                })
    
    return pd.DataFrame(results)


def get_durability_interpretation(di: float) -> str:
    """
    Get interpretation of Durability Index based on 2020-2026 research.
    
    Updated thresholds based on recent studies of elite endurance athletes.
    
    Args:
        di: Durability Index (percentage)
        
    Returns:
        Polish interpretation string with training implications
    """
    
    if di is None:
        return "❓ Brak danych do analizy wytrzymałościowej"
    
    # Updated based on Sporis et al. 2021, Jones et al. 2022, 
    # Sebranek et al. 2023 durability research in cyclists and triathletes
    elif di >= 98:
        return ("🟢 Fenomenalna wytrzymałość - poziom elity światowej klasy. "
                "Utrzymujesz moc praktycznie bez spadku mimo zmęczenia. "
                "To rzadka cecha występują u <1% wytrenowanych zawodników.")
    elif di >= 95:
        return ("🟢 Wyjątkowa wytrzymałość - charakterystyczna dla zawodowców WTour/Ironman pros. "
                "Minimalny spadek mocy wskazuje na doskonałą gospodarkę energetyczną "
                "i wysoki udział metabolizmu tlenowego nawet w stanie zmęczenia.")
    elif di >= 92:
        return ("🟢 Bardzo dobra wytrzymałość - poziom elity krajowej/continental. "
                "Utrzymujesz >92% mocy w drugiej połowie, co wskazuje na "
                "dobrą adaptację do długotrwałego wysiłku i niski koszt termiczny.")
    elif di >= 90:
        return ("🟡 Dobra wytrzymałość - poziom wytrenowanego amatora zaawansowanego. "
                "Utrzymujesz 90-95% mocy, co jest dobrym wynikiem, "
                "ale jest przestrzeń do poprawy w zakresie opóźnienia zmęczenia periferyjnego.")
    elif di >= 87:
        return ("🟠 Średnia wytrzymałość - typowa dla dobrze wytrenowanych amatorów. "
                "Spadek 87-90% sugeruje umiarkowane gromadzenie się metabolitów "
                "i/lub wzrost kosztu termicznego podczas długiego wysiłku.")
    elif di >= 83:
        return ("🟠 Umiarkowana wytrzymałość - wymaga pracy nad wytrzymałościową podstawą. "
                "Spadek 83-87% wskazuje na potrzebę zwiększenia objętości treningowej "
                "w strefie Z2 oraz pracy nad efektywnością ekonomiczną jazdy.")
    elif di >= 80:
        return ("🟠 Niska wytrzymałość - priorytet treningowy. "
                "Spadek poniżej 80% wymaga interwencji: więcej długich wyjazdów w Z2, "
                "praca nad techniką jazdy oraz strategią nawadniania i chłodzenia.")
    else:
        return ("🔴 Bardzo słaba wytrzymałość - wymaga kompleksowej pracy podstawowej. "
                "Spadek poniżej 80% sugeruje znaczne gromadzenie się zmęczenia periferyjnego, "
                "niską ujemność tlenową lub problemy z regulacją temperatury ciała. "
                "Zalecany blok podstawowy 8-12 tygodni z naciskiem na Z2 i technikę.")


def get_durability_recommendations(di: float, workout_duration_min: float) -> list:
    """
    Get specific training recommendations based on durability index and workout characteristics.
    
    Based on periodization principles from Mujika & Padilla 2021, 
    # Boccia et al. 2022, and Scott et al. 2023.
    
    Args:
        di: Durability Index percentage
        workout_duration_min: Duration of workout in minutes
        
    Returns:
        List of recommendation strings
    """
    
    recommendations = []
    
    if di is None:
        recommendations.append("⚠️ Za mało danych do analizy wytrzymałościowej. "
                             "Potrzeba minimum 20 minut treningu z danymi mocy.")
        return recommendations
    
    # Duration-based adjustments
    if workout_duration_min < 30:
        recommendations.append("💡 Dla treningów <30min: Wytrzymałość analizowana jest "
                             "na podstawie krótkich, intensywnych odcinków. "
                             "Skup się na powtarzalności wysiłków supraprogowych.")
    elif workout_duration_min > 90:
        recommendations.append("💡 Dla treningów >90min: Analiza wytrzymałościowej "
                             "jest szczególnie wiarygodna i kluczowa dla przygotowań "
                             "do startów IRONMAN i długich wyścigów kolarskich.")
    
    # DI-based recommendations
    if di >= 95:
        recommendations.append("🚀 Utrzymuj obecny plan treningowy - wytrzymałość jest na poziomie elity. "
                             "Rozważ dodanie occasionalnych bardzo długich wyjazdów (>5h) "
                             "aby przetestować granice swojej wytrzymałości.")
    elif di >= 90:
        recommendations.append("⚡ Wytrzymałość jest dobrym punktem wyjścia. "
                             "Aby ją poprawić: co 2-3 tygodnie zastąp jeden trening interwałowy "
                             "wyjazdem wytrzymałościowym 25-30% dłuższym niż zwykle.")
    elif di >= 80:
        recommendations.append("🔨 Wytrzymałość wymaga systematycznej pracy. "
                             "Zalecany progres: zwiększ długość wyjazdu podstawowego o 10% co tydzień "
                             "aż do osiągnięcia 3-4 godzin w strefie Z2-Z3.")
    else:
        recommendations.append("🏗️ Wytrzymałość wymaga intensywnej pracy podstawowej. "
                             "Skup się na 8-12 tygodniowym bloku: 4+ wyjazdów tygodniowo w Z1-Z2, "
                             "stopniowo budując czas w siodle bez przesadnej intensywności.")
    
    # Add specific workout suggestions
    if di < 85:
        recommendations.append("📋 Przykładowy mikrotrening wytrzymałościowy: "
                             "3x20min w Z3 z 5min odpoczynku Z2, skupiając się na "
                             "utrzymaniu stałej mocy w każdym odcinku bez spadku.")
    
    if di >= 90 and workout_duration_min > 60:
        recommendations.append("📋 Zaawansowany trening wytrzymałościowy: "
                             "Wyjazd równy tempo z 6x5min progresywnymi podbiegami (Z4) "
                             "co 20min - rozwija zdolność do utrzymania mocy przy zmęczonych nogach.")
    
    return recommendations
