"""
Training Distribution / Time-in-Zone Analysis Module.
Implements comprehensive time-in-zone analysis for power, heart rate, and SmO2.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from .common import ensure_pandas
from .power import calculate_power_zones_time


def calculate_training_distribution(
    df: pd.DataFrame,
    cp: float,
    hr_max: Optional[float] = None,
    hr_rest: Optional[float] = None,
    smo2_min: Optional[float] = None,
    smo2_max: Optional[float] = None,
    zones_dict: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive training distribution across multiple modalities.
    
    Args:
        df: DataFrame with 'watts', 'hr', and/or 'smo2' columns
        cp: Critical Power in watts
        hr_max: Maximum heart rate (for HR zone calculation)
        hr_rest: Resting heart rate (for HRR calculation)
        smo2_min: Minimum SmO2 value (for SmO2 zone calculation)
        smo2_max: Maximum SmO2 value (for SmO2 zone calculation)
        zones_dict: Optional custom zone definitions
        
    Returns:
        Dictionary containing time-in-zone analysis for all available metrics
    """
    df = ensure_pandas(df)
    
    if df is None:
        return {}
    
    results = {
        'power': {},
        'heart_rate': {},
        'smO2': {},
        'summary': {}
    }
    
    # Power zones analysis
    if 'watts' in df.columns and cp > 0:
        results['power'] = calculate_power_zones_time(df, cp, zones_dict)
        results['power']['total_seconds'] = len(df)
        results['power']['cp_used'] = cp
    
    # Heart rate zones analysis
    if 'hr' in df.columns and hr_max is not None and hr_max > 0:
        results['heart_rate'] = calculate_hr_zones_time(
            df, hr_max, hr_rest, zones_dict
        )
        results['heart_rate']['total_seconds'] = len(df)
        results['heart_rate']['hr_max_used'] = hr_max
        if hr_rest is not None:
            results['heart_rate']['hr_rest_used'] = hr_rest
    
    # SmO2 zones analysis
    if 'smo2' in df.columns and smo2_min is not None and smo2_max is not None and smo2_max > smo2_min:
        results['smO2'] = calculate_smo2_zones_time(
            df, smo2_min, smo2_max, zones_dict
        )
        results['smO2']['total_seconds'] = len(df)
        results['smO2']['smo2_range_used'] = (smo2_min, smo2_max)
    
    # Calculate summary statistics
    results['summary'] = calculate_training_summary(results)
    
    return results


def calculate_hr_zones_time(
    df: pd.DataFrame,
    hr_max: float,
    hr_rest: Optional[float] = None,
    zones_dict: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, int]:
    """
    Calculate time spent in heart rate zones.
    
    Args:
        df: DataFrame with 'hr' column
        hr_max: Maximum heart rate
        hr_rest: Resting heart rate (optional, for HRR calculation)
        zones_dict: Optional custom zone definitions
        
    Returns:
        Dict mapping zone name to seconds spent
    """
    df = ensure_pandas(df)
    
    if 'hr' not in df.columns or hr_max <= 0:
        return {}
    
    # Use heart rate reserve if resting HR provided, otherwise use absolute HR
    if hr_rest is not None and hr_max > hr_rest:
        hrr = hr_max - hr_rest
        # Calculate %HRR for each data point
        hr_values = ((df['hr'] - hr_rest) / hrr * 100).fillna(0)
        # Zones are in %HRR
        if zones_dict is None:
            zones = {
                "Z1 Very Light": (0, 30),      # <30% HRR
                "Z2 Light": (30, 40),          # 30-40% HRR
                "Z3 Moderate": (40, 60),       # 40-60% HRR
                "Z4 Hard": (60, 80),           # 60-80% HRR
                "Z5 Very Hard": (80, 100),     # 80-100% HRR
            }
        else:
            zones = zones_dict
    else:
        # Use absolute %HRmax
        hr_values = (df['hr'] / hr_max * 100).fillna(0)
        if zones_dict is None:
            zones = {
                "Z1 Very Light": (0, 50),      # <50% HRmax
                "Z2 Light": (50, 60),          # 50-60% HRmax
                "Z3 Moderate": (60, 70),       # 60-70% HRmax
                "Z4 Hard": (70, 80),           # 70-80% HRmax
                "Z5 Very Hard": (80, 90),      # 80-90% HRmax
                "Z6 Maximum": (90, 100),       # 90-100% HRmax
            }
        else:
            zones = zones_dict
    
    results = {}
    
    for zone_name, (low_pct, high_pct) in zones.items():
        low_value = low_pct  # Already in percentage
        high_value = high_pct  # Already in percentage
        
        mask = (hr_values >= low_value) & (hr_values < high_value)
        seconds_in_zone = mask.sum()
        results[zone_name] = int(seconds_in_zone)
    
    return results


def calculate_smo2_zones_time(
    df: pd.DataFrame,
    smo2_min: float,
    smo2_max: float,
    zones_dict: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, int]:
    """
    Calculate time spent in SmO2 zones.
    
    Args:
        df: DataFrame with 'smo2' column
        smo2_min: Minimum SmO2 value
        smo2_max: Maximum SmO2 value
        zones_dict: Optional custom zone definitions
        
    Returns:
        Dict mapping zone name to seconds spent
    """
    df = ensure_pandas(df)
    
    if 'smo2' not in df.columns or smo2_max <= smo2_min:
        return {}
    
    # Normalize SmO2 to 0-100% scale
    smo2_range = smo2_max - smo2_min
    if smo2_range > 0:
        smo2_normalized = ((df['smo2'] - smo2_min) / smo2_range * 100).fillna(0)
    else:
        smo2_normalized = pd.Series(0, index=df.index)
    
    if zones_dict is None:
        # SmO2 zones: higher values = better oxygenation
        zones = {
            "Very High": (80, 100),    # 80-100% normalized SmO2
            "High": (60, 80),          # 60-80% normalized SmO2
            "Moderate": (40, 60),      # 40-60% normalized SmO2
            "Low": (20, 40),           # 20-40% normalized SmO2
            "Very Low": (0, 20),       # 0-20% normalized SmO2
        }
    else:
        zones = zones_dict
    
    results = {}
    
    for zone_name, (low_pct, high_pct) in zones.items():
        low_value = low_pct  # Already in percentage
        high_value = high_pct  # Already in percentage
        
        mask = (smo2_normalized >= low_value) & (smo2_normalized < high_value)
        seconds_in_zone = mask.sum()
        results[zone_name] = int(seconds_in_zone)
    
    return results


def calculate_training_summary(training_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics from training distribution data.
    
    Args:
        training_data: Output from calculate_training_distribution
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_workout_time_min': 0,
        'primary_zone': {},
        'zone_balance_score': 0,
        'intensity_distribution': {},
        'recommendations': []
    }
    
    # Calculate total workout time from available data
    total_seconds = 0
    for modality in ['power', 'heart_rate', 'smO2']:
        if modality in training_data and 'total_seconds' in training_data[modality]:
            total_seconds = training_data[modality]['total_seconds']
            break
    
    if total_seconds > 0:
        summary['total_workout_time_min'] = round(total_seconds / 60, 1)
    
    # Find primary zone (zone with most time spent) for each modality
    for modality in ['power', 'heart_rate', 'smO2']:
        if modality in training_data and training_data[modality]:
            # Remove non-zone entries
            zone_data = {k: v for k, v in training_data[modality].items() 
                        if k not in ['total_seconds', 'cp_used', 'hr_max_used', 'hr_rest_used', 'smo2_range_used']}
            
            if zone_data:
                primary_zone = max(zone_data, key=zone_data.get)
                primary_zone_time = zone_data[primary_zone]
                primary_zone_pct = round((primary_zone_time / total_seconds) * 100, 1) if total_seconds > 0 else 0
                
                summary['primary_zone'][modality] = {
                    'zone': primary_zone,
                    'time_seconds': primary_zone_time,
                    'percentage': primary_zone_pct
                }
    
    # Calculate intensity distribution (time in zones <VT1, VT1-VT2, >VT2 equivalent)
    # This is a simplified version - in practice would use actual VT1/VT2
    if 'power' in training_data and training_data['power']:
        power_data = training_data['power']
        zone_times = {k: v for k, v in power_data.items() 
                     if k not in ['total_seconds', 'cp_used']}
        
        if zone_times and total_seconds > 0:
            # Approximate: Z1+Z2 = Easy, Z3+Z4 = Moderate, Z5+Z6 = Hard
            easy_time = sum(v for k, v in zone_times.items() 
                           if 'Z1' in k or 'Z2' in k or 'Recovery' in k or 'Endurance' in k)
            moderate_time = sum(v for k, v in zone_times.items() 
                               if 'Z3' in k or 'Z4' in k or 'Tempo' in k or 'Threshold' in k)
            hard_time = sum(v for k, v in zone_times.items() 
                           if 'Z5' in k or 'Z6' in k or 'VO2' in k or 'Anaerobic' in k)
            
            summary['intensity_distribution'] = {
                'easy_percent': round((easy_time / total_seconds) * 100, 1),
                'moderate_percent': round((moderate_time / total_seconds) * 100, 1),
                'hard_percent': round((hard_time / total_seconds) * 100, 1)
            }
    
    # Calculate zone balance score (how evenly distributed across zones)
    # Higher score = more evenly distributed (ideal for base training)
    # Lower score = polarized/highly concentrated (ideal for race prep)
    if 'power' in training_data and training_data['power']:
        power_data = training_data['power']
        zone_times = [v for k, v in power_data.items() 
                     if k not in ['total_seconds', 'cp_used'] and isinstance(v, (int, float))]
        
        if len(zone_times) > 1 and total_seconds > 0:
            # Calculate coefficient of variation of zone times
            mean_time = np.mean(zone_times)
            if mean_time > 0:
                std_time = np.std(zone_times)
                cv = std_time / mean_time
                # Convert to 0-100 score where 100 = perfectly even distribution
                summary['zone_balance_score'] = max(0, min(100, 100 * (1 - cv)))
    
    # Generate recommendations based on distribution
    summary['recommendations'] = generate_training_recommendations(summary)
    
    return summary


def generate_training_recommendations(summary: Dict[str, Any]) -> List[str]:
    """
    Generate training recommendations based on training distribution analysis.
    
    Args:
        summary: Training summary from calculate_training_summary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    total_time = summary.get('total_workout_time_min', 0)
    intensity_dist = summary.get('intensity_distribution', {})
    zone_balance = summary.get('zone_balance_score', 50)
    
    if total_time == 0:
        recommendations.append("⚠️ Za mało danych do analizy rozkładu treningowego.")
        return recommendations
    
    # Intensity distribution recommendations
    easy_pct = intensity_dist.get('easy_percent', 0)
    moderate_pct = intensity_dist.get('moderate_percent', 0)
    hard_pct = intensity_dist.get('hard_percent', 0)
    
    if easy_pct < 70 and total_time > 60:  # For longer workouts
        recommendations.append(
            f"🔄 Rozważ zwiększenie części łatwej (Z1-Z2) z {easy_pct:.1f}% do 70-80% "
            "dla lepszej regeneracji i adaptacji tlenowej."
        )
    elif easy_pct > 85:
        recommendations.append(
            f"⚡ Część łatwa stanowi {easy_pct:.1f}% treningu. "
            "Rozważ dodanie większej ilości pracy w Z3-Z4 dla zwiększenia progu mleczanowego."
        )
    
    if hard_pct > 20 and total_time > 90:
        recommendations.append(
            f"💪 Intensywna część (Z5-Z6) wynosi {hard_pct:.1f}%. "
            "Upewnij się, że masz odpowiednią regenerację między ciężkimi odcinkami."
        )
    
    # Zone balance recommendations
    if zone_balance < 30:
        recommendations.append(
            f"🎯 Bardzo nierównomierny rozkład stref (score: {zone_balance:.0f}/100). "
            "Trening jest silnie spolaryzowany. Dobrze dla przygotowań wyścigowych, "
            "ale może prowadzić do przetrenowania przy długotrwałym stosowaniu."
        )
    elif zone_balance > 70:
        recommendations.append(
            f"⚖️ Bardzo równomierny rozkład stref (score: {zone_balance:.0f}/100). "
            "Idealny dla okresu podstawowego i regeneracji aktywnej."
        )
    
    # Primary zone recommendations
    primary_power = summary.get('primary_zone', {}).get('power', {})
    if primary_power:
        zone_name = primary_power.get('zone', '')
        zone_pct = primary_power.get('percentage', 0)
        
        if 'Z1' in zone_name or 'Recovery' in zone_name:
            if zone_pct > 50:
                recommendations.append(
                    f"🚴‍♂️ Głównie strefa regeneracyjna ({zone_pct:.1f}%). "
                    "Rozważ włączenie interwałów tempa lub podbiegów dla bodźca treningowego."
                )
        elif 'Z5' in zone_name or 'Anaerobic' in zone_name or 'VO2' in zone_name:
            if zone_pct > 40:
                recommendations.append(
                    f"🔥 Dominuje strefa beztlenowa ({zone_pct:.1f}%). "
                    "Taki rozkład jest typowy dla wyścigów krytycznych lub sprintów. "
                    "Uważaj na akumulację zmęczenia."
                )
    
    # Add general recommendations based on workout duration
    if total_time < 30:
        recommendations.append(
            "⏱️ Krótkie treningi (<30min): Skup się na jakości i intensywności. "
            "Rozważ pracę nad techniką i neuromuskularną koordynacją."
        )
    elif total_time > 120:
        recommendations.append(
            "⏳ Długie treningi (>2h): Zwróć uwagę na nawodnienie, odżywianie i utrzymanie koncentracji. "
            "Rozważ podział na sekcje z różnymi celami treningowymi."
        )
    
    if not recommendations:
        recommendations.append(
            "✅ Rozkład treningowy wygląda zrównoważony i odpowiedni do celów treningowych. "
            "Kontynuuj obecny plan i monitoruj postępy."
        )
    
    return recommendations


def get_zone_color_mapping() -> Dict[str, str]:
    """
    Get standard color mapping for training zones.
    
    Returns:
        Dictionary mapping zone names to colors
    """
    return {
        "Z1 Very Light": "#E8F4FD",  # Light blue
        "Z1 Recovery": "#E8F4FD",
        "Z2 Light": "#BBDEFB",       # Blue
        "Z2 Endurance": "#BBDEFB",
        "Z3 Moderate": "#90CAF9",    # Light blue
        "Z3 Tempo": "#90CAF9",
        "Z4 Hard": "#64B5F6",        # Blue
        "Z4 Threshold": "#64B5F6",
        "Z5 VO2max": "#42A5F5",      # Darker blue
        "Z5 Very Hard": "#42A5F5",
        "Z6 Anaerobic": "#2196F3",   # Dark blue
        "Z6 Maximum": "#2196F3",
        "Very High": "#C8E6C9",      # Light green
        "High": "#A5D6A7",           # Green
        "Moderate": "#81C784",       # Light green
        "Low": "#66BB6A",            # Green
        "Very Low": "#43A047",       # Darker green
    }


if __name__ == "__main__":
    # For testing purposes
    pass
