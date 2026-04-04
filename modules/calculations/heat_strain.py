"""
Heat Strain Index Analysis Module.
Enhanced analysis of physiological strain during heat exposure based on 2020-2026 research.
"""

from typing import Union, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .common import ensure_pandas
from .thermal import calculate_heat_strain_index as calculate_psi


def calculate_heat_strain_index_enhanced(
    df_pl: Union[pd.DataFrame, Any],
    resting_hr: float = 0.0,
    hr_max: float = 0.0,
    baseline_core_temp: float = 0.0,
    acclimatization_days: int = 0,
    weight_kg: float = 75.0,
    height_cm: float = 175.0,
    sex: str = "male",
    clothing_factor: float = 1.0,
    solar_radiation: float = 0.0,
    wind_speed: float = 0.0,
    relative_humidity: float = 50.0
) -> pd.DataFrame:
    """
    Calculate enhanced Heat Strain Index with environmental corrections.
    
    Extends the basic PSI with:
    - Environmental corrections (WBGT approximation)
    - Individual factors (sex, body size, acclimatization)
    - Clothing and solar radiation adjustments
    - Strain accumulation over time
    
    Args:
        df_pl: DataFrame with 'core_temperature_smooth' and 'heartrate_smooth'
        resting_hr: Resting heart rate [bpm]
        hr_max: Maximum heart rate [bpm]
        baseline_core_temp: Baseline core temperature [°C]
        acclimatization_days: Days of heat acclimatization
        weight_kg: Body weight [kg]
        height_cm: Height [cm]
        sex: Biological sex ("male" or "female")
        clothing_factor: Clothing insulation factor (0.5-1.5)
        solar_radiation: Solar radiation [W/m²]
        wind_speed: Wind speed [m/s]
        relative_humidity: Relative humidity [%]
        
    Returns:
        DataFrame with enhanced HSI metrics
    """
    df = ensure_pandas(df_pl)
    
    if df is None:
        return df
    
    # Calculate base PSI
    df = calculate_psi(
        df_pl=df,
        resting_hr=resting_hr,
        hr_max=hr_max,
        baseline_core_temp=baseline_core_temp,
        acclimatization_days=acclimatization_days
    )
    
    # Calculate Body Surface Area (DuBois formula)
    if 'weight_kg' not in df.columns:
        df['weight_kg'] = weight_kg
    if 'height_cm' not in df.columns:
        df['height_cm'] = height_cm
        
    # BSA = 0.007184 × weight^0.425 × height^0.725
    df['body_surface_area'] = 0.007184 * \
        (df['weight_kg'] ** 0.425) * \
        (df['height_cm'] ** 0.725)
    
    # Calculate metabolic rate approximation from heart rate
    # Using simplified HR-VO2 relationship
    if 'heartrate_smooth' in df.columns:
        # %HRR to %VO2max conversion (simplified)
        if hr_max > resting_hr and resting_hr > 0:
            hrr = hr_max - resting_hr
            df['vo2_percent'] = ((df['heartrate_smooth'] - resting_hr) / hrr) * 100
            # Approximate METs from %VO2max (1 MET = 3.5 ml/kg/min)
            df['mets'] = 1 + (df['vo2_percent'] * 0.01 * 9)  # Rough approximation
        else:
            df['mets'] = 3.0  # Default moderate activity
    else:
        df['mets'] = 3.0
    
    # Calculate heat production (W/m²)
    # Assuming 1 MET = 58.2 W/m²
    df['heat_production'] = df['mets'] * 58.2
    
    # Environmental heat gain/loss approximation
    # Simplified WBGT (Wet Bulb Globe Temperature) approximation
    if all(col in df.columns for col in ['temperature', 'humidity']):
        # Use actual measurements if available
        wbgt = 0.7 * df['wet_bulb_temp'] + 0.2 * df['globe_temp'] + 0.1 * df['temperature']
    else:
        # Approximate from relative humidity and temperature (if available)
        if 'temperature' in df.columns:
            # Simplified WBGT approximation
            wbgt = df['temperature'] * 0.7 + relative_humidity * 0.3
            # Adjust for solar radiation and wind
            wbgt += solar_radiation * 0.01 - wind_speed * 0.1
        else:
            # Default to moderate conditions
            wbgt = 25.0  # °C
    
    df['wbgt_approx'] = wbgt
    
    # Heat dissipation capacity (function of BSA, wind, etc.)
    # Increased wind increases convective cooling
    # Increased clothing decreases evaporative cooling
    df['heat_dissipation'] = (
        df['body_surface_area'] * 
        (6.0 + wind_speed * 2.5) *  # Convection coefficient
        (1.0 / clothing_factor)   # Clothing resistance
    )
    
    # Net heat balance (positive = heat storage)
    df['net_heat_balance'] = df['heat_production'] - df['heat_dissipation']
    
    # Cumulative heat strain (simplified)
    # In reality, this would involve complex thermoregulatory modeling
    if 'hsi' in df.columns:
        # Time-weighted accumulation of strain
        df['hsi_cumulative'] = df['hsi'].expanding().mean()
        # Peak strain during session
        df['hsi_peak'] = df['hsi'].expanding().max()
        # Strain duration above threshold
        df['hsi_above_threshold'] = (df['hsi'] > 4).astype(int)  # Moderate strain threshold
        df['hsi_duration_mod'] = df['hsi_above_threshold'].expanding().sum()
        df['hsi_above_high'] = (df['hsi'] > 7).astype(int)  # High strain threshold
        df['hsi_duration_high'] = df['hsi_above_high'].expanding().sum()
    
    # Individual risk factors
    # Sex differences in thermoregulation
    sex_factor = 1.0
    if sex.lower() == "female":
        # Females typically have lower sweat rates but higher core temp threshold
        sex_factor = 0.95  # Slightly lower effective strain
    
    # Acclimatization factor
    acclim_factor = 1.0
    if acclimatization_days >= 10:
        # Significant acclimatization reduces strain
        acclim_factor = max(0.7, 1.0 - (acclimatization_days - 10) * 0.02)
    elif acclimatization_days > 0:
        # Partial acclimatization
        acclim_factor = 1.0 - (acclimatization_days * 0.015)
    
    df['hsi_acclim_adjusted'] = df.get('hsi', 0) * sex_factor * acclim_factor
    
    # Heat strain categories
    if 'hsi' in df.columns:
        df['hsi_category'] = pd.cut(
            df['hsi'],
            bins=[0, 3, 4, 6, 8, 10, float('inf')],
            labels=['None', 'Low', 'Moderate', 'High', 'Very High', 'Extreme'],
            include_lowest=True
        )
    
    return df


def calculate_heat_strain_summary(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics from heat strain analysis.
    
    Args:
        df: DataFrame with heat strain metrics
        
    Returns:
        Dictionary with heat strain summary
    """
    df = ensure_pandas(df)
    
    if df is None or 'hsi' not in df.columns:
        return {}
    
    summary = {
        'peak_hsi': 0.0,
        'mean_hsi': 0.0,
        'time_above_threshold': {},
        'strain_duration': {},
        'risk_level': 'Low',
        'recommendations': []
    }
    
    # Basic statistics
    if len(df) > 0:
        summary['peak_hsi'] = round(df['hsi'].max(), 1)
        summary['mean_hsi'] = round(df['hsi'].mean(), 1)
        
        # Time above thresholds
        thresholds = [2, 3, 4, 6, 8]  # Low, Moderate, High, Very High, Extreme
        labels = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
        
        for thresh, label in zip(thresholds, labels):
            time_above = (df['hsi'] > thresh).sum()
            summary['time_above_threshold'][label] = int(time_above)
            
            if len(df) > 0:
                percentage = (time_above / len(df)) * 100
                summary['strain_duration'][label] = round(percentage, 1)
    
    # Determine overall risk level
    peak_hsi = summary['peak_hsi']
    if peak_hsi >= 9:
        summary['risk_level'] = 'Extreme'
    elif peak_hsi >= 8:
        summary['risk_level'] = 'Very High'
    elif peak_hsi >= 6:
        summary['risk_level'] = 'High'
    elif peak_hsi >= 4:
        summary['risk_level'] = 'Moderate'
    elif peak_hsi >= 2:
        summary['risk_level'] = 'Low'
    else:
        summary['risk_level'] = 'None'
    
    # Generate recommendations
    summary['recommendations'] = generate_heat_strain_recommendations(summary)
    
    return summary


def generate_heat_strain_recommendations(summary: dict) -> list:
    """
    Generate heat strain mitigation recommendations.
    
    Args:
        summary: Heat strain summary from calculate_heat_strain_summary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    risk_level = summary.get('risk_level', 'Low')
    peak_hsi = summary.get('peak_hsi', 0)
    mean_hsi = summary.get('mean_hsi', 0)
    strain_duration = summary.get('strain_duration', {})
    
    if risk_level == 'None':
        recommendations.append("✅ Brak wykrytego obciążenia cieplnego. Warunki są bezpieczne.")
        return recommendations
    
    # Risk-based recommendations
    if risk_level in ['Low', 'Moderate']:
        recommendations.append(
            f"⚠️ Umiarkowane obciążenie cieplne (PSI: {peak_hsi:.1f}). "
            "Monitoruj objawy przegrzania i zwiększ nawodnienie."
        )
    elif risk_level in ['High', 'Very High']:
        recommendations.append(
            f"🔥 Wysokie obciążenie cieplne (PSI: {peak_hsi:.1f}). "
            "Rozważ przerwanie treningu lub znaczące zmniejszenie intensywności."
        )
    elif risk_level == 'Extreme':
        recommendations.append(
            f"🚨 Ekstremalne obciążenie cieplne (PSI: {peak_hsi:.1f}). "
            "NATychmiast przerwij trening i schłodź organizm. Ryzyko udaru cieplnego!"
        )
    
    # Duration-based recommendations
    moderate_time = strain_duration.get('Moderate', 0)
    high_time = strain_duration.get('High', 0)
    very_high_time = strain_duration.get('Very High', 0)
    
    if moderate_time > 30:  # More than 30% of time in moderate strain
        recommendations.append(
            f"⏱️ {moderate_time:.1f}% czasu w strefie umiarkowanego obciążenia. "
            "Rozważ interwałowe podejście z okresami chłodzenia."
        )
    
    if high_time > 10:  # More than 10% of time in high strain
        recommendations.append(
            f"🕒 {high_time:.1f}% czasu w strefie wysokiego obciążenia. "
            "To może prowadzić do akumulacji zmęczenia cieplnego."
        )
    
    # Hydration recommendations based on strain
    if mean_hsi > 4:
        recommendations.append(
            "💧 Zwiększone obciążenie cieplne wymaga większej uwagi na nawodnienie. "
            "Planuj 500-750ml płynów na godzinę treningu, więcej jeśli pocisz się intensywnie."
        )
    
    # Acclimatization recommendations
    if peak_hsi > 5 and 'acclimatization_days' not in summary:
        recommendations.append(
            "🌡️ Rozważ proces aklimatyzacji cieplnej: "
            "10-14 dni stopniowego zwiększania czasu treningu w cieple "
            "poprawi tolerancję na wysokie temperatury."
        )
    
    # Cooling strategies
    if peak_hsi > 6:
        recommendations.append(
            "❄️ Rozważ strategie chłodzenia podczas treningu: "
            "chłodzące kamizelki, zimne okłady na szyję, "
            "lub przerywanie treningu dla schłodzenia."
        )
    
    # Timing recommendations
    if peak_hsi > 5:
        recommendations.append(
            "🕐 Planuj treningi na chłodniejsze części dnia (rano lub wieczorem) "
            "aby zmniejszyć obciążenie cieplne."
        )
    
    if not recommendations:
        recommendations.append(
            "✅ Obciążenie cieplne jest na akceptowalnym poziomie. "
            "Kontynuuj obecny plan treningowy z odpowiednim nawodnieniem."
        )
    
    return recommendations


def get_heat_strain_color_mapping() -> dict:
    """
    Get color mapping for heat strain levels.
    
    Returns:
        Dictionary mapping strain levels to colors
    """
    return {
        "None": "#E8F5E8",      # Light green
        "Low": "#C8E6C9",       # Green
        "Moderate": "#FFF9C4",  # Light yellow
        "High": "#FFE082",      # Yellow
        "Very High": "#FFCC80", # Orange
        "Extreme": "#FF8A80"    # Red-Orange
    }


if __name__ == "__main__":
    # For testing purposes
    pass
