
import pandas as pd
import numpy as np

from modules.calculations.thresholds import detect_vt_transition_zone


def create_mock_data():
    """Create mock data simulating a ramp test (Trend + Low Noise)"""
    # 20 minutes (1200s), 1s interval
    time = np.arange(0, 1200, 1)
    
    # Power ramp: 100W to 400W
    watts = 100 + (time / 1200) * 300
    
    true_ve = np.zeros_like(time, dtype=float)
    current_val = 20.0
    
    # Construct piecewise linear trend
    for t in range(len(time)):
        if t < 400:
            slope = 0.02
        elif t < 800:
            slope = 0.06
        else:
            slope = 0.16
        
        if t > 0:
            current_val += slope
        
        true_ve[t] = current_val

    # Low noise to verify precision
    noise = np.random.normal(0, 0.1, size=len(time)) 
    ve = true_ve + noise
        
    df = pd.DataFrame({
        'time': time,
        'watts': watts,
        'tymeventilation': ve,
        'hr': 100 + (time/1200)*80 
    })
    
    return df


def test_detection():
    """Test VT1/VT2 detection on clean mock data."""
    df = create_mock_data()
    vt1, vt2 = detect_vt_transition_zone(df)
    
    # Assertions for clean data
    assert vt1 is not None, "VT1 not detected on clean data"
    assert vt1.range_watts[0] > 180, f"VT1 start {vt1.range_watts[0]} too early"
    assert vt1.range_watts[1] < 220, f"VT1 end {vt1.range_watts[1]} too late"
    
    assert vt2 is not None, "VT2 not detected on clean data"
    assert vt2.range_watts[0] > 280, f"VT2 start {vt2.range_watts[0]} too early"
    assert vt2.range_watts[1] < 320, f"VT2 end {vt2.range_watts[1]} too late"
