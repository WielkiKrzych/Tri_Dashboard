# Tests configuration for Tri_Dashboard
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_power_df():
    """Sample cycling data with power, HR, cadence."""
    np.random.seed(42)
    n = 600  # 10 minutes of data
    
    return pd.DataFrame({
        'time': np.arange(n, dtype=float),
        'watts': np.random.normal(200, 30, n).clip(0, 400),
        'heartrate': np.random.normal(140, 10, n).clip(60, 200),
        'cadence': np.random.normal(90, 5, n).clip(60, 120),
        'smo2': np.random.normal(60, 10, n).clip(20, 95),
    })


@pytest.fixture
def sample_power_df_with_smooth(sample_power_df):
    """Sample data with smoothed columns."""
    df = sample_power_df.copy()
    df['watts_smooth'] = df['watts'].rolling(30, min_periods=1).mean()
    df['heartrate_smooth'] = df['heartrate'].rolling(30, min_periods=1).mean()
    df['cadence_smooth'] = df['cadence'].rolling(30, min_periods=1).mean()
    df['time_min'] = df['time'] / 60
    return df


@pytest.fixture
def sample_hrv_df():
    """Sample HRV data with R-R intervals."""
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        'time': np.arange(n, dtype=float),
        'rr': np.random.normal(800, 50, n).clip(600, 1200),  # R-R in ms
    })


@pytest.fixture
def cp_value():
    """Standard CP value for testing."""
    return 280  # Watts


@pytest.fixture
def w_prime_value():
    """Standard W' value for testing."""
    return 20000  # Joules
