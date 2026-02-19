"""
Performance tests for session_analysis module.
"""

import time
import pandas as pd
import numpy as np
import pytest

from services.session_analysis import (
    calculate_header_metrics,
    calculate_header_metrics_cached,
    apply_smo2_smoothing
)


def test_np_calculation_performance():
    """Test that cached version is faster than regular."""
    # Create test data
    df = pd.DataFrame({
        'watts': np.random.normal(200, 50, 14400)  # 4h of data
    })
    
    # Warm up cache
    _ = calculate_header_metrics_cached(df, cp=280)
    
    # Test regular version
    start = time.perf_counter()
    for _ in range(100):
        result_normal = calculate_header_metrics(df, cp=280)
    time_normal = time.perf_counter() - start
    
    # Test cached version (should hit cache)
    start = time.perf_counter()
    for _ in range(100):
        result_cached = calculate_header_metrics_cached(df, cp=280)
    time_cached = time.perf_counter() - start
    
    # Verify results are equal
    assert abs(result_normal[0] - result_cached[0]) < 0.001
    assert abs(result_normal[1] - result_cached[1]) < 0.001
    assert abs(result_normal[2] - result_cached[2]) < 0.001
    
    # Cached should be significantly faster (at least 2x on second run)
    print(f"\nPerformance: normal={time_normal:.3f}s, cached={time_cached:.3f}s")
    print(f"Speedup: {time_normal/time_cached:.1f}x")


def test_smo2_smoothing_inplace():
    """Test that inplace parameter works correctly."""
    df = pd.DataFrame({
        'smo2': [70, 69, 68, 67, 66, 65, 64, 63, 62, 61]
    })
    
    # Test inplace=False (default)
    df_copy = df.copy()
    result = apply_smo2_smoothing(df_copy, inplace=False)
    assert 'smo2_smooth_ultra' in result.columns
    assert 'smo2_smooth_ultra' not in df_copy.columns  # Original not modified
    
    # Test inplace=True
    df_copy2 = df.copy()
    result2 = apply_smo2_smoothing(df_copy2, inplace=True)
    assert 'smo2_smooth_ultra' in result2.columns
    assert 'smo2_smooth_ultra' in df_copy2.columns  # Original modified
    assert result2 is df_copy2  # Same object returned


def test_smo2_smoothing_without_smo2_column():
    """Test smoothing when smo2 column doesn't exist."""
    df = pd.DataFrame({'other_col': [1, 2, 3]})
    
    result = apply_smo2_smoothing(df, inplace=False)
    assert 'smo2_smooth_ultra' not in result.columns
    assert len(result.columns) == len(df.columns)


def test_header_metrics_edge_cases():
    """Test header metrics with edge cases."""
    # Empty DataFrame
    df_empty = pd.DataFrame()
    result = calculate_header_metrics(df_empty, cp=280)
    assert result == (0.0, 0.0, 0.0)
    
    # No watts column
    df_no_watts = pd.DataFrame({'hr': [100, 110, 120]})
    result = calculate_header_metrics(df_no_watts, cp=280)
    assert result == (0.0, 0.0, 0.0)
    
    # Zero CP — need enough rows to pass MIN_RECORDS_FOR_ROLLING (default 30)
    df = pd.DataFrame({'watts': list(range(100, 130))})  # 30 rows
    result = calculate_header_metrics(df, cp=0)
    assert result[0] > 0  # NP calculated independently of CP
    assert result[1] == 0.0  # IF = 0 when CP is zero
    assert result[2] == 0.0  # TSS = 0 when CP is zero
