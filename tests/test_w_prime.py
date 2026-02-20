"""Tests for modules/calculations/w_prime.py — W' Balance calculations."""

import pytest
import numpy as np
import pandas as pd
from modules.calculations.w_prime import (
    calculate_w_prime_fast,
    _calculate_w_prime_balance_cached,
    calculate_w_prime_balance,
    get_recovery_recommendation,
    estimate_w_prime_reconstitution,
)
from modules.utils import _serialize_df_to_parquet_bytes


# =========================================================================
# calculate_w_prime_fast (Numba JIT)
# =========================================================================

class TestCalculateWPrimeFast:
    def test_below_cp_stays_at_max(self):
        """Power below CP → W' should stay at capacity."""
        n = 100
        watts = np.full(n, 150.0)  # Below CP
        time = np.arange(n, dtype=np.float64)
        cp, w_prime = 200.0, 20000.0

        result = calculate_w_prime_fast(watts, time, cp, w_prime)
        assert result[-1] == pytest.approx(w_prime)

    def test_above_cp_depletes(self):
        """Power above CP → W' should deplete."""
        n = 100
        watts = np.full(n, 300.0)  # Above CP=200
        time = np.arange(n, dtype=np.float64)
        cp, w_prime = 200.0, 20000.0

        result = calculate_w_prime_fast(watts, time, cp, w_prime)
        assert result[-1] < w_prime
        # After 100s at 300W with CP=200, depletion = 100*100 = 10000J
        assert result[-1] == pytest.approx(w_prime - 100 * 100, abs=200)

    def test_never_below_zero(self):
        """W' should never go below 0."""
        n = 500
        watts = np.full(n, 400.0)  # Way above CP
        time = np.arange(n, dtype=np.float64)
        cp, w_prime = 200.0, 20000.0

        result = calculate_w_prime_fast(watts, time, cp, w_prime)
        assert np.all(result >= 0)

    def test_never_above_capacity(self):
        """W' should never exceed original capacity."""
        n = 100
        watts = np.full(n, 50.0)  # Way below CP
        time = np.arange(n, dtype=np.float64)
        cp, w_prime = 200.0, 20000.0

        result = calculate_w_prime_fast(watts, time, cp, w_prime)
        assert np.all(result <= w_prime)


# =========================================================================
# _calculate_w_prime_balance_cached
# =========================================================================

class TestCalculateWPrimeBalanceCached:
    def test_with_watts_column(self):
        df = pd.DataFrame({
            "watts": [100.0, 150.0, 200.0, 250.0, 300.0],
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        df_bytes = _serialize_df_to_parquet_bytes(df)
        result = _calculate_w_prime_balance_cached(df_bytes, 200.0, 20000.0)
        assert "w_prime_balance" in result.columns
        assert len(result) == 5

    def test_without_watts_column(self):
        df = pd.DataFrame({"hr": [120, 130, 140]})
        df_bytes = _serialize_df_to_parquet_bytes(df)
        result = _calculate_w_prime_balance_cached(df_bytes, 200.0, 20000.0)
        assert "w_prime_balance" in result.columns
        assert result["w_prime_balance"].isna().all()

    def test_auto_generates_time(self):
        df = pd.DataFrame({"watts": [100.0, 200.0, 300.0]})
        df_bytes = _serialize_df_to_parquet_bytes(df)
        result = _calculate_w_prime_balance_cached(df_bytes, 200.0, 20000.0)
        assert "w_prime_balance" in result.columns


# =========================================================================
# calculate_w_prime_balance
# =========================================================================

class TestCalculateWPrimeBalance:
    def test_from_pandas_dataframe(self):
        df = pd.DataFrame({
            "watts": [100.0, 200.0, 300.0, 200.0, 100.0],
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        result = calculate_w_prime_balance(df, 200.0, 20000.0)
        assert isinstance(result, pd.DataFrame)
        assert "w_prime_balance" in result.columns

    def test_from_dict(self):
        data = {
            "watts": [100.0, 200.0, 300.0],
            "time": [0.0, 1.0, 2.0],
        }
        result = calculate_w_prime_balance(data, 200.0, 20000.0)
        assert isinstance(result, pd.DataFrame)
        assert "w_prime_balance" in result.columns

    def test_auto_time_column(self):
        df = pd.DataFrame({"watts": [100.0, 200.0, 300.0]})
        result = calculate_w_prime_balance(df, 200.0, 20000.0)
        assert "w_prime_balance" in result.columns


# =========================================================================
# get_recovery_recommendation
# =========================================================================

class TestGetRecoveryRecommendation:
    @pytest.mark.parametrize("score,expected_zone", [
        (95, "Pełna gotowość"),
        (80, "Dobra gotowość"),
        (60, "Częściowe odzyskanie"),
        (40, "Zmęczenie"),
        (20, "Wyczerpanie"),
    ])
    def test_all_bands(self, score, expected_zone):
        zone, desc = get_recovery_recommendation(score)
        assert expected_zone in zone
        assert len(desc) > 0


# =========================================================================
# estimate_w_prime_reconstitution
# =========================================================================

class TestEstimateWPrimeReconstitution:
    def test_no_recovery_time(self):
        result = estimate_w_prime_reconstitution(50.0, 0)
        assert result == pytest.approx(50.0)

    def test_full_recovery_with_long_time(self):
        result = estimate_w_prime_reconstitution(50.0, 10000, tau=400)
        assert result > 95.0

    def test_partial_recovery(self):
        result = estimate_w_prime_reconstitution(50.0, 400, tau=400)
        # After 1 tau, ~63% of depletion recovered
        # remaining 50% + 50% * 0.632 ≈ 81.6%
        assert 75.0 < result < 90.0

    def test_zero_depletion(self):
        result = estimate_w_prime_reconstitution(0.0, 400)
        assert result == pytest.approx(100.0)
