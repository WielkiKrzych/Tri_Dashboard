"""Tests for modules/social/comparison.py — anonymization and percentile rankings."""

import pytest
import pandas as pd
import numpy as np
from modules.social.comparison import (
    DataAnonymizer,
    ComparisonService,
    AnonymizedProfile,
    PercentileRanking,
)


# =========================================================================
# DataAnonymizer
# =========================================================================

class TestDataAnonymizer:
    @pytest.fixture
    def anon(self):
        return DataAnonymizer()

    def test_create_anonymous_id_deterministic(self, anon):
        id1 = anon.create_anonymous_id("user@example.com")
        id2 = anon.create_anonymous_id("user@example.com")
        assert id1 == id2
        assert len(id1) == 16

    def test_create_anonymous_id_different_inputs(self, anon):
        id1 = anon.create_anonymous_id("user1@example.com")
        id2 = anon.create_anonymous_id("user2@example.com")
        assert id1 != id2

    def test_get_bracket_in_range(self, anon):
        result = anon.get_bracket(32, anon.AGE_BRACKETS)
        assert result == "30-35"

    def test_get_bracket_first(self, anon):
        result = anon.get_bracket(20, anon.AGE_BRACKETS)
        assert result == "18-25"

    def test_get_bracket_out_of_range(self, anon):
        result = anon.get_bracket(150, anon.AGE_BRACKETS)
        assert result == "unknown"

    def test_anonymize_session(self, anon):
        df = pd.DataFrame({
            "watts": np.random.randint(100, 300, 300),
        })
        metrics = {"avg_watts": 200, "np": 210, "tss": 50}
        result = anon.anonymize_session(df, metrics, age=32, weight=75.0, gender="M")
        assert "age_bracket" in result
        assert "ftp_wkg" in result
        assert result["gender"] == "M"

    def test_anonymize_session_zero_weight_defaults(self, anon):
        df = pd.DataFrame({"watts": [200] * 60})
        metrics = {"avg_watts": 200, "np": 200}
        result = anon.anonymize_session(df, metrics, age=30, weight=0, gender="M")
        assert result["ftp_wkg"] == pytest.approx(200 / 70)  # Default weight 70

    def test_anonymize_profile(self, anon):
        mmp = {"5s": 800, "1m": 500, "5m": 350, "20m": 280, "vo2max": 55}
        result = anon.anonymize_profile("user1", 32, 75.0, "M", 300.0, mmp)
        assert isinstance(result, AnonymizedProfile)
        assert result.ftp_wkg == pytest.approx(300.0 / 75.0)
        assert result.mmp_5s_wkg == pytest.approx(800 / 75.0)


# =========================================================================
# ComparisonService
# =========================================================================

class TestComparisonService:
    @pytest.fixture
    def service(self):
        return ComparisonService()

    def test_get_ftp_percentile_male_mid(self, service):
        result = service.get_ftp_percentile(4.0, "M")
        assert isinstance(result, PercentileRanking)
        assert 65 <= result.percentile <= 85  # Interpolated range
        assert result.metric == "FTP/kg"

    def test_get_ftp_percentile_female(self, service):
        result = service.get_ftp_percentile(3.0, "F")
        assert isinstance(result, PercentileRanking)
        assert 40 <= result.percentile <= 75  # Interpolated range

    def test_get_ftp_percentile_elite(self, service):
        result = service.get_ftp_percentile(7.0, "M")
        assert result.percentile == 99

    def test_get_ftp_percentile_beginner(self, service):
        result = service.get_ftp_percentile(1.0, "M")
        assert result.percentile < 10

    def test_get_cycling_category(self, service):
        assert service._get_cycling_category(7.5) == "World Tour Pro"
        assert service._get_cycling_category(4.0) == "Cat 4"
        assert service._get_cycling_category(1.0) == "Beginner"

    def test_get_vo2max_percentile_male_30s(self, service):
        result = service.get_vo2max_percentile(42.0, 35, "M")
        assert isinstance(result, PercentileRanking)
        assert result.metric == "VO2max"
        assert 30 <= result.percentile <= 80

    def test_get_vo2max_percentile_elite(self, service):
        result = service.get_vo2max_percentile(65.0, 30, "M")
        assert result.percentile == 95
        assert result.category == "Elite"

    def test_get_fitness_level(self, service):
        assert service._get_fitness_level(95) == "Elite"
        assert service._get_fitness_level(80) == "Excellent"
        assert service._get_fitness_level(55) == "Good"
        assert service._get_fitness_level(30) == "Average"
        assert service._get_fitness_level(10) == "Below Average"

    def test_get_summary_rankings(self, service):
        rankings = service.get_summary_rankings(
            ftp=300, weight=75.0, vo2max=50.0, age=35, gender="M"
        )
        assert len(rankings) == 2
        assert rankings[0].metric == "FTP/kg"
        assert rankings[1].metric == "VO2max"

    def test_get_summary_rankings_zero_weight(self, service):
        rankings = service.get_summary_rankings(
            ftp=300, weight=0, vo2max=50.0, age=35, gender="M"
        )
        assert len(rankings) == 1  # Only VO2max
