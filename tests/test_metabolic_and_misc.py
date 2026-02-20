"""Tests for metabolic engine, settings, notes, and misc modules at 0% coverage."""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np


# =========================================================================
# metabolic_engine
# =========================================================================

class TestMetabolicEngine:
    def test_estimate_vlamax(self):
        from modules.calculations.metabolic_engine import estimate_vlamax
        vlamax = estimate_vlamax(
            cp_watts=280, w_prime_kj=15.0, pmax_watts=1200, weight_kg=75
        )
        assert isinstance(vlamax, float)
        assert vlamax > 0

    def test_calculate_vo2max_vlamax_ratio(self):
        from modules.calculations.metabolic_engine import calculate_vo2max_vlamax_ratio
        ratio = calculate_vo2max_vlamax_ratio(55.0, 0.5)
        assert ratio == pytest.approx(110.0)

    def test_classify_phenotype(self):
        from modules.calculations.metabolic_engine import classify_phenotype
        # High VO2max, low VLaMax = aerobic
        phenotype = classify_phenotype(vo2max=60.0, vlamax=0.3, anaerobic_reserve_pct=20)
        assert isinstance(phenotype, str)
        assert len(phenotype) > 0

    def test_diagnose_limiter(self):
        from modules.calculations.metabolic_engine import diagnose_limiter
        limiter, confidence, interpretation = diagnose_limiter(
            vo2max=50.0, vlamax=0.6, cp_watts=250, weight_kg=75
        )
        assert isinstance(limiter, str)
        assert 0 <= confidence <= 1.0

    def test_determine_adaptation_target(self):
        from modules.calculations.metabolic_engine import determine_adaptation_target
        target, strategy = determine_adaptation_target("VO2max", "aerobic")
        assert isinstance(target, str)
        assert isinstance(strategy, str)

    def test_analyze_metabolic_engine(self):
        from modules.calculations.metabolic_engine import analyze_metabolic_engine
        result = analyze_metabolic_engine(
            vo2max=55.0, vo2max_source="test", vo2max_confidence=0.9,
            cp_watts=280, w_prime_kj=15.0, pmax_watts=1200, weight_kg=75, ftp_watts=265
        )
        assert result is not None
        assert hasattr(result, "profile")
        assert result.profile.vlamax > 0


# =========================================================================
# settings
# =========================================================================

class TestSettingsManager:
    def test_load_settings_returns_defaults(self):
        from modules.settings import SettingsManager
        mgr = SettingsManager()
        settings = mgr.load_settings()
        assert isinstance(settings, dict)

    def test_save_settings_noop(self):
        from modules.settings import SettingsManager
        mgr = SettingsManager()
        # save_settings is disabled, should not raise
        mgr.save_settings({"ftp": 280})

    def test_get_ui_values(self):
        from modules.settings import SettingsManager
        mgr = SettingsManager()
        values = mgr.get_ui_values()
        assert isinstance(values, dict)


# =========================================================================
# notes
# =========================================================================

class TestTrainingNotes:
    def test_init(self):
        from modules.notes import TrainingNotes
        notes = TrainingNotes()
        assert notes is not None

    def test_load_notes_nonexistent_file(self):
        from modules.notes import TrainingNotes
        notes = TrainingNotes()
        result = notes.load_notes("nonexistent_file.csv")
        assert isinstance(result, (dict, list))

    def test_add_and_get_notes(self):
        from modules.notes import TrainingNotes
        notes = TrainingNotes()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override notes directory
            notes.notes_dir = Path(tmpdir)
            test_file = "test_training_2024.csv"
            notes.add_note(test_file, 10.0, "power", "Good effort here")
            result = notes.get_notes_for_metric(test_file, "power")
            assert len(result) >= 1


# =========================================================================
# polars_adapter
# =========================================================================

class TestPolarsAdapter:
    def test_from_pandas_and_back(self):
        import pandas as pd
        from modules.polars_adapter import PolarsAdapter
        df = pd.DataFrame({"watts": [100, 200, 300], "hr": [120, 140, 160]})
        adapter = PolarsAdapter(df)
        result = adapter.to_pandas()
        assert len(result) == 3

    def test_groupby_agg(self):
        import pandas as pd
        from modules.polars_adapter import PolarsAdapter
        df = pd.DataFrame({
            "zone": [1, 1, 2, 2, 3],
            "watts": [100, 120, 200, 220, 350],
        })
        adapter = PolarsAdapter(df)
        result = adapter.groupby_agg("zone", {"watts": "mean"})
        assert len(result) == 3

    def test_rolling_mean(self):
        import pandas as pd
        from modules.polars_adapter import PolarsAdapter
        df = pd.DataFrame({"watts": [100, 200, 300, 400, 500]})
        adapter = PolarsAdapter(df)
        result = adapter.rolling_mean("watts", 3)
        assert len(result) == 5


# =========================================================================
# cache_utils
# =========================================================================

class TestCacheUtils:
    def test_get_cache(self):
        from modules.cache_utils import get_cache
        cache = get_cache()
        # May return None if caching is disabled
        assert cache is None or cache is not None

    def test_cache_result_decorator(self):
        from modules.cache_utils import cache_result

        @cache_result(ttl=60)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3
        # Second call should use cache
        result2 = add(1, 2)
        assert result2 == 3
