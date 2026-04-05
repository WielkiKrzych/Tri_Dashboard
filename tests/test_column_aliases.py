import pandas as pd
import pytest

from modules.calculations.column_aliases import (
    BREATH_RATE_ALIASES,
    HR_ALIASES,
    POWER_ALIASES,
    normalize_columns,
    resolve_all_aliases,
    resolve_breath_rate_column,
    resolve_hr_column,
    resolve_power_column,
)


class TestNormalizeColumns:
    def test_lowercases_and_strips(self):
        df = pd.DataFrame({"  HR  ": [1], "Power": [2]})
        result = normalize_columns(df)
        assert list(result.columns) == ["hr", "power"]
        assert result is df

    def test_already_clean(self):
        df = pd.DataFrame({"hr": [1], "watts": [2]})
        normalize_columns(df)
        assert list(df.columns) == ["hr", "watts"]


class TestResolveHrColumn:
    @pytest.mark.parametrize("alias", list(HR_ALIASES))
    def test_recognizes_all_hr_aliases(self, alias):
        df = pd.DataFrame({alias: [72, 80, 90]})
        result = resolve_hr_column(df)
        assert result == "hr"
        assert "hr" in df.columns

    @pytest.mark.parametrize("alias", list(HR_ALIASES))
    def test_case_insensitive(self, alias):
        df = pd.DataFrame({alias: [72]})
        normalize_columns(df)
        result = resolve_hr_column(df)
        assert result == "hr"

    def test_canonical_hr_unchanged(self):
        df = pd.DataFrame({"hr": [72]})
        result = resolve_hr_column(df)
        assert result == "hr"
        assert list(df.columns) == ["hr"]

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"watts": [100]})
        assert resolve_hr_column(df) is None

    def test_whitespace_aliases(self):
        df = pd.DataFrame({"  heart_rate  ": [72]})
        normalize_columns(df)
        resolve_hr_column(df)
        assert "hr" in df.columns


class TestResolvePowerColumn:
    @pytest.mark.parametrize("alias", list(POWER_ALIASES))
    def test_recognizes_power_aliases(self, alias):
        df = pd.DataFrame({alias: [200]})
        result = resolve_power_column(df)
        assert result == "watts"
        assert "watts" in df.columns

    def test_canonical_watts_unchanged(self):
        df = pd.DataFrame({"watts": [200]})
        assert resolve_power_column(df) == "watts"

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"hr": [72]})
        assert resolve_power_column(df) is None


class TestResolveBreathRateColumn:
    @pytest.mark.parametrize("alias", list(BREATH_RATE_ALIASES))
    def test_recognizes_breath_rate_aliases(self, alias):
        df = pd.DataFrame({alias: [30]})
        result = resolve_breath_rate_column(df)
        assert result == "tymebreathrate"
        assert "tymebreathrate" in df.columns


class TestResolveAllAliases:
    def test_resolves_everything(self):
        df = pd.DataFrame(
            {
                "heart_rate": [72],
                "power": [200],
                "br": [30],
                "smo2": [65],
            }
        )
        resolve_all_aliases(df)
        assert "hr" in df.columns
        assert "watts" in df.columns
        assert "tymebreathrate" in df.columns
        assert "smo2" in df.columns

    def test_idempotent(self):
        df = pd.DataFrame({"heart_rate": [72], "power": [200]})
        resolve_all_aliases(df)
        cols_first = list(df.columns)
        resolve_all_aliases(df)
        cols_second = list(df.columns)
        assert cols_first == cols_second

    def test_mixed_case_input(self):
        df = pd.DataFrame({"Heart_Rate": [72], "POWER": [200]})
        resolve_all_aliases(df)
        assert "hr" in df.columns
        assert "watts" in df.columns
