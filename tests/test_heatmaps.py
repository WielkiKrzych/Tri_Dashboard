"""
Tests for Power Zone Heatmap Module.

Tests cover:
- Zone assignment correctness
- Pivot table row/column sums
- Different resolutions (hourly, weekday_hourly)
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from modules.heatmaps import (
    get_power_zones,
    assign_power_zone,
    power_zone_heatmap,
    plot_power_zone_heatmap,
)


class TestGetPowerZones:
    """Tests for get_power_zones function."""
    
    def test_default_zones_count(self):
        """Should return 7 default zones."""
        zones = get_power_zones(ftp=250)
        assert len(zones) == 7
    
    def test_zone_bounds(self):
        """Zone bounds should be calculated from FTP."""
        ftp = 300
        zones = get_power_zones(ftp)
        
        # Z1: 0-55% FTP
        assert zones[0].lower_watts == 0
        assert zones[0].upper_watts == ftp * 0.55
        
        # Z4: 90-105% FTP (Threshold)
        assert zones[3].lower_watts == ftp * 0.90
        assert zones[3].upper_watts == ftp * 1.05


class TestAssignPowerZone:
    """Tests for assign_power_zone function."""
    
    def test_z2_assignment(self):
        """Power at 65% FTP should be Z2."""
        ftp = 300
        zones = get_power_zones(ftp)
        power = ftp * 0.65  # 195W for 300 FTP
        
        zone = assign_power_zone(power, zones)
        
        assert zone == "Z2 Endurance"
    
    def test_z4_threshold(self):
        """Power at FTP (100%) should be Z4."""
        ftp = 280
        zones = get_power_zones(ftp)
        
        zone = assign_power_zone(ftp, zones)
        
        assert zone == "Z4 Threshold"
    
    def test_above_all_zones(self):
        """Power way above FTP should be Z7."""
        ftp = 300
        zones = get_power_zones(ftp)
        power = ftp * 2.0  # 200% FTP
        
        zone = assign_power_zone(power, zones)
        
        assert zone == "Z7 Neuromuscular"
    
    def test_zero_power(self):
        """Zero power should return None."""
        zones = get_power_zones(250)
        
        zone = assign_power_zone(0, zones)
        
        assert zone is None
    
    def test_nan_power(self):
        """NaN power should return None."""
        zones = get_power_zones(250)
        
        zone = assign_power_zone(np.nan, zones)
        
        assert zone is None


class TestPowerZoneHeatmapHourly:
    """Tests for power_zone_heatmap with hourly resolution."""
    
    def test_synthetic_uniform_distribution(self):
        """Uniform power distribution should have equal time per hour."""
        # Generate 24 hours of data at 1Hz (86400 seconds)
        n_samples = 24 * 60  # 1 minute per hour for speed
        
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [base_time + timedelta(seconds=i*60) for i in range(n_samples)]
        
        # Power at Z2 (200W for 300 FTP)
        power = [200.0] * n_samples
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'watts': power
        })
        
        pivot, metadata = power_zone_heatmap(df, ftp=300, resolution="hourly")
        
        # All time should be in Z2
        assert pivot.loc["Z2 Endurance"].sum() == n_samples
        assert pivot.loc["Z1 Recovery"].sum() == 0
    
    def test_hour_distribution(self):
        """Data from different hours should appear in correct columns."""
        # Create data for hours 0, 6, 12, 18
        data = []
        base_date = datetime(2024, 1, 1)
        
        for hour in [0, 6, 12, 18]:
            ts = base_date.replace(hour=hour)
            for i in range(60):  # 1 minute each
                data.append({
                    'timestamp': ts + timedelta(seconds=i),
                    'watts': 250  # Z3 for 300 FTP
                })
        
        df = pd.DataFrame(data)
        pivot, metadata = power_zone_heatmap(df, ftp=300, resolution="hourly")
        
        # Z3 should have 60 seconds in each of hours 0, 6, 12, 18
        assert pivot.loc["Z3 Tempo", 0] == 60
        assert pivot.loc["Z3 Tempo", 6] == 60
        assert pivot.loc["Z3 Tempo", 12] == 60
        assert pivot.loc["Z3 Tempo", 18] == 60
        
        # Other hours should be 0
        assert pivot.loc["Z3 Tempo", 3] == 0


class TestPowerZoneHeatmapWeekdayHourly:
    """Tests for power_zone_heatmap with weekday_hourly resolution."""
    
    def test_weekday_assignment(self):
        """Data from different weekdays should appear in correct columns."""
        data = []
        
        # Monday (2024-01-01 is Monday) at 10:00
        monday = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(60):
            data.append({
                'timestamp': monday + timedelta(seconds=i),
                'watts': 250
            })
        
        # Wednesday at 10:00
        wednesday = datetime(2024, 1, 3, 10, 0, 0)
        for i in range(60):
            data.append({
                'timestamp': wednesday + timedelta(seconds=i),
                'watts': 250
            })
        
        df = pd.DataFrame(data)
        pivot, metadata = power_zone_heatmap(df, ftp=300, resolution="weekday_hourly")
        
        # Check Monday and Wednesday columns
        assert pivot.loc["Z3 Tempo", "Mon_10"] == 60
        assert pivot.loc["Z3 Tempo", "Wed_10"] == 60


class TestPlotPowerZoneHeatmap:
    """Tests for plot_power_zone_heatmap function."""
    
    def test_returns_figure(self):
        """Should return a Plotly Figure."""
        # Simple pivot table
        pivot = pd.DataFrame(
            [[60, 120], [30, 90]],
            index=["Z1 Recovery", "Z2 Endurance"],
            columns=[0, 1]
        )
        
        fig = plot_power_zone_heatmap(pivot, resolution="hourly")
        
        assert fig is not None
        assert hasattr(fig, 'update_layout')
