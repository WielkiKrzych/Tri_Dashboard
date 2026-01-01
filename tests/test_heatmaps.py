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


class TestPowerZoneHeatmap:
    """Tests for power_zone_heatmap (per minute)."""
    
    def test_synthetic_uniform_distribution(self):
        """Uniform power distribution should have equal time per minute (1.0)."""
        # Generate 10 minutes of data at 1Hz
        n_samples = 10 * 60
        
        # Power at Z2
        power = [200.0] * n_samples
        
        df = pd.DataFrame({
            'time': np.arange(n_samples),
            'watts': power
        })
        
        pivot, metadata = power_zone_heatmap(df, ftp=300)
        
        # All time should be in Z2
        # Sum of minutes in Z2 should be 10.0
        assert pivot.loc["Z2 Endurance"].sum() == 10.0
        assert pivot.loc["Z1 Recovery"].sum() == 0
    
    def test_minute_distribution(self):
        """Data from different minutes should appear in correct columns."""
        # Create data for minute 0 and minute 5
        data = []
        
        # Minute 0
        for i in range(60):
            data.append({'time': i, 'watts': 250})
        
        # Minute 5
        for i in range(60):
            data.append({'time': 300 + i, 'watts': 250})
            
        df = pd.DataFrame(data)
        pivot, metadata = power_zone_heatmap(df, ftp=300)
        
        # Z3 should have 1.0 minute in columns 0 and 5
        assert pivot.loc["Z3 Tempo", 0] == 1.0
        assert pivot.loc["Z3 Tempo", 5] == 1.0
        
        # Other minutes should be 0
        assert pivot.loc["Z3 Tempo", 3] == 0


class TestPlotPowerZoneHeatmap:
    """Tests for plot_power_zone_heatmap function."""
    
    def test_returns_figure(self):
        """Should return a Plotly Figure."""
        # Simple pivot table
        pivot = pd.DataFrame(
            [[1.0, 1.0], [0.5, 0.5]],
            index=["Z1 Recovery", "Z2 Endurance"],
            columns=[0, 1]
        )
        
        fig = plot_power_zone_heatmap(pivot)
        
        assert fig is not None
        assert hasattr(fig, 'update_layout')
