#!/usr/bin/env python3
"""Test VT detection with user's CSV file"""

import sys
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import VT detection function
from modules.calculations.ventilatory import detect_vt_vslope_savgol
from modules.calculations.test_validator import validate_ramp_test


def test_vt_detection():
    """Test VT detection with user's CSV"""

    csv_path = "/Users/wielkikrzych/Desktop/RampTest 03.01.2026.csv"

    print("=" * 80)
    print("Testing VT Detection with User's CSV")
    print("=" * 80)

    # Load CSV
    print(f"\n1. Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Check required columns
    required_cols = ["watts", "tymeventilation", "time", "hr"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"   ✗ Missing required columns: {missing_cols}")
        return False

    print(f"   ✓ All required columns present")

    # Show data summary
    print(f"\n2. Data Summary:")
    print(
        f"   HR: {df['hr'].min():.1f} - {df['hr'].max():.1f} bpm (median: {df['hr'].median():.1f})"
    )
    print(f"   Power: {df['watts'].min():.1f} - {df['watts'].max():.1f} W")
    print(f"   VE: {df['tymeventilation'].min():.1f} - {df['tymeventilation'].max():.1f} L/min")
    print(f"   Time: {df['time'].min():.1f} - {df['time'].max():.1f} min")

    # Test VT detection
    print(f"\n3. Testing VT Detection (CPET method):")
    try:
        cpet_result = detect_vt_vslope_savgol(
            df,
            step_range=None,  # Auto-detect steps
            power_column="watts",
            ve_column="tymeventilation",
            time_column="time",
            min_power_watts=None,
        )
        print(f"   ✓ VT Detection successful")

        # Show results
        vt1 = cpet_result.get("VT1", {})
        vt2 = cpet_result.get("VT2", {})

        print(f"\n4. VT Results:")
        if vt1:
            print(f"   VT1 (Ventilatory Threshold 1):")
            print(f"      Power: {vt1.get('power', 'N/A')} W")
            print(f"      HR: {vt1.get('hr', 'N/A')} bpm")
            print(f"      VE: {vt1.get('ve', 'N/A')} L/min")
        else:
            print(f"   VT1: Not detected")

        if vt2:
            print(f"\n   VT2 (Ventilatory Threshold 2):")
            print(f"      Power: {vt2.get('power', 'N/A')} W")
            print(f"      HR: {vt2.get('hr', 'N/A')} bpm")
            print(f"      VE: {vt2.get('ve', 'N/A')} L/min")
        else:
            print(f"\n   VT2: Not detected")

        # Check for analysis notes
        notes = cpet_result.get("analysis_notes", [])
        if notes:
            print(f"\n5. Analysis Notes:")
            for note in notes:
                print(f"   {note}")

        # Test validator
        print(f"\n6. Testing Test Validator:")
        validation = validate_ramp_test(
            df,
            power_column="watts",
            ve_column="tymeventilation",
            hr_column="hr",
            time_column="time",
            cp_watts=350,  # Example CP value
            w_prime_joules=20000,  # Example W' value
        )

        print(f"   Overall Validity: {validation.validity}")
        print(f"   Confidence Level: {validation.confidence}")

        print(f"\n   Quality Metrics:")
        for metric, value in validation.quality_metrics.items():
            print(f"      {metric}: {value}")

        if validation.issues:
            print(f"\n   Issues:")
            for issue in validation.issues:
                print(f"      - {issue}")

        print(f"\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"   ✗ VT Detection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vt_detection()
    sys.exit(0 if success else 1)
