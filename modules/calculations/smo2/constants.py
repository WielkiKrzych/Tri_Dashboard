"""
SmO2 Constants Module

Thresholds and constants for SmO2 analysis.
"""

# Limiter classification thresholds
LIMITER_THRESHOLDS = {
    "slope_severe": -8.0,      # SmO2 drop > 8%/100W = severe local
    "slope_moderate": -4.0,    # SmO2 drop > 4%/100W = moderate local
    "halftime_fast": 30.0,     # < 30s = good capillarization
    "halftime_slow": 90.0,     # > 90s = poor capillarization
    "coupling_strong": -0.7,   # Strong negative = central driver
    "coupling_weak": -0.3,     # Weak = local driver
}

# T1 Detection criteria
T1_CRITERIA = {
    "trend_threshold": -0.4,   # dSmO2/dt < -0.4%/min
    "cv_threshold": 4.0,       # CV < 4%
    "min_steps": 2,            # ≥2 consecutive steps
}

# T2 Detection criteria  
T2_CRITERIA = {
    "trend_threshold": -1.5,   # dSmO2/dt < -1.5%/min
    "min_curvature": 0.0003,   # min |curvature|
    "osc_increase": 1.3,       # osc amp increase ≥30%
    "power_gap": 1.20,         # T1 + 20%
}

# Artifact rejection
ARTIFACT_THRESHOLDS = {
    "cv_max": 6.0,             # Reject if CV > 6%
    "min_samples_per_step": 30,
    "smoothing_window": 45,    # Median smoothing window [s]
}

# Training recommendations by limiter type
RECOMMENDATIONS = {
    "local": (
        "Sweet Spot 2×20min @ 88-94% FTP",
        "Siłowy 4×8min @ 50-60rpm pod LT1",
        "Objętość Z2 3-4h ciągła"
    ),
    "central": (
        "VO₂max 4-6×4min @ 106-120% FTP",
        "Tempo 60-90min @ 80-90% FTP",
        "Z2 długie 4-5h"
    ),
    "metabolic": (
        "Z2 bardzo długie 4-5h @ 60-70% FTP",
        "Treningi na czczo 1.5-2h",
        "Tempo pod LT1 2×30min"
    ),
    "unknown": ("Trening zrównoważony",)
}
