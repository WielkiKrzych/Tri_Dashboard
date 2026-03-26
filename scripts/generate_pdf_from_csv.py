#!/usr/bin/env python3
"""
Standalone PDF generator: CSV → full analysis → PDF.

Usage:
    python scripts/generate_pdf_from_csv.py <csv_path> <output_pdf_path>

Bypasses Streamlit gating for CLI usage.
"""
import sys
import os
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock streamlit before any imports that use it
import types
st_mock = types.ModuleType("streamlit")
st_mock.session_state = {"report_generation_requested": True, "report_exports_ready": False}
st_mock.cache_data = lambda *a, **kw: (lambda f: f)
st_mock.cache_resource = lambda *a, **kw: (lambda f: f)
st_mock.warning = lambda *a, **kw: None
st_mock.error = lambda *a, **kw: None
st_mock.info = lambda *a, **kw: None
st_mock.success = lambda *a, **kw: None
st_mock.spinner = lambda *a, **kw: type('ctx', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})()
sys.modules["streamlit"] = st_mock

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/generate_pdf_from_csv.py <csv_path> <output_pdf_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_pdf = Path(sys.argv[2])

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    import pandas as pd
    import numpy as np

    # 1. Load CSV using pandas directly (avoids file-object issues)
    logger.info("Loading CSV: %s", csv_path)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize columns same as load_data would
    from modules.utils import normalize_columns_pandas
    df = normalize_columns_pandas(df)
    if "time" not in df.columns:
        df["time"] = np.arange(len(df)).astype(float)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # 2. Run full ramp test pipeline
    from modules.calculations.pipeline import run_ramp_test_pipeline
    logger.info("Running ramp test pipeline...")

    # Map column names for the pipeline
    hr_col = "heartrate" if "heartrate" in df.columns else ("hr" if "hr" in df.columns else "hr")

    result = run_ramp_test_pipeline(
        df,
        power_column="watts",
        hr_column=hr_col,
        ve_column="tymeventilation",
        smo2_column="smo2",
        time_column="time",
        test_date=datetime.now().strftime("%Y-%m-%d"),
    )

    # 3. Build report data
    data = result.to_dict()

    # Add time series
    ts_map = {
        "watts": "power_watts", "power": "power_watts",
        "hr": "hr_bpm", "heartrate": "hr_bpm",
        "smo2": "smo2_pct",
        "tymeventilation": "ve_lmin", "ve": "ve_lmin",
        "torque": "torque_nm",
        "cadence": "cadence_rpm", "cad": "cadence_rpm",
        "core_temperature": "core_temp",
        "heatstrainindex": "hsi", "heat_strain_index": "hsi",
    }
    ts_data = {}
    if "time" in df.columns:
        ts_data["time_sec"] = df["time"].tolist()
    else:
        ts_data["time_sec"] = list(range(len(df)))

    for df_col, json_key in ts_map.items():
        if df_col in df.columns and json_key not in ts_data:
            ts_data[json_key] = df[df_col].fillna(0).tolist()
    data["time_series"] = ts_data

    # 4. Run advanced analyses
    analysis_df = df.copy()

    # SmO2 advanced
    if "smo2" in analysis_df.columns:
        try:
            from modules.calculations.smo2_advanced import analyze_smo2_advanced, format_smo2_metrics_for_report
            adf = analysis_df.copy()
            adf["SmO2"] = adf["smo2"]
            if "seconds" not in adf.columns and "time" in adf.columns:
                adf["seconds"] = range(len(adf))
            smo2_metrics = analyze_smo2_advanced(adf)
            data["smo2_advanced"] = format_smo2_metrics_for_report(smo2_metrics)
            logger.info("SmO2 advanced analysis complete")
        except Exception as e:
            logger.warning("SmO2 advanced failed: %s", e)

    # Cardiovascular
    hr_col = next((c for c in ["hr", "heartrate"] if c in analysis_df.columns), None)
    if hr_col:
        try:
            from modules.calculations.cardio_advanced import analyze_cardiovascular, format_cardio_metrics_for_report
            adf = analysis_df.copy()
            if "hr" not in adf.columns and "heartrate" in adf.columns:
                adf["hr"] = adf["heartrate"]
            cardio_metrics = analyze_cardiovascular(adf)
            data["cardio_advanced"] = format_cardio_metrics_for_report(cardio_metrics)
            logger.info("Cardio analysis complete")
        except Exception as e:
            logger.warning("Cardio analysis failed: %s", e)

    # Ventilation
    ve_col = next((c for c in ["tymeventilation", "ve"] if c in analysis_df.columns), None)
    if ve_col:
        try:
            from modules.calculations.vent_advanced import analyze_ventilation, format_vent_metrics_for_report
            vent_metrics = analyze_ventilation(analysis_df.copy())
            data["vent_advanced"] = format_vent_metrics_for_report(vent_metrics)
            logger.info("Ventilation analysis complete")
        except Exception as e:
            logger.warning("Ventilation analysis failed: %s", e)

    # Biomechanical occlusion
    has_power = "watts" in analysis_df.columns
    has_smo2 = "smo2" in analysis_df.columns
    has_cadence = "cadence" in analysis_df.columns or "cad" in analysis_df.columns
    has_torque = "torque" in analysis_df.columns

    if (has_torque or (has_power and has_cadence)) and has_smo2:
        try:
            from modules.calculations.biomech_occlusion import analyze_biomech_occlusion, format_occlusion_for_report
            if has_torque:
                torque = analysis_df["torque"].values
            else:
                power = analysis_df["watts"].values
                cad_col = "cadence" if "cadence" in analysis_df.columns else "cad"
                cadence_vals = analysis_df[cad_col].values
                angular_vel = 2 * np.pi * cadence_vals / 60
                angular_vel[angular_vel < 0.1] = 0.1
                torque = power / angular_vel

            smo2_vals = analysis_df["smo2"].values
            cad_vals = analysis_df["cadence"].values if "cadence" in analysis_df.columns else (analysis_df["cad"].values if "cad" in analysis_df.columns else None)
            occlusion = analyze_biomech_occlusion(torque, smo2_vals, cad_vals)
            data["biomech_occlusion"] = format_occlusion_for_report(occlusion)
            logger.info("Biomech occlusion analysis complete")
        except Exception as e:
            logger.warning("Biomech analysis failed: %s", e)

    # Thermoregulation
    core_col = next((c for c in ["core_temperature", "core_temp", "temp"] if c in analysis_df.columns), None)
    if core_col:
        try:
            from modules.calculations.thermoregulation import analyze_thermoregulation, format_thermo_for_report
            core_temp = analysis_df[core_col].values
            time_sec = np.arange(len(core_temp))
            hr_vals = analysis_df["hr"].values if "hr" in analysis_df.columns else (analysis_df["heartrate"].values if "heartrate" in analysis_df.columns else None)
            power_vals = analysis_df["watts"].values if "watts" in analysis_df.columns else None
            hsi_col = next((c for c in ["hsi", "heatstrainindex", "heat_strain_index"] if c in analysis_df.columns), None)
            hsi_vals = analysis_df[hsi_col].values if hsi_col else None

            thermo = analyze_thermoregulation(core_temp, time_sec, hr_vals, power_vals, hsi_vals)
            data["thermo_analysis"] = format_thermo_for_report(thermo)
            logger.info("Thermoregulation analysis complete")
        except Exception as e:
            logger.warning("Thermoregulation failed: %s", e)

    # Cardiac drift
    power_col = next((c for c in ["watts", "power"] if c in analysis_df.columns), None)
    hr_col2 = next((c for c in ["hr", "heartrate"] if c in analysis_df.columns), None)
    if power_col and hr_col2:
        try:
            from modules.calculations.cardiac_drift import analyze_cardiac_drift, format_drift_for_report
            power_arr = analysis_df[power_col].values
            hr_arr = analysis_df[hr_col2].values
            time_arr = np.arange(len(power_arr))
            core_arr = analysis_df[core_col].values if core_col else None
            smo2_arr = analysis_df["smo2"].values if "smo2" in analysis_df.columns else None
            hsi_arr = analysis_df[hsi_col].values if hsi_col else None

            drift_profile = analyze_cardiac_drift(power_arr, hr_arr, time_arr, core_arr, smo2_arr, hsi_arr)
            if "thermo_analysis" not in data:
                data["thermo_analysis"] = {}
            data["thermo_analysis"]["cardiac_drift"] = format_drift_for_report(drift_profile)
            logger.info("Cardiac drift analysis complete")
        except Exception as e:
            logger.warning("Cardiac drift failed: %s", e)

    # VO2max estimate
    if "watts" in analysis_df.columns:
        try:
            from modules.calculations.metrics import calculate_vo2max
            mmp_5min = analysis_df["watts"].rolling(window=300, min_periods=60).mean().max()
            weight = 75  # default
            if pd.notna(mmp_5min) and mmp_5min > 0:
                vo2max_est = calculate_vo2max(mmp_5min, weight)
                if "metrics" not in data:
                    data["metrics"] = {}
                data["metrics"]["vo2max"] = round(vo2max_est, 2)
                data["metrics"]["vo2max_metadata"] = {
                    "value": round(vo2max_est, 2),
                    "mmp_5min_watts": round(float(mmp_5min), 1),
                    "method": "rolling_300s_mean_max",
                    "source": "cli_generator",
                    "confidence": 0.70,
                    "formula": "Sitko et al. 2021",
                    "weight_kg": weight,
                }
                logger.info("VO2max estimated: %.1f ml/kg/min", vo2max_est)
        except Exception as e:
            logger.warning("VO2max calculation failed: %s", e)

    # Canonical physiology
    try:
        from modules.calculations.canonical_physio import build_canonical_physiology, format_canonical_for_report
        canonical = build_canonical_physiology(data, data.get("time_series", {}))
        data["canonical_physiology"] = format_canonical_for_report(canonical)
        logger.info("Canonical physiology built")

        from modules.calculations.metabolic_engine import analyze_metabolic_engine, format_metabolic_strategy_for_report
        cp_watts = canonical.cp_watts.value
        vo2max = canonical.vo2max.value
        weight_kg = canonical.weight_kg.value
        w_prime_kj = canonical.w_prime_kj.value or 15
        pmax = canonical.pmax_watts.value

        if cp_watts > 0:
            metabolic_strategy = analyze_metabolic_engine(
                vo2max=vo2max,
                vo2max_source=canonical.vo2max.source,
                vo2max_confidence=canonical.vo2max.confidence,
                cp_watts=cp_watts,
                w_prime_kj=w_prime_kj,
                pmax_watts=pmax,
                weight_kg=weight_kg,
                ftp_watts=canonical.ftp_watts.value,
            )
            formatted = format_metabolic_strategy_for_report(metabolic_strategy)
            formatted["profile"]["vo2max_alternatives"] = canonical.vo2max.alternatives
            formatted["profile"]["data_quality"] = (
                "good" if canonical.vo2max.confidence >= 0.7
                else ("moderate" if canonical.vo2max.confidence >= 0.5 else "low")
            )
            data["metabolic_strategy"] = formatted
            logger.info("Metabolic strategy complete")
    except Exception as e:
        logger.warning("Canonical/metabolic failed: %s", e)

    # 5. Metadata
    from modules.calculations.version import RAMP_METHOD_VERSION
    from modules.reporting.report_io import CANONICAL_SCHEMA, CANONICAL_VERSION

    session_id = str(uuid.uuid4())
    data["metadata"] = {
        "test_date": datetime.now().strftime("%Y-%m-%d"),
        "analysis_timestamp": datetime.now().isoformat(),
        "method_version": RAMP_METHOD_VERSION,
        "session_id": session_id,
        "athlete_id": None,
        "notes": f"Generated from {csv_path.name}",
        "analyzer": "Tri_Dashboard/cli_generator",
    }

    final_json = {"$schema": CANONICAL_SCHEMA, "version": CANONICAL_VERSION, **data}

    # 6. Generate figures
    from modules.reporting.figures import generate_all_ramp_figures
    temp_dir = tempfile.mkdtemp()
    fig_config = {"method_version": RAMP_METHOD_VERSION}
    logger.info("Generating figures...")
    figure_paths = generate_all_ramp_figures(final_json, temp_dir, fig_config, source_df=df)
    logger.info("Figures generated: %s", list(figure_paths.keys()))

    # 7. Generate PDF
    from modules.reporting.pdf import generate_ramp_pdf, PDFConfig
    pdf_config = PDFConfig(is_conditional=False)
    logger.info("Generating PDF...")
    generate_ramp_pdf(final_json, figure_paths, str(output_pdf), pdf_config)
    logger.info("PDF saved to: %s", output_pdf)

    print(f"\nPDF generated successfully: {output_pdf}")


if __name__ == "__main__":
    main()
