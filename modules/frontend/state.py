"""
Frontend State Management.

Centralized state manager to handle Session State with type safety.
"""

from typing import Any

import streamlit as st
from modules.settings import SettingsManager


class StateManager:
    """Manages application state and settings."""

    # Default values for ALL session_state keys used across the app.
    # Organized by category. Keys are initialized in init_session_state()
    # so that every module can safely read them without guard clauses.
    _defaults: dict[str, Any] = {
        # --- Runtime (app.py) ---
        "data_loaded": False,
        "selected_athlete_id": "default",
        "session_type": None,
        "ramp_classification": None,
        "current_file_hash": None,
        # --- Report flags ---
        "report_generation_requested": False,
        "report_exports_ready": False,
        # --- Analysis ranges: SmO2 (modules/ui/smo2.py) ---
        "smo2_start_sec": 600,
        "smo2_end_sec": 1200,
        # --- Analysis ranges: Ventilation (modules/ui/vent.py) ---
        "vent_start_sec": 600,
        "vent_end_sec": 1200,
        "br_start_sec": 600,
        "br_end_sec": 1200,
        "tv_start_sec": 600,
        "tv_end_sec": 1200,
        # --- Threshold detection (modules/ui/threshold_analysis_ui.py) ---
        "threshold_result": None,
        "detected_vt1": None,
        "detected_vt2": None,
        # --- Ventilation thresholds (modules/ui/vent_thresholds.py) ---
        "vt_min_power_watts": 0,
        "cpet_vt_result": {},
        # --- Pending pipeline data (vent_thresholds, ramp_archive, report) ---
        "pending_pipeline_result": None,
        "pending_source_df": None,
        "pending_uploaded_file_name": None,
        "pending_cp_input": 0,
        # --- Manual thresholds (modules/ui/manual_thresholds.py) ---
        "manual_vt1_watts": None,
        "manual_vt2_watts": None,
        "test_start_power": 100,
        "test_end_power": 400,
        "step_increment": 20,
        "test_duration": "45:00",
        # --- HRV / DFA (modules/ui/hrv.py) ---
        "df_dfa": None,
        "dfa_error": None,
        # --- Environment (modules/ui/environment_ui.py) ---
        "current_weather": None,
        # --- History comparison (modules/ui/compare.py) ---
        "history_metrics": [],
    }

    def __init__(self):
        self.settings_manager = SettingsManager()
        self._keys_map = {
            "weight": "rider_weight",
            "height": "rider_height",
            "age": "rider_age",
            "gender_m": "is_male",
            "vt1_w": "vt1_watts",
            "vt2_w": "vt2_watts",
            "vt1_v": "vt1_vent",
            "vt2_v": "vt2_vent",
            "cp_in": "cp",
            "wp_in": "w_prime",
            "crank": "crank_length",
        }

    def init_session_state(self) -> None:
        """Initialize session state with all registered defaults, then overlay saved settings."""
        # 1. Set ALL registered defaults
        for key, default in self._defaults.items():
            st.session_state.setdefault(key, default)

        # 2. Overlay persisted settings (only for _keys_map entries)
        saved = self.settings_manager.load_settings()
        for ui_key, json_key in self._keys_map.items():
            if json_key in saved:
                st.session_state[ui_key] = saved[json_key]

    def load_profile_into_state(self, profile_id: str) -> None:
        """Load an athlete profile's parameters into session state."""
        from modules.db.athlete_profiles import AthleteProfileStore

        store = AthleteProfileStore()
        profile = store.get_profile(profile_id)
        if not profile:
            return
        settings_dict = profile.to_settings_dict()
        for ui_key, json_key in self._keys_map.items():
            st.session_state[ui_key] = settings_dict.get(json_key)

    def save_settings_callback(self) -> None:
        """Callback to save current UI values to persistence."""
        current_values = {}
        for ui_key, json_key in self._keys_map.items():
            if ui_key in st.session_state:
                current_values[json_key] = st.session_state[ui_key]

        selected_id = st.session_state.get("selected_athlete_id")
        if selected_id:
            from modules.db.athlete_profiles import AthleteProfileStore, AthleteProfile

            store = AthleteProfileStore()
            profile = store.get_profile(selected_id)
            if profile:
                updated = AthleteProfile.from_settings_dict(
                    name=profile.name,
                    settings=current_values,
                    profile_id=profile.id,
                )
                store.update_profile(updated)

        self.settings_manager.save_settings(current_values)

    def cleanup_old_data(self) -> None:
        """Clean up old DataFrames from session state."""
        keys_to_check = ["_prev_df_plot", "_prev_df_resampled", "_prev_file_name", "data_loaded"]
        for key in keys_to_check:
            if key in st.session_state:
                del st.session_state[key]

    def set_data_loaded(self) -> None:
        st.session_state["data_loaded"] = True

    def is_data_loaded(self) -> bool:
        return st.session_state.get("data_loaded", False)
