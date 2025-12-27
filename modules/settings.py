import json
import os
import streamlit as st

SETTINGS_FILE = 'user_settings.json'

class SettingsManager:
    def __init__(self, file_path=SETTINGS_FILE):
        self.file_path = file_path
        self.default_settings = {
            "rider_weight": 95.0,
            "rider_height": 180,
            "rider_age": 30,
            "is_male": True,
            "cp": 410,
            "w_prime": 31000,
            "vt1_watts": 280,
            "vt2_watts": 400,
            "vt1_vent": 79.0,
            "vt2_vent": 136.0,
            "crank_length": 160.0
        }

    def load_settings(self):
        """Wczytuje ustawienia z pliku JSON. Jeśli plik nie istnieje, zwraca domyślne."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**self.default_settings, **settings}
            except Exception as e:
                st.error(f"Błąd wczytywania ustawień: {e}")
                return self.default_settings
        else:
            return self.default_settings

    def save_settings(self, settings_dict):
        """Zapisuje słownik ustawień do pliku JSON."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(settings_dict, f, indent=4)
            return True
        except Exception as e:
            st.error(f"Błąd zapisywania ustawień: {e}")
            return False

    def get_ui_values(self):
        """Pomocnik do pobierania wartości do UI (Session State lub Load)."""
        # Jeśli settings są już w session_state, użyj ich. Jak nie, wczytaj z pliku.
        if 'user_settings' not in st.session_state:
            st.session_state['user_settings'] = self.load_settings()
        return st.session_state['user_settings']

    def update_from_ui(self, key, value):
        """Callback do aktualizacji konkretnego ustawienia."""
        if 'user_settings' not in st.session_state:
             st.session_state['user_settings'] = self.load_settings()
        
        st.session_state['user_settings'][key] = value
        self.save_settings(st.session_state['user_settings'])
