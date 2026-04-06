"""
Frontend Layout Manager.

Handles the main application shell, sidebar, and high-level routing.
"""

import logging

import streamlit as st
from typing import Tuple, Any

from modules.config import Config
from .state import StateManager

logger = logging.getLogger(__name__)


class AppLayout:
    """Main application layout and shell."""

    def __init__(self, state_manager: StateManager):
        self.state = state_manager


    def render_sidebar(self) -> Tuple[Any, dict]:
        """Render the application sidebar.

        Returns:
            Tuple of (uploaded_file, user_params_dict)
        """
        st.sidebar.header("Ustawienia Zawodnika")

        params = {}

        with st.sidebar.expander("⚙️ Parametry Fizyczne", expanded=True):
            params["rider_weight"] = st.number_input(
                "Waga Zawodnika [kg]",
                step=0.5,
                min_value=30.0,
                max_value=200.0,
                key="weight",
                on_change=self.state.save_settings_callback,
            )
            params["rider_height"] = st.number_input(
                "Wzrost [cm]",
                step=1,
                min_value=100,
                max_value=250,
                key="height",
                on_change=self.state.save_settings_callback,
            )
            params["rider_age"] = st.number_input(
                "Wiek [lata]",
                step=1,
                min_value=10,
                max_value=100,
                key="age",
                on_change=self.state.save_settings_callback,
            )
            params["is_male"] = st.checkbox(
                "Mężczyzna?", key="gender_m", on_change=self.state.save_settings_callback
            )

            st.markdown("---")
            params["vt1_watts"] = st.number_input(
                "VT1 (Próg Tlenowy) [W]",
                min_value=0,
                value=0,
                key="vt1_w",
                on_change=self.state.save_settings_callback,
            )
            params["vt2_watts"] = st.number_input(
                "VT2 (Próg Beztlenowy/FTP) [W]",
                min_value=0,
                value=0,
                key="vt2_w",
                on_change=self.state.save_settings_callback,
            )
            st.divider()

            st.markdown("### 🫁 Wentylacja [L/min]")
            params["vt1_vent"] = st.number_input(
                "VT1 (Próg Tlenowy) [L/min]",
                min_value=0.0,
                key="vt1_v",
                on_change=self.state.save_settings_callback,
            )
            params["vt2_vent"] = st.number_input(
                "VT2 (Próg Beztlenowy) [L/min]",
                min_value=0.0,
                key="vt2_v",
                on_change=self.state.save_settings_callback,
            )

        st.sidebar.divider()
        params["cp"] = st.sidebar.number_input(
            "Moc Krytyczna (CP) [W]",
            min_value=1,
            key="cp_in",
            on_change=self.state.save_settings_callback,
        )
        params["w_prime"] = st.sidebar.number_input(
            "W' (W Prime) [J]",
            min_value=0,
            key="wp_in",
            on_change=self.state.save_settings_callback,
        )
        st.sidebar.divider()
        params["crank_length"] = st.sidebar.number_input(
            "Długość korby [mm]", key="crank", on_change=self.state.save_settings_callback
        )
        # File upload with size limit (50 MB max) and content validation
        MAX_FILE_SIZE_MB = 50
        uploaded_file = st.sidebar.file_uploader(
            "Wgraj plik (CSV / TXT)", type=["csv", "txt"], accept_multiple_files=False
        )
        if uploaded_file and uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.sidebar.error(f"Plik za duzy. Maksymalny rozmiar: {MAX_FILE_SIZE_MB} MB")
            uploaded_file = None
        if uploaded_file is not None:
            uploaded_file = self._validate_upload_content(uploaded_file)

        return uploaded_file, params

    @staticmethod
    def _validate_upload_content(uploaded_file):
        """Server-side content validation for uploaded files.

        Checks that the file content is valid UTF-8 text with CSV structure.
        Returns the file if valid, None otherwise.
        """
        try:
            header = uploaded_file.read(1024)
            uploaded_file.seek(0)

            # Must be decodable as UTF-8 text
            try:
                header_text = header.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                st.sidebar.error("Plik nie jest prawidlowym plikiem tekstowym (CSV/TXT).")
                return None

            # Must contain at least one comma or tab (CSV/TSV structure)
            if "," not in header_text and "\t" not in header_text:
                st.sidebar.error("Plik nie zawiera danych CSV (brak separatorow).")
                return None

            # Must not contain null bytes (binary file disguised as CSV)
            if "\x00" in header_text:
                logger.warning("Upload rejected: binary content detected (null bytes)")
                st.sidebar.error("Plik zawiera dane binarne — tylko CSV/TXT.")
                return None

            return uploaded_file

        except Exception as e:
            logger.error("Upload content validation failed: %s", e, exc_info=True)
            st.sidebar.error("Blad walidacji pliku. Sprawdz format.")
            return None

    def render_header(self) -> None:
        """Render the main header."""
        st.title(f"{Config.APP_ICON} {Config.APP_TITLE}")

    def render_export_section(self, uploaded_file, data_bundle) -> None:
        """Render the export section in sidebar."""
        # Logic extracted from app.py
        # Passed data_bundle is a dict of whatever is needed for export
        if not uploaded_file:
            return

        st.sidebar.markdown("---")
        st.sidebar.header("📄 Export Raportu")

        # NOTE: This part requires importing report functions.
        # To keep layout decoupled, better to pass a callback or specific UI renderer.
        # For now, let's keep it minimal and assume app.py handles the actual button logic
        # OR we import specifically here.
        pass  # Leaving empty to avoid circular imports. Best handled in app.py or dedicated logic.
