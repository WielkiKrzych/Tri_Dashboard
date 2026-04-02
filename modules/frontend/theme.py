"""
Frontend Theme Management.

Handles CSS loading and theme configuration (dark mode only).
"""

import logging

import streamlit as st
from modules.config import Config

logger = logging.getLogger(__name__)

THEMES = {
    "dark": "style.css",
}


class ThemeManager:
    """Manages application theme and styling."""

    @staticmethod
    def load_css(theme: str = "dark") -> None:
        """Load global CSS from file based on theme selection.

        Args:
            theme: Theme name ('dark' or 'light'). Defaults to 'dark'.
        """
        css_filename = THEMES.get(theme, "style.css")
        css_file = Config.BASE_DIR / css_filename

        # Validate CSS file stays within project directory
        if not css_file.resolve().is_relative_to(Config.BASE_DIR.resolve()):
            logger.warning("CSS path escapes project directory, falling back to default")
            css_file = Config.BASE_DIR / "style.css"

        try:
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            logger.error("Failed to load CSS: %s", e)
            st.error("Nie udalo sie zaladowac stylow CSS.")

    @staticmethod
    def set_page_config() -> None:
        """Apply Streamlit page configuration."""
        st.set_page_config(
            page_title=Config.APP_TITLE, layout=Config.APP_LAYOUT, page_icon=Config.APP_ICON
        )
