"""Shared utilities for UI modules."""
from typing import Union, Optional, Dict, Any
import pandas as pd

# Re-export ensure_pandas for UI modules
from modules.calculations import ensure_pandas

__all__ = ['ensure_pandas']
