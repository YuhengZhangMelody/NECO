"""
OpenOOD postprocessors package.

Important:
- Do NOT import every postprocessor here, because many of them have optional
  dependencies (libmr, statsmodels, faiss, ...).
- We expose only the BasePostprocessor and the get_postprocessor factory.
"""

from .base_postprocessor import BasePostprocessor  # noqa: F401
from .utils import get_postprocessor  # noqa: F401

__all__ = [
    "BasePostprocessor",
    "get_postprocessor",
]
