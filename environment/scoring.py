"""
Utilities for hackathon-compliant score handling.

Task and episode scores must stay strictly inside the open interval (0, 1).
"""

from __future__ import annotations

import math


MIN_OPENENV_SCORE = 0.01
MAX_OPENENV_SCORE = 0.99


def clamp_openenv_score(value: float, *, precision: int = 4) -> float:
    """Clamp a score into the hackathon's required open interval."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = MIN_OPENENV_SCORE

    if not math.isfinite(numeric):
        numeric = MIN_OPENENV_SCORE

    bounded = min(MAX_OPENENV_SCORE, max(MIN_OPENENV_SCORE, numeric))
    return round(bounded, precision)


def format_openenv_score(value: float) -> str:
    """Format a score for stdout using the required 2-decimal contract."""
    return f"{clamp_openenv_score(value, precision=2):.2f}"
