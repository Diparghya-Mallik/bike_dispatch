"""Small shared helpers for score-based allocation policies (Phase 4+)."""

from __future__ import annotations

from typing import List


def min_max_normalize(values: List[float]) -> List[float]:
    """Scale values to [0, 1] via min-max normalization.

    If all values are equal (including a single-element or empty list),
    returns all zeros rather than dividing by zero -- in that case the
    term carries no discriminating information for this decision.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]