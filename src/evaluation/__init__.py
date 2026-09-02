"""Rider/driver/fairness/system metrics, independent of allocation logic.

Phase 3 adds the core metric bundle (rider_metrics, driver_metrics,
fairness_metrics, system_metrics) and the ``evaluate_simulation`` entry
point in evaluator.py. Cross-seed statistics (confidence intervals,
effect sizes) are added in statistics.py at Phase 8.
"""

from .evaluator import evaluate_simulation

__all__ = ["evaluate_simulation"]