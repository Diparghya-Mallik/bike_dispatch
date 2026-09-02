"""Evaluation engine entry point (spec section 12).

``evaluate_simulation`` is the one function most callers need: it takes
a ``SimulationResult`` (produced by ``Simulator.run()``, from *any*
allocation policy) and returns the four-way metric bundle -- rider,
driver, fairness, system -- as a plain nested dict, ready to be
flattened into a DataFrame row for Phase 7's multi-seed experiment
runner and Phase 8's statistical analysis.

This module deliberately contains no statistics beyond single-run
summaries (means, percentiles, Jain's/Gini). Confidence intervals,
effect sizes, and cross-seed aggregation belong to
``src/evaluation/statistics.py`` (Phase 8), not here -- this module
only ever sees one simulation at a time.
"""

from __future__ import annotations

from typing import Dict

from src.evaluation.driver_metrics import compute_driver_metrics
from src.evaluation.fairness_metrics import compute_fairness_metrics
from src.evaluation.rider_metrics import compute_rider_metrics
from src.evaluation.system_metrics import compute_system_metrics
from src.simulation.simulator import SimulationResult


def evaluate_simulation(result: SimulationResult) -> Dict[str, object]:
    """Compute the full rider/driver/fairness/system metric bundle."""
    ride_results = list(result.ride_results.values())

    return {
        "rider": compute_rider_metrics(ride_results),
        "driver": compute_driver_metrics(result.drivers),
        "fairness": compute_fairness_metrics(result.drivers, result.events),
        "system": compute_system_metrics(ride_results, result.drivers, result.total_runtime),
    }