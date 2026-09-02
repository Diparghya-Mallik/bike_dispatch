"""Driver-side metrics (spec section 14).

Operates purely on ``Driver`` objects (their post-simulation state:
idle_time, busy_time, online_time, rides_completed). No allocation
logic, no knowledge of the policy that produced this state.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.models.driver import Driver


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def idle_time_stats(drivers: Iterable[Driver]) -> Dict[str, float]:
    return _summary([d.idle_time for d in drivers])


def utilization_stats(drivers: Iterable[Driver]) -> Dict[str, float]:
    return _summary([d.utilization() for d in drivers])


def rides_per_driver_stats(drivers: Iterable[Driver]) -> Dict[str, float]:
    """Distribution of rides completed per driver, plus coefficient of
    variation (std/mean) -- spec section 14 explicitly asks to track
    the *distribution*, not just the mean, since that's where allocation
    unfairness shows up."""
    drivers = list(drivers)
    values = [d.rides_completed for d in drivers]
    summary = _summary(values)
    mean = summary["mean"]
    std = summary["std"]
    summary["coefficient_of_variation"] = (std / mean) if mean else None
    return summary


def compute_driver_metrics(drivers: Iterable[Driver]) -> Dict[str, object]:
    drivers = list(drivers)
    return {
        "num_drivers": len(drivers),
        "idle_time": idle_time_stats(drivers),
        "utilization": utilization_stats(drivers),
        "rides_per_driver": rides_per_driver_stats(drivers),
    }