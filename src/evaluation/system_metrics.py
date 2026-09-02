"""System-level metrics (spec section 16).

Aggregate, whole-simulation numbers -- how many rides got done, how
much driving it took, and how expensive the allocation decisions
themselves were to compute. This last part (allocation compute time)
is what will let Phase 6+ compare greedy heuristics against OR-Tools
optimization on cost, not just quality.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.models.driver import Driver
from src.models.events import RideResult


def _percentile_or_none(values: List[float], p: float):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p))


def compute_system_metrics(
    ride_results: Iterable[RideResult],
    drivers: Iterable[Driver],
    total_runtime: float,
) -> Dict[str, object]:
    ride_results = list(ride_results)
    drivers = list(drivers)

    served = [r for r in ride_results if r.served]
    unserved = [r for r in ride_results if not r.served]

    total_busy = sum(d.busy_time for d in drivers)
    total_online = sum(d.online_time for d in drivers)
    overall_utilization = (total_busy / total_online) if total_online > 0 else 0.0

    total_pickup_distance = sum(
        r.pickup_distance for r in ride_results if r.pickup_distance is not None
    )

    compute_times = [
        r.allocation_compute_time
        for r in ride_results
        if r.allocation_compute_time is not None
    ]
    mean_compute_time = (
        float(np.mean(compute_times)) if compute_times else None
    )

    return {
        "completed_rides": len(served),
        "unserved_requests": len(unserved),
        "total_requests": len(ride_results),
        "total_pickup_distance": total_pickup_distance,
        "overall_driver_utilization": overall_utilization,
        "mean_allocation_compute_time_s": mean_compute_time,
        "p95_allocation_compute_time_s": _percentile_or_none(compute_times, 95),
        "total_simulation_runtime_s": total_runtime,
    }