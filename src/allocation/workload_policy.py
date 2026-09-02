"""P4 -- Workload Aware Allocation (spec section 10).

Considers driver workload rather than raw ride count, since raw counts
are misleading when drivers have been online for very different
lengths of time:

    Workload_i = RidesCompleted_i / OnlineHours_i

IMPORTANT: ``Driver.online_time`` is only recomputed by
``Simulator.run()`` once, at the very end of the simulation -- reading
it mid-simulation would silently return 0.0 for every driver the whole
time, collapsing this policy into pure ETA minimization without
raising any error. To avoid that trap, this policy computes each
driver's online time itself, as ``ride.request_time -
driver.went_online_at`` -- ``ride.request_time`` *is* "now" at the
moment ``select_driver`` is called.

A driver with (near-)zero online time so far gets Workload = 0
(treated as maximally under-worked / most eligible) rather than
raising a division error.

Combines workload with pickup ETA via the same per-decision, min-max
normalized weighted score used by P3:

    Score(d) = w_eta * ETA_norm(d) + w_workload * Workload_norm(d)

Both terms are *minimized*. Ties are broken by driver_id.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

from src.allocation.base import AllocationPolicy
from src.allocation.scoring_utils import min_max_normalize
from src.models.driver import Driver
from src.models.ride import Ride


class ETAProvider(Protocol):
    def travel_time(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        ...


def _workload(driver: Driver, now: float) -> float:
    online_time = now - driver.went_online_at
    if online_time <= 1e-9:
        return 0.0
    return driver.rides_completed / online_time


class WorkloadAwarePolicy(AllocationPolicy):
    """Weighted score of pickup ETA and driver workload, both minimized."""

    name = "workload_aware"

    def __init__(self, eta_provider: ETAProvider, w_eta: float = 0.5, w_workload: float = 0.5):
        if w_eta < 0 or w_workload < 0:
            raise ValueError("weights must be non-negative")
        self.eta_provider = eta_provider
        self.w_eta = w_eta
        self.w_workload = w_workload

    def select_driver(self, ride: Ride, available_drivers: List[Driver]) -> Optional[Driver]:
        if not available_drivers:
            return None
        if len(available_drivers) == 1:
            return available_drivers[0]

        now = ride.request_time
        etas = [
            self.eta_provider.travel_time(d.location, ride.pickup_location)
            for d in available_drivers
        ]
        workloads = [_workload(d, now) for d in available_drivers]

        eta_norm = min_max_normalize(etas)
        workload_norm = min_max_normalize(workloads)

        scores = [
            self.w_eta * eta_norm[i] + self.w_workload * workload_norm[i]
            for i in range(len(available_drivers))
        ]

        best_score = min(scores)
        tied = [
            available_drivers[i]
            for i, s in enumerate(scores)
            if s == best_score
        ]
        return min(tied, key=lambda d: d.driver_id)