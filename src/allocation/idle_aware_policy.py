"""P3 -- Idle-Time Aware Allocation (spec section 10).

Favors drivers who have been idle longer, while still weighing pickup
ETA, via a normalized weighted score:

    Score(d) = -w_eta * ETA_norm(d) + w_idle * IdleTime_norm(d)

IMPORTANT: ``Driver.idle_time`` only accumulates when a driver
*transitions out* of AVAILABLE (see ``Driver.transition_to``) -- it's
banked retroactively, at the end of an idle stretch. A driver who has
been idle since the simulation started, and has never yet been
assigned a ride, has an *unbanked* idle stretch in progress and shows
``idle_time == 0.0`` the entire time, no matter how long they've
actually been waiting. Reading ``driver.idle_time`` directly here would
silently ignore exactly the drivers this policy is supposed to favor
most. To avoid that, this policy computes each available driver's
*live* idle time as their banked ``idle_time`` plus however long their
current (in-progress) idle stretch has lasted:

    live_idle_time(d) = d.idle_time + (now - d.last_state_change_at)

where ``now`` is ``ride.request_time`` -- the simulator invokes
``select_driver`` synchronously at request time, so that *is* now. This
is safe because every driver passed in ``available_drivers`` is, by
construction, currently in the AVAILABLE state.

Both terms are min-max normalized across the *currently available*
drivers for this specific ride. The policy picks the driver that
*maximizes* the score: low ETA and high idle time both push the score
up. Ties are broken by driver_id.
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


def _live_idle_time(driver: Driver, now: float) -> float:
    """Banked idle_time plus the in-progress idle stretch, if any.

    Assumes ``driver`` is currently AVAILABLE (true for every driver the
    simulator ever passes to select_driver).
    """
    in_progress = max(now - driver.last_state_change_at, 0.0)
    return driver.idle_time + in_progress


class IdleAwarePolicy(AllocationPolicy):
    """Weighted score of pickup ETA (minimize) and idle time (maximize)."""

    name = "idle_aware"

    def __init__(self, eta_provider: ETAProvider, w_eta: float = 0.7, w_idle: float = 0.3):
        if w_eta < 0 or w_idle < 0:
            raise ValueError("weights must be non-negative")
        self.eta_provider = eta_provider
        self.w_eta = w_eta
        self.w_idle = w_idle

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
        idles = [_live_idle_time(d, now) for d in available_drivers]

        eta_norm = min_max_normalize(etas)
        idle_norm = min_max_normalize(idles)

        scores = [
            -self.w_eta * eta_norm[i] + self.w_idle * idle_norm[i]
            for i in range(len(available_drivers))
        ]

        best_score = max(scores)
        tied = [
            available_drivers[i]
            for i, s in enumerate(scores)
            if s == best_score
        ]
        return min(tied, key=lambda d: d.driver_id)