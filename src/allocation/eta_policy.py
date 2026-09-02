"""P2 -- Minimum ETA Allocation (spec section 10).

Selects the available driver with the smallest estimated time of
arrival to the ride's pickup location. Initially ETA = distance /
average_speed, computed via an ``eta_provider`` (typically the
``City``) rather than assumed here -- so a future, more realistic ETA
model can be swapped in without changing this policy. With a single
constant city speed, this is numerically identical to P1 Nearest
Driver; the two are kept separate because they diverge once ETA stops
being a pure function of distance.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

from src.allocation.base import AllocationPolicy
from src.models.driver import Driver
from src.models.ride import Ride


class ETAProvider(Protocol):
    def travel_time(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        ...


class ETAPolicy(AllocationPolicy):
    """Selects the available driver with the minimum pickup ETA."""

    name = "min_eta"

    def __init__(self, eta_provider: ETAProvider):
        self.eta_provider = eta_provider

    def select_driver(self, ride: Ride, available_drivers: List[Driver]) -> Optional[Driver]:
        if not available_drivers:
            return None
        return min(
            available_drivers,
            key=lambda d: (
                self.eta_provider.travel_time(d.location, ride.pickup_location),
                d.driver_id,
            ),
        )