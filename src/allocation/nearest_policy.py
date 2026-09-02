"""P1 -- Nearest Driver Allocation (spec section 10).

Selects the available driver with the smallest Euclidean distance to
the ride's pickup location:

    d* = argmin_d distance(d, r)

This is the primary simple baseline and mimics naive proximity-based
dispatch. It requires a ``City`` (or anything exposing a ``distance``
method) to compute distances -- it does not assume Euclidean geometry
itself, so a future road-network City (Phase 10) can be dropped in
without changing this policy.

Ties (multiple drivers at the same minimum distance) are broken by
``driver_id`` so results are deterministic given the same driver pool,
independent of dict/list ordering.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

from src.allocation.base import AllocationPolicy
from src.models.driver import Driver
from src.models.ride import Ride


class DistanceProvider(Protocol):
    def distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        ...


class NearestDriverPolicy(AllocationPolicy):
    """Selects the closest available driver to the ride's pickup point."""

    name = "nearest_driver"

    def __init__(self, distance_provider: DistanceProvider):
        self.distance_provider = distance_provider

    def select_driver(
        self, ride: Ride, available_drivers: List[Driver]
    ) -> Optional[Driver]:
        if not available_drivers:
            return None
        return min(
            available_drivers,
            key=lambda d: (
                self.distance_provider.distance(d.location, ride.pickup_location),
                d.driver_id,
            ),
        )
