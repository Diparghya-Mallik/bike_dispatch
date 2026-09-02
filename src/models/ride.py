"""Ride request model.

A Ride represents a single rider's request for a trip. This module only
models the request itself and its lifecycle status -- it does not decide
which driver is assigned (that is the allocation policy's job, see
src/allocation/) and it does not record what actually happened during
the trip (that is the Event / ride-result record, see events.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

Coordinate = Tuple[float, float]


class RideStatus(Enum):
    """Lifecycle status of a ride request."""

    REQUESTED = auto()
    ASSIGNED = auto()
    DRIVER_EN_ROUTE = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    UNSERVED = auto()  # No driver could be assigned (e.g. none available)


@dataclass
class Ride:
    """A single ride request.

    Attributes:
        ride_id: Unique identifier for this ride request.
        request_time: Simulation time the request was made.
        pickup_location: (x, y) coordinate where the rider is waiting.
        destination: (x, y) coordinate of the ride's destination.
        status: Current RideStatus.
        estimated_fare: Optional placeholder for a later fare model.
        trip_distance: Euclidean distance from pickup to destination.
            Populated by the generator/simulator, not computed here, to
            keep this module free of geometry assumptions.
        trip_duration: Optional estimated/actual trip duration.
        assigned_driver_id: driver_id of the assigned driver, if any.
    """

    ride_id: str
    request_time: float
    pickup_location: Coordinate
    destination: Coordinate
    status: RideStatus = RideStatus.REQUESTED
    estimated_fare: Optional[float] = None
    trip_distance: Optional[float] = None
    trip_duration: Optional[float] = None
    assigned_driver_id: Optional[str] = None

    def mark_assigned(self, driver_id: str) -> None:
        self.assigned_driver_id = driver_id
        self.status = RideStatus.ASSIGNED

    def mark_unserved(self) -> None:
        self.status = RideStatus.UNSERVED

    def mark_en_route(self) -> None:
        self.status = RideStatus.DRIVER_EN_ROUTE

    def mark_in_progress(self) -> None:
        self.status = RideStatus.IN_PROGRESS

    def mark_completed(self) -> None:
        self.status = RideStatus.COMPLETED
