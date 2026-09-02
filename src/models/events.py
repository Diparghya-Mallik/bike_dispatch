"""Event model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional


class EventType(Enum):
    RIDE_REQUESTED = auto()
    DRIVER_ASSIGNED = auto()
    PICKUP_STARTED = auto()
    RIDER_PICKED_UP = auto()
    RIDE_STARTED = auto()
    RIDE_COMPLETED = auto()
    DRIVER_AVAILABLE = auto()
    RIDE_UNSERVED = auto()


@dataclass(frozen=True)
class Event:
    event_type: EventType
    time: float
    ride_id: Optional[str] = None
    driver_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        record = {
            "event_type": self.event_type.name,
            "time": self.time,
            "ride_id": self.ride_id,
            "driver_id": self.driver_id,
        }
        if self.metadata:
            record.update(self.metadata)
        return record


@dataclass
class RideResult:
    """Per-ride outcome record, per spec section 7."""

    ride_id: str
    request_time: float
    assignment_time: Optional[float] = None
    pickup_time: Optional[float] = None
    ride_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    driver_id: Optional[str] = None
    pickup_distance: Optional[float] = None
    pickup_eta: Optional[float] = None
    served: bool = False
    allocation_compute_time: Optional[float] = None
    """Wall-clock seconds spent inside policy.select_driver() for this
    ride. Populated by the simulator regardless of whether a driver was
    found, so the evaluation engine's system metrics (spec section 16:
    "Average allocation computation time", "P95 allocation computation
    time") can measure allocation cost even for unserved requests."""

    def waiting_time(self) -> Optional[float]:
        if self.pickup_time is None:
            return None
        return self.pickup_time - self.request_time

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ride_id": self.ride_id,
            "request_time": self.request_time,
            "assignment_time": self.assignment_time,
            "pickup_time": self.pickup_time,
            "ride_start_time": self.ride_start_time,
            "completion_time": self.completion_time,
            "driver_id": self.driver_id,
            "pickup_distance": self.pickup_distance,
            "pickup_eta": self.pickup_eta,
            "served": self.served,
            "waiting_time": self.waiting_time(),
            "allocation_compute_time": self.allocation_compute_time,
        }