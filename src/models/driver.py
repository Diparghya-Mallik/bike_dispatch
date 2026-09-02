"""Driver model.

A Driver represents a single bike-taxi driver in the simulation. Drivers
move between a small set of states as they are assigned rides, pick up
riders, complete trips, and return to being available.

State machine (see project spec, section 7):

    AVAILABLE -> ASSIGNED -> PICKING_UP -> ON_RIDE -> AVAILABLE

Only drivers in the AVAILABLE state are eligible to receive new ride
assignments. This module intentionally contains no allocation logic --
it only tracks driver state and simple bookkeeping (idle/busy time,
rides completed, earnings placeholder, current location).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

Coordinate = Tuple[float, float]


class DriverState(Enum):
    """The finite set of states a driver can be in."""

    AVAILABLE = auto()
    ASSIGNED = auto()
    PICKING_UP = auto()
    ON_RIDE = auto()


# Legal transitions, used by Driver.transition_to() to catch modeling bugs
# early rather than silently allowing an invalid state change.
_VALID_TRANSITIONS = {
    DriverState.AVAILABLE: {DriverState.ASSIGNED},
    DriverState.ASSIGNED: {DriverState.PICKING_UP, DriverState.AVAILABLE},
    DriverState.PICKING_UP: {DriverState.ON_RIDE, DriverState.AVAILABLE},
    DriverState.ON_RIDE: {DriverState.AVAILABLE},
}


@dataclass
class Driver:
    """A single bike-taxi driver.

    Attributes:
        driver_id: Unique identifier for this driver.
        location: Current (x, y) coordinate of the driver.
        status: Current DriverState.
        online_time: Total time (sim units) the driver has been online.
        idle_time: Total time spent in the AVAILABLE state.
        busy_time: Total time spent in ASSIGNED/PICKING_UP/ON_RIDE states.
        rides_completed: Count of rides fully completed.
        earnings: Cumulative (placeholder) earnings.
        current_ride: ride_id of the ride currently being handled, if any.
        went_online_at: Sim time the driver came online (for bookkeeping).
        last_state_change_at: Sim time of the most recent state transition,
            used internally to accumulate idle_time / busy_time correctly.
    """

    driver_id: str
    location: Coordinate
    status: DriverState = DriverState.AVAILABLE
    online_time: float = 0.0
    idle_time: float = 0.0
    busy_time: float = 0.0
    rides_completed: int = 0
    earnings: float = 0.0
    current_ride: Optional[str] = None
    went_online_at: float = 0.0
    last_state_change_at: float = field(default=0.0, repr=False)

    def is_available(self) -> bool:
        return self.status is DriverState.AVAILABLE

    def transition_to(self, new_status: DriverState, now: float) -> None:
        """Move the driver to a new state, updating idle/busy bookkeeping.

        Args:
            new_status: The state to transition into.
            now: Current simulation time, used to accrue idle/busy time
                since the last transition.

        Raises:
            ValueError: If the transition is not a legal one per the
                driver state machine.
        """
        if new_status not in _VALID_TRANSITIONS[self.status]:
            raise ValueError(
                f"Illegal driver transition for {self.driver_id}: "
                f"{self.status.name} -> {new_status.name}"
            )

        elapsed = now - self.last_state_change_at
        if elapsed < 0:
            raise ValueError(
                f"Non-monotonic simulation time for driver {self.driver_id}: "
                f"now={now} < last_state_change_at={self.last_state_change_at}"
            )

        if self.status is DriverState.AVAILABLE:
            self.idle_time += elapsed
        else:
            self.busy_time += elapsed

        self.status = new_status
        self.last_state_change_at = now

    def move_to(self, location: Coordinate) -> None:
        """Update the driver's current location."""
        self.location = location

    def complete_ride(self, now: float) -> None:
        """Convenience helper: finish a ride and return to AVAILABLE."""
        self.transition_to(DriverState.AVAILABLE, now)
        self.rides_completed += 1
        self.current_ride = None

    def accrue_online_time(self, now: float) -> None:
        """Recompute online_time as elapsed time since going online.

        Should be called by the simulator/evaluation layer when a
        snapshot of online_time is needed (e.g. at simulation end or
        periodically), rather than incrementally, to avoid double
        counting.
        """
        self.online_time = now - self.went_online_at

    def utilization(self) -> float:
        """Fraction of online time spent busy. Returns 0.0 if never online."""
        if self.online_time <= 0:
            return 0.0
        return self.busy_time / self.online_time
