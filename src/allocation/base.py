"""Allocation policy interface (spec section 9).

The simulator calls a policy without knowing which specific strategy is
being used, so policies can be swapped without modifying the simulation
engine. Concrete policies (P0 random, P1 nearest, P2 min-ETA, ...) are
implemented in later phases, each as its own module in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.driver import Driver
from src.models.ride import Ride


class AllocationPolicy(ABC):
    """Common interface every allocation strategy must implement."""

    #: Human-readable identifier used in experiment configs and result logs.
    name: str = "base_policy"

    @abstractmethod
    def select_driver(
        self, ride: Ride, available_drivers: List[Driver]
    ) -> Optional[Driver]:
        """Choose a driver for the given ride.

        Args:
            ride: The ride request needing a driver.
            available_drivers: Drivers currently in the AVAILABLE state.
                May be empty.

        Returns:
            The chosen Driver, or None if no assignment should be made
            (e.g. no available drivers, or the policy chooses to defer).
        """
        raise NotImplementedError
