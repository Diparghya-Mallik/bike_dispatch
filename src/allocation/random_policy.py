"""P0 -- Random Allocation (spec section 10).

Randomly selects any available driver, with no regard for distance,
ETA, idle time, or workload. This is a deliberately weak baseline: any
policy that can't beat random allocation on rider-side metrics isn't
doing anything useful, and any policy that can't beat it on fairness
metrics isn't doing anything useful either (random allocation is, by
construction, roughly fair in expectation over many rides).

Purpose per spec: weak baseline / sanity check.
"""

from __future__ import annotations

import random
from typing import List, Optional

from src.allocation.base import AllocationPolicy
from src.models.driver import Driver
from src.models.ride import Ride


class RandomPolicy(AllocationPolicy):
    """Selects uniformly at random among available drivers.

    Takes an explicit ``random.Random`` instance rather than touching
    global random state, so allocation randomness can be seeded and
    reproduced independently of demand-generation randomness (spec
    section 20/25: reproducibility via explicit seeds).
    """

    name = "random"

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()

    def select_driver(
        self, ride: Ride, available_drivers: List[Driver]
    ) -> Optional[Driver]:
        if not available_drivers:
            return None
        return self.rng.choice(available_drivers)
