"""City geometry.

Per spec section 6, the initial modeling assumption is a plain 2D
coordinate space rather than a real road network, with Euclidean
distance between points. This module intentionally contains nothing
else -- no GIS, no road graphs -- so that later phases can swap in a
realistic road-network implementation behind the same small interface
(``distance`` and ``random_point``) without touching the simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class City:
    """A rectangular 2D city of given width/height, with Euclidean distance.

    Attributes:
        width: City extent along the x-axis.
        height: City extent along the y-axis.
        average_speed: Assumed constant travel speed (distance units per
            simulation time unit), used to convert distance to travel
            time. This is a simplifying Phase-1 assumption; later phases
            may make speed variable or road-network-based.
    """

    width: float = 100.0
    height: float = 100.0
    average_speed: float = 1.0

    def distance(self, a: Coordinate, b: Coordinate) -> float:
        """Euclidean distance between two points."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def travel_time(self, a: Coordinate, b: Coordinate) -> float:
        """Time to travel from a to b at the city's average speed."""
        if self.average_speed <= 0:
            raise ValueError("average_speed must be positive")
        return self.distance(a, b) / self.average_speed

    def random_point(self, rng: random.Random) -> Coordinate:
        """A uniformly random point within the city bounds.

        Args:
            rng: A seeded random.Random instance, so callers control
                reproducibility explicitly rather than this method
                touching global random state.
        """
        return (rng.uniform(0.0, self.width), rng.uniform(0.0, self.height))

    def contains(self, point: Coordinate) -> bool:
        x, y = point
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height
