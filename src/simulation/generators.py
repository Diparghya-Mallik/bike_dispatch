"""Synthetic demand and driver generation.

Produces reproducible ``Ride`` request streams and initial ``Driver``
placements for a given ``City``. All randomness is drawn from an
explicit, seeded ``random.Random`` instance passed in by the caller (see
spec section 20: "Never compare two policies using only one random
simulation" and section 25: "Keep random number generation
reproducible") -- nothing here touches global random state.

Ride arrivals follow a Poisson process (i.e. exponential inter-arrival
times) at a configurable constant rate. Spatial placement of pickups
supports a "uniform" pattern and a simple "clustered" pattern (spec
section 19). Temporal patterns (peak / peak+recovery) and additional
spatial patterns can be layered on top of ``generate_rides`` in later
phases by varying the rate over time; Phase 1 keeps the rate constant.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.models.driver import Driver
from src.models.ride import Ride
from src.simulation.environment import City

Coordinate = Tuple[float, float]


@dataclass
class DemandConfig:
    """Configuration for synthetic ride-request generation.

    Attributes:
        arrival_rate: Expected number of ride requests per unit of
            simulation time (Poisson rate, lambda).
        duration: Total simulation time over which to generate requests.
        spatial_pattern: "uniform" or "clustered".
        num_clusters: Number of demand clusters, used only when
            spatial_pattern == "clustered".
        cluster_std: Standard deviation (in city distance units) of
            pickup locations around a cluster center.
    """

    arrival_rate: float
    duration: float
    spatial_pattern: str = "uniform"
    num_clusters: int = 3
    cluster_std: float = 8.0

    def __post_init__(self) -> None:
        if self.arrival_rate < 0:
            raise ValueError("arrival_rate must be non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.spatial_pattern not in ("uniform", "clustered"):
            raise ValueError(
                f"Unknown spatial_pattern: {self.spatial_pattern!r}"
            )


class DemandGenerator:
    """Generates reproducible synthetic ride requests for a City."""

    def __init__(self, city: City, config: DemandConfig):
        self.city = city
        self.config = config

    def _clip(self, point: Coordinate) -> Coordinate:
        x = min(max(point[0], 0.0), self.city.width)
        y = min(max(point[1], 0.0), self.city.height)
        return (x, y)

    def _sample_location(
        self, rng: random.Random, cluster_centers: Optional[List[Coordinate]]
    ) -> Coordinate:
        if self.config.spatial_pattern == "uniform" or not cluster_centers:
            return self.city.random_point(rng)

        center = rng.choice(cluster_centers)
        point = (
            rng.gauss(center[0], self.config.cluster_std),
            rng.gauss(center[1], self.config.cluster_std),
        )
        return self._clip(point)

    def generate_rides(self, rng: random.Random) -> List[Ride]:
        """Generate the full stream of ride requests for the simulation.

        Args:
            rng: Seeded random.Random instance controlling all draws.

        Returns:
            Rides sorted by request_time, each with a distinct pickup
            and destination location (and trip_distance populated).
        """
        cluster_centers: Optional[List[Coordinate]] = None
        if self.config.spatial_pattern == "clustered":
            cluster_centers = [
                self.city.random_point(rng)
                for _ in range(self.config.num_clusters)
            ]

        rides: List[Ride] = []
        t = 0.0
        counter = itertools.count(1)

        while True:
            if self.config.arrival_rate <= 0:
                break
            inter_arrival = rng.expovariate(self.config.arrival_rate)
            t += inter_arrival
            if t >= self.config.duration:
                break

            pickup = self._sample_location(rng, cluster_centers)
            destination = self._sample_location(rng, cluster_centers)
            # Avoid a zero-length trip if pickup == destination by chance.
            while destination == pickup:
                destination = self._sample_location(rng, cluster_centers)

            ride_id = f"ride_{next(counter):06d}"
            ride = Ride(
                ride_id=ride_id,
                request_time=t,
                pickup_location=pickup,
                destination=destination,
            )
            ride.trip_distance = self.city.distance(pickup, destination)
            rides.append(ride)

        return rides

    def generate_drivers(
        self, count: int, rng: random.Random, went_online_at: float = 0.0
    ) -> List[Driver]:
        """Generate initial driver placements, uniformly at random.

        Driver supply is intentionally always placed uniformly regardless
        of ``spatial_pattern`` -- that config only controls where *demand*
        appears, so supply/demand mismatch scenarios (spec section 19)
        can be studied.

        Args:
            count: Number of drivers to generate.
            rng: Seeded random.Random instance.
            went_online_at: Simulation time at which all drivers are
                considered to have come online.
        """
        drivers: List[Driver] = []
        for i in range(count):
            location = self.city.random_point(rng)
            driver = Driver(
                driver_id=f"driver_{i:04d}",
                location=location,
                went_online_at=went_online_at,
                last_state_change_at=went_online_at,
            )
            drivers.append(driver)
        return drivers
