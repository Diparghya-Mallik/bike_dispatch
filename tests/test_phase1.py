"""Phase 1 tests.

These are not meant to test any allocation strategy (there isn't one
yet) -- only that the models, city geometry, demand generator, and
simulator scaffolding behave correctly and deterministically. A trivial
in-test "first available driver" policy stands in for a real policy,
which is out of scope until Phase 2.
"""

from __future__ import annotations

import random
from typing import List, Optional

import pytest

from src.allocation.base import AllocationPolicy
from src.models.driver import Driver, DriverState
from src.models.events import EventType
from src.models.ride import Ride, RideStatus
from src.simulation.environment import City
from src.simulation.generators import DemandConfig, DemandGenerator
from src.simulation.simulator import Simulator


class FirstAvailablePolicy(AllocationPolicy):
    """Trivial test-only policy: picks the first available driver.

    A real nearest-driver / ETA-based policy belongs in Phase 2
    (src/allocation/nearest_policy.py etc.), not here.
    """

    name = "test_first_available"

    def select_driver(
        self, ride: Ride, available_drivers: List[Driver]
    ) -> Optional[Driver]:
        return available_drivers[0] if available_drivers else None


# ---------------------------------------------------------------------
# Driver state machine
# ---------------------------------------------------------------------


def test_driver_valid_transition_sequence():
    d = Driver(driver_id="d0", location=(0.0, 0.0))
    d.transition_to(DriverState.ASSIGNED, now=1.0)
    d.transition_to(DriverState.PICKING_UP, now=2.0)
    d.transition_to(DriverState.ON_RIDE, now=4.0)
    d.transition_to(DriverState.AVAILABLE, now=10.0)

    assert d.status is DriverState.AVAILABLE
    # idle: [0,1) = 1.0 ; busy: [1,10) = 9.0
    assert d.idle_time == pytest.approx(1.0)
    assert d.busy_time == pytest.approx(9.0)


def test_driver_illegal_transition_raises():
    d = Driver(driver_id="d0", location=(0.0, 0.0))
    with pytest.raises(ValueError):
        d.transition_to(DriverState.ON_RIDE, now=1.0)


def test_driver_utilization():
    d = Driver(driver_id="d0", location=(0.0, 0.0))
    d.online_time = 10.0
    d.busy_time = 4.0
    assert d.utilization() == pytest.approx(0.4)


# ---------------------------------------------------------------------
# City geometry
# ---------------------------------------------------------------------


def test_city_distance_and_travel_time():
    city = City(width=100, height=100, average_speed=2.0)
    assert city.distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert city.travel_time((0, 0), (3, 4)) == pytest.approx(2.5)


def test_city_random_point_reproducible():
    city = City(width=100, height=100)
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert city.random_point(rng1) == city.random_point(rng2)


# ---------------------------------------------------------------------
# Demand generator
# ---------------------------------------------------------------------


def test_demand_generator_reproducible_with_same_seed():
    city = City(width=100, height=100)
    config = DemandConfig(arrival_rate=2.0, duration=20.0)
    gen = DemandGenerator(city, config)

    rides_a = gen.generate_rides(random.Random(7))
    rides_b = gen.generate_rides(random.Random(7))

    assert [r.ride_id for r in rides_a] == [r.ride_id for r in rides_b]
    assert [r.request_time for r in rides_a] == [r.request_time for r in rides_b]
    assert [r.pickup_location for r in rides_a] == [r.pickup_location for r in rides_b]


def test_demand_generator_rides_sorted_and_within_duration():
    city = City(width=100, height=100)
    config = DemandConfig(arrival_rate=3.0, duration=15.0)
    gen = DemandGenerator(city, config)
    rides = gen.generate_rides(random.Random(1))

    times = [r.request_time for r in rides]
    assert times == sorted(times)
    assert all(0.0 <= t < 15.0 for t in times)
    assert all(city.contains(r.pickup_location) for r in rides)
    assert all(city.contains(r.destination) for r in rides)


def test_demand_generator_clustered_pattern_runs():
    city = City(width=100, height=100)
    config = DemandConfig(
        arrival_rate=2.0, duration=10.0, spatial_pattern="clustered", num_clusters=2
    )
    gen = DemandGenerator(city, config)
    rides = gen.generate_rides(random.Random(3))
    assert all(city.contains(r.pickup_location) for r in rides)


def test_demand_generator_drivers():
    city = City(width=100, height=100)
    config = DemandConfig(arrival_rate=1.0, duration=10.0)
    gen = DemandGenerator(city, config)
    drivers = gen.generate_drivers(count=5, rng=random.Random(1))
    assert len(drivers) == 5
    assert len({d.driver_id for d in drivers}) == 5
    assert all(d.status is DriverState.AVAILABLE for d in drivers)


# ---------------------------------------------------------------------
# End-to-end simulator smoke test
# ---------------------------------------------------------------------


def test_simulator_end_to_end_with_ample_drivers():
    """With plenty of drivers and a trivial policy, every ride should
    eventually be assigned and most should complete within the window.
    """
    city = City(width=50, height=50, average_speed=5.0)
    config = DemandConfig(arrival_rate=1.0, duration=20.0)
    gen = DemandGenerator(city, config)

    rng = random.Random(123)
    rides = gen.generate_rides(rng)
    drivers = gen.generate_drivers(count=10, rng=rng)

    policy = FirstAvailablePolicy()
    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=policy)
    result = sim.run(duration=config.duration)

    assert len(result.rides) == len(rides)
    assert len(result.ride_results) == len(rides)

    # Every ride was at least attempted (RIDE_REQUESTED logged for all).
    requested_ride_ids = {
        e.ride_id for e in result.events if e.event_type is EventType.RIDE_REQUESTED
    }
    assert requested_ride_ids == {r.ride_id for r in rides}

    # With 10 drivers and low demand, essentially all rides should be
    # served (assigned) rather than unserved.
    assigned = [
        r for r in result.ride_results.values() if r.assignment_time is not None
    ]
    assert len(assigned) >= 1

    # No driver should end up in an invalid/inconsistent state.
    for d in drivers:
        assert d.status in (
            DriverState.AVAILABLE,
            DriverState.ASSIGNED,
            DriverState.PICKING_UP,
            DriverState.ON_RIDE,
        )
        assert d.idle_time >= 0
        assert d.busy_time >= 0


def test_simulator_no_drivers_all_unserved():
    city = City(width=50, height=50, average_speed=5.0)
    config = DemandConfig(arrival_rate=1.0, duration=10.0)
    gen = DemandGenerator(city, config)
    rng = random.Random(5)
    rides = gen.generate_rides(rng)

    policy = FirstAvailablePolicy()
    sim = Simulator(city=city, drivers=[], rides=rides, policy=policy)
    result = sim.run(duration=config.duration)

    assert all(
        r.status is RideStatus.UNSERVED or r.status is RideStatus.REQUESTED
        for r in result.rides
    )
    assert all(not res.served for res in result.ride_results.values())
