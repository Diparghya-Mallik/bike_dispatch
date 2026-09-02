"""Phase 2 tests: concrete allocation policies (P0 random, P1 nearest).

Statistical comparisons between policies (e.g. "nearest reduces mean
pickup distance vs random") belong to the evaluation engine (Phase 3)
and statistical analysis (Phase 8), not here -- these tests only check
that each policy is internally correct and plugs into the Phase 1
simulator without error.
"""

from __future__ import annotations

import random

from src.allocation.nearest_policy import NearestDriverPolicy
from src.allocation.random_policy import RandomPolicy
from src.models.driver import Driver
from src.models.ride import Ride
from src.simulation.environment import City
from src.simulation.generators import DemandConfig, DemandGenerator
from src.simulation.simulator import Simulator


def _ride(pickup=(0.0, 0.0)) -> Ride:
    return Ride(
        ride_id="ride_test",
        request_time=0.0,
        pickup_location=pickup,
        destination=(10.0, 10.0),
    )


def _drivers(*locations) -> list[Driver]:
    return [
        Driver(driver_id=f"driver_{i}", location=loc)
        for i, loc in enumerate(locations)
    ]


# ---------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------


def test_random_policy_returns_none_when_no_drivers():
    policy = RandomPolicy(rng=random.Random(1))
    assert policy.select_driver(_ride(), []) is None


def test_random_policy_only_picks_from_available_drivers():
    drivers = _drivers((0, 0), (5, 5), (9, 9))
    policy = RandomPolicy(rng=random.Random(1))
    for _ in range(20):
        chosen = policy.select_driver(_ride(), drivers)
        assert chosen in drivers


def test_random_policy_deterministic_with_seed():
    drivers = _drivers((0, 0), (5, 5), (9, 9), (2, 8), (7, 1))
    policy_a = RandomPolicy(rng=random.Random(99))
    policy_b = RandomPolicy(rng=random.Random(99))

    picks_a = [policy_a.select_driver(_ride(), drivers).driver_id for _ in range(10)]
    picks_b = [policy_b.select_driver(_ride(), drivers).driver_id for _ in range(10)]

    assert picks_a == picks_b


# ---------------------------------------------------------------------
# NearestDriverPolicy
# ---------------------------------------------------------------------


def test_nearest_policy_returns_none_when_no_drivers():
    city = City(width=100, height=100)
    policy = NearestDriverPolicy(distance_provider=city)
    assert policy.select_driver(_ride(), []) is None


def test_nearest_policy_picks_closest_driver():
    city = City(width=100, height=100)
    policy = NearestDriverPolicy(distance_provider=city)

    ride = _ride(pickup=(50.0, 50.0))
    drivers = _drivers((0.0, 0.0), (51.0, 51.0), (90.0, 90.0))

    chosen = policy.select_driver(ride, drivers)
    assert chosen.driver_id == "driver_1"  # (51, 51) is closest to (50, 50)


def test_nearest_policy_breaks_ties_by_driver_id():
    city = City(width=100, height=100)
    policy = NearestDriverPolicy(distance_provider=city)

    ride = _ride(pickup=(0.0, 0.0))
    # driver_0 and driver_1 are equidistant from the pickup point.
    drivers = [
        Driver(driver_id="driver_9", location=(3.0, 4.0)),
        Driver(driver_id="driver_1", location=(4.0, 3.0)),
    ]
    chosen = policy.select_driver(ride, drivers)
    assert chosen.driver_id == "driver_1"  # lexicographically smaller id wins


# ---------------------------------------------------------------------
# Integration: each policy plugged into the Phase 1 simulator
# ---------------------------------------------------------------------


def test_random_policy_runs_end_to_end_in_simulator():
    city = City(width=100, height=100, average_speed=5.0)
    config = DemandConfig(arrival_rate=2.0, duration=20.0)
    gen = DemandGenerator(city, config)
    rng = random.Random(11)
    rides = gen.generate_rides(rng)
    drivers = gen.generate_drivers(count=10, rng=rng)

    policy = RandomPolicy(rng=random.Random(11))
    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=policy)
    result = sim.run(duration=config.duration)

    assert len(result.ride_results) == len(rides)
    assigned = [r for r in result.ride_results.values() if r.assignment_time is not None]
    assert len(assigned) > 0


def test_nearest_policy_runs_end_to_end_in_simulator():
    city = City(width=100, height=100, average_speed=5.0)
    config = DemandConfig(arrival_rate=2.0, duration=20.0)
    gen = DemandGenerator(city, config)
    rng = random.Random(11)
    rides = gen.generate_rides(rng)
    drivers = gen.generate_drivers(count=10, rng=rng)

    policy = NearestDriverPolicy(distance_provider=city)
    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=policy)
    result = sim.run(duration=config.duration)

    assert len(result.ride_results) == len(rides)
    assigned = [r for r in result.ride_results.values() if r.assignment_time is not None]
    assert len(assigned) > 0

    # Every served ride's pickup_distance should be non-negative and
    # consistent with the driver actually being closest at assignment time
    # is hard to re-verify after the fact (drivers move), but it should at
    # minimum be a finite, sane number.
    for r in result.ride_results.values():
        if r.pickup_distance is not None:
            assert r.pickup_distance >= 0.0
