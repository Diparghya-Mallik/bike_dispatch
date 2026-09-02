"""Phase 4 tests: ETA (P2), idle-aware (P3), and workload-aware (P4) policies."""

from __future__ import annotations

import random

import pytest

from src.allocation.eta_policy import ETAPolicy
from src.allocation.idle_aware_policy import IdleAwarePolicy
from src.allocation.workload_policy import WorkloadAwarePolicy
from src.models.driver import Driver
from src.models.ride import Ride
from src.simulation.environment import City
from src.simulation.generators import DemandConfig, DemandGenerator
from src.simulation.simulator import Simulator


def _ride(pickup=(0.0, 0.0), request_time=0.0) -> Ride:
    return Ride(ride_id="ride_test", request_time=request_time, pickup_location=pickup, destination=(10.0, 10.0))


def test_eta_policy_returns_none_when_no_drivers():
    city = City(width=100, height=100)
    policy = ETAPolicy(eta_provider=city)
    assert policy.select_driver(_ride(), []) is None


def test_eta_policy_picks_min_eta_driver():
    city = City(width=100, height=100, average_speed=5.0)
    policy = ETAPolicy(eta_provider=city)

    ride = _ride(pickup=(50.0, 50.0))
    drivers = [
        Driver(driver_id="far", location=(0.0, 0.0)),
        Driver(driver_id="near", location=(51.0, 51.0)),
    ]
    chosen = policy.select_driver(ride, drivers)
    assert chosen.driver_id == "near"


def test_eta_policy_matches_nearest_with_constant_speed():
    city = City(width=100, height=100, average_speed=3.0)
    policy = ETAPolicy(eta_provider=city)

    rng = random.Random(4)
    drivers = [Driver(driver_id=f"d{i}", location=(rng.uniform(0, 100), rng.uniform(0, 100))) for i in range(8)]
    ride = _ride(pickup=(40.0, 60.0))

    chosen_by_eta = policy.select_driver(ride, drivers)
    chosen_by_distance = min(drivers, key=lambda d: (city.distance(d.location, ride.pickup_location), d.driver_id))
    assert chosen_by_eta.driver_id == chosen_by_distance.driver_id


def test_idle_aware_single_driver_shortcut():
    city = City(width=100, height=100)
    policy = IdleAwarePolicy(eta_provider=city)
    drivers = [Driver(driver_id="only", location=(99.0, 99.0))]
    chosen = policy.select_driver(_ride(pickup=(0.0, 0.0)), drivers)
    assert chosen.driver_id == "only"


def test_idle_aware_favors_idle_driver_when_idle_weight_dominates():
    city = City(width=100, height=100, average_speed=5.0)
    policy = IdleAwarePolicy(eta_provider=city, w_eta=0.1, w_idle=0.9)

    ride = _ride(pickup=(0.0, 0.0))
    close_but_not_idle = Driver(driver_id="close", location=(1.0, 1.0))
    close_but_not_idle.idle_time = 1.0
    far_but_idle = Driver(driver_id="far", location=(90.0, 90.0))
    far_but_idle.idle_time = 100.0

    chosen = policy.select_driver(ride, [close_but_not_idle, far_but_idle])
    assert chosen.driver_id == "far"


def test_idle_aware_favors_close_driver_when_eta_weight_dominates():
    city = City(width=100, height=100, average_speed=5.0)
    policy = IdleAwarePolicy(eta_provider=city, w_eta=0.9, w_idle=0.1)

    ride = _ride(pickup=(0.0, 0.0))
    close_but_not_idle = Driver(driver_id="close", location=(1.0, 1.0))
    close_but_not_idle.idle_time = 1.0
    far_but_idle = Driver(driver_id="far", location=(90.0, 90.0))
    far_but_idle.idle_time = 100.0

    chosen = policy.select_driver(ride, [close_but_not_idle, far_but_idle])
    assert chosen.driver_id == "close"


def test_idle_aware_uses_live_idle_time_not_stale_banked_attribute():
    """Regression test: Driver.idle_time only accumulates when a driver
    transitions OUT of AVAILABLE, so a driver idle since t=0 who has
    never been assigned yet shows idle_time == 0.0 the whole time. The
    policy must add the in-progress idle stretch
    (now - last_state_change_at), not just read the banked idle_time,
    or it will never actually favor a never-yet-dispatched driver."""
    city = City(width=100, height=100, average_speed=5.0)
    policy = IdleAwarePolicy(eta_provider=city, w_eta=0.0, w_idle=1.0)

    ride = _ride(pickup=(0.0, 0.0), request_time=20.0)
    never_dispatched = Driver(driver_id="never_dispatched", location=(5.0, 5.0), last_state_change_at=0.0)
    just_freed = Driver(driver_id="just_freed", location=(5.0, 5.0), last_state_change_at=18.0)

    assert never_dispatched.idle_time == 0.0
    assert just_freed.idle_time == 0.0

    chosen = policy.select_driver(ride, [never_dispatched, just_freed])
    assert chosen.driver_id == "never_dispatched"


def test_idle_aware_ties_broken_by_driver_id():
    city = City(width=100, height=100)
    policy = IdleAwarePolicy(eta_provider=city)
    ride = _ride(pickup=(0.0, 0.0))
    drivers = [
        Driver(driver_id="zeta", location=(5.0, 5.0)),
        Driver(driver_id="alpha", location=(5.0, 5.0)),
    ]
    for d in drivers:
        d.idle_time = 3.0
    chosen = policy.select_driver(ride, drivers)
    assert chosen.driver_id == "alpha"


def test_idle_aware_rejects_negative_weights():
    city = City(width=100, height=100)
    with pytest.raises(ValueError):
        IdleAwarePolicy(eta_provider=city, w_eta=-1.0)


def test_workload_aware_single_driver_shortcut():
    city = City(width=100, height=100)
    policy = WorkloadAwarePolicy(eta_provider=city)
    drivers = [Driver(driver_id="only", location=(99.0, 99.0))]
    chosen = policy.select_driver(_ride(pickup=(0.0, 0.0)), drivers)
    assert chosen.driver_id == "only"


def test_workload_aware_favors_underworked_driver_when_eta_tied():
    city = City(width=100, height=100, average_speed=5.0)
    policy = WorkloadAwarePolicy(eta_provider=city, w_eta=0.0, w_workload=1.0)

    ride = _ride(pickup=(0.0, 0.0), request_time=10.0)
    overworked = Driver(driver_id="overworked", location=(5.0, 5.0), went_online_at=0.0)
    overworked.rides_completed = 5
    fresh = Driver(driver_id="fresh", location=(5.0, 5.0), went_online_at=0.0)
    fresh.rides_completed = 0

    chosen = policy.select_driver(ride, [overworked, fresh])
    assert chosen.driver_id == "fresh"


def test_workload_aware_uses_request_time_not_stale_online_time_attribute():
    """Regression test: Driver.online_time is only refreshed by
    Simulator.run() at the very end, so the policy must derive online
    time from ride.request_time - went_online_at, not read the (stale,
    always-0 mid-simulation) online_time attribute directly."""
    city = City(width=100, height=100, average_speed=5.0)
    policy = WorkloadAwarePolicy(eta_provider=city, w_eta=0.0, w_workload=1.0)

    ride = _ride(pickup=(0.0, 0.0), request_time=20.0)
    busy = Driver(driver_id="busy", location=(5.0, 5.0), went_online_at=0.0)
    busy.rides_completed = 10
    idle = Driver(driver_id="idle", location=(5.0, 5.0), went_online_at=0.0)
    idle.rides_completed = 0

    chosen = policy.select_driver(ride, [busy, idle])
    assert chosen.driver_id == "idle"


def test_workload_aware_zero_online_time_treated_as_zero_workload():
    city = City(width=100, height=100, average_speed=5.0)
    policy = WorkloadAwarePolicy(eta_provider=city)
    ride = _ride(pickup=(0.0, 0.0), request_time=0.0)
    driver = Driver(driver_id="brand_new", location=(1.0, 1.0), went_online_at=0.0)
    driver.rides_completed = 0
    chosen = policy.select_driver(ride, [driver])
    assert chosen.driver_id == "brand_new"


def test_workload_aware_rejects_negative_weights():
    city = City(width=100, height=100)
    with pytest.raises(ValueError):
        WorkloadAwarePolicy(eta_provider=city, w_workload=-0.5)


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda city: ETAPolicy(eta_provider=city),
        lambda city: IdleAwarePolicy(eta_provider=city),
        lambda city: WorkloadAwarePolicy(eta_provider=city),
    ],
)
def test_phase4_policy_runs_end_to_end_in_simulator(policy_factory):
    city = City(width=100, height=100, average_speed=5.0)
    config = DemandConfig(arrival_rate=2.0, duration=20.0)
    gen = DemandGenerator(city, config)
    rng = random.Random(11)
    rides = gen.generate_rides(rng)
    drivers = gen.generate_drivers(count=10, rng=rng)

    policy = policy_factory(city)
    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=policy)
    result = sim.run(duration=config.duration)

    assert len(result.ride_results) == len(rides)
    assigned = [r for r in result.ride_results.values() if r.assignment_time is not None]
    assert len(assigned) > 0