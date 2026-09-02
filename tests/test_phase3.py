"""Phase 3 tests: evaluation engine (rider/driver/fairness/system metrics).

These tests build small, hand-constructed inputs where the correct
answer is known, rather than relying only on end-to-end simulation
runs -- so a bug in metric math can't hide behind simulation noise.
"""

from __future__ import annotations

import random

import pytest

from src.allocation.nearest_policy import NearestDriverPolicy
from src.allocation.random_policy import RandomPolicy
from src.evaluation.driver_metrics import compute_driver_metrics, rides_per_driver_stats
from src.evaluation.evaluator import evaluate_simulation
from src.evaluation.fairness_metrics import (
    gini_coefficient,
    jains_fairness_index,
    opportunity_rates,
)
from src.evaluation.rider_metrics import compute_rider_metrics
from src.evaluation.system_metrics import compute_system_metrics
from src.models.driver import Driver
from src.models.events import Event, EventType, RideResult
from src.simulation.environment import City
from src.simulation.generators import DemandConfig, DemandGenerator
from src.simulation.simulator import Simulator


def test_rider_metrics_assignment_and_unserved_rate():
    results = [
        RideResult(ride_id="r1", request_time=0, assignment_time=1, pickup_time=2, served=True),
        RideResult(ride_id="r2", request_time=0, assignment_time=1, pickup_time=3, served=True),
        RideResult(ride_id="r3", request_time=0, served=False),
    ]
    metrics = compute_rider_metrics(results)
    assert metrics["assignment_rate"] == pytest.approx(2 / 3)
    assert metrics["unserved_rate"] == pytest.approx(1 / 3)


def test_rider_metrics_pickup_time_stats():
    results = [
        RideResult(ride_id=f"r{i}", request_time=0, pickup_time=float(i))
        for i in range(1, 11)
    ]
    metrics = compute_rider_metrics(results)
    stats = metrics["pickup_time"]
    assert stats["n"] == 10
    assert stats["mean"] == pytest.approx(5.5)
    assert stats["median"] == pytest.approx(5.5)


def test_rider_metrics_empty_input():
    metrics = compute_rider_metrics([])
    assert metrics["total_requests"] == 0
    assert metrics["assignment_rate"] == 0.0
    assert metrics["pickup_time"]["n"] == 0


def test_driver_metrics_basic():
    drivers = []
    for i, (busy, online, rides) in enumerate([(4, 10, 2), (8, 10, 4), (2, 10, 1)]):
        d = Driver(driver_id=f"d{i}", location=(0, 0))
        d.busy_time = busy
        d.online_time = online
        d.rides_completed = rides
        drivers.append(d)

    metrics = compute_driver_metrics(drivers)
    assert metrics["num_drivers"] == 3
    assert metrics["utilization"]["mean"] == pytest.approx((0.4 + 0.8 + 0.2) / 3)
    assert metrics["rides_per_driver"]["mean"] == pytest.approx((2 + 4 + 1) / 3)


def test_rides_per_driver_coefficient_of_variation():
    drivers = [Driver(driver_id=f"d{i}", location=(0, 0)) for i in range(4)]
    for d, n in zip(drivers, [2, 2, 2, 2]):
        d.rides_completed = n
    stats = rides_per_driver_stats(drivers)
    assert stats["std"] == pytest.approx(0.0)
    assert stats["coefficient_of_variation"] == pytest.approx(0.0)


def test_jains_index_perfect_equality():
    assert jains_fairness_index([5, 5, 5, 5]) == pytest.approx(1.0)


def test_jains_index_maximal_inequality():
    assert jains_fairness_index([10, 0, 0, 0]) == pytest.approx(1 / 4)


def test_jains_index_empty_and_all_zero():
    assert jains_fairness_index([]) == 1.0
    assert jains_fairness_index([0, 0, 0]) == 1.0


def test_gini_perfect_equality_is_zero():
    assert gini_coefficient([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)


def test_gini_maximal_inequality_approaches_bound():
    assert gini_coefficient([10, 0, 0, 0]) == pytest.approx(0.75)


def test_gini_empty_and_all_zero():
    assert gini_coefficient([]) == 0.0
    assert gini_coefficient([0, 0, 0]) == 0.0


def test_opportunity_rates_from_events():
    events = [
        Event(EventType.RIDE_REQUESTED, time=0, ride_id="r1", metadata={"available_driver_ids": ["d0"]}),
        Event(EventType.DRIVER_ASSIGNED, time=0, ride_id="r1", driver_id="d0"),
        Event(EventType.RIDE_REQUESTED, time=1, ride_id="r2", metadata={"available_driver_ids": ["d0", "d1"]}),
        Event(EventType.DRIVER_ASSIGNED, time=1, ride_id="r2", driver_id="d1"),
    ]
    rates = opportunity_rates(events, ["d0", "d1", "d2"])
    assert rates["d0"] == pytest.approx(1 / 2)
    assert rates["d1"] == pytest.approx(1 / 1)
    assert rates["d2"] is None


def test_system_metrics_basic_counts():
    results = [
        RideResult(ride_id="r1", request_time=0, served=True, pickup_distance=5.0, allocation_compute_time=0.001),
        RideResult(ride_id="r2", request_time=0, served=False, allocation_compute_time=0.002),
    ]
    d = Driver(driver_id="d0", location=(0, 0))
    d.busy_time = 5.0
    d.online_time = 10.0

    metrics = compute_system_metrics(results, [d], total_runtime=1.23)
    assert metrics["completed_rides"] == 1
    assert metrics["unserved_requests"] == 1
    assert metrics["total_pickup_distance"] == pytest.approx(5.0)
    assert metrics["overall_driver_utilization"] == pytest.approx(0.5)
    assert metrics["mean_allocation_compute_time_s"] == pytest.approx(0.0015)
    assert metrics["total_simulation_runtime_s"] == pytest.approx(1.23)


def test_evaluate_simulation_end_to_end_structure_and_ranges():
    city = City(width=100, height=100, average_speed=5.0)
    config = DemandConfig(arrival_rate=2.0, duration=20.0)
    gen = DemandGenerator(city, config)
    rng = random.Random(7)
    rides = gen.generate_rides(rng)
    drivers = gen.generate_drivers(count=10, rng=rng)

    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=NearestDriverPolicy(city))
    result = sim.run(duration=config.duration)
    metrics = evaluate_simulation(result)

    for section in ("rider", "driver", "fairness", "system"):
        assert section in metrics

    assert 0.0 <= metrics["rider"]["assignment_rate"] <= 1.0
    assert 0.0 <= metrics["rider"]["unserved_rate"] <= 1.0
    assert 0.0 <= metrics["driver"]["utilization"]["mean"] <= 1.0
    assert 0.0 <= metrics["fairness"]["rides_received"]["jains_index"] <= 1.0
    assert metrics["system"]["total_requests"] == len(rides)
    assert metrics["system"]["total_simulation_runtime_s"] >= 0.0


def test_evaluate_simulation_works_for_random_policy_too():
    """Sanity check that the evaluator has no hidden dependency on which
    policy produced the SimulationResult (spec section 12)."""
    city = City(width=100, height=100, average_speed=5.0)
    config = DemandConfig(arrival_rate=2.0, duration=20.0)
    gen = DemandGenerator(city, config)
    rng = random.Random(7)
    rides = gen.generate_rides(rng)
    drivers = gen.generate_drivers(count=10, rng=rng)

    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=RandomPolicy(random.Random(7)))
    result = sim.run(duration=config.duration)
    metrics = evaluate_simulation(result)
    assert metrics["system"]["total_requests"] == len(rides)