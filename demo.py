"""Phase 1-4 demo: run all five policies on identical rides/drivers,
evaluate them, and print a side-by-side comparison.
"""

from __future__ import annotations

import copy
import random

from src.allocation.eta_policy import ETAPolicy
from src.allocation.idle_aware_policy import IdleAwarePolicy
from src.allocation.nearest_policy import NearestDriverPolicy
from src.allocation.random_policy import RandomPolicy
from src.allocation.workload_policy import WorkloadAwarePolicy
from src.evaluation.evaluator import evaluate_simulation
from src.simulation.environment import City
from src.simulation.generators import DemandConfig, DemandGenerator
from src.simulation.simulator import Simulator

SEED = 42
CITY_WIDTH, CITY_HEIGHT = 100, 100
AVERAGE_SPEED = 5.0
NUM_DRIVERS = 30  # enough supply that requests usually see multiple
                  # available drivers, so P3/P4's extra scoring terms
                  # actually get a chance to matter
ARRIVAL_RATE = 2.0
DURATION = 30.0


def run_and_summarize(name, city, drivers, rides, policy):
    sim = Simulator(city=city, drivers=drivers, rides=rides, policy=policy)
    result = sim.run(duration=DURATION)
    metrics = evaluate_simulation(result)

    rider = metrics["rider"]
    driver = metrics["driver"]
    fairness = metrics["fairness"]
    system = metrics["system"]

    print(f"\n--- {name} ---")
    print(
        f"Completed: {system['completed_rides']}/{system['total_requests']} "
        f"({100 * system['completed_rides'] / system['total_requests']:.1f}%)   "
        f"Unserved: {system['unserved_requests']}"
    )
    if rider["pickup_time"]["n"]:
        print(
            f"Pickup wait  - mean: {rider['pickup_time']['mean']:.2f}  "
            f"p90: {rider['pickup_time']['p90']:.2f}"
        )
    if rider["pickup_distance"]["n"]:
        print(f"Pickup dist  - mean: {rider['pickup_distance']['mean']:.2f}")

    print(
        f"Driver util  - mean: {driver['utilization']['mean']:.2f}  "
        f"rides/driver mean: {driver['rides_per_driver']['mean']:.2f}  "
        f"CV: {driver['rides_per_driver']['coefficient_of_variation']:.2f}"
    )
    print(
        f"Fairness     - rides Jain's: {fairness['rides_received']['jains_index']:.3f}  "
        f"Gini: {fairness['rides_received']['gini']:.3f}"
    )
    return metrics


def main() -> None:
    city = City(width=CITY_WIDTH, height=CITY_HEIGHT, average_speed=AVERAGE_SPEED)
    config = DemandConfig(arrival_rate=ARRIVAL_RATE, duration=DURATION)
    generator = DemandGenerator(city, config)

    rng = random.Random(SEED)
    rides_template = generator.generate_rides(rng)
    drivers_template = generator.generate_drivers(count=NUM_DRIVERS, rng=rng)

    print("=" * 70)
    print(f"City: {CITY_WIDTH}x{CITY_HEIGHT}, speed={AVERAGE_SPEED}")
    print(f"Drivers: {NUM_DRIVERS}   Duration: {DURATION}   Seed: {SEED}")
    print(f"Generated {len(rides_template)} ride requests")
    print("=" * 70)

    policies = [
        ("P0 Random", lambda c: RandomPolicy(rng=random.Random(SEED))),
        ("P1 Nearest Driver", lambda c: NearestDriverPolicy(distance_provider=c)),
        ("P2 Min ETA", lambda c: ETAPolicy(eta_provider=c)),
        ("P3 Idle-Aware", lambda c: IdleAwarePolicy(eta_provider=c, w_eta=0.4, w_idle=0.6)),
        ("P4 Workload-Aware", lambda c: WorkloadAwarePolicy(eta_provider=c, w_eta=0.4, w_workload=0.6)),
    ]

    for name, policy_factory in policies:
        rides = copy.deepcopy(rides_template)
        drivers = copy.deepcopy(drivers_template)
        policy = policy_factory(city)
        run_and_summarize(name, city, drivers, rides, policy)

    print("\n" + "=" * 70)
    print(
        "Note: single-seed comparison only, for eyeballing. Real policy "
        "comparisons need Phase 7 (multi-seed runner) and Phase 8 "
        "(statistical analysis: CIs, effect sizes)."
    )


if __name__ == "__main__":
    main()