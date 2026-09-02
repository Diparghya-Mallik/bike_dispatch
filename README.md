# Bike-Taxi Driver Allocation & Dispatch Research

A research-oriented simulation, optimization, and evaluation framework
for studying bike-taxi ride allocation strategies (inspired by, but not
affiliated with or claiming to model, real platforms such as Rapido).

This is **not** a ride-hailing application. No mobile app, web
dashboard, auth, payments, or real-time GPS. The goal is controlled,
reproducible experiments comparing allocation/optimization strategies
across rider service quality, driver utilization, and driver
opportunity fairness. See `PROJECT_SPEC.md` for the full research
design and roadmap.

## Status: Phase 2 complete

Implemented so far:

- `src/models/driver.py` — `Driver` dataclass and its state machine
  (`AVAILABLE -> ASSIGNED -> PICKING_UP -> ON_RIDE -> AVAILABLE`).
- `src/models/ride.py` — `Ride` request dataclass and lifecycle status.
- `src/models/events.py` — `Event` log entries and per-ride `RideResult`
  records.
- `src/simulation/environment.py` — `City`: 2D coordinate space,
  Euclidean distance, travel time.
- `src/simulation/simulator.py` — SimPy-based discrete-event
  `Simulator`. Contains **no allocation logic** — it calls
  `policy.select_driver(ride, available_drivers)` and reacts to the
  result, so any `AllocationPolicy` can be plugged in without changing
  this file.
- `src/simulation/generators.py` — reproducible synthetic demand
  (Poisson arrivals; uniform or clustered pickup/destination
  placement) and driver placement, all seeded via an explicit
  `random.Random`.
- `src/allocation/base.py` — the `AllocationPolicy` interface.
- `src/allocation/random_policy.py` — **P0 Random Allocation**: uniform
  random choice among available drivers, seeded via an explicit
  `random.Random`. Weak baseline / sanity check.
- `src/allocation/nearest_policy.py` — **P1 Nearest Driver**: picks the
  available driver with minimum Euclidean distance to pickup, via a
  `distance_provider` (typically the `City`) rather than assuming
  Euclidean geometry itself. Ties broken deterministically by
  `driver_id`.
- `tests/test_phase1.py` — unit tests for the state machine, city
  geometry, generator reproducibility, and an end-to-end simulator
  smoke test using a trivial test-only policy.
- `tests/test_phase2.py` — unit tests for both policies in isolation
  (determinism, tie-breaking, empty-driver-pool handling) plus
  integration tests running each through the Phase 1 simulator.
- `demo.py` — runs the same seeded rides/drivers through both P0 and P1
  and prints a side-by-side comparison (completion rate, pickup wait,
  pickup distance, rides per driver). Not part of the experiment
  framework — just a readable single-run sanity check.

Not yet implemented (see spec for phase order): event logging →
evaluation engine (Phase 3), ETA / idle / workload-aware policies
(Phase 4), multi-objective allocation (Phase 5), OR-Tools global
assignment (Phase 6), experiment runner across seeds/scenarios
(Phase 7), statistical analysis (Phase 8), Pareto analysis and
visualization (Phase 9), and realistic road networks / ML prediction
(Phase 10).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest
```

## Design notes / assumptions carried into later phases

- City is a plain 2D coordinate space (`width x height`) with constant
  average speed; Euclidean distance stands in for real routing until
  Phase 10.
- The simulator assigns a driver at most once, at the moment of
  request, by asking the policy for a decision against the drivers
  available *right then*. It does not currently queue or retry
  requests that find no available driver — they are immediately marked
  `UNSERVED`. Retry/queueing behavior, if desired, should be an
  explicit, documented policy-level or simulator-level choice in a
  later phase, not a silent default.
- Rides whose pickup+trip would finish after the configured
  `duration` are cut off mid-flight by SimPy's clock stopping; their
  `RideResult.served` stays `False`. Evaluation code in later phases
  should consider excluding a warm-up/cool-down window rather than
  treating these as true "no driver available" failures.
- `Driver.online_time` is recomputed via `accrue_online_time()` rather
  than incremented continuously, to avoid double counting; call it
  before reading `online_time` or `utilization()` outside of
  `Simulator.run()`.
