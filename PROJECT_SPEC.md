# Bike-Taxi Driver Allocation & Dispatch Research Project

## 1. Project Overview

Build a **research-oriented simulation, optimization, and evaluation framework for bike-taxi ride allocation**, inspired by platforms such as Rapido.

The project should investigate how different ride-allocation strategies affect:

* Rider pickup/waiting time
* Driver idle time
* Driver utilization
* Ride distribution among drivers
* Driver opportunity fairness
* System efficiency
* Computational cost

The project is **not an end-to-end ride-hailing application**.

Do NOT build:

* Mobile applications
* Web dashboards
* Authentication systems
* Payment systems
* Real-time GPS tracking
* Production APIs
* Databases unless later required for a specific research experiment

The primary objective is to conduct **controlled, reproducible experiments** comparing different allocation and optimization strategies.

---

# 2. Core Research Question

Investigate:

> **How do different real-time bike-taxi ride allocation strategies affect the trade-off between rider service quality, driver utilization, driver opportunity fairness, and overall system efficiency under varying demand and driver-availability conditions?**

The research should determine whether fairness-aware allocation can improve driver-side outcomes without causing an unacceptable deterioration in rider experience.

---

# 3. Research Questions

The framework should eventually allow investigation of questions such as:

1. Does nearest-driver allocation minimize rider pickup time?
2. Does nearest-driver allocation produce unequal driver utilization?
3. Can incorporating driver idle time improve allocation fairness?
4. Can workload-aware allocation distribute rides more equitably?
5. What is the trade-off between rider pickup time and driver fairness?
6. Does global optimization outperform greedy allocation?
7. How do different policies behave during demand spikes?
8. How do policies behave when drivers are scarce?
9. How sensitive are results to the relative weighting of different objectives?
10. Is there a Pareto frontier between rider experience and driver fairness?
11. Does optimization provide enough improvement to justify its additional computational cost?
12. Can ML-based ETA or demand prediction improve allocation performance?

---

# 4. Research Philosophy

Treat this as a **simulation and experimental research project**, not an application-development project.

The system should follow:

```text
Synthetic / Public Data
        ↓
Scenario Generation
        ↓
Simulation
        ↓
Allocation Policy
        ↓
Event Logging
        ↓
Evaluation Engine
        ↓
Statistical Analysis
        ↓
Visualization
        ↓
Comparison / Conclusions
```

All experiments must be reproducible.

Use fixed random seeds where appropriate.

Do not make unsupported claims about real Rapido operations. The system should be described as a **generic bike-taxi dispatch model inspired by real-world ride-hailing systems**.

---

# 5. Technology Stack

Use **Python as the primary language**.

Initial stack:

* Python 3.10+
* NumPy
* Pandas
* SimPy
* SciPy
* Matplotlib
* Seaborn

Optimization:

* Google OR-Tools

Potential later additions:

* scikit-learn
* XGBoost
* NetworkX
* GeoPandas
* OSMnx

Do not introduce additional dependencies unless they have a clear research purpose.

---

# 6. Initial Modeling Assumption

Initially model the city as a **2D coordinate space** rather than a real road network.

Example:

```text
City dimensions: 100 × 100
```

Drivers and ride requests have coordinates:

```python
(x, y)
```

Use Euclidean distance initially:

[
d =
\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
]

Later, the project may introduce realistic road-network distances and travel times.

Do not introduce GIS complexity during the initial simulation phase.

---

# 7. Core Entities

## Driver

Each driver should contain at least:

```text
driver_id
location
status
online_time
idle_time
busy_time
rides_completed
earnings
current_ride
```

Possible driver states:

```text
AVAILABLE
ASSIGNED
PICKING_UP
ON_RIDE
```

State transition:

```text
AVAILABLE
    ↓
ASSIGNED
    ↓
PICKING_UP
    ↓
ON_RIDE
    ↓
AVAILABLE
```

Only AVAILABLE drivers can receive new rides.

---

## Ride Request

Each ride should contain:

```text
ride_id
request_time
pickup_location
destination
```

Later fields may include:

```text
estimated_fare
trip_distance
trip_duration
```

---

## Ride Result / Event Record

Record what actually happened:

```text
ride_id
request_time
assignment_time
pickup_time
ride_start_time
completion_time
driver_id
pickup_distance
pickup_eta
```

---

# 8. Simulation Engine

Use **SimPy** for discrete-event simulation.

The simulation should represent events such as:

```text
RIDE_REQUESTED
DRIVER_ASSIGNED
PICKUP_STARTED
RIDER_PICKED_UP
RIDE_STARTED
RIDE_COMPLETED
DRIVER_AVAILABLE
```

The simulator should maintain the evolving state of:

* Drivers
* Ride requests
* Driver availability
* Ride assignments
* Driver locations
* Driver idle/busy time

The simulator must not contain hard-coded allocation logic.

---

# 9. Allocation Policy Architecture

Use a common interface.

Conceptually:

```python
class AllocationPolicy:
    def select_driver(self, ride, available_drivers):
        ...
```

The simulator should call the policy without knowing which specific strategy is being used.

This allows policies to be swapped without modifying the simulation engine.

---

# 10. Allocation Policies

Implement policies progressively.

## P0 — Random Allocation

Randomly select an available driver.

Purpose:

* Weak baseline
* Sanity check

---

## P1 — Nearest Driver

Select:

[
d^* =
\arg\min_d distance(d,r)
]

Purpose:

* Primary simple baseline
* Mimics proximity-based dispatch

---

## P2 — Minimum ETA

Select:

[
d^* =
\arg\min_d ETA(d,r)
]

Initially:

[
ETA =
\frac{distance}{average\ speed}
]

Later ETA can become more realistic.

---

## P3 — Idle-Time Aware

Favor drivers who have been idle longer while considering pickup ETA.

Example objective:

[
Score =
-w_1 ETA_{norm}
+w_2 IdleTime_{norm}
]

The exact formulation should be documented and tested.

---

## P4 — Workload Aware

Consider driver workload:

[
Workload_i =
\frac{RidesCompleted_i}
{OnlineHours_i}
]

Avoid using raw ride count alone when online time differs substantially.

---

## P5 — Earnings Aware

Potentially consider normalized earnings rate:

[
EarningsRate_i =
\frac{Earnings_i}
{OnlineHours_i}
]

Initially treat earnings-aware allocation as a secondary experiment because a realistic earnings model requires assumptions about fares, incentives, commissions, etc.

---

## P6 — Multi-Objective Allocation

Combine normalized objectives:

[
Score(d,r)=
w_1S_{ETA}
+w_2S_{idle}
+w_3S_{workload}
+w_4S_{fairness}
]

Weights should satisfy:

[
\sum_iw_i=1
]

Do not assume one weight combination is universally optimal.

Run systematic weight-sensitivity experiments.

---

# 11. Optimization Layer

Optimization must be separated from heuristic allocation policies.

## O1 — Global Assignment

Represent assignment using:

[
x_{ij}\in{0,1}
]

where:

[
x_{ij}=1
]

means driver (i) receives ride (j).

Objective:

[
\min
\sum_i\sum_j C_{ij}x_{ij}
]

subject to:

[
\sum_jx_{ij}\leq1
]

[
\sum_ix_{ij}\leq1
]

This should initially use pickup ETA as the cost.

Implement using **OR-Tools**.

---

## O2 — Global Multi-Objective Assignment

Extend the cost:

[
C_{ij}
======

w_1C^{ETA}_{ij}
+w_2C^{idle}_i
+w_3C^{workload}_i
+w_4C^{fairness}_i
]

Compare against greedy allocation.

Measure:

* Solution quality
* Rider experience
* Driver fairness
* Computational cost

---

## O3 — Rolling-Horizon Optimization

Potential advanced phase.

Instead of optimizing only the current request:

```text
Current state
    ↓
Look ahead over short horizon
    ↓
Optimize
    ↓
Execute decision
    ↓
Observe new state
    ↓
Re-optimize
```

Do not implement this until the basic simulator and global assignment model are stable.

---

# 12. Evaluation Engine

The evaluation engine must be completely independent of allocation logic.

Architecture:

```text
Simulation
    ↓
Event Records
    ↓
Evaluation Engine
    ├── Rider Metrics
    ├── Driver Metrics
    ├── Fairness Metrics
    └── System Metrics
```

---

# 13. Rider Metrics

Calculate:

### Assignment rate

[
AssignmentRate =
\frac{AssignedRides}{TotalRequests}
]

### Unserved request rate

[
UnservedRate =
\frac{UnservedRequests}{TotalRequests}
]

### Pickup time

[
PickupTime =
DriverArrivalTime -
RequestTime
]

Report:

* Mean
* Median
* P90
* P95
* P99

### Pickup distance

Average driver distance traveled to reach the rider.

---

# 14. Driver Metrics

Calculate:

### Idle time

[
IdleTime =
AvailableTime -
BusyTime
]

### Utilization

[
Utilization_i =
\frac{BusyTime_i}
{OnlineTime_i}
]

### Rides per driver

Track distribution rather than only the average.

Potential metrics:

* Mean
* Median
* Standard deviation
* Coefficient of variation

---

# 15. Fairness Metrics

Implement:

## Jain's Fairness Index

[
J =
\frac{(\sum_i x_i)^2}
{n\sum_i x_i^2}
]

Calculate it for:

* rides received
* utilization
* potentially earnings

Higher values indicate greater equality.

---

## Gini coefficient

Calculate inequality for:

* rides
* utilization
* earnings when modeled

Lower values indicate greater equality.

---

## Opportunity fairness

Define:

[
OpportunityRate_i =
\frac{RidesReceived_i}
{EligibleOpportunities_i}
]

Compare the distribution across drivers.

This is particularly important because the project is concerned with **fair allocation opportunities**, not simply equal earnings.

---

# 16. System Metrics

Calculate:

* Completed rides
* Total pickup distance
* Overall driver utilization
* Unserved requests
* Average allocation computation time
* P95 allocation computation time
* Total simulation runtime

Computational cost is essential when comparing greedy algorithms with global optimization.

---

# 17. Do Not Initially Create a Single Composite Score

Do not immediately combine everything into:

```text
overall_score = ...
```

Keep metrics separate.

The purpose of the research is to discover trade-offs.

For example:

```text
Rider experience
        ↕
Driver fairness
        ↕
System efficiency
```

Analyze these trade-offs using Pareto analysis.

---

# 18. Experiment Framework

Experiments must be automatically reproducible.

Each experiment should specify:

```text
number of drivers
number of rides / arrival rate
simulation duration
city dimensions
demand pattern
driver availability
allocation policy
optimization method
policy weights
random seed
```

Example configuration:

```yaml
simulation:
  duration: 60
  city_width: 100
  city_height: 100

drivers:
  count: 100

demand:
  arrival_rate: 100

allocation:
  policy: idle_aware

weights:
  eta: 0.7
  idle: 0.3

seed: 42
```

---

# 19. Experimental Scenarios

At minimum test:

### Demand

```text
Low
Medium
High
Extreme
```

### Driver supply

```text
Shortage
Balanced
Surplus
```

### Spatial demand

```text
Uniform
Clustered
Highly clustered
```

### Temporal demand

```text
Constant
Peak
Peak + recovery
```

---

# 20. Random Seeds

Each configuration should be run multiple times.

Initially:

```text
20–30 seeds per configuration
```

Never compare two policies using only one random simulation.

Store individual runs before calculating aggregate statistics.

---

# 21. Statistical Analysis

For every important comparison calculate:

* Mean
* Standard deviation
* 95% confidence interval
* Relative improvement
* Effect size
* Appropriate statistical significance test

Example reporting style:

> Idle-aware allocation reduced median driver idle time by 24.8% relative to nearest-driver allocation while increasing mean rider pickup time by 0.31 minutes.

Avoid reporting only p-values.

---

# 22. Pareto Analysis

Treat important objectives separately.

For example:

[
f_1 = PickupTime
]

[
f_2 = DriverInequality
]

[
f_3 = IdleTime
]

Identify Pareto-optimal solutions.

The goal is to determine whether there are policies that substantially improve driver fairness with only a small rider-experience penalty.

---

# 23. Project Architecture

Use the following structure:

```text
bike-taxi-dispatch-research/
│
├── src/
│   ├── models/
│   │   ├── driver.py
│   │   ├── ride.py
│   │   └── events.py
│   │
│   ├── simulation/
│   │   ├── environment.py
│   │   ├── simulator.py
│   │   └── generators.py
│   │
│   ├── allocation/
│   │   ├── base.py
│   │   ├── random_policy.py
│   │   ├── nearest_policy.py
│   │   ├── eta_policy.py
│   │   ├── idle_aware_policy.py
│   │   ├── workload_policy.py
│   │   └── multi_objective.py
│   │
│   ├── optimization/
│   │   ├── assignment.py
│   │   └── multi_objective.py
│   │
│   ├── evaluation/
│   │   ├── rider_metrics.py
│   │   ├── driver_metrics.py
│   │   ├── fairness_metrics.py
│   │   ├── system_metrics.py
│   │   ├── statistics.py
│   │   └── evaluator.py
│   │
│   └── experiments/
│       ├── scenarios.py
│       ├── runner.py
│       └── parameter_grid.py
│
├── tests/
│
├── notebooks/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
├── plots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 24. Development Order

Build in this exact order.

### Phase 1

```text
Driver model
Ride model
Event model
Simulation environment
Synthetic demand generator
```

### Phase 2

```text
Random allocation
Nearest-driver allocation
```

### Phase 3

```text
Event logging
Evaluation engine
Basic metrics
```

### Phase 4

```text
ETA allocation
Idle-aware allocation
Workload-aware allocation
```

### Phase 5

```text
Multi-objective allocation
Weight experiments
```

### Phase 6

```text
OR-Tools assignment
Global optimization
```

### Phase 7

```text
Experiment runner
Multiple seeds
Scenario generation
```

### Phase 8

```text
Statistical analysis
Confidence intervals
Effect sizes
```

### Phase 9

```text
Pareto analysis
Visualization
```

### Phase 10

Only after the above is stable:

```text
Realistic road networks
ML ETA prediction
Demand prediction
Rolling-horizon optimization
```

---

# 25. Coding Principles

Follow these rules throughout the project:

1. Keep simulation, allocation, optimization, and evaluation independent.
2. Avoid global variables.
3. Use type hints where practical.
4. Use dataclasses for simple data models.
5. Write unit tests for important mathematical functions.
6. Keep random number generation reproducible.
7. Don't put research logic exclusively inside notebooks.
8. Use notebooks primarily for analysis and visualization.
9. Document assumptions explicitly.
10. Never hard-code experimental results.
11. Every experiment must be reproducible from its configuration.
12. Don't introduce complexity before validating the simpler model.

---

# 26. Research Progression

The final research pipeline should eventually look like:

```text
                 SCENARIO
                    │
                    ▼
             DEMAND GENERATOR
                    │
                    ▼
                SIMULATOR
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
       GREEDY POLICY   OPTIMIZER
             │             │
             └──────┬──────┘
                    ▼
              EVENT LOGGER
                    │
                    ▼
             EVALUATION ENGINE
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        RIDER     DRIVER    FAIRNESS
        METRICS   METRICS   METRICS
          │         │         │
          └─────────┼─────────┘
                    ▼
            STATISTICAL ANALYSIS
                    │
                    ▼
             PARETO ANALYSIS
                    │
                    ▼
              RESEARCH RESULTS
```

---

# 27. Ultimate Research Goal

Do **not** aim to conclude:

> "Our algorithm is better."

Instead, aim to answer something like:

> **Under what combinations of demand intensity, driver availability, and allocation objectives does fairness-aware dispatch provide meaningful improvements in driver opportunity equity without unacceptable degradation of rider service quality?**

That is a much stronger research question.

The project should ultimately produce **evidence**, not merely an algorithm.
