"""Fairness metrics (spec section 15).

Three notions of fairness, deliberately kept separate rather than
combined into one score (spec section 17):

  - Jain's Fairness Index: a general-purpose equality measure over any
    per-driver quantity (rides received, utilization, earnings).
  - Gini coefficient: a general-purpose inequality measure, same inputs.
  - Opportunity fairness: specifically, how evenly ride *opportunities*
    (not just rides won) were distributed -- i.e. was a driver even in
    the running for a given request. This is the metric the spec calls
    out as "particularly important... not simply equal earnings."

All three operate on plain per-driver value lists / the raw event log
-- no allocation logic, no assumptions about which policy produced the
data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from src.models.driver import Driver
from src.models.events import Event, EventType


def jains_fairness_index(values: Iterable[float]) -> float:
    """Jain's Fairness Index: (sum x_i)^2 / (n * sum x_i^2).

    Ranges from 1/n (maximally unfair -- all in one driver) to 1.0
    (perfectly fair -- all equal). By convention, returns 1.0 for an
    empty input or when every value is zero (nobody got anything, but
    that "nothing" was distributed equally).
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return 1.0
    arr = np.asarray(values, dtype=float)
    sum_sq = float(np.sum(arr) ** 2)
    sq_sum = float(np.sum(arr ** 2))
    if sq_sum == 0:
        return 1.0
    return sum_sq / (n * sq_sum)


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of inequality: 0 = perfect equality, ~(n-1)/n =
    maximal inequality (one driver has everything).

    Uses the standard rank-based formula on the sorted array. Returns
    0.0 for an empty input or when every value is zero.
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=float))
    total = float(np.sum(arr))
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * total) - (n + 1) / n)


def opportunity_rates(events: Iterable[Event], driver_ids: Iterable[str]) -> Dict[str, float]:
    """Per-driver opportunity rate: rides received / eligible opportunities.

    An "eligible opportunity" for a driver is a ride request that
    occurred while that driver was AVAILABLE (recorded on the
    RIDE_REQUESTED event's ``available_driver_ids`` metadata). "Rides
    received" counts DRIVER_ASSIGNED events for that driver.

    A driver with zero eligible opportunities gets a rate of None
    (undefined, not zero -- they were simply never in the running).
    """
    events = list(events)
    driver_ids = list(driver_ids)

    eligible_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        if event.event_type is not EventType.RIDE_REQUESTED:
            continue
        available = (event.metadata or {}).get("available_driver_ids", [])
        for driver_id in available:
            eligible_counts[driver_id] += 1

    received_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        if event.event_type is EventType.DRIVER_ASSIGNED and event.driver_id:
            received_counts[event.driver_id] += 1

    rates: Dict[str, float] = {}
    for driver_id in driver_ids:
        eligible = eligible_counts.get(driver_id, 0)
        received = received_counts.get(driver_id, 0)
        rates[driver_id] = (received / eligible) if eligible > 0 else None
    return rates


def compute_fairness_metrics(drivers: Iterable[Driver], events: Iterable[Event]) -> Dict[str, object]:
    """Full fairness metric bundle: Jain's index + Gini for rides
    received and utilization, plus the opportunity-fairness breakdown.
    """
    drivers = list(drivers)
    events = list(events)

    rides = [d.rides_completed for d in drivers]
    utilizations = [d.utilization() for d in drivers]

    rates_by_driver = opportunity_rates(events, [d.driver_id for d in drivers])
    defined_rates = [r for r in rates_by_driver.values() if r is not None]

    return {
        "rides_received": {
            "jains_index": jains_fairness_index(rides),
            "gini": gini_coefficient(rides),
        },
        "utilization": {
            "jains_index": jains_fairness_index(utilizations),
            "gini": gini_coefficient(utilizations),
        },
        "opportunity_fairness": {
            "rates_by_driver": rates_by_driver,
            "jains_index": jains_fairness_index(defined_rates),
            "gini": gini_coefficient(defined_rates),
            "num_drivers_with_opportunities": len(defined_rates),
        },
    }