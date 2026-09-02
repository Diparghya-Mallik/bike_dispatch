"""Rider-side metrics (spec section 13).

Operates purely on ``RideResult`` records -- it has no knowledge of
which allocation policy produced them, per spec section 12's
requirement that the evaluation engine be independent of allocation
logic.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.models.events import RideResult


def _percentile_summary(values: List[float]) -> Dict[str, float]:
    """Mean/median/p90/p95/p99 summary, or an all-None dict if empty."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def assignment_rate(ride_results: Iterable[RideResult]) -> float:
    """Fraction of requests that received a driver assignment.

    Note: this counts *assignment*, not full completion -- a ride can be
    assigned and still be mid-trip (or, in principle, later cancelled in
    a future phase). Use ``ride_results`` served flags for completion.
    """
    results = list(ride_results)
    if not results:
        return 0.0
    assigned = sum(1 for r in results if r.assignment_time is not None)
    return assigned / len(results)


def unserved_rate(ride_results: Iterable[RideResult]) -> float:
    """Fraction of requests that never received a driver assignment."""
    return 1.0 - assignment_rate(ride_results)


def pickup_time_stats(ride_results: Iterable[RideResult]) -> Dict[str, float]:
    """Mean/median/p90/p95/p99 of rider pickup (waiting) time, over
    rides that were actually picked up (pickup_time is not None)."""
    waits = [r.waiting_time() for r in ride_results if r.waiting_time() is not None]
    return _percentile_summary(waits)


def pickup_distance_stats(ride_results: Iterable[RideResult]) -> Dict[str, float]:
    """Mean/median/p90/p95/p99 of driver distance traveled to pickup."""
    distances = [r.pickup_distance for r in ride_results if r.pickup_distance is not None]
    return _percentile_summary(distances)


def compute_rider_metrics(ride_results: Iterable[RideResult]) -> Dict[str, object]:
    """Full rider-side metric bundle for a set of RideResults."""
    results = list(ride_results)
    return {
        "total_requests": len(results),
        "assignment_rate": assignment_rate(results),
        "unserved_rate": unserved_rate(results),
        "pickup_time": pickup_time_stats(results),
        "pickup_distance": pickup_distance_stats(results),
    }