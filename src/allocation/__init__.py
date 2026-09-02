"""Allocation policy interfaces and implementations.

Phase 2: P0 random, P1 nearest driver.
Phase 4 adds: P2 minimum ETA, P3 idle-time aware, P4 workload aware.
Later phases add earnings-aware (P5) and multi-objective (P6) policies.
The simulator only ever depends on the AllocationPolicy interface,
never on a specific strategy.
"""

from .base import AllocationPolicy
from .random_policy import RandomPolicy
from .nearest_policy import NearestDriverPolicy
from .eta_policy import ETAPolicy
from .idle_aware_policy import IdleAwarePolicy
from .workload_policy import WorkloadAwarePolicy

__all__ = [
    "AllocationPolicy",
    "RandomPolicy",
    "NearestDriverPolicy",
    "ETAPolicy",
    "IdleAwarePolicy",
    "WorkloadAwarePolicy",
]