"""Allocation policy interfaces and implementations.

Phase 2 adds the first two concrete policies (P0 random, P1 nearest
driver). Later phases add ETA, idle-aware, workload-aware, earnings-
aware, and multi-objective policies, each in its own module -- the
simulator only ever depends on the ``AllocationPolicy`` interface, never
on a specific strategy.
"""

from .base import AllocationPolicy
from .random_policy import RandomPolicy
from .nearest_policy import NearestDriverPolicy

__all__ = ["AllocationPolicy", "RandomPolicy", "NearestDriverPolicy"]
