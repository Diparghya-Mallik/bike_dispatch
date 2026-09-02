"""Discrete-event simulator (spec section 8).

The Simulator drives a SimPy environment forward, moving rides through
their lifecycle and drivers through theirs, emitting an Event for every
state change. It contains **no allocation logic**: it calls
``policy.select_driver(ride, available_drivers)`` and acts on the
result, so any AllocationPolicy can be swapped in without touching this
file.

Phase 3 additions (for the evaluation engine):
  - Each RIDE_REQUESTED event records ``available_driver_ids`` -- the
    drivers who were AVAILABLE at that instant -- so opportunity
    fairness (spec section 15) can be computed later: an "eligible
    opportunity" for a driver is a ride request that happened while
    that driver was available, regardless of who got picked.
  - Each RideResult records ``allocation_compute_time`` -- the wall-clock
    time spent inside policy.select_driver() -- so system metrics (spec
    section 16) can report allocation cost.
  - SimulationResult records ``total_runtime`` -- the wall-clock time to
    run the whole simulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import simpy

from src.allocation.base import AllocationPolicy
from src.models.driver import Driver, DriverState
from src.models.events import Event, EventType, RideResult
from src.models.ride import Ride
from src.simulation.environment import City


@dataclass
class SimulationResult:
    events: List[Event]
    ride_results: Dict[str, RideResult]
    drivers: List[Driver]
    rides: List[Ride]
    duration: float
    total_runtime: float = 0.0

    def events_as_records(self) -> List[dict]:
        return [e.as_dict() for e in self.events]

    def ride_results_as_records(self) -> List[dict]:
        return [r.as_dict() for r in self.ride_results.values()]


class Simulator:
    def __init__(self, city: City, drivers: List[Driver], rides: List[Ride], policy: AllocationPolicy):
        self.city = city
        self.drivers = drivers
        self.rides = rides
        self.policy = policy

        self._drivers_by_id = {d.driver_id: d for d in drivers}
        self.events: List[Event] = []
        self.ride_results: Dict[str, RideResult] = {
            r.ride_id: RideResult(ride_id=r.ride_id, request_time=r.request_time)
            for r in rides
        }
        self.env: Optional[simpy.Environment] = None

    def _log(self, event_type: EventType, ride_id=None, driver_id=None, **metadata) -> None:
        self.events.append(
            Event(
                event_type=event_type,
                time=self.env.now,
                ride_id=ride_id,
                driver_id=driver_id,
                metadata=metadata or None,
            )
        )

    def _available_drivers(self) -> List[Driver]:
        return [d for d in self.drivers if d.is_available()]

    def _handle_ride(self, ride: Ride):
        env = self.env
        if env.now < ride.request_time:
            yield env.timeout(ride.request_time - env.now)

        result = self.ride_results[ride.ride_id]
        available = self._available_drivers()
        self._log(
            EventType.RIDE_REQUESTED,
            ride_id=ride.ride_id,
            available_driver_ids=[d.driver_id for d in available],
        )

        start = time.perf_counter()
        driver = self.policy.select_driver(ride, available)
        result.allocation_compute_time = time.perf_counter() - start

        if driver is None:
            ride.mark_unserved()
            result.served = False
            self._log(EventType.RIDE_UNSERVED, ride_id=ride.ride_id)
            return

        now = env.now
        driver.transition_to(DriverState.ASSIGNED, now)
        driver.current_ride = ride.ride_id
        ride.mark_assigned(driver.driver_id)
        result.assignment_time = now
        result.driver_id = driver.driver_id
        self._log(EventType.DRIVER_ASSIGNED, ride_id=ride.ride_id, driver_id=driver.driver_id)

        yield from self._run_trip(driver, ride, result)

    def _run_trip(self, driver: Driver, ride: Ride, result: RideResult):
        env = self.env

        now = env.now
        driver.transition_to(DriverState.PICKING_UP, now)
        pickup_distance = self.city.distance(driver.location, ride.pickup_location)
        pickup_travel_time = self.city.travel_time(driver.location, ride.pickup_location)
        result.pickup_eta = pickup_travel_time
        self._log(EventType.PICKUP_STARTED, ride_id=ride.ride_id, driver_id=driver.driver_id, eta=pickup_travel_time)

        yield env.timeout(pickup_travel_time)

        now = env.now
        driver.move_to(ride.pickup_location)
        result.pickup_time = now
        result.pickup_distance = pickup_distance
        ride.mark_en_route()
        self._log(EventType.RIDER_PICKED_UP, ride_id=ride.ride_id, driver_id=driver.driver_id)

        driver.transition_to(DriverState.ON_RIDE, now)
        ride.mark_in_progress()
        result.ride_start_time = now
        self._log(EventType.RIDE_STARTED, ride_id=ride.ride_id, driver_id=driver.driver_id)

        trip_time = self.city.travel_time(ride.pickup_location, ride.destination)
        yield env.timeout(trip_time)

        now = env.now
        driver.move_to(ride.destination)
        ride.mark_completed()
        result.completion_time = now
        result.served = True
        self._log(EventType.RIDE_COMPLETED, ride_id=ride.ride_id, driver_id=driver.driver_id)

        driver.complete_ride(now)
        self._log(EventType.DRIVER_AVAILABLE, driver_id=driver.driver_id)

    def run(self, duration: float) -> SimulationResult:
        wall_start = time.perf_counter()
        self.env = simpy.Environment()

        for ride in self.rides:
            self.env.process(self._handle_ride(ride))

        self.env.run(until=duration)

        for driver in self.drivers:
            driver.accrue_online_time(duration)

        total_runtime = time.perf_counter() - wall_start

        return SimulationResult(
            events=self.events,
            ride_results=self.ride_results,
            drivers=self.drivers,
            rides=self.rides,
            duration=duration,
            total_runtime=total_runtime,
        )