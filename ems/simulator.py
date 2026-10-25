
"""Event-driven simulator for emergency ambulance operations."""

from dataclasses import dataclass, field
import heapq
import math
import random

from ems.ambulance import Ambulance, AmbulanceStatus
from ems.city import City, Location
from ems.config import CityConfig
from ems.demand import DemandGenerator, EmergencyCall


@dataclass(order=True)
class SimulationEvent:
    """An event scheduled at a particular simulation time."""

    time_min: float
    sequence: int
    event_type: str = field(compare=False)
    ambulance_id: int = field(default=-1, compare=False)
    call_id: int = field(default=-1, compare=False)
    destination: Location | None = field(default=None, compare=False)


@dataclass
class CallOutcome:
    """Response-time outcome for a dispatched call."""

    call_id: int
    arrival_time_min: float
    response_time_min: float
    ambulance_id: int


class AmbulanceSimulator:
    """Simulate ambulance dispatch, incident service, and relocation."""

    def __init__(
        self,
        config: CityConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or CityConfig()
        self.city = City(self.config)
        self.rng = random.Random(seed)
        self.demand_generator = DemandGenerator(self.config, seed=seed)

        self.current_time_min = 0.0
        self._event_sequence = 0
        self._events: list[SimulationEvent] = []

        self.pending_calls: list[EmergencyCall] = []
        self.outcomes: list[CallOutcome] = []
        self.dropped_calls = 0
        self.relocation_count = 0
        self.relocation_minutes = 0.0

        if len(self.config.home_bases) != self.config.num_ambulances:
            raise ValueError(
                "home_bases must contain one entry per ambulance"
            )

        if any(
            not 0 <= base_id < len(self.city.bases)
            for base_id in self.config.home_bases
        ):
            raise ValueError("Every home base index must be valid")

        self.ambulances = [
            Ambulance(
                ambulance_id=index,
                home_base_id=base_id,
                location=self.city.bases[base_id],
            )
            for index, base_id in enumerate(self.config.home_bases)
        ]

    def _schedule_event(
        self,
        time_min: float,
        event_type: str,
        ambulance_id: int = -1,
        call_id: int = -1,
        destination: Location | None = None,
    ) -> None:
        """Add an event to the priority queue."""
        self._event_sequence += 1
        heapq.heappush(
            self._events,
            SimulationEvent(
                time_min=time_min,
                sequence=self._event_sequence,
                event_type=event_type,
                ambulance_id=ambulance_id,
                call_id=call_id,
                destination=destination,
            ),
        )

    def _travel_time(
        self,
        origin: Location,
        destination: Location,
        speed_kmh: float,
    ) -> float:
        """Sample travel time with lognormal variation."""
        mean_time = self.city.travel_time_minutes(
            origin, destination, speed_kmh
        )

        if mean_time <= 0:
            return 0.0

        sigma = self.config.travel_noise_sigma
        log_mean = math.log(mean_time) - 0.5 * sigma**2
        return self.rng.lognormvariate(log_mean, sigma)

    def relocate_ambulance(
        self,
        ambulance_id: int,
        target_base_id: int,
    ) -> bool:
        """Start relocating an idle ambulance to a candidate base."""
        if not 0 <= ambulance_id < len(self.ambulances):
            raise ValueError("Invalid ambulance ID")

        if not 0 <= target_base_id < len(self.city.bases):
            raise ValueError("Invalid target base ID")

        ambulance = self.ambulances[ambulance_id]

        if not ambulance.is_available(self.current_time_min):
            return False

        destination = self.city.bases[target_base_id]

        if ambulance.location == destination:
            return False

        travel_time = self._travel_time(
            ambulance.location,
            destination,
            self.config.reposition_speed_kmh,
        )

        ambulance.start_repositioning()
        ambulance.available_at_min = self.current_time_min + travel_time
        self.relocation_count += 1
        self.relocation_minutes += travel_time

        self._schedule_event(
            ambulance.available_at_min,
            "finish_repositioning",
            ambulance_id=ambulance_id,
            destination=destination,
        )
        return True

    def _dispatch(self, call: EmergencyCall) -> bool:
        """Dispatch the nearest available ambulance to a call."""
        available = [
            unit
            for unit in self.ambulances
            if unit.is_available(self.current_time_min)
        ]

        if not available:
            return False

        call_location = Location(*call.location)
        ambulance = min(
            available,
            key=lambda unit: self.city.road_distance(
                unit.location, call_location
            ),
        )

        ambulance.assign_call(call.call_id)

        travel_time = self._travel_time(
            ambulance.location,
            call_location,
            self.config.emergency_speed_kmh,
        )
        response_time = self.config.dispatch_delay_min + travel_time

        self._schedule_event(
            self.current_time_min + response_time,
            "arrive_scene",
            ambulance_id=ambulance.ambulance_id,
            call_id=call.call_id,
            destination=call_location,
        )

        self.outcomes.append(
            CallOutcome(
                call_id=call.call_id,
                arrival_time_min=call.arrival_time_min,
                response_time_min=response_time,
                ambulance_id=ambulance.ambulance_id,
            )
        )
        return True

    def _sample_lognormal_duration(
        self,
        mean: float,
        sigma: float,
    ) -> float:
        """Sample a positive duration with the requested arithmetic mean."""
        if mean <= 0:
            raise ValueError("Duration mean must be positive")
        if sigma < 0:
            raise ValueError("Duration sigma cannot be negative")
        if sigma == 0:
            return mean

        return self.rng.lognormvariate(
            math.log(mean) - 0.5 * sigma**2,
            sigma,
        )

    def _handle_event(self, event: SimulationEvent) -> None:
        """Process one scheduled event."""
        self.current_time_min = event.time_min
        ambulance = self.ambulances[event.ambulance_id]

        if event.event_type == "finish_repositioning":
            if event.destination is None:
                raise RuntimeError("Relocation event has no destination")

            ambulance.finish_repositioning(event.destination)
            ambulance.available_at_min = self.current_time_min
            return

        if event.event_type == "arrive_scene":
            if event.destination is None:
                raise RuntimeError("Scene arrival has no location")

            ambulance.location = event.destination
            ambulance.status = AmbulanceStatus.AT_SCENE

            scene_duration = self._sample_lognormal_duration(
                self.config.scene_time_mean_min,
                self.config.scene_time_sigma,
            )

            self._schedule_event(
                self.current_time_min + scene_duration,
                "finish_scene",
                ambulance_id=ambulance.ambulance_id,
                call_id=event.call_id,
            )
            return

        if event.event_type == "finish_scene":
            should_transport = (
                self.rng.random() < self.config.transport_probability
            )

            if not should_transport:
                ambulance.finish_call(ambulance.location)
                ambulance.available_at_min = self.current_time_min
                return

            hospital_id = self.city.nearest_hospital(ambulance.location)
            hospital = self.city.hospitals[hospital_id]
            ambulance.status = AmbulanceStatus.TO_HOSPITAL

            travel_time = self._travel_time(
                ambulance.location,
                hospital,
                self.config.emergency_speed_kmh,
            )

            self._schedule_event(
                self.current_time_min + travel_time,
                "arrive_hospital",
                ambulance_id=ambulance.ambulance_id,
                call_id=event.call_id,
                destination=hospital,
            )
            return

        if event.event_type == "arrive_hospital":
            if event.destination is None:
                raise RuntimeError("Hospital arrival has no location")

            ambulance.location = event.destination
            ambulance.status = AmbulanceStatus.AT_HOSPITAL

            handover_duration = self._sample_lognormal_duration(
                self.config.hospital_handover_mean_min,
                self.config.hospital_handover_sigma,
            )

            self._schedule_event(
                self.current_time_min + handover_duration,
                "finish_handover",
                ambulance_id=ambulance.ambulance_id,
                call_id=event.call_id,
            )
            return

        if event.event_type == "finish_handover":
            ambulance.finish_call(ambulance.location)
            ambulance.available_at_min = self.current_time_min

    
    def run_day(self, decision_callback=None) -> list[CallOutcome]:
        """Run one day, optionally requesting periodic redeployment decisions.

        The callback receives the simulator at each decision epoch.
        It may call relocate_ambulance() to move an idle unit.
        """
        self.current_time_min = 0.0
        self._event_sequence = 0
        self._events = []
        self.outcomes = []
        self.dropped_calls = 0
        self.relocation_count = 0
        self.relocation_minutes = 0.0
        self.pending_calls = self.demand_generator.generate_day()

        for ambulance in self.ambulances:
            ambulance.location = self.city.bases[ambulance.home_base_id]
            ambulance.status = AmbulanceStatus.IDLE
            ambulance.assigned_call_id = None
            ambulance.available_at_min = 0.0

        call_index = 0
        interval = self.config.redeployment_interval_min
        next_decision_time = interval

        while call_index < len(self.pending_calls) or self._events:
            next_call_time = (
                self.pending_calls[call_index].arrival_time_min
                if call_index < len(self.pending_calls)
                else float("inf")
            )
            next_event_time = (
                self._events[0].time_min
                if self._events
                else float("inf")
            )
            next_time = min(next_call_time, next_event_time)

            if (
                decision_callback is not None
                and next_decision_time <= next_time
                and next_decision_time < self.config.episode_duration_hours * 60
            ):
                self.current_time_min = next_decision_time
                decision_callback(self)
                next_decision_time += interval
                continue

            if next_call_time <= next_event_time:
                self.current_time_min = next_call_time
                call = self.pending_calls[call_index]
                call_index += 1

                if not self._dispatch(call):
                    self.dropped_calls += 1
            else:
                event = heapq.heappop(self._events)
                self._handle_event(event)

        return self.outcomes

