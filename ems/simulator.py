
"""Event-driven simulator for emergency ambulance operations."""

from dataclasses import dataclass, field
import heapq
import random

from ems.ambulance import Ambulance, AmbulanceStatus
from ems.city import City, Location
from ems.config import CityConfig
from ems.demand import DemandGenerator, EmergencyCall


@dataclass(order=True)
class SimulationEvent:
    """An event scheduled to occur at a particular simulation time."""

    time_min: float
    sequence: int
    event_type: str = field(compare=False)
    ambulance_id: int = field(default=-1, compare=False)
    call_id: int = field(default=-1, compare=False)


@dataclass
class CallOutcome:
    """Record the outcome of serving an emergency call."""

    call_id: int
    arrival_time_min: float
    response_time_min: float
    ambulance_id: int


class AmbulanceSimulator:
    """Run a single-day ambulance simulation."""

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

        self.ambulances = [
            Ambulance(
                ambulance_id=index,
                home_base_id=base_id,
                location=self.city.bases[base_id],
            )
            for index, base_id in enumerate(self.config.home_bases)
        ]

        if len(self.ambulances) != self.config.num_ambulances:
            raise ValueError(
                "home_bases must contain one entry per ambulance"
            )

    def _schedule_event(
        self,
        time_min: float,
        event_type: str,
        ambulance_id: int = -1,
        call_id: int = -1,
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
        return self.rng.lognormvariate(
            meanvar := __import__("math").log(mean_time)
            - 0.5 * sigma**2,
            sigma,
        )

    def _dispatch(self, call: EmergencyCall) -> bool:
        """Dispatch the nearest idle ambulance, if one is available."""
        available = [
            ambulance
            for ambulance in self.ambulances
            if ambulance.is_available(self.current_time_min)
        ]

        if not available:
            return False

        ambulance = min(
            available,
            key=lambda unit: self.city.road_distance(
                unit.location, Location(*call.location)
            ),
        )

        ambulance.assign_call(call.call_id)
        call_location = Location(*call.location)

        travel_time = self._travel_time(
            ambulance.location,
            call_location,
            self.config.emergency_speed_kmh,
        )
        response_time = self.config.dispatch_delay_min + travel_time

        self._schedule_event(
            self.current_time_min + response_time,
            "arrive_scene",
            ambulance.ambulance_id,
            call.call_id,
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

    def _handle_event(self, event: SimulationEvent) -> None:
        """Apply the state transition associated with an event."""
        self.current_time_min = event.time_min
        ambulance = self.ambulances[event.ambulance_id]

        if event.event_type == "arrive_scene":
            ambulance.status = AmbulanceStatus.AT_SCENE

            scene_duration = self.rng.lognormvariate(
                __import__("math").log(
                    self.config.scene_time_mean_min
                ) - 0.5 * self.config.scene_time_sigma**2,
                self.config.scene_time_sigma,
            )
            self._schedule_event(
                self.current_time_min + scene_duration,
                "finish_service",
                ambulance.ambulance_id,
                event.call_id,
            )

        elif event.event_type == "finish_service":
            ambulance.finish_call(ambulance.location)
            ambulance.available_at_min = self.current_time_min

    def run_day(self) -> list[CallOutcome]:
        """Simulate one day and return outcomes for dispatched calls."""
        self.pending_calls = self.demand_generator.generate_day()
        call_index = 0

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

            if next_call_time <= next_event_time:
                self.current_time_min = next_call_time
                call = self.pending_calls[call_index]
                call_index += 1

                if not self._dispatch(call):
                    # This initial version drops calls when every unit is busy.
                    continue
            else:
                event = heapq.heappop(self._events)
                self._handle_event(event)

        return self.outcomes
