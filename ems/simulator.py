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
    """An event scheduled at a particular simulation time.

    Events are ordered chronologically by time_min. When events occur at the exact same
    time, the priority field breaks ties deterministically:
      0: Service and relocation completion events (freeing up ambulances)
      1: Emergency call arrival events (dispatching ambulances)
      2: Periodic decision epochs (evaluating fleet and redeploying)
    Sequence ensures stable FIFO ordering among identical time and priority.
    """

    time_min: float
    priority: int = 0
    sequence: int = 0
    event_type: str = field(default="", compare=False)
    ambulance_id: int = field(default=-1, compare=False)
    call_id: int = field(default=-1, compare=False)
    destination: Location | None = field(default=None, compare=False)
    call: EmergencyCall | None = field(default=None, compare=False)


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
        self.seed = seed

        if len(self.config.home_bases) != self.config.num_ambulances:
            raise ValueError(
                "home_bases must contain one entry per ambulance"
            )

        if any(
            not 0 <= base_id < len(self.city.bases)
            for base_id in self.config.home_bases
        ):
            raise ValueError("Every home base index must be valid")

        self.reset(seed=self.seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the simulator state and random number generator for a new episode."""
        if seed is not None:
            self.seed = seed

        self.rng = random.Random(self.seed)
        self.demand_generator = DemandGenerator(self.config, seed=self.seed)

        self.current_time_min = 0.0
        self._event_sequence = 0
        self._events: list[SimulationEvent] = []

        self.pending_calls: list[EmergencyCall] = []
        self.outcomes: list[CallOutcome] = []
        self.dropped_calls = 0
        self.relocation_count = 0
        self.relocation_minutes = 0.0

        self.ambulances = [
            Ambulance(
                ambulance_id=index,
                home_base_id=base_id,
                location=self.city.bases[base_id],
                status=AmbulanceStatus.IDLE,
                assigned_call_id=None,
                available_at_min=0.0,
            )
            for index, base_id in enumerate(self.config.home_bases)
        ]

    def _schedule_event(
        self,
        time_min: float,
        event_type: str,
        priority: int = 0,
        ambulance_id: int = -1,
        call_id: int = -1,
        destination: Location | None = None,
        call: EmergencyCall | None = None,
    ) -> None:
        """Add an event to the priority queue."""
        self._event_sequence += 1
        heapq.heappush(
            self._events,
            SimulationEvent(
                time_min=time_min,
                priority=priority,
                sequence=self._event_sequence,
                event_type=event_type,
                ambulance_id=ambulance_id,
                call_id=call_id,
                destination=destination,
                call=call,
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
        """Start relocating an idle ambulance to a candidate base.

        Returns True if relocation successfully started; False if the unit is busy,
        the target is invalid, or the unit is already at the target base.
        """
        if not 0 <= ambulance_id < len(self.ambulances):
            raise ValueError("Invalid ambulance ID")

        if not 0 <= target_base_id < len(self.city.bases):
            raise ValueError("Invalid target base ID")

        ambulance = self.ambulances[ambulance_id]

        if not ambulance.is_available(self.current_time_min):
            return False

        destination = self.city.bases[target_base_id]

        if ambulance.location.is_close_to(destination):
            return False

        travel_time = self._travel_time(
            ambulance.location,
            destination,
            self.config.reposition_speed_kmh,
        )

        arrival_time = self.current_time_min + travel_time
        ambulance.start_repositioning(available_at_min=arrival_time)
        self.relocation_count += 1
        self.relocation_minutes += travel_time

        self._schedule_event(
            arrival_time,
            "finish_repositioning",
            priority=0,
            ambulance_id=ambulance_id,
            destination=destination,
        )
        return True

    def _dispatch(self, call: EmergencyCall) -> bool:
        """Dispatch the nearest available ambulance to a call.

        Returns True if dispatched; False if dropped under the documented immediate-drop
        policy (when all ambulances are busy or relocating).
        """
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
        arrive_scene_time = (
            self.current_time_min
            + self.config.dispatch_delay_min
            + travel_time
        )
        response_time = arrive_scene_time - call.arrival_time_min

        self._schedule_event(
            arrive_scene_time,
            "arrive_scene",
            priority=0,
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
        """Process one scheduled completion event."""
        self.current_time_min = event.time_min
        ambulance = self.ambulances[event.ambulance_id]

        if event.event_type == "finish_repositioning":
            if event.destination is None:
                raise RuntimeError("Relocation event has no destination")

            ambulance.finish_repositioning(
                event.destination, current_time_min=self.current_time_min
            )
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
                priority=0,
                ambulance_id=ambulance.ambulance_id,
                call_id=event.call_id,
            )
            return

        if event.event_type == "finish_scene":
            should_transport = (
                self.rng.random() < self.config.transport_probability
            )

            if not should_transport:
                ambulance.finish_call(
                    ambulance.location, current_time_min=self.current_time_min
                )
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
                priority=0,
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
                priority=0,
                ambulance_id=ambulance.ambulance_id,
                call_id=event.call_id,
            )
            return

        if event.event_type == "finish_handover":
            ambulance.finish_call(
                ambulance.location, current_time_min=self.current_time_min
            )

    def run_day(
        self,
        decision_callback=None,
        seed: int | None = None,
        calls: list[EmergencyCall] | None = None,
        reset: bool = True,
    ) -> list[CallOutcome]:
        """Run one day, optionally requesting periodic redeployment decisions.

        The callback receives the simulator at each decision epoch.
        It may call relocate_ambulance() to move an idle unit.
        """
        if reset:
            self.reset(seed=seed if seed is not None else self.seed)

        if calls is not None:
            self.pending_calls = list(calls)
        else:
            self.pending_calls = self.demand_generator.generate_day()

        # Schedule all call arrivals (priority 1)
        for call in self.pending_calls:
            self._schedule_event(
                call.arrival_time_min,
                "call_arrival",
                priority=1,
                call_id=call.call_id,
                call=call,
            )

        # Schedule periodic decision epochs (priority 2)
        if decision_callback is not None:
            interval = self.config.redeployment_interval_min
            duration = self.config.episode_duration_hours * 60.0
            next_t = interval
            while next_t < duration:
                self._schedule_event(next_t, "decision_epoch", priority=2)
                next_t += interval

        while self._events:
            event = heapq.heappop(self._events)
            self.current_time_min = event.time_min

            if event.event_type == "decision_epoch":
                if decision_callback is not None:
                    decision_callback(self)
            elif event.event_type == "call_arrival":
                if event.call is not None and not self._dispatch(event.call):
                    self.dropped_calls += 1
            else:
                self._handle_event(event)

        return self.outcomes
