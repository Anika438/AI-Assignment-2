
"""Generate emergency-call locations and arrival times."""

from dataclasses import dataclass
import math
import random

from ems.city import City
from ems.config import CityConfig, hotspot_weights


@dataclass(frozen=True)
class EmergencyCall:
    """Represent one emergency call in the simulation."""

    call_id: int
    arrival_time_min: float
    location: tuple[float, float]


class DemandGenerator:
    """Generate emergency calls using time-varying spatial demand."""

    def __init__(
        self,
        config: CityConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or CityConfig()
        self.city = City(self.config)
        self.rng = random.Random(seed)
        self.next_call_id = 0

        self.hourly_average = (
            self.config.expected_calls_per_day / 24.0
        )

    def hourly_rate(self, hour: float) -> float:
        """Estimate the expected calls per hour at a given hour."""
        hour_index = int(hour) % 24
        return (
            self.hourly_average
            * self.config.hourly_profile[hour_index]
        )

    def _sample_poisson(self, rate: float) -> int:
        """Sample a Poisson random variable using Knuth's algorithm."""
        if rate < 0:
            raise ValueError("rate cannot be negative")
        if rate == 0:
            return 0

        threshold = math.exp(-rate)
        product = 1.0
        count = 0

        while product > threshold:
            count += 1
            product *= self.rng.random()

        return count - 1

    def _sample_hotspot(self, hour: float) -> tuple[float, float]:
        """Choose a hotspot and sample a location around it."""
        weights = hotspot_weights(hour)
        total_weight = sum(weights)
        target = self.rng.random() * total_weight

        cumulative_weight = 0.0
        selected_hotspot = self.config.hotspots[-1]

        for hotspot, weight in zip(self.config.hotspots, weights):
            cumulative_weight += weight
            if target <= cumulative_weight:
                selected_hotspot = hotspot
                break

        center_x, center_y, spread, _name = selected_hotspot

        x = self.rng.gauss(center_x, spread)
        y = self.rng.gauss(center_y, spread)

        # Keep sampled coordinates inside the city.
        x = min(max(x, 0.0), self.config.city_size_km)
        y = min(max(y, 0.0), self.config.city_size_km)

        return x, y

    def _sample_location(self, hour: float) -> tuple[float, float]:
        """Sample a call location from hotspots or the city background."""
        if self.rng.random() < self.config.background_demand_weight:
            return (
                self.rng.uniform(0.0, self.config.city_size_km),
                self.rng.uniform(0.0, self.config.city_size_km),
            )

        return self._sample_hotspot(hour)

    def generate_day(self) -> list[EmergencyCall]:
        """Generate one day's calls and return them in arrival order."""
        calls: list[EmergencyCall] = []

        for hour in range(24):
            rate = self.hourly_rate(hour)
            number_of_calls = self._sample_poisson(rate)

            for _ in range(number_of_calls):
                arrival_time = hour * 60 + self.rng.uniform(0.0, 60.0)
                location = self._sample_location(float(hour))

                calls.append(
                    EmergencyCall(
                        call_id=self.next_call_id,
                        arrival_time_min=arrival_time,
                        location=location,
                    )
                )
                self.next_call_id += 1

        calls.sort(key=lambda call: call.arrival_time_min)
        return calls
