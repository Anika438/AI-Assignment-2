
"""Build numerical state features for ambulance redeployment."""

import math

from ems.city import City, Location
from ems.config import CityConfig, hotspot_weights


class StateBuilder:
    """Convert city and ambulance conditions into an RL state vector."""

    def __init__(self, config: CityConfig | None = None) -> None:
        self.config = config or CityConfig()
        self.city = City(self.config)

    def build_state(
        self,
        current_time_min: float,
        ambulance_locations: list[Location],
        available_flags: list[bool],
    ) -> list[float]:
        """Build a fixed-length state vector from current conditions."""
        if len(ambulance_locations) != self.config.num_ambulances:
            raise ValueError("Provide one location per ambulance")

        if len(available_flags) != self.config.num_ambulances:
            raise ValueError("Provide one availability flag per ambulance")

        hour = (current_time_min / 60.0) % 24.0
        angle = 2.0 * math.pi * hour / 24.0

        state = [
            math.sin(angle),
            math.cos(angle),
            self.config.hourly_profile[int(hour)],
            sum(available_flags) / len(available_flags),
        ]

        weights = hotspot_weights(hour)
        max_weight = max(weights) or 1.0
        normalized_weights = [weight / max_weight for weight in weights]
        state.extend(normalized_weights)

        for base in self.city.bases:
            nearby_demand = sum(
                weight
                for (x, y, spread, _name), weight in zip(
                    self.config.hotspots, weights
                )
                if self.city.manhattan_distance(
                    base, Location(x, y)
                ) <= 2.0 * spread
            )
            state.append(nearby_demand / max(sum(weights), 1e-8))

        for base in self.city.bases:
            available_nearby = sum(
                1.0
                for location, is_available in zip(
                    ambulance_locations, available_flags
                )
                if is_available
                and self.city.manhattan_distance(location, base) <= 3.0
            )
            state.append(available_nearby / self.config.num_ambulances)

        return state
