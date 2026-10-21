
"""Geographic utilities for the ambulance simulation."""

from dataclasses import dataclass
import math

from ems.config import CityConfig


@dataclass(frozen=True)
class Location:
    """A point in the city, measured in kilometres from the origin."""

    x: float
    y: float


class City:
    """Represent the simulated city and its candidate locations."""

    def __init__(self, config: CityConfig | None = None) -> None:
        self.config = config or CityConfig()
        self.size_km = self.config.city_size_km
        self.grid_size = self.config.grid_size

        self.bases = [
            Location(x, y) for x, y in self.config.bases
        ]
        self.hospitals = [
            Location(x, y) for x, y in self.config.hospitals
        ]

        self.zone_width_km = self.size_km / self.grid_size

    @property
    def num_zones(self) -> int:
        """Return the number of demand zones in the city."""
        return self.grid_size**2

    def zone_to_location(self, zone_id: int) -> Location:
        """Return the centre coordinate of a demand zone.

        Zone IDs are numbered row by row, starting at the bottom-left:
        0, 1, ..., grid_size**2 - 1.
        """
        if not 0 <= zone_id < self.num_zones:
            raise ValueError(
                f"zone_id must be between 0 and {self.num_zones - 1}"
            )

        row, column = divmod(zone_id, self.grid_size)

        return Location(
            x=(column + 0.5) * self.zone_width_km,
            y=(row + 0.5) * self.zone_width_km,
        )

    def manhattan_distance(
        self,
        first: Location,
        second: Location,
    ) -> float:
        """Calculate grid-road distance before applying the road factor."""
        return abs(first.x - second.x) + abs(first.y - second.y)

    def road_distance(
        self,
        first: Location,
        second: Location,
    ) -> float:
        """Estimate road distance in kilometres."""
        return (
            self.manhattan_distance(first, second)
            * self.config.road_distance_factor
        )

    def travel_time_minutes(
        self,
        first: Location,
        second: Location,
        speed_kmh: float,
    ) -> float:
        """Calculate travel time for a given speed, excluding delays."""
        if speed_kmh <= 0:
            raise ValueError("speed_kmh must be positive")

        distance_km = self.road_distance(first, second)
        return distance_km / speed_kmh * 60.0

    def nearest_base(self, location: Location) -> int:
        """Return the index of the nearest candidate base."""
        return min(
            range(len(self.bases)),
            key=lambda index: self.manhattan_distance(
                location, self.bases[index]
            ),
        )

    def nearest_hospital(self, location: Location) -> int:
        """Return the index of the nearest hospital."""
        return min(
            range(len(self.hospitals)),
            key=lambda index: self.road_distance(
                location, self.hospitals[index]
            ),
        )
