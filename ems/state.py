"""Build numerical state features for ambulance redeployment."""

import math

from ems.city import City, Location
from ems.config import CityConfig, hotspot_weights


class StateBuilder:
    """Convert city and ambulance conditions into a normalized RL state vector."""

    def __init__(self, config: CityConfig | None = None) -> None:
        self.config = config or CityConfig()
        self.city = City(self.config)
        # Fleet features: 3 (time) + 2 (fleet) + 5 (hotspots) + 12 (base demand)
        #                + 12 (occupancy) + 12 (coverage) + 12 (deficit) + 1 (global cov) + 1 (hospital) = 60
        # Candidate donor features: 1 (at hospital) + 12 (road dist to bases) + 12 (at base) = 25
        # Total = 85
        self.fleet_features_size = 60
        self.donor_features_size = 25
        self.state_size = self.fleet_features_size + self.donor_features_size

    def build_state(
        self,
        current_time_min: float,
        ambulance_locations: list[Location],
        available_flags: list[bool],
        candidate_location: Location | None = None,
    ) -> list[float]:
        """Build a fixed-length normalized state vector from current conditions."""
        if len(ambulance_locations) != self.config.num_ambulances:
            raise ValueError("Provide one location per ambulance")

        if len(available_flags) != self.config.num_ambulances:
            raise ValueError("Provide one availability flag per ambulance")

        hour = (current_time_min / 60.0) % 24.0
        angle = 2.0 * math.pi * hour / 24.0

        max_hourly = max(self.config.hourly_profile) or 1.0
        avail_count = sum(available_flags)
        num_amb = float(self.config.num_ambulances)

        # 1. Time-of-day features (3)
        state: list[float] = [
            math.sin(angle),
            math.cos(angle),
            self.config.hourly_profile[int(hour)] / max_hourly,
        ]

        # 2. Fleet availability (2)
        state.append(avail_count / num_amb)
        state.append((num_amb - avail_count) / num_amb)

        # 3. Hotspot demand weights (5)
        weights = hotspot_weights(hour)
        max_weight = max(weights) or 1.0
        state.extend([weight / max_weight for weight in weights])

        # 4. Nearby demand around candidate bases (12)
        sum_weights = sum(weights) or 1.0
        nearby_demands = []
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
            norm_dem = nearby_demand / sum_weights
            nearby_demands.append(norm_dem)
            state.append(norm_dem)

        # 5. Exact base occupancy (12): units stationed directly at candidate bases
        for base in self.city.bases:
            exact_count = sum(
                1.0
                for location, is_available in zip(
                    ambulance_locations, available_flags
                )
                if is_available and location.is_close_to(base)
            )
            state.append(min(exact_count, 3.0) / 3.0)

        # 6. Local coverage around bases (12): units within 3.36 km (8-min reach)
        coverage_radius_km = 3.36
        local_unit_counts = []
        for base in self.city.bases:
            available_nearby = sum(
                1.0
                for location, is_available in zip(
                    ambulance_locations, available_flags
                )
                if is_available
                and self.city.manhattan_distance(location, base)
                <= coverage_radius_km
            )
            local_unit_counts.append(available_nearby)
            state.append(available_nearby / num_amb)

        # 7. Base demand deficit (12): unmet demand at each base
        for dem, units in zip(nearby_demands, local_unit_counts):
            deficit = dem / (1.0 + units)
            state.append(deficit)

        # 8. Global coverage fraction (1)
        avail_locations = [
            loc
            for loc, avail in zip(ambulance_locations, available_flags)
            if avail
        ]
        global_cov = self.city.demand_coverage_fraction(
            avail_locations, hour, self.config.response_target_min
        )
        state.append(global_cov)

        # 9. Units stranded at hospitals (1)
        stranded_hosp = sum(
            1.0
            for loc, avail in zip(ambulance_locations, available_flags)
            if avail and any(loc.is_close_to(h) for h in self.city.hospitals)
        )
        state.append(stranded_hosp / num_amb)

        # 10. Candidate donor features (25)
        max_road_dist = (
            self.config.city_size_km * 2.0 * self.config.road_distance_factor
        )
        if candidate_location is not None:
            # Is candidate at hospital? (1)
            at_hosp = 1.0 if any(
                candidate_location.is_close_to(h) for h in self.city.hospitals
            ) else 0.0
            state.append(at_hosp)

            # Road distance to each of the 12 bases normalized in [0, 1] (12)
            for base in self.city.bases:
                dist = self.city.road_distance(candidate_location, base)
                state.append(min(dist / max_road_dist, 1.0))

            # Is candidate currently stationed at base b? (12)
            for base in self.city.bases:
                at_b = 1.0 if candidate_location.is_close_to(base) else 0.0
                state.append(at_b)
        else:
            state.extend([0.0] * self.donor_features_size)

        return state
