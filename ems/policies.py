"""Baseline redeployment policies for emergency ambulance operations.

All policies implement a `decide(simulator)` callback that can be directly passed
to `AmbulanceSimulator.run_day(decision_callback=policy.decide)`.
"""

from ems.city import City, Location
from ems.config import hotspot_weights


class NoRedeploymentPolicy:
    """Baseline 1: No redeployment.

    Ambulances remain at whichever location they become idle (either their initial home
    base, or the scene / hospital of their last served call) until dispatched again.
    Performs exactly 0 relocations.
    """

    def decide(self, simulator) -> None:
        """No relocation action taken."""
        pass


class FixedHomeBasePolicy:
    """Baseline 2: Fixed-Home Base return policy.

    Whenever an ambulance completes service and becomes idle at a hospital or scene,
    it is dispatched to return to its assigned home station.
    """

    def __init__(self, city: City | None = None) -> None:
        self.city = city

    def choose_base(self, home_base_id: int) -> int:
        """Return the ambulance's original home base index."""
        return home_base_id

    def decide(self, simulator) -> None:
        """Relocate idle ambulances displaced from their home base back home."""
        city = self.city or simulator.city
        for ambulance in simulator.ambulances:
            if ambulance.is_available(simulator.current_time_min):
                home_loc = city.bases[ambulance.home_base_id]
                if not ambulance.location.is_close_to(home_loc):
                    simulator.relocate_ambulance(
                        ambulance.ambulance_id, ambulance.home_base_id
                    )


class NearestBasePolicy:
    """Baseline 3: Nearest-Base standby policy.

    When an ambulance becomes idle away from any candidate base (e.g. at a hospital
    or incident scene), it relocates to the closest candidate base to minimize
    deadhead travel time while restoring network readiness.
    """

    def __init__(self, city: City | None = None) -> None:
        self.city = city

    def choose_base(self, ambulance_location: Location) -> int:
        """Return the index of the closest candidate base."""
        city = self.city or City()
        return city.nearest_base(ambulance_location)

    def decide(self, simulator) -> None:
        """Relocate idle ambulances that are not currently at a base to the nearest base."""
        city = self.city or simulator.city
        for ambulance in simulator.ambulances:
            if ambulance.is_available(simulator.current_time_min):
                at_any_base = any(
                    ambulance.location.is_close_to(base_loc)
                    for base_loc in city.bases
                )
                if not at_any_base:
                    target_base_id = city.nearest_base(ambulance.location)
                    simulator.relocate_ambulance(
                        ambulance.ambulance_id, target_base_id
                    )


class GreedyCoveragePolicy:
    """Baseline 4: Greedy Demand-Coverage heuristic (MEXCLP-inspired).

    Evaluates spatial demand deficits across the city based on time-of-day
    hotspot intensity. If a base with high demand has zero ambulances covering it,
    it relocates an idle unit from an over-covered location (e.g. a hospital or
    a base with multiple idle units) to that base.
    """

    def __init__(
        self,
        city: City | None = None,
        min_deficit_threshold: float = 0.20,
    ) -> None:
        self.city = city
        self.min_deficit_threshold = min_deficit_threshold

    def decide(self, simulator) -> None:
        """Identify high-deficit base and relocate a surplus idle unit if available."""
        city = self.city or simulator.city
        hour = (simulator.current_time_min / 60.0) % 24.0
        weights = hotspot_weights(hour)

        available = [
            unit
            for unit in simulator.ambulances
            if unit.is_available(simulator.current_time_min)
        ]
        if not available:
            return

        # Base occupancy (exact matches)
        base_occupancy = [0] * len(city.bases)
        for unit in available:
            for b_idx, base_loc in enumerate(city.bases):
                if unit.location.is_close_to(base_loc):
                    base_occupancy[b_idx] += 1
                    break

        # Compute deficit for each base
        coverage_radius_km = 3.36  # Reachable within 8 min (36 km/h, 1.25 road factor, 1 min dispatch)
        deficits = []
        for b_idx, base_loc in enumerate(city.bases):
            local_demand = sum(
                w
                for (hx, hy, sp, _), w in zip(
                    simulator.config.hotspots, weights
                )
                if city.manhattan_distance(base_loc, Location(hx, hy))
                <= 2.0 * sp
            )
            local_units = sum(
                1
                for unit in available
                if city.manhattan_distance(unit.location, base_loc)
                <= coverage_radius_km
            )
            # Deficit: expected demand weighted by lack of coverage
            deficit = local_demand / (1.0 + local_units)
            deficits.append(deficit)

        max_deficit = max(deficits)
        best_base_id = deficits.index(max_deficit)

        # Only relocate if deficit exceeds threshold and target base has 0 units
        if (
            max_deficit < self.min_deficit_threshold
            or base_occupancy[best_base_id] > 0
        ):
            return

        # Select donor ambulance: prioritize units stranded at hospitals
        donor = None
        for unit in available:
            is_at_hospital = any(
                unit.location.is_close_to(h) for h in city.hospitals
            )
            if is_at_hospital:
                donor = unit
                break

        # If none at hospital, pick from a base with duplicate units
        if donor is None:
            for unit in available:
                for b_idx, base_loc in enumerate(city.bases):
                    if (
                        unit.location.is_close_to(base_loc)
                        and base_occupancy[b_idx] > 1
                    ):
                        donor = unit
                        break
                if donor is not None:
                    break

        if donor is not None:
            simulator.relocate_ambulance(donor.ambulance_id, best_base_id)

