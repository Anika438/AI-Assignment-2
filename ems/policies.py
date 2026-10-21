
"""Simple baseline policies for ambulance redeployment."""

from ems.city import City, Location


class NearestBasePolicy:
    """Choose the candidate base nearest to an ambulance."""

    def __init__(self, city: City) -> None:
        self.city = city

    def choose_base(self, ambulance_location: Location) -> int:
        """Return the index of the closest candidate base."""
        return self.city.nearest_base(ambulance_location)


class FixedHomeBasePolicy:
    """Always recommend the ambulance's original home base."""

    @staticmethod
    def choose_base(home_base_id: int) -> int:
        """Return the ambulance's original base index."""
        return home_base_id
