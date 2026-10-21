
"""Represent individual ambulances in the simulation."""

from dataclasses import dataclass
from enum import Enum

from ems.city import Location


class AmbulanceStatus(str, Enum):
    """Possible states of an ambulance."""

    IDLE = "idle"
    RESPONDING = "responding"
    AT_SCENE = "at_scene"
    TO_HOSPITAL = "to_hospital"
    AT_HOSPITAL = "at_hospital"
    REPOSITIONING = "repositioning"


@dataclass
class Ambulance:
    """Track an ambulance's state throughout an episode."""

    ambulance_id: int
    home_base_id: int
    location: Location
    status: AmbulanceStatus = AmbulanceStatus.IDLE
    assigned_call_id: int | None = None
    available_at_min: float = 0.0

    def is_available(self, current_time_min: float) -> bool:
        """Check whether the ambulance can accept a new call."""
        return (
            self.status == AmbulanceStatus.IDLE
            and self.available_at_min <= current_time_min
            and self.assigned_call_id is None
        )

    def assign_call(self, call_id: int) -> None:
        """Mark this ambulance as responding to a call."""
        if self.status != AmbulanceStatus.IDLE:
            raise ValueError("Only an idle ambulance can be assigned")
        if self.assigned_call_id is not None:
            raise ValueError("Ambulance already has an assigned call")

        self.assigned_call_id = call_id
        self.status = AmbulanceStatus.RESPONDING

    def finish_call(self, new_location: Location) -> None:
        """Release the ambulance after completing its current call."""
        if self.assigned_call_id is None:
            raise ValueError("Ambulance has no assigned call")

        self.location = new_location
        self.assigned_call_id = None
        self.status = AmbulanceStatus.IDLE

    def start_repositioning(self) -> None:
        """Mark an idle ambulance as moving to a different base."""
        if self.status != AmbulanceStatus.IDLE:
            raise ValueError("Only an idle ambulance can reposition")
        if self.assigned_call_id is not None:
            raise ValueError("An assigned ambulance cannot reposition")

        self.status = AmbulanceStatus.REPOSITIONING

    def finish_repositioning(self, new_location: Location) -> None:
        """Finish relocation and update the ambulance's location."""
        if self.status != AmbulanceStatus.REPOSITIONING:
            raise ValueError("Ambulance is not repositioning")

        self.location = new_location
        self.status = AmbulanceStatus.IDLE
