
"""Connect the RL agent to simulator redeployment decisions."""

from ems.agent import LinearQAgent
from ems.simulator import AmbulanceSimulator
from ems.state import StateBuilder


class RLPolicy:
    """Select relocation actions using the learned Q-values."""

    def __init__(
        self,
        simulator: AmbulanceSimulator,
        agent: LinearQAgent,
        explore: bool = False,
    ) -> None:
        self.simulator = simulator
        self.agent = agent
        self.explore = explore
        self.state_builder = StateBuilder(simulator.config)

    def state(self) -> list[float]:
        """Build a state from the current simulator conditions."""
        return self.state_builder.build_state(
            current_time_min=self.simulator.current_time_min,
            ambulance_locations=[
                unit.location for unit in self.simulator.ambulances
            ],
            available_flags=[
                unit.is_available(self.simulator.current_time_min)
                for unit in self.simulator.ambulances
            ],
        )

    def decide(self, simulator: AmbulanceSimulator) -> tuple[list[float], int] | None:
        """Choose a base for one idle ambulance and attempt relocation."""
        available = [
            unit for unit in simulator.ambulances
            if unit.is_available(simulator.current_time_min)
        ]
        if not available:
            return None

        state = self.state()
        ambulance = available[0]
        action = self.agent.choose_action(state, explore=self.explore)

        simulator.relocate_ambulance(
            ambulance.ambulance_id,
            action,
        )
        return state, action
