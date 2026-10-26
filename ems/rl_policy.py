"""Connect the RL agent to simulator redeployment decisions."""

from ems.agent import LinearQAgent
from ems.city import Location
from ems.simulator import AmbulanceSimulator
from ems.state import StateBuilder


class RLPolicy:
    """Select relocation actions using learned Q-values with a first-class STAY option.

    Applies a relocation hurdle and base occupancy constraints so that moves are only
    executed when expected benefit exceeds staying plus the transit hurdle, to candidate
    bases that currently lack coverage.
    """

    def __init__(
        self,
        simulator: AmbulanceSimulator,
        agent: LinearQAgent,
        explore: bool = False,
        relocation_hurdle: float = 0.05,
    ) -> None:
        self.simulator = simulator
        self.agent = agent
        self.explore = explore
        self.relocation_hurdle = relocation_hurdle
        self.state_builder = StateBuilder(simulator.config)

    def state(
        self, candidate_location: Location | None = None
    ) -> list[float]:
        """Build a normalized state vector from current simulator conditions."""
        return self.state_builder.build_state(
            current_time_min=self.simulator.current_time_min,
            ambulance_locations=[
                unit.location for unit in self.simulator.ambulances
            ],
            available_flags=[
                unit.is_available(self.simulator.current_time_min)
                for unit in self.simulator.ambulances
            ],
            candidate_location=candidate_location,
        )

    def select_candidate_ambulance(
        self,
        simulator: AmbulanceSimulator,
    ):
        """Identify the best idle candidate ambulance for potential relocation.

        Prioritizes units stranded at hospitals or bases with duplicate occupancy.
        """
        available = [
            unit
            for unit in simulator.ambulances
            if unit.is_available(simulator.current_time_min)
        ]
        if not available:
            return None

        # 1. Units stranded at hospitals (top priority to redeploy back to network)
        for unit in available:
            if any(unit.location.is_close_to(h) for h in simulator.city.hospitals):
                return unit

        # 2. Units at candidate bases with duplicate coverage (> 1 idle unit)
        for base in simulator.city.bases:
            units_at_base = [
                u for u in available if u.location.is_close_to(base)
            ]
            if len(units_at_base) > 1:
                return units_at_base[0]

        # 3. Otherwise first available unit
        return available[0]

    def decide(
        self, simulator: AmbulanceSimulator
    ) -> tuple[list[float], int]:
        """Choose an action (0=STAY, 1..12=relocate to Base 0..11) and execute if valid.

        Returns (state, executed_action) where executed_action reflects the action
        actually applied to the simulator environment.
        """
        donor = self.select_candidate_ambulance(simulator)
        donor_loc = donor.location if donor is not None else None
        state = self.state(candidate_location=donor_loc)

        # If no unit is available to move, action is strictly STAY (0)
        if donor is None:
            return state, 0

        # Calculate exact base occupancy among idle units
        base_occupancy = [0] * len(simulator.city.bases)
        for u in simulator.ambulances:
            if u.is_available(simulator.current_time_min):
                for b_idx, b_loc in enumerate(simulator.city.bases):
                    if u.location.is_close_to(b_loc):
                        base_occupancy[b_idx] += 1
                        break

        # Action 0 is STAY. Action 1..12 correspond to Bases 0..11.
        # Only allow relocations to empty bases that do not already have the donor unit
        valid_actions = [0]
        for b_idx, base_loc in enumerate(simulator.city.bases):
            if base_occupancy[b_idx] == 0 and not donor.location.is_close_to(base_loc):
                valid_actions.append(b_idx + 1)

        chosen_action = self.agent.choose_action(
            state, explore=self.explore, valid_actions=valid_actions
        )

        # Relocation hurdle: only initiate relocation during exploitation if expected utility
        # exceeds stay value by at least the relocation hurdle
        if not self.explore and chosen_action > 0:
            q_values = self.agent.q_values(state)
            stay_value = q_values[0]
            move_value = q_values[chosen_action]
            if move_value < stay_value + self.relocation_hurdle:
                chosen_action = 0

        executed_action = 0
        if chosen_action > 0:
            target_base_id = chosen_action - 1
            success = simulator.relocate_ambulance(
                donor.ambulance_id, target_base_id
            )
            if success:
                executed_action = chosen_action
            else:
                executed_action = 0

        return state, executed_action

        # Relocation hurdle: only initiate relocation during exploitation if expected utility
        # exceeds stay value by at least the relocation hurdle
        if not self.explore and chosen_action > 0:
            q_values = self.agent.q_values(state)
            stay_value = q_values[0]
            move_value = q_values[chosen_action]
            if move_value < stay_value + self.relocation_hurdle:
                chosen_action = 0

        executed_action = 0
        if chosen_action > 0:
            target_base_id = chosen_action - 1
            success = simulator.relocate_ambulance(
                donor.ambulance_id, target_base_id
            )
            if success:
                executed_action = chosen_action
            else:
                executed_action = 0

        return state, executed_action

