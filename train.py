
"""Entry point for the ambulance RL project."""

from ems.agent import LinearQAgent
from ems.city import City
from ems.config import CityConfig
from ems.metrics import calculate_metrics
from ems.simulator import AmbulanceSimulator
from ems.state import StateBuilder


def main() -> None:
    """Run a basic integration check of the project components."""
    config = CityConfig()
    city = City(config)
    state_builder = StateBuilder(config)

    simulator = AmbulanceSimulator(config=config, seed=42)
    outcomes = simulator.run_day()
    metrics = calculate_metrics(
        outcomes,
        target_min=config.response_target_min,
    )

    ambulance_locations = [
        ambulance.location for ambulance in simulator.ambulances
    ]
    available_flags = [
        ambulance.is_available(simulator.current_time_min)
        for ambulance in simulator.ambulances
    ]

    state = state_builder.build_state(
        current_time_min=simulator.current_time_min,
        ambulance_locations=ambulance_locations,
        available_flags=available_flags,
    )

    agent = LinearQAgent(
        state_size=len(state),
        num_actions=len(city.bases),
        seed=42,
    )
    action = agent.choose_action(state, explore=False)

    print("Ambulance redeployment project: integration check")
    print(f"Demand zones: {city.num_zones}")
    print(f"Candidate bases: {len(city.bases)}")
    print(f"Ambulances: {len(simulator.ambulances)}")
    print(f"State features: {len(state)}")
    print(f"Agent action (base index): {action}")
    print(f"Calls dispatched: {metrics.calls_dispatched}")
    print(f"Mean response time: {metrics.mean_response_time_min:.2f} min")
    print(
        "Fraction reached within target: "
        f"{metrics.target_response_fraction:.1%}"
    )


if __name__ == "__main__":
    main()
