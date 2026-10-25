
"""Training utilities for the linear Q-learning agent."""

import random

from ems.agent import LinearQAgent
from ems.city import City, Location
from ems.config import CityConfig
from ems.rewards import RewardConfig, calculate_redeployment_reward
from ems.state import StateBuilder


def train_agent(
    episodes: int = 200,
    seed: int = 42,
) -> tuple[LinearQAgent, list[float]]:
    """Train an agent on simplified ambulance relocation decisions."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    config = CityConfig()
    city = City(config)
    state_builder = StateBuilder(config)
    rng = random.Random(seed)

    agent = LinearQAgent(
        state_size=33,
        num_actions=len(city.bases),
        learning_rate=0.005,
        discount_factor=0.90,
        epsilon=0.25,
        seed=seed,
    )

    reward_config = RewardConfig()
    episode_rewards: list[float] = []

    for episode in range(episodes):
        hour = rng.uniform(0.0, 24.0)
        current_time_min = hour * 60.0

        ambulance_locations = [
            city.bases[base_id] for base_id in config.home_bases
        ]
        available_flags = [True] * config.num_ambulances

        state = state_builder.build_state(
            current_time_min,
            ambulance_locations,
            available_flags,
        )

        ambulance_id = rng.randrange(config.num_ambulances)
        ambulance_location = ambulance_locations[ambulance_id]
        action = agent.choose_action(state)

        destination = city.bases[action]
        relocation_time = city.travel_time_minutes(
            ambulance_location,
            destination,
            config.reposition_speed_kmh,
        )

        # Approximate coverage reward: favour bases near active hotspots.
        weights = __import__(
            "ems.config", fromlist=["hotspot_weights"]
        ).hotspot_weights(hour)

        coverage = sum(
            weight
            for (x, y, spread, _name), weight in zip(
                config.hotspots, weights
            )
            if city.manhattan_distance(destination, Location(x, y))
            <= 2.0 * spread
        )

        # The score is a proxy, not an observed emergency response time.
        estimated_response = max(
            1.0,
            15.0 - 2.0 * coverage,
        )

        reward = calculate_redeployment_reward(
            response_time_min=estimated_response,
            relocation_time_min=relocation_time,
            config=reward_config,
        )

        next_locations = list(ambulance_locations)
        next_locations[ambulance_id] = destination

        next_state = state_builder.build_state(
            current_time_min=(current_time_min + 30.0) % 1440.0,
            ambulance_locations=next_locations,
            available_flags=available_flags,
        )

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=True,
        )

        episode_rewards.append(reward)

        # Reduce exploration gradually while preserving some randomness.
        agent.epsilon = max(0.05, 0.25 * (1.0 - episode / episodes))

    return agent, episode_rewards
