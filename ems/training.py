"""Train Q-learning from actual simulator decision transitions."""

from ems.agent import LinearQAgent
from ems.config import CityConfig
from ems.metrics import calculate_metrics
from ems.rewards import RewardConfig, calculate_interval_reward
from ems.rl_policy import RLPolicy
from ems.simulator import AmbulanceSimulator


class SimulatorTrainer:
    """Collect transitions and update the agent during each simulated day."""

    def __init__(
        self,
        simulator: AmbulanceSimulator,
        agent: LinearQAgent,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.agent = agent
        self.reward_config = reward_config or RewardConfig()
        self.policy = RLPolicy(simulator, agent, explore=True)
        self.previous: tuple[list[float], int] | None = None
        self.outcome_count = 0
        self.dropped_count = 0
        self.relocation_minutes = 0.0
        self.episode_reward = 0.0

    def interval_reward(self) -> float:
        """Compute the bounded, grounded reward for the elapsed interval."""
        sim = self.simulator
        new_outcomes = sim.outcomes[self.outcome_count:]
        new_drops = sim.dropped_calls - self.dropped_count
        new_relocation = sim.relocation_minutes - self.relocation_minutes

        hour = (sim.current_time_min / 60.0) % 24.0
        avail_locations = [
            unit.location
            for unit in sim.ambulances
            if unit.is_available(sim.current_time_min)
        ]
        coverage = sim.city.demand_coverage_fraction(
            avail_locations, hour, sim.config.response_target_min
        )

        response_times = [
            outcome.response_time_min for outcome in new_outcomes
        ]
        reward = calculate_interval_reward(
            served_response_times=response_times,
            dropped_calls=new_drops,
            relocation_minutes=new_relocation,
            coverage_fraction=coverage,
            config=self.reward_config,
        )

        self.outcome_count = len(sim.outcomes)
        self.dropped_count = sim.dropped_calls
        self.relocation_minutes = sim.relocation_minutes
        self.episode_reward += reward
        return reward

    def decide(self, simulator: AmbulanceSimulator) -> None:
        """Process transition from previous interval, then select and execute action."""
        donor = self.policy.select_candidate_ambulance(simulator)
        donor_loc = donor.location if donor is not None else None
        current_state = self.policy.state(candidate_location=donor_loc)

        if self.previous is not None:
            old_state, old_action = self.previous
            reward = self.interval_reward()
            self.agent.update(
                state=old_state,
                action=old_action,
                reward=reward,
                next_state=current_state,
                done=False,
            )
            self.previous = None
        else:
            self.outcome_count = len(simulator.outcomes)
            self.dropped_count = simulator.dropped_calls
            self.relocation_minutes = simulator.relocation_minutes

        # Only initiate a new transition when an eligible candidate unit is available
        if donor is not None:
            state, executed_action = self.policy.decide(simulator)
            self.previous = (state, executed_action)

    def finish_episode(self) -> None:
        """Update terminal transition at episode boundary."""
        if self.previous is None:
            return

        state, action = self.previous
        reward = self.interval_reward()
        terminal_state = self.policy.state()

        self.agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=terminal_state,
            done=True,
        )
        self.previous = None


def train_agent(
    episodes: int = 100,
    seed: int = 1000,
    learning_rate: float = 0.005,
    discount_factor: float = 0.95,
    verbose: bool = True,
) -> tuple[LinearQAgent, list[float]]:
    """Train and return the agent and per-episode accumulated rewards.

    Note: seed specifies the starting training seed (e.g. 1000..1000+episodes).
    Evaluation should be run on disjoint held-out seeds (e.g. 42..61).
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    config = CityConfig()
    from ems.state import StateBuilder
    state_builder = StateBuilder(config)

    # 13 actions: 0 = STAY, 1..12 = Candidate bases 0..11
    num_actions = len(config.bases) + 1

    agent = LinearQAgent(
        state_size=state_builder.state_size,
        num_actions=num_actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=0.40,
        epsilon_min=0.02,
        epsilon_decay=0.93,
        seed=seed,
    )

    rewards: list[float] = []

    for episode in range(episodes):
        sim = AmbulanceSimulator(config=config, seed=seed + episode)
        trainer = SimulatorTrainer(sim, agent)

        sim.run_day(decision_callback=trainer.decide)
        trainer.finish_episode()

        rewards.append(trainer.episode_reward)
        agent.decay_epsilon()

        metrics = calculate_metrics(
            sim.outcomes,
            target_min=config.response_target_min,
        )

        if verbose and (
            episode < 5
            or (episode + 1) % 5 == 0
            or episode == episodes - 1
        ):
            print(
                f"Episode {episode + 1:2d}/{episodes} | "
                f"eps: {agent.epsilon:.3f} | "
                f"mean resp: {metrics.mean_response_time_min:.2f} min | "
                f"p90: {metrics.percentile_90_response_time_min:.2f} min | "
                f"drops: {sim.dropped_calls} | "
                f"relocs: {sim.relocation_count} | "
                f"reward: {trainer.episode_reward:.2f}"
            )

    return agent, rewards
