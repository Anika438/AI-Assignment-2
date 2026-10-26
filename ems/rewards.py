"""Reward functions for ambulance redeployment decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights controlling the reward signal.

    Components and trade-offs:
    - response_target_min (8.0 min): Benchmark standard response time.
    - response_penalty_weight (1.0): Normalizes response time relative to target.
      Meeting target yields -1.0; 4 min yields -0.5; slow responses are clipped at -2.5.
    - dropped_call_penalty (2.0): Penalty for unserved calls to heavily prioritize fleet availability.
    - coverage_reward_weight (1.0): Positive reward in [0, 1] for maintaining spatial demand coverage.
    - relocation_cost_weight (0.5): Penalty per 30 minutes of ambulance repositioning time.
      A 15-minute relocation costs -0.25, ensuring units only move if expected coverage
      and response time benefits exceed the transit cost.
    """

    response_target_min: float = 8.0
    response_penalty_weight: float = 1.0
    dropped_call_penalty: float = 2.0
    coverage_reward_weight: float = 1.0
    relocation_cost_weight: float = 0.5


def calculate_interval_reward(
    served_response_times: list[float],
    dropped_calls: int,
    relocation_minutes: float,
    coverage_fraction: float,
    config: RewardConfig | None = None,
) -> float:
    """Compute bounded, balanced reward for a simulation decision interval.

    Returns a bounded scalar typically in [-3.0, +1.0] balancing service quality,
    coverage, and travel costs.
    """
    settings = config or RewardConfig()

    if settings.response_target_min <= 0:
        raise ValueError("response_target_min must be positive")
    if dropped_calls < 0 or relocation_minutes < 0:
        raise ValueError("dropped_calls and relocation_minutes cannot be negative")

    # 1. Response time component (per served call average, clipped to [0, 2.5])
    if served_response_times:
        avg_resp_ratio = sum(
            min(max(0.0, t) / settings.response_target_min, 2.5)
            for t in served_response_times
        ) / len(served_response_times)
        response_component = -settings.response_penalty_weight * avg_resp_ratio
    else:
        response_component = 0.0

    # 2. Dropped call component
    drop_component = -settings.dropped_call_penalty * float(dropped_calls)

    # 3. Spatial coverage component: in [0, 1]
    coverage_component = settings.coverage_reward_weight * max(
        0.0, min(1.0, coverage_fraction)
    )

    # 4. Relocation travel cost: normalized by nominal 30 min interval
    reloc_component = -settings.relocation_cost_weight * (
        relocation_minutes / 30.0
    )

    total_reward = (
        response_component
        + drop_component
        + coverage_component
        + reloc_component
    )
    return total_reward


def calculate_redeployment_reward(
    response_time_min: float,
    relocation_time_min: float,
    config: RewardConfig | None = None,
) -> float:
    """Legacy helper: Score a single decision using response time and relocation cost."""
    settings = config or RewardConfig()

    if response_time_min < 0 or relocation_time_min < 0:
        raise ValueError("Times cannot be negative")
    if settings.response_target_min <= 0:
        raise ValueError("Response target must be positive")

    response_penalty = (
        settings.response_penalty_weight
        * response_time_min
        / settings.response_target_min
    )
    relocation_penalty = (
        settings.relocation_cost_weight * relocation_time_min / 30.0
    )

    return -(response_penalty + relocation_penalty)
