
"""Reward functions for ambulance redeployment decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights controlling the reward signal."""

    response_target_min: float = 8.0
    response_penalty_weight: float = 1.0
    relocation_penalty_per_min: float = 0.05


def calculate_redeployment_reward(
    response_time_min: float,
    relocation_time_min: float,
    config: RewardConfig | None = None,
) -> float:
    """Score a decision using response time and relocation cost.

    Lower response times are better. Relocation is penalized to
    discourage unnecessary movement between standby bases.
    """
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
        settings.relocation_penalty_per_min * relocation_time_min
    )

    return -(response_penalty + relocation_penalty)
