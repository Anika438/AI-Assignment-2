"""Unit tests for reward calculation and scaling."""

import unittest

from ems.rewards import RewardConfig, calculate_interval_reward, calculate_redeployment_reward


class TestRewards(unittest.TestCase):
    """Verify reward calculation, scaling, and operational penalties."""

    def setUp(self):
        self.config = RewardConfig(
            response_target_min=8.0,
            response_penalty_weight=1.0,
            dropped_call_penalty=2.0,
            coverage_reward_weight=1.0,
            relocation_cost_weight=0.5,
        )

    def test_ideal_performance_reward(self):
        """Zero drops, zero relocations, high coverage, and 4-min response times."""
        # 4 min response on 8 min target gives avg ratio 0.5 -> response penalty = -0.5
        # coverage 1.0 -> coverage reward = +1.0
        # total = +0.5
        reward = calculate_interval_reward(
            served_response_times=[4.0, 4.0, 4.0],
            dropped_calls=0,
            relocation_minutes=0.0,
            coverage_fraction=1.0,
            config=self.config,
        )
        self.assertAlmostEqual(reward, 0.5, places=5)

    def test_dropped_call_penalty(self):
        """Each dropped call applies a severe penalty of -2.0."""
        r0 = calculate_interval_reward(
            served_response_times=[8.0],
            dropped_calls=0,
            relocation_minutes=0.0,
            coverage_fraction=0.8,
            config=self.config,
        )
        r1 = calculate_interval_reward(
            served_response_times=[8.0],
            dropped_calls=1,
            relocation_minutes=0.0,
            coverage_fraction=0.8,
            config=self.config,
        )
        self.assertAlmostEqual(r0 - r1, 2.0, places=5)

    def test_relocation_travel_cost(self):
        """Relocation travel time of 30 minutes incurs -0.5 penalty."""
        r0 = calculate_interval_reward(
            served_response_times=[],
            dropped_calls=0,
            relocation_minutes=0.0,
            coverage_fraction=0.0,
            config=self.config,
        )
        r30 = calculate_interval_reward(
            served_response_times=[],
            dropped_calls=0,
            relocation_minutes=30.0,
            coverage_fraction=0.0,
            config=self.config,
        )
        self.assertAlmostEqual(r0 - r30, 0.5, places=5)

    def test_reward_bounds(self):
        """Reward remains bounded even under extreme response times."""
        slow_responses = [100.0, 200.0, 300.0]  # clipped to 2.5
        reward = calculate_interval_reward(
            served_response_times=slow_responses,
            dropped_calls=0,
            relocation_minutes=0.0,
            coverage_fraction=0.0,
            config=self.config,
        )
        # response component is clipped at -2.5
        self.assertAlmostEqual(reward, -2.5, places=5)


if __name__ == "__main__":
    unittest.main()

