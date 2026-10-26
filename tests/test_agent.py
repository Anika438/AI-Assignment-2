"""Unit tests for LinearQAgent and numerical stability."""

import math
import unittest

from ems.agent import LinearQAgent


class TestLinearQAgent(unittest.TestCase):
    """Verify linear Q-learning updates, stability, and action selection."""

    def setUp(self):
        self.state_size = 85
        self.num_actions = 13
        self.agent = LinearQAgent(
            state_size=self.state_size,
            num_actions=self.num_actions,
            learning_rate=0.01,
            discount_factor=0.90,
            seed=42,
        )

    def test_state_size_validation(self):
        """Passing mismatched state size must raise ValueError."""
        with self.assertRaises(ValueError):
            self.agent.q_values([0.0] * 10)

    def test_non_finite_state_rejected(self):
        """Non-finite state features must raise ValueError."""
        bad_state = [0.0] * self.state_size
        bad_state[0] = float("nan")
        with self.assertRaises(ValueError):
            self.agent.q_values(bad_state)

    def test_td_error_clipping_and_weight_bounds(self):
        """Extreme rewards or TD errors must be clipped without exploding weights."""
        state = [0.5] * self.state_size
        next_state = [0.5] * self.state_size

        # Pass a huge reward
        self.agent.update(
            state=state,
            action=0,
            reward=1000.0,
            next_state=next_state,
            done=False,
        )

        for w in self.agent.weights[0]:
            self.assertTrue(math.isfinite(w))
            self.assertLessEqual(abs(w), self.agent.max_weight)

    def test_action_masking(self):
        """Agent must only select from valid_actions when provided."""
        state = [0.1] * self.state_size
        # Force greedy selection
        self.agent.epsilon = 0.0

        # Only allow actions 0 and 5
        for _ in range(20):
            action = self.agent.choose_action(
                state, explore=False, valid_actions=[0, 5]
            )
            self.assertIn(action, [0, 5])

    def test_stay_preference_on_tie(self):
        """When multiple actions are tied at 0 (e.g. initial weights), prefer STAY (0)."""
        state = [0.0] * self.state_size
        self.agent.epsilon = 0.0
        action = self.agent.choose_action(state, explore=False)
        self.assertEqual(action, 0)


if __name__ == "__main__":
    unittest.main()
