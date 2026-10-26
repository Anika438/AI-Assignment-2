"""Unit tests for RLPolicy and StateBuilder."""

import unittest

from ems.agent import LinearQAgent
from ems.config import CityConfig
from ems.rl_policy import RLPolicy
from ems.simulator import AmbulanceSimulator
from ems.state import StateBuilder


class TestRLPolicy(unittest.TestCase):
    """Verify RL policy decision making, state extraction, and executed action tracking."""

    def setUp(self):
        self.config = CityConfig()
        self.builder = StateBuilder(self.config)
        self.agent = LinearQAgent(
            state_size=self.builder.state_size,
            num_actions=len(self.config.bases) + 1,
            seed=42,
        )

    def test_state_builder_dimension_and_bounds(self):
        """StateBuilder must produce exactly state_size features all within [-1, 1] or [0, 1]."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        locations = [u.location for u in sim.ambulances]
        avail_flags = [u.is_available(0.0) for u in sim.ambulances]

        state = self.builder.build_state(0.0, locations, avail_flags)
        self.assertEqual(len(state), self.builder.state_size)
        self.assertEqual(self.builder.state_size, 85)

        for feat in state:
            self.assertGreaterEqual(feat, -1.0)
            self.assertLessEqual(feat, 1.0)

    def test_stay_action_does_not_relocate(self):
        """When agent chooses STAY (0), no relocation occurs."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        policy = RLPolicy(sim, self.agent, explore=False)
        # Force agent weights to 0 so it chooses STAY (0)
        state, action = policy.decide(sim)
        self.assertEqual(action, 0)
        self.assertEqual(sim.relocation_count, 0)

    def test_executed_action_reflects_reality(self):
        """The executed action returned by policy.decide must match what happened in simulator."""
        sim = AmbulanceSimulator(config=self.config, seed=42)
        policy = RLPolicy(sim, self.agent, explore=False)

        state, executed_action = policy.decide(sim)
        if executed_action == 0:
            self.assertEqual(sim.relocation_count, 0)
        else:
            self.assertEqual(sim.relocation_count, 1)


if __name__ == "__main__":
    unittest.main()
