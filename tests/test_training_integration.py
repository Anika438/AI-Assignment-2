"""Unit tests for training and evaluation pipeline integration."""

import json
from pathlib import Path
import tempfile
import unittest

from ems.agent import LinearQAgent
from ems.config import CityConfig
from ems.metrics import calculate_metrics
from ems.rl_policy import RLPolicy
from ems.simulator import AmbulanceSimulator
from ems.training import train_agent


class TestTrainingIntegration(unittest.TestCase):
    """Verify train_agent, model serialization, and evaluation execution."""

    def test_train_agent_smoke_and_serialization(self):
        """Train agent for a brief 2-episode run, verify weights, and round-trip JSON serialization."""
        episodes = 2
        agent, rewards = train_agent(
            episodes=episodes,
            seed=5000,
            learning_rate=0.01,
            discount_factor=0.95,
            verbose=False,
        )

        self.assertEqual(len(rewards), episodes)
        self.assertEqual(agent.state_size, 85)
        self.assertEqual(agent.num_actions, 13)
        self.assertLess(agent.epsilon, 0.40)  # Epsilon decayed

        # Test serialization and loading
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "test_model.json"
            payload = {
                "state_size": agent.state_size,
                "num_actions": agent.num_actions,
                "weights": agent.weights,
                "epsilon": agent.epsilon,
                "episode_rewards": rewards,
            }
            model_file.write_text(json.dumps(payload), encoding="utf-8")

            # Load into a fresh agent
            loaded_data = json.loads(model_file.read_text(encoding="utf-8"))
            loaded_agent = LinearQAgent(
                state_size=loaded_data["state_size"],
                num_actions=loaded_data["num_actions"],
            )
            loaded_agent.weights = loaded_data["weights"]
            loaded_agent.epsilon = 0.0

            # Verify identical Q-values
            test_state = [0.1] * 85
            q_orig = agent.q_values(test_state)
            q_loaded = loaded_agent.q_values(test_state)
            for qo, ql in zip(q_orig, q_loaded):
                self.assertAlmostEqual(qo, ql, places=6)

            # Verify evaluation with loaded agent in simulator
            config = CityConfig()
            sim = AmbulanceSimulator(config=config, seed=42)
            policy = RLPolicy(sim, loaded_agent, explore=False)
            outcomes = sim.run_day(decision_callback=policy.decide)

            metrics = calculate_metrics(outcomes, target_min=config.response_target_min)
            self.assertGreater(metrics.calls_dispatched, 0)
            self.assertGreater(metrics.mean_response_time_min, 0.0)


if __name__ == "__main__":
    unittest.main()

