
"""Train and save the ambulance redeployment agent."""

import json
from pathlib import Path

from ems.training import train_agent


def main() -> None:
    episodes = 30
    agent, rewards = train_agent(episodes=episodes, seed=42)

    output_path = Path("models") / "linear_q_agent.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
    "state_size": agent.state_size,
    "num_actions": agent.num_actions,
    "weights": agent.weights,
    "epsilon": agent.epsilon,
    "episode_rewards": rewards,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved trained agent to {output_path}")
    print(f"First reward: {rewards[0]:.3f}")
    print(f"Last reward:  {rewards[-1]:.3f}")


if __name__ == "__main__":
    main()
