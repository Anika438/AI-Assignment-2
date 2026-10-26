"""Train and save the ambulance redeployment reinforcement learning agent."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ems.training import train_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Linear Q-Learning agent for ambulance redeployment."
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=1000,
        help="Starting seed for training episodes (disjoint from test seeds)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="Discount factor"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/linear_q_agent.json",
        help="Path to save trained agent JSON",
    )
    args = parser.parse_args()

    print(
        f"Starting RL training for {args.episodes} episodes (training seeds {args.train_seed}..{args.train_seed + args.episodes - 1})..."
    )
    agent, rewards = train_agent(
        episodes=args.episodes,
        seed=args.train_seed,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        verbose=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_seed_start": args.train_seed,
        "train_seed_end": args.train_seed + args.episodes - 1,
        "episodes": args.episodes,
        "learning_rate": args.lr,
        "discount_factor": args.gamma,
        "state_size": agent.state_size,
        "num_actions": agent.num_actions,
        "action_meanings": {
            0: "STAY (no relocation)",
            **{i: f"Relocate to candidate Base {i - 1}" for i in range(1, agent.num_actions)},
        },
        "epsilon_final": agent.epsilon,
        "weights": agent.weights,
        "episode_rewards": rewards,
    }

    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSuccessfully saved trained agent and metadata to: {output_path}")
    print(f"First episode reward: {rewards[0]:.2f}")
    print(f"Final episode reward: {rewards[-1]:.2f}")
    print("\nTo evaluate this model against baselines on held-out seeds, run:")
    print("  py evaluate.py")


if __name__ == "__main__":
    main()
