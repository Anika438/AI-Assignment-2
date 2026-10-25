
"""Compare the learned policy with fixed-home and nearest-base baselines."""

import json
from pathlib import Path

from ems.agent import LinearQAgent
from ems.config import CityConfig
from ems.metrics import calculate_metrics
from ems.policies import FixedHomeBasePolicy, NearestBasePolicy
from ems.rl_policy import RLPolicy
from ems.simulator import AmbulanceSimulator


SEEDS = [42, 43, 44, 45, 46]


def load_agent(path: str, seed: int = 42) -> LinearQAgent:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    agent = LinearQAgent(
        state_size=payload["state_size"],
        num_actions=payload["num_actions"],
        seed=seed,
    )
    agent.weights = payload["weights"]
    agent.epsilon = 0.0
    return agent


def evaluate_policy(name: str, policy_name: str, agent=None) -> None:
    config = CityConfig()
    rows = []

    for seed in SEEDS:
        simulator = AmbulanceSimulator(config=config, seed=seed)

        if policy_name == "rl":
            policy = RLPolicy(simulator, agent, explore=False)
            simulator.run_day(decision_callback=policy.decide)
        else:
            # These baseline policies are currently simple policy objects;
            # they do not automatically relocate units during the simulation.
            simulator.run_day()

        metrics = calculate_metrics(
            simulator.outcomes,
            target_min=config.response_target_min,
        )
        rows.append(
            (
                len(simulator.pending_calls),
                len(simulator.outcomes),
                simulator.dropped_calls,
                metrics.mean_response_time_min,
                metrics.percentile_90_response_time_min,
                100 * metrics.target_response_fraction,
            )
        )

    print(f"\n{name}")
    print("Seed  Generated  Dispatched  Dropped  Mean(min)  P90(min)  Within target")
    for seed, row in zip(SEEDS, rows):
        generated, dispatched, dropped, mean, p90, within = row
        print(
            f"{seed:>4}  {generated:>9}  {dispatched:>10}  {dropped:>7}  "
            f"{mean:>9.2f}  {p90:>8.2f}  {within:>11.1f}%"
        )

    print(
        "Average mean response: "
        f"{sum(row[3] for row in rows) / len(rows):.2f} min"
    )


def main() -> None:
    model_path = Path("models/linear_q_agent.json")
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `py train.py` first."
        )

    agent = load_agent(str(model_path))
    evaluate_policy("RL policy", "rl", agent)


if __name__ == "__main__":
    main()
