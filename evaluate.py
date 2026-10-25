
"""Evaluate the current ambulance simulator and report its limitations."""

from ems.config import CityConfig
from ems.metrics import calculate_metrics
from ems.simulator import AmbulanceSimulator


def evaluate_once(seed: int) -> dict[str, float | int]:
    """Run one simulation day and return response-time metrics."""
    config = CityConfig()
    simulator = AmbulanceSimulator(config=config, seed=seed)
    outcomes = simulator.run_day()
    metrics = calculate_metrics(
        outcomes,
        target_min=config.response_target_min,
    )

    return {
        "seed": seed,
        "calls_generated": len(simulator.pending_calls),
        "calls_dispatched": metrics.calls_dispatched,
        "calls_dropped": simulator.dropped_calls,
        "mean_response_min": metrics.mean_response_time_min,
        "median_response_min": metrics.median_response_time_min,
        "p90_response_min": metrics.percentile_90_response_time_min,
        "within_target_fraction": metrics.target_response_fraction,
    }


def main() -> None:
    """Run repeatable simulator checks across several random seeds."""
    seeds = [42, 43, 44, 45, 46]
    results = [evaluate_once(seed) for seed in seeds]

    print("Ambulance simulator evaluation")
    print("-" * 72)
    print(
        f"{'Seed':>6} {'Generated':>10} {'Dispatched':>11} "
        f"{'Dropped':>8} {'Mean (min)':>11} {'P90 (min)':>10} "
        f"{'Within 8m':>10}"
    )

    for result in results:
        print(
            f"{result['seed']:>6} "
            f"{result['calls_generated']:>10} "
            f"{result['calls_dispatched']:>11} "
            f"{result['calls_dropped']:>8} "
            f"{result['mean_response_min']:>11.2f} "
            f"{result['p90_response_min']:>10.2f} "
            f"{result['within_target_fraction']:>9.1%}"
        )

    print()
    print("RL policy: NOT EVALUATED")
    print(
        "The current training loop uses a proxy reward and is not "
        "connected to simulator decisions."
    )
    print(
        "Also note: the current simulator simplifies incident service "
        "and drops calls when all ambulances are busy."
    )


if __name__ == "__main__":
    main()
