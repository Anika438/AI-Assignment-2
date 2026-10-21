
"""Performance metrics for ambulance simulation outcomes."""

from dataclasses import dataclass

from ems.simulator import CallOutcome


@dataclass(frozen=True)
class SimulationMetrics:
    """Summary statistics for a simulation run."""

    calls_dispatched: int
    mean_response_time_min: float
    median_response_time_min: float
    percentile_90_response_time_min: float
    target_response_fraction: float


def calculate_metrics(
    outcomes: list[CallOutcome],
    target_min: float = 8.0,
) -> SimulationMetrics:
    """Calculate response-time statistics for dispatched calls."""
    if target_min <= 0:
        raise ValueError("target_min must be positive")

    if not outcomes:
        return SimulationMetrics(
            calls_dispatched=0,
            mean_response_time_min=0.0,
            median_response_time_min=0.0,
            percentile_90_response_time_min=0.0,
            target_response_fraction=0.0,
        )

    response_times = sorted(
        outcome.response_time_min for outcome in outcomes
    )

    count = len(response_times)
    mean_time = sum(response_times) / count

    middle = count // 2
    if count % 2 == 0:
        median_time = (
            response_times[middle - 1] + response_times[middle]
        ) / 2.0
    else:
        median_time = response_times[middle]

    # Nearest-rank percentile: rank = ceil(0.90 * count).
    rank = max(1, int(0.90 * count + 0.999999))
    percentile_90 = response_times[rank - 1]

    within_target = sum(
        time <= target_min for time in response_times
    )

    return SimulationMetrics(
        calls_dispatched=count,
        mean_response_time_min=mean_time,
        median_response_time_min=median_time,
        percentile_90_response_time_min=percentile_90,
        target_response_fraction=within_target / count,
    )
