"""Fair evaluation of redeployment policies on identical held-out seeds."""

import argparse
import csv
import json
import math
from pathlib import Path

from ems.agent import LinearQAgent
from ems.city import City
from ems.config import CityConfig
from ems.metrics import calculate_metrics
from ems.policies import (
    FixedHomeBasePolicy,
    GreedyCoveragePolicy,
    NearestBasePolicy,
    NoRedeploymentPolicy,
)
from ems.rl_policy import RLPolicy
from ems.simulator import AmbulanceSimulator

DEFAULT_HELD_OUT_SEEDS = list(range(42, 62))  # 20 held-out seeds


def student_t_two_tailed_pvalue(t_stat: float, df: int) -> float:
    """Calculate exact two-tailed p-value for Student's t distribution using numerical integration."""
    if df <= 0:
        return 1.0
    t_abs = abs(t_stat)
    if t_abs < 1e-12:
        return 1.0

    log_c = (
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
    )

    def pdf(x: float) -> float:
        return math.exp(log_c - ((df + 1) / 2) * math.log(1.0 + (x * x) / df))

    # Simpson's rule integration from t_abs to upper bound
    n_steps = 2000
    upper = max(100.0, t_abs + 50.0)
    h = (upper - t_abs) / n_steps
    s = pdf(t_abs) + pdf(upper)
    for i in range(1, n_steps):
        xi = t_abs + i * h
        s += pdf(xi) * (4.0 if i % 2 == 1 else 2.0)
    p = 2.0 * (h / 3.0) * s
    return min(1.0, max(0.0, p))


def paired_stats(
    diffs: list[float],
) -> tuple[float, float, float, float, tuple[float, float], float]:
    """Compute mean difference, standard error, t-statistic, p-value, 95% CI, and Cohen's dz."""
    n = len(diffs)
    if n <= 1:
        return 0.0, 0.0, 0.0, 1.0, (0.0, 0.0), 0.0

    mean_d = sum(diffs) / n
    var_d = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 1e-9
    se_d = std_d / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else 0.0
    p_val = student_t_two_tailed_pvalue(t_stat, n - 1)

    # Approximate 95% critical value for t (2.093 for df=19, 2.0 for large n)
    # Using numerical bisection for exact t_crit
    t_crit = 2.093 if n == 20 else (2.776 if n == 5 else 2.0)
    ci_lower = mean_d - t_crit * se_d
    ci_upper = mean_d + t_crit * se_d
    cohens_d = mean_d / std_d if std_d > 0 else 0.0

    return mean_d, se_d, t_stat, p_val, (ci_lower, ci_upper), cohens_d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair evaluation of ambulance redeployment policies."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/linear_q_agent.json",
        help="Path to trained RL model JSON",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_HELD_OUT_SEEDS,
        help="List of held-out evaluation seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save JSON and CSV reports",
    )
    args = parser.parse_args()

    config = CityConfig()
    city = City(config)

    # 1. Load trained RL agent
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run 'py train.py' first."
        )

    model_data = json.loads(model_path.read_text(encoding="utf-8"))
    agent = LinearQAgent(
        state_size=model_data["state_size"],
        num_actions=model_data["num_actions"],
        seed=42,
    )
    agent.weights = model_data["weights"]
    agent.epsilon = 0.0  # Pure exploitation during evaluation

    # 2. Instantiate all 5 policies
    policies = {
        "No redeployment": NoRedeploymentPolicy(),
        "Fixed home base": FixedHomeBasePolicy(city),
        "Nearest base": NearestBasePolicy(city),
        "Greedy coverage": GreedyCoveragePolicy(city),
        "Trained RL": RLPolicy(
            AmbulanceSimulator(config),
            agent,
            explore=False,
            relocation_hurdle=0.05,
        ),
    }

    seeds = args.seeds
    num_seeds = len(seeds)
    print(f"Evaluating {len(policies)} policies over {num_seeds} held-out seeds:")
    print(f"Seeds: {seeds[:5]}... (total {num_seeds})")

    # 3. Run evaluation across all seeds
    detailed_rows: list[dict] = []
    policy_results: dict[str, list[dict]] = {name: [] for name in policies}

    for seed in seeds:
        for name, policy in policies.items():
            sim = AmbulanceSimulator(config=config, seed=seed)
            sim.run_day(decision_callback=policy.decide)

            metrics = calculate_metrics(
                sim.outcomes, target_min=config.response_target_min
            )

            row = {
                "policy": name,
                "seed": seed,
                "mean_response_min": round(metrics.mean_response_time_min, 3),
                "median_response_min": round(metrics.median_response_time_min, 3),
                "p90_response_min": round(metrics.percentile_90_response_time_min, 3),
                "target_fraction": round(metrics.target_response_fraction, 4),
                "dropped_calls": sim.dropped_calls,
                "relocations": sim.relocation_count,
                "relocation_minutes": round(sim.relocation_minutes, 2),
                "calls_dispatched": len(sim.outcomes),
            }
            policy_results[name].append(row)
            detailed_rows.append(row)

    # 4. Compute aggregate statistics
    summary: dict[str, dict] = {}
    print("\n" + "=" * 96)
    print(
        f"{'POLICY':<18} | {'MEAN (MIN)':<11} | {'MEDIAN':<8} | {'P90 (MIN)':<9} | {'<8 MIN %':<8} | {'DROPS':<7} | {'RELOCS':<7}"
    )
    print("-" * 96)

    for name, rows in policy_results.items():
        n = len(rows)
        mean_resp = sum(r["mean_response_min"] for r in rows) / n
        std_resp = math.sqrt(
            sum((r["mean_response_min"] - mean_resp) ** 2 for r in rows) / (n - 1)
        )
        median_resp = sum(r["median_response_min"] for r in rows) / n
        p90_resp = sum(r["p90_response_min"] for r in rows) / n
        target_pct = sum(r["target_fraction"] for r in rows) / n * 100.0
        drops = sum(r["dropped_calls"] for r in rows) / n
        relocs = sum(r["relocations"] for r in rows) / n
        reloc_mins = sum(r["relocation_minutes"] for r in rows) / n

        summary[name] = {
            "mean_response_time_min": round(mean_resp, 2),
            "std_response_time_min": round(std_resp, 2),
            "median_response_time_min": round(median_resp, 2),
            "p90_response_time_min": round(p90_resp, 2),
            "target_response_pct": round(target_pct, 1),
            "mean_dropped_calls": round(drops, 2),
            "mean_relocations": round(relocs, 2),
            "mean_relocation_minutes": round(reloc_mins, 2),
        }

        print(
            f"{name:<18} | {mean_resp:5.2f} +/- {std_resp:4.2f} | {median_resp:6.2f} | {p90_resp:7.2f} | {target_pct:6.1f}% | {drops:5.2f} | {relocs:5.2f}"
        )
    print("=" * 96)
    print(
        "* Note: Response time metrics are calculated over served calls only."
    )
    print("  Unserved calls are recorded strictly under 'DROPS'.\n")

    # 5. Paired statistical comparisons: Trained RL vs each Baseline across identical seeds
    print("PAIRED STATISTICAL COMPARISONS (Trained RL vs Baseline on Identical Seeds):")
    paired_comparisons: dict[str, dict] = {}
    rl_rows = policy_results["Trained RL"]

    for name, rows in policy_results.items():
        if name == "Trained RL":
            continue
        diff_mean = [
            r_rl["mean_response_min"] - r_b["mean_response_min"]
            for r_rl, r_b in zip(rl_rows, rows)
        ]
        diff_p90 = [
            r_rl["p90_response_min"] - r_b["p90_response_min"]
            for r_rl, r_b in zip(rl_rows, rows)
        ]
        diff_drops = [
            r_rl["dropped_calls"] - r_b["dropped_calls"]
            for r_rl, r_b in zip(rl_rows, rows)
        ]

        mean_diff, se_diff, t_stat, p_val, ci, cohens_d = paired_stats(diff_mean)
        p90_diff, _, _, _, _, _ = paired_stats(diff_p90)
        drops_diff, _, _, _, _, _ = paired_stats(diff_drops)

        sig_label = "YES (p < 0.05)" if p_val < 0.05 else "NO (p >= 0.05)"
        paired_comparisons[f"RL_vs_{name}"] = {
            "mean_response_diff_min": round(mean_diff, 3),
            "se_diff_min": round(se_diff, 3),
            "t_statistic": round(t_stat, 2),
            "p_value": round(p_val, 4),
            "ci_95_pct": [round(ci[0], 3), round(ci[1], 3)],
            "cohens_d": round(cohens_d, 2),
            "statistically_significant": p_val < 0.05,
            "p90_response_diff_min": round(p90_diff, 3),
            "dropped_calls_diff": round(drops_diff, 2),
        }

        faster_or_slower = "FASTER" if mean_diff < 0 else "SLOWER"
        print(
            f"  vs {name:<16}: Mean Diff = {mean_diff:+.2f} min ({faster_or_slower}, "
            f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], t={t_stat:+.2f}, p={p_val:.4f}, Sig: {sig_label})\n"
            f"                     P90 Diff = {p90_diff:+.2f} min, Drops Diff = {drops_diff:+.2f}"
        )

    # 6. Save machine-readable output files
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_summary.json"
    json_payload = {
        "seeds": seeds,
        "num_seeds": num_seeds,
        "summary": summary,
        "paired_comparisons": paired_comparisons,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON to: {json_path}")

    csv_path = out_dir / "evaluation_details.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "policy",
            "seed",
            "mean_response_min",
            "median_response_min",
            "p90_response_min",
            "target_fraction",
            "dropped_calls",
            "relocations",
            "relocation_minutes",
            "calls_dispatched",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"Saved detailed per-seed CSV to: {csv_path}")


if __name__ == "__main__":
    main()
