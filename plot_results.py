"""Generate publication-ready figures for the 2-page report and documentation."""

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2,
    })

    # 1. Load Summary Results
    summary_path = Path("results/evaluation_summary.json")
    if not summary_path.exists():
        print("Summary JSON not found. Run py evaluate.py first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    policies = summary_data["summary"]
    policy_names = ["No redeployment", "Fixed home base", "Nearest base", "Greedy coverage", "Trained RL"]
    short_names = ["No Redepl.", "Fixed Home", "Nearest Base", "Greedy (MEXCLP)", "Trained RL (Ours)"]
    colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    mean_resp = [policies[p]["mean_response_time_min"] for p in policy_names]
    std_resp = [policies[p]["std_response_time_min"] for p in policy_names]
    p90_resp = [policies[p]["p90_response_time_min"] for p in policy_names]
    drops = [policies[p]["mean_dropped_calls"] for p in policy_names]
    relocs = [policies[p]["mean_relocations"] for p in policy_names]

    # Figure 1: Mean & P90 Response Times (Grouped Bar Chart)
    x = np.arange(len(short_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, mean_resp, width, yerr=std_resp, capsize=4, label="Mean Response Time", color="#4C72B0", alpha=0.9)
    rects2 = ax.bar(x + width/2, p90_resp, width, label="90th Percentile (P90) Response", color="#C44E52", alpha=0.9)

    ax.set_ylabel("Response Time (minutes)")
    ax.set_title("Response Time Comparison Across 20 Held-Out Demand Seeds (Seeds 42–61)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 22)

    # Add data labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}m", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}m", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig1_path = figures_dir / "response_times.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Relocations vs Dropped Calls Trade-Off
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for i, name in enumerate(short_names):
        marker = "*" if "Ours" in name else "o"
        size = 200 if "Ours" in name else 120
        ax.scatter(relocs[i], drops[i], color=colors[i], s=size, marker=marker, label=name, zorder=5)
        offset_y = 0.15 if i % 2 == 0 else -0.2
        ax.annotate(name, (relocs[i], drops[i]), textcoords="offset points", xytext=(0, 10 if offset_y > 0 else -15),
                    ha="center", fontsize=9, fontweight="bold", color=colors[i])

    ax.set_xlabel("Relocations per Day (Fleet Churn)")
    ax.set_ylabel("Dropped Calls per Day (Unserved Calls)")
    ax.set_title("Fleet Relocation Churn vs. Dropped Calls")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-5, 105)
    ax.set_ylim(2.5, 6.5)

    plt.tight_layout()
    fig2_path = figures_dir / "relocations_vs_drops.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Training Rewards Progression
    model_path = Path("models/linear_q_agent.json")
    if model_path.exists():
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
        
        rewards = model_data.get("episode_rewards", [])
        if rewards:
            episodes = np.arange(1, len(rewards) + 1)
            # 10-episode rolling average
            window = 10
            rolling_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
            rolling_x = np.arange(window, len(rewards) + 1)

            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.plot(episodes, rewards, color="#4C72B0", alpha=0.35, label="Raw Episode Reward")
            ax.plot(rolling_x, rolling_avg, color="#2ca02c", linewidth=2.5, label=f"{window}-Episode Moving Average")
            ax.set_xlabel("Training Episode (Day)")
            ax.set_ylabel("Cumulative Bounded Interval Reward")
            ax.set_title("Double Linear Q-Learning Training Reward Curve (Seeds 1000–1099)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(frameon=True, loc="lower right")

            plt.tight_layout()
            fig3_path = figures_dir / "training_rewards.png"
            plt.savefig(fig3_path)
            plt.close()
            print(f"Saved: {fig3_path}")

    # Figure 4: Box Plot Distribution across 20 seeds
    details_path = Path("results/evaluation_details.csv")
    if details_path.exists():
        data_by_policy = {p: [] for p in policy_names}
        with open(details_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["policy"] in data_by_policy:
                    data_by_policy[row["policy"]].append(float(row["mean_response_min"]))

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        box_data = [data_by_policy[p] for p in policy_names]
        bp = ax.boxplot(box_data, tick_labels=short_names, patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Mean Response Time (minutes)")
        ax.set_title("Response Time Variance Across 20 Held-Out Evaluation Seeds")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig4_path = figures_dir / "response_distributions.png"
        plt.savefig(fig4_path)
        plt.close()
        print(f"Saved: {fig4_path}")


if __name__ == "__main__":
    main()
