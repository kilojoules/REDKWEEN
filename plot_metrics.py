"""Plot metrics from a REDKWEEN loop experiment.

Reads metrics.jsonl and generates:
  1. ../images/redkween_asr_curve.png   — ASR line plot over rounds
  2. ../images/redkween_wins_per_round.png — Bar chart of wins vs candidates per round

Usage:
  python plot_metrics.py                          # Legacy: reads results/metrics.jsonl
  python plot_metrics.py --experiment-dir experiments/A050_buffered
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "images")


def load_metrics(metrics_file):
    if not os.path.exists(metrics_file):
        print(f"[Error] {metrics_file} not found. Run the REDKWEEN loop first.")
        sys.exit(1)

    records = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"[Error] {metrics_file} is empty.")
        sys.exit(1)

    records.sort(key=lambda r: r["round"])
    return records


def plot_asr_curve(records):
    rounds = [r["round"] for r in records]
    asrs = [r["asr"] * 100 for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, asrs, "o-", color="#ef4444", linewidth=2, markersize=8)
    ax.fill_between(rounds, asrs, alpha=0.15, color="#ef4444")
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax.set_title("REDKWEEN: Attack Success Rate Over Rounds", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "redkween_asr_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_wins_per_round(records):
    rounds = [r["round"] for r in records]
    candidates = [r["candidates"] for r in records]
    wins = [r["wins"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    x = range(len(rounds))

    ax.bar([i - bar_width / 2 for i in x], candidates, bar_width, label="Candidates", color="#94a3b8")
    ax.bar([i + bar_width / 2 for i in x], wins, bar_width, label="Wins (Jailbreaks)", color="#ef4444")

    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("REDKWEEN: Wins vs Candidates Per Round", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "redkween_wins_per_round.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot REDKWEEN loop metrics")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Path to experiment directory (e.g. experiments/A050_buffered). "
                             "If not set, reads results/metrics.jsonl (legacy).")
    args = parser.parse_args()

    if args.experiment_dir:
        metrics_file = os.path.join(args.experiment_dir, "metrics.jsonl")
    else:
        metrics_file = os.path.join(_SCRIPT_DIR, "results", "metrics.jsonl")

    records = load_metrics(metrics_file)
    plot_asr_curve(records)
    plot_wins_per_round(records)
    print("Done.")
