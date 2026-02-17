"""Plot metrics from the Chaos Loop experiment.

Reads metrics.jsonl and generates:
  1. ../images/chaos_asr_curve.png   — ASR line plot over rounds
  2. ../images/chaos_wins_per_round.png — Bar chart of wins vs candidates per round
"""

import json
import os
import sys

import matplotlib.pyplot as plt

METRICS_FILE = "metrics.jsonl"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        print(f"[Error] {METRICS_FILE} not found. Run the chaos loop first.")
        sys.exit(1)

    records = []
    with open(METRICS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"[Error] {METRICS_FILE} is empty.")
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
    ax.set_title("Chaos-1B: Attack Success Rate Over Rounds", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "chaos_asr_curve.png")
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
    ax.set_title("Chaos-1B: Wins vs Candidates Per Round", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "chaos_wins_per_round.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    records = load_metrics()
    plot_asr_curve(records)
    plot_wins_per_round(records)
    print("Done.")
