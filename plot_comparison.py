"""Generate comparison line charts across all matchups.

Produces:
  images/frozen_asr.png   — Phase 1 (frozen victim) ASR over rounds
  images/selfplay_asr.png — Phase 2 (self-play) ASR over rounds
"""

import json
import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "images")

FROZEN_EXPERIMENTS = [
    ("1B vs 8B", "experiments/frozen_victim_v2/metrics.jsonl"),
    ("3B vs 8B", "experiments/frozen_3b_adv/metrics.jsonl"),
    ("8B vs 8B", "experiments/frozen_8b_v_8b/metrics.jsonl"),
    ("8B vs 3B", "experiments/frozen_8b_adv_3b_vic/metrics.jsonl"),
]

SELFPLAY_EXPERIMENTS = [
    ("3B vs 8B", "experiments/selfplay_3b_v_8b/metrics.jsonl"),
    ("8B vs 8B", "experiments/selfplay_8b_v_8b/metrics.jsonl"),
    ("8B vs 3B", "experiments/selfplay_8b_v_3b/metrics.jsonl"),
]

COLORS = {
    "1B vs 8B": "#6366f1",  # indigo
    "3B vs 8B": "#f59e0b",  # amber
    "8B vs 8B": "#ef4444",  # red
    "8B vs 3B": "#10b981",  # emerald
}


def load_metrics(path):
    full = os.path.join(SCRIPT_DIR, path)
    with open(full) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["round"])
    return records


def plot_chart(experiments, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, path in experiments:
        records = load_metrics(path)
        rounds = [r["round"] for r in records]
        asrs = [r["asr"] * 100 for r in records]
        color = COLORS[label]
        ax.plot(rounds, asrs, "o-", label=label, color=color,
                linewidth=2, markersize=5, alpha=0.9)

    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylim(-2, 55)
    ax.legend(fontsize=11, title="Adversary vs Victim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_chart(FROZEN_EXPERIMENTS,
               "Phase 1: Adversary Learning (Frozen Victim)",
               "frozen_asr.png")
    plot_chart(SELFPLAY_EXPERIMENTS,
               "Phase 2: Self-Play (Both Adapt)",
               "selfplay_asr.png")
    print("Done.")
