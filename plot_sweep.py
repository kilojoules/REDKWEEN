"""Sweep visualization for A-parameter experiments.

Reads metrics from experiments/*/metrics.jsonl and baselines_comparison.json
to produce four key figures:

1. ASR vs A — Two lines (memoryless, buffered) showing mean ASR per A value.
2. ASR vs Round per A — Subplots showing ASR evolution over rounds.
3. Diversity vs A — Attack diversity metric per A and mode.
4. Baseline comparison bar chart — All methods side by side.

Usage:
    python plot_sweep.py
    python plot_sweep.py --experiments-dir experiments
    python plot_sweep.py --output-dir ../images
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_metrics(experiments_dir: str) -> dict[str, list[dict]]:
    """Load metrics.jsonl from each experiment subdirectory.

    Returns {experiment_name: [list of round records]}.
    """
    all_metrics = {}
    if not os.path.isdir(experiments_dir):
        return all_metrics

    for name in sorted(os.listdir(experiments_dir)):
        metrics_path = os.path.join(experiments_dir, name, "metrics.jsonl")
        if os.path.exists(metrics_path):
            records = []
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if records:
                all_metrics[name] = records
    return all_metrics


def parse_experiment_name(name: str) -> tuple[float, str] | None:
    """Extract A value and mode from experiment name like 'A030_buffered'."""
    match = re.match(r"A(\d{3})_(\w+)$", name)
    if match:
        A = int(match.group(1)) / 100.0
        mode = match.group(2)
        return A, mode
    return None


def plot_asr_vs_a(all_metrics: dict, output_dir: str):
    """Figure 1: ASR vs A — two lines for memoryless and buffered."""
    # Group by (A, mode) -> mean ASR across rounds
    mode_data: dict[str, dict[float, float]] = defaultdict(dict)

    for name, records in all_metrics.items():
        parsed = parse_experiment_name(name)
        if parsed is None:
            continue
        A, mode = parsed
        asrs = [r["asr"] for r in records]
        mean_asr = sum(asrs) / len(asrs) if asrs else 0.0
        mode_data[mode][A] = mean_asr

    if not mode_data:
        print("   No sweep data found for ASR vs A plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"buffered": "#3b82f6", "memoryless": "#ef4444"}
    markers = {"buffered": "o", "memoryless": "s"}

    for mode in sorted(mode_data.keys()):
        data = mode_data[mode]
        a_vals = sorted(data.keys())
        asrs = [data[a] * 100 for a in a_vals]
        ax.plot(a_vals, asrs,
                marker=markers.get(mode, "^"),
                color=colors.get(mode, "#666"),
                linewidth=2, markersize=8, label=mode)

    ax.set_xlabel("A (Zoo Sampling Probability)", fontsize=13)
    ax.set_ylabel("Mean ASR (%)", fontsize=13)
    ax.set_title("Attack Success Rate vs A Parameter", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "sweep_asr_vs_a.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_asr_vs_round(all_metrics: dict, output_dir: str):
    """Figure 2: ASR vs Round — one subplot per A value."""
    # Group by A value
    a_groups: dict[float, dict[str, list[dict]]] = defaultdict(dict)

    for name, records in all_metrics.items():
        parsed = parse_experiment_name(name)
        if parsed is None:
            continue
        A, mode = parsed
        a_groups[A][mode] = records

    if not a_groups:
        print("   No sweep data found for ASR vs Round plot")
        return

    a_values = sorted(a_groups.keys())
    n_plots = len(a_values)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    colors = {"buffered": "#3b82f6", "memoryless": "#ef4444"}

    for idx, A in enumerate(a_values):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        for mode, records in sorted(a_groups[A].items()):
            rounds = [r["round"] for r in records]
            asrs = [r["asr"] * 100 for r in records]
            ax.plot(rounds, asrs, "o-",
                    color=colors.get(mode, "#666"),
                    linewidth=1.5, markersize=5, label=mode)

        ax.set_title(f"A = {A:.2f}", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("ASR Over Rounds by A Value", fontsize=15, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(output_dir, "sweep_asr_vs_round.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_diversity_vs_a(experiments_dir: str, output_dir: str):
    """Figure 3: Attack diversity vs A for each mode."""
    mode_data: dict[str, dict[float, float]] = defaultdict(dict)

    if not os.path.isdir(experiments_dir):
        print("   No experiments directory found for diversity plot")
        return

    for name in sorted(os.listdir(experiments_dir)):
        div_path = os.path.join(experiments_dir, name, "diversity_metrics.json")
        if not os.path.exists(div_path):
            continue

        parsed = parse_experiment_name(name)
        if parsed is None:
            continue
        A, mode = parsed

        with open(div_path) as f:
            metrics = json.load(f)
        if "mean_cosine_distance" in metrics:
            mode_data[mode][A] = metrics["mean_cosine_distance"]

    if not mode_data:
        print("   No diversity data found (run eval_extended.py --diversity first)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"buffered": "#3b82f6", "memoryless": "#ef4444"}
    markers = {"buffered": "o", "memoryless": "s"}

    for mode in sorted(mode_data.keys()):
        data = mode_data[mode]
        a_vals = sorted(data.keys())
        divs = [data[a] for a in a_vals]
        ax.plot(a_vals, divs,
                marker=markers.get(mode, "^"),
                color=colors.get(mode, "#666"),
                linewidth=2, markersize=8, label=mode)

    ax.set_xlabel("A (Zoo Sampling Probability)", fontsize=13)
    ax.set_ylabel("Mean Cosine Distance", fontsize=13)
    ax.set_title("Attack Diversity vs A Parameter", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "sweep_diversity_vs_a.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_baseline_comparison(output_dir: str):
    """Figure 4: Baseline comparison bar chart."""
    baselines_path = "baselines_comparison.json"
    if not os.path.exists(baselines_path):
        print(f"   {baselines_path} not found (run run_baselines.py first)")
        return

    with open(baselines_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("   No baseline results found")
        return

    methods = [r["method"] for r in results]
    asrs = [r["asr"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color-code: Chaos in red, others in blue/gray
    colors = []
    for method in methods:
        if "Chaos" in method:
            colors.append("#ef4444")
        elif method.startswith("Direct"):
            colors.append("#94a3b8")
        else:
            colors.append("#3b82f6")

    bars = ax.bar(range(len(methods)), asrs, color=colors, edgecolor="white")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax.set_title("Baseline Comparison: Attack Success Rates", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels on bars
    for bar, asr in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{asr:.0f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    path = os.path.join(output_dir, "baseline_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot A-parameter sweep results")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Root experiments directory")
    parser.add_argument("--output-dir", type=str, default="../images",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== SWEEP VISUALIZATION ===")

    # Load sweep metrics
    all_metrics = load_sweep_metrics(args.experiments_dir)
    print(f"   Loaded metrics from {len(all_metrics)} experiments")

    if all_metrics:
        plot_asr_vs_a(all_metrics, args.output_dir)
        plot_asr_vs_round(all_metrics, args.output_dir)

    plot_diversity_vs_a(args.experiments_dir, args.output_dir)
    plot_baseline_comparison(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
