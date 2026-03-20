"""Post-hoc analysis comparing stealth vs naive adversarial attacks.

Reads metrics from a stealth experiment and a baseline (naive) experiment,
then produces:
  1. ASR + stealth metrics over rounds (dual-axis plot)
  2. Probe score distributions: stealth vs naive wins
  3. Attack text statistics: length, strategy diversity, lexical features
  4. Probe evasion trajectory
  5. Qualitative side-by-side comparison of stealth vs naive attacks

Usage:
  python stealth_analysis.py \
      --stealth-dir experiments/stealth_v1 \
      --baseline-dir experiments/frozen_victim_v2 \
      --probe-dir results/sae/frozen_victim_v2

  python stealth_analysis.py \
      --stealth-dir experiments/stealth_v1 \
      --probe-dir results/sae/frozen_victim_v2 \
      --rescore-baseline experiments/frozen_victim_v2
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(_SCRIPT_DIR, "images")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(experiment_dir):
    """Load metrics.jsonl from an experiment directory."""
    metrics_path = os.path.join(experiment_dir, "metrics.jsonl")
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_round_attacks(experiment_dir, round_num):
    """Load all attacks from a specific round."""
    path = os.path.join(experiment_dir, "rounds", f"round_{round_num}.jsonl")
    if not os.path.exists(path):
        return []
    attacks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                attacks.append(json.loads(line))
    return attacks


def load_all_round_attacks(experiment_dir):
    """Load attacks from all rounds."""
    rounds_dir = os.path.join(experiment_dir, "rounds")
    if not os.path.isdir(rounds_dir):
        return []

    all_attacks = []
    for fname in sorted(os.listdir(rounds_dir)):
        if fname.startswith("round_") and fname.endswith(".jsonl"):
            r = int(fname.split("_")[1].split(".")[0])
            path = os.path.join(rounds_dir, fname)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["round"] = r
                        all_attacks.append(rec)
    return all_attacks


# ---------------------------------------------------------------------------
# Retroactive probe scoring (score baseline attacks through the probe)
# ---------------------------------------------------------------------------

def rescore_attacks(experiment_dir, probe_dir, victim_model=None):
    """Score all attacks from an experiment through the SAE probe.

    Returns list of dicts with 'attack', 'unsafe', 'round', 'probe_score'.
    """
    from stealth_loop import JailbreakProbe
    from model_utils import load_model, extract_hidden_states, unload_model
    from sae_analysis import _detect_victim_model

    print(f"\n=== RESCORING ATTACKS FROM {experiment_dir} ===")

    # Load probe
    probe = JailbreakProbe.from_pretrained(probe_dir)

    # Load all attack data
    all_data = load_all_round_attacks(experiment_dir)
    if not all_data:
        print("[Error] No round data found")
        return []

    attacks = [d["attack"] for d in all_data]

    # Detect victim model
    if victim_model is None:
        victim_model = _detect_victim_model(experiment_dir)
    print(f"  Victim: {victim_model}")

    # Load victim and extract hidden states
    model, tokenizer = load_model(victim_model)

    prompts = []
    for attack in attacks:
        messages = [{"role": "user", "content": attack}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    print(f"  Extracting hidden states for {len(prompts)} attacks...")
    batch_size = 50
    all_hidden = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        h = extract_hidden_states(model, tokenizer, batch)
        all_hidden.append(h)
        done = min(i + batch_size, len(prompts))
        print(f"    {done}/{len(prompts)}")

    hidden_states = torch.cat(all_hidden, dim=0)
    unload_model(model, tokenizer)

    # Score
    scores = probe.score(hidden_states)

    # Merge scores into data
    for i, d in enumerate(all_data):
        d["probe_score"] = float(scores[i])

    print(f"  Scored {len(all_data)} attacks: "
          f"mean probe={scores.mean():.3f}, std={scores.std():.3f}")

    return all_data


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

def compute_text_stats(attacks_data):
    """Compute text-level statistics for a set of attacks."""
    texts = [d["attack"] for d in attacks_data]
    lengths = [len(t.split()) for t in texts]
    char_lengths = [len(t) for t in texts]

    # Strategy keyword detection (heuristic)
    strategy_keywords = {
        "fiction": ["novel", "story", "chapter", "character", "fiction",
                    "narrative", "scene"],
        "code": ["```", "function", "def ", "class ", "import ", "code review",
                 "variable"],
        "roleplay": ["roleplay", "character", "act as", "pretend", "stage",
                     "theatrical", "director"],
        "education": ["CTF", "exercise", "educational", "challenge", "tutorial",
                      "learn"],
        "obfuscation": ["synonym", "metaphor", "indirect", "encoded",
                        "obfusc", "disguise"],
        "multi_turn": ["step 1", "step 2", "first,", "next,", "then,",
                       "sub-question", "decomp"],
        "technical": ["troubleshoot", "diagnostic", "sysadmin", "procedure",
                      "recovery"],
    }

    strategy_counts = Counter()
    for text in texts:
        text_lower = text.lower()
        for strategy, keywords in strategy_keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                strategy_counts[strategy] += 1

    return {
        "n": len(texts),
        "mean_word_length": np.mean(lengths) if lengths else 0,
        "median_word_length": np.median(lengths) if lengths else 0,
        "std_word_length": np.std(lengths) if lengths else 0,
        "mean_char_length": np.mean(char_lengths) if char_lengths else 0,
        "strategy_distribution": dict(strategy_counts),
        "unique_first_words": len(set(t.split()[0].lower() for t in texts
                                      if t.split())),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stealth_trajectory(stealth_metrics, baseline_metrics, output_name):
    """Dual-axis plot: ASR + probe metrics over rounds."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- ASR comparison ---
    s_rounds = [m["round"] for m in stealth_metrics]
    s_asr = [m["asr"] for m in stealth_metrics]
    s_stealth_asr = [m.get("stealth_asr", 0) for m in stealth_metrics]

    ax1.plot(s_rounds, s_asr, "o-", color="#ef4444", linewidth=2,
             label="Stealth: Total ASR", markersize=6)
    ax1.plot(s_rounds, s_stealth_asr, "s--", color="#f97316", linewidth=2,
             label="Stealth: Stealth ASR (probe < 0.5)", markersize=5)

    if baseline_metrics:
        b_rounds = [m["round"] for m in baseline_metrics]
        b_asr = [m["asr"] for m in baseline_metrics]
        ax1.plot(b_rounds, b_asr, "^-", color="#6366f1", linewidth=2,
                 label="Naive: Total ASR", markersize=6, alpha=0.7)

    ax1.set_ylabel("Attack Success Rate", fontsize=12)
    ax1.set_title("Stealth vs Naive Adversary: ASR Trajectory",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Probe metrics ---
    s_probe_all = [m.get("mean_probe_all", 0) for m in stealth_metrics]
    s_probe_wins = [m.get("mean_probe_wins", 0) for m in stealth_metrics]
    s_evasion = [m.get("evasion_rate", 0) for m in stealth_metrics]

    ax2.plot(s_rounds, s_probe_wins, "o-", color="#ef4444", linewidth=2,
             label="Mean probe score (wins)")
    ax2.plot(s_rounds, s_probe_all, "x--", color="#f97316", linewidth=1.5,
             label="Mean probe score (all)", alpha=0.7)
    ax2.fill_between(s_rounds, 0, s_evasion, color="#22c55e", alpha=0.15,
                     label="Evasion rate")
    ax2.plot(s_rounds, s_evasion, "s-", color="#22c55e", linewidth=2,
             label="Evasion rate (wins w/ probe < 0.5)")

    ax2.axhline(y=0.5, color="gray", linewidth=1, linestyle=":",
                label="Detection threshold")
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Score / Rate", fontsize=12)
    ax2.set_title("Probe Evasion Metrics", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, f"stealth_trajectory_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_probe_distributions(stealth_data, baseline_data, output_name):
    """Histogram of probe scores: stealth wins vs naive wins."""
    import matplotlib.pyplot as plt

    stealth_win_scores = [d["probe_score"] for d in stealth_data
                          if d.get("unsafe")]
    stealth_safe_scores = [d["probe_score"] for d in stealth_data
                           if not d.get("unsafe")]
    baseline_win_scores = [d["probe_score"] for d in baseline_data
                           if d.get("unsafe")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stealth wins vs naive wins
    ax = axes[0]
    if baseline_win_scores:
        ax.hist(baseline_win_scores, bins=30, alpha=0.6, color="#6366f1",
                label=f"Naive wins (n={len(baseline_win_scores)})",
                density=True, edgecolor="white")
    ax.hist(stealth_win_scores, bins=30, alpha=0.6, color="#ef4444",
            label=f"Stealth wins (n={len(stealth_win_scores)})",
            density=True, edgecolor="white")
    ax.axvline(x=0.5, color="gray", linewidth=2, linestyle="--",
               label="Detection threshold")
    ax.set_xlabel("Probe Score (detection probability)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Successful Jailbreaks: Probe Score Distribution",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: stealth wins vs stealth failures
    ax = axes[1]
    ax.hist(stealth_safe_scores, bins=30, alpha=0.6, color="#3b82f6",
            label=f"Failed attacks (n={len(stealth_safe_scores)})",
            density=True, edgecolor="white")
    ax.hist(stealth_win_scores, bins=30, alpha=0.6, color="#ef4444",
            label=f"Stealth wins (n={len(stealth_win_scores)})",
            density=True, edgecolor="white")
    ax.axvline(x=0.5, color="gray", linewidth=2, linestyle="--",
               label="Detection threshold")
    ax.set_xlabel("Probe Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Stealth Experiment: Wins vs Failures",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(IMAGE_DIR, f"stealth_probe_dist_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_text_comparison(stealth_stats, baseline_stats, output_name):
    """Bar charts comparing text-level features of stealth vs naive attacks."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: word length distribution
    ax = axes[0]
    labels = ["Mean", "Median", "Std"]
    stealth_vals = [stealth_stats["mean_word_length"],
                    stealth_stats["median_word_length"],
                    stealth_stats["std_word_length"]]
    baseline_vals = [baseline_stats["mean_word_length"],
                     baseline_stats["median_word_length"],
                     baseline_stats["std_word_length"]]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, baseline_vals, w, color="#6366f1", label="Naive",
           edgecolor="white")
    ax.bar(x + w / 2, stealth_vals, w, color="#ef4444", label="Stealth",
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Word Count", fontsize=11)
    ax.set_title("Attack Length Statistics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: strategy distribution
    ax = axes[1]
    all_strategies = sorted(
        set(list(stealth_stats["strategy_distribution"].keys())
            + list(baseline_stats["strategy_distribution"].keys()))
    )
    s_counts = [stealth_stats["strategy_distribution"].get(s, 0)
                for s in all_strategies]
    b_counts = [baseline_stats["strategy_distribution"].get(s, 0)
                for s in all_strategies]

    # Normalize by total
    s_total = stealth_stats["n"] or 1
    b_total = baseline_stats["n"] or 1
    s_frac = [c / s_total for c in s_counts]
    b_frac = [c / b_total for c in b_counts]

    x = np.arange(len(all_strategies))
    ax.barh(x - w / 2, b_frac, w, color="#6366f1", label="Naive",
            edgecolor="white")
    ax.barh(x + w / 2, s_frac, w, color="#ef4444", label="Stealth",
            edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(all_strategies, fontsize=9)
    ax.set_xlabel("Fraction of Attacks", fontsize=11)
    ax.set_title("Strategy Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    path = os.path.join(IMAGE_DIR, f"stealth_text_compare_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_evasion_by_round(stealth_data, output_name):
    """Per-round breakdown of evasion rate and probe score."""
    import matplotlib.pyplot as plt

    rounds = sorted(set(d["round"] for d in stealth_data))
    round_stats = {}
    for r in rounds:
        r_data = [d for d in stealth_data if d["round"] == r]
        wins = [d for d in r_data if d.get("unsafe")]
        safe = [d for d in r_data if not d.get("unsafe")]

        win_scores = [d["probe_score"] for d in wins]
        safe_scores = [d["probe_score"] for d in safe]

        round_stats[r] = {
            "asr": len(wins) / len(r_data) if r_data else 0,
            "mean_probe_wins": np.mean(win_scores) if win_scores else 0,
            "mean_probe_safe": np.mean(safe_scores) if safe_scores else 0,
            "evasion_rate": (sum(1 for s in win_scores if s < 0.5)
                            / len(wins) if wins else 0),
            "n_wins": len(wins),
        }

    fig, ax = plt.subplots(figsize=(10, 5))

    x = list(round_stats.keys())
    probe_wins = [round_stats[r]["mean_probe_wins"] for r in x]
    probe_safe = [round_stats[r]["mean_probe_safe"] for r in x]
    evasion = [round_stats[r]["evasion_rate"] for r in x]

    ax.plot(x, probe_wins, "o-", color="#ef4444", linewidth=2,
            label="Mean probe (wins)", markersize=6)
    ax.plot(x, probe_safe, "s--", color="#3b82f6", linewidth=1.5,
            label="Mean probe (failures)", markersize=5, alpha=0.7)
    ax.fill_between(x, 0, evasion, color="#22c55e", alpha=0.15)
    ax.plot(x, evasion, "^-", color="#22c55e", linewidth=2,
            label="Evasion rate", markersize=5)
    ax.axhline(y=0.5, color="gray", linewidth=1, linestyle=":")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Score / Rate", fontsize=12)
    ax.set_title("Probe Evasion by Round", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(IMAGE_DIR, f"stealth_evasion_round_{output_name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Qualitative comparison
# ---------------------------------------------------------------------------

def print_qualitative_comparison(stealth_data, baseline_data, n=5):
    """Print side-by-side examples of stealth vs naive successful attacks."""
    stealth_wins = sorted(
        [d for d in stealth_data if d.get("unsafe")],
        key=lambda d: d.get("probe_score", 1.0)
    )
    naive_wins = sorted(
        [d for d in baseline_data if d.get("unsafe")],
        key=lambda d: d.get("probe_score", 0.0),
        reverse=True
    )

    print("\n" + "=" * 80)
    print("QUALITATIVE COMPARISON: STEALTHIEST vs MOST DETECTABLE ATTACKS")
    print("=" * 80)

    print(f"\n--- TOP {n} STEALTHIEST WINS (lowest probe scores) ---")
    for i, d in enumerate(stealth_wins[:n]):
        print(f"\n  [{i+1}] Round {d['round']} | probe={d['probe_score']:.3f}")
        print(f"      {d['attack'][:200]}")

    print(f"\n--- TOP {n} MOST DETECTABLE NAIVE WINS (highest probe scores) ---")
    for i, d in enumerate(naive_wins[:n]):
        print(f"\n  [{i+1}] Round {d['round']} | probe={d['probe_score']:.3f}")
        print(f"      {d['attack'][:200]}")

    # Also show overall stats
    stealth_scores = [d["probe_score"] for d in stealth_wins]
    naive_scores = [d["probe_score"] for d in naive_wins]

    print(f"\n--- SUMMARY ---")
    print(f"Stealth wins: n={len(stealth_wins)}, "
          f"mean probe={np.mean(stealth_scores):.3f}, "
          f"median={np.median(stealth_scores):.3f}, "
          f"evasion rate={sum(1 for s in stealth_scores if s < 0.5)/len(stealth_scores):.1%}"
          if stealth_scores else "no stealth wins")
    print(f"Naive wins:   n={len(naive_wins)}, "
          f"mean probe={np.mean(naive_scores):.3f}, "
          f"median={np.median(naive_scores):.3f}, "
          f"evasion rate={sum(1 for s in naive_scores if s < 0.5)/len(naive_scores):.1%}"
          if naive_scores else "no naive wins")


# ---------------------------------------------------------------------------
# Full analysis report
# ---------------------------------------------------------------------------

def generate_report(stealth_data, baseline_data, stealth_metrics,
                    baseline_metrics, output_dir):
    """Generate a JSON report summarizing the comparison."""
    stealth_wins = [d for d in stealth_data if d.get("unsafe")]
    baseline_wins = [d for d in baseline_data if d.get("unsafe")]

    s_scores = [d["probe_score"] for d in stealth_wins]
    b_scores = [d["probe_score"] for d in baseline_wins]

    # Compute per-round probe score trajectory for stealth wins
    s_rounds = sorted(set(d["round"] for d in stealth_data))
    probe_trajectory = {}
    for r in s_rounds:
        r_wins = [d for d in stealth_wins if d["round"] == r]
        if r_wins:
            scores = [d["probe_score"] for d in r_wins]
            probe_trajectory[r] = {
                "n_wins": len(r_wins),
                "mean_probe": round(np.mean(scores), 4),
                "evasion_rate": round(
                    sum(1 for s in scores if s < 0.5) / len(scores), 4
                ),
            }

    report = {
        "stealth": {
            "total_attacks": len(stealth_data),
            "total_wins": len(stealth_wins),
            "overall_asr": round(len(stealth_wins) / len(stealth_data), 4)
                          if stealth_data else 0,
            "mean_probe_wins": round(np.mean(s_scores), 4) if s_scores else 0,
            "median_probe_wins": round(np.median(s_scores), 4) if s_scores else 0,
            "evasion_rate": round(
                sum(1 for s in s_scores if s < 0.5) / len(s_scores), 4
            ) if s_scores else 0,
            "text_stats": compute_text_stats(stealth_wins),
            "probe_trajectory": probe_trajectory,
        },
        "baseline": {
            "total_attacks": len(baseline_data),
            "total_wins": len(baseline_wins),
            "overall_asr": round(len(baseline_wins) / len(baseline_data), 4)
                          if baseline_data else 0,
            "mean_probe_wins": round(np.mean(b_scores), 4) if b_scores else 0,
            "median_probe_wins": round(np.median(b_scores), 4) if b_scores else 0,
            "evasion_rate": round(
                sum(1 for s in b_scores if s < 0.5) / len(b_scores), 4
            ) if b_scores else 0,
            "text_stats": compute_text_stats(baseline_wins),
        },
        "comparison": {
            "probe_score_delta": round(
                (np.mean(b_scores) - np.mean(s_scores)), 4
            ) if s_scores and b_scores else 0,
            "evasion_rate_delta": round(
                (sum(1 for s in s_scores if s < 0.5) / len(s_scores)
                 - sum(1 for s in b_scores if s < 0.5) / len(b_scores)), 4
            ) if s_scores and b_scores else 0,
            "asr_delta": round(
                (len(stealth_wins) / len(stealth_data)
                 - len(baseline_wins) / len(baseline_data)), 4
            ) if stealth_data and baseline_data else 0,
        },
    }

    report_path = os.path.join(output_dir, "stealth_report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare stealth vs naive adversarial attacks"
    )
    parser.add_argument("--stealth-dir", type=str, required=True,
                        help="Stealth experiment directory")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Baseline experiment directory (for metrics)")
    parser.add_argument("--probe-dir", type=str, required=True,
                        help="SAE probe directory")
    parser.add_argument("--rescore-baseline", type=str, default=None,
                        help="Rescore baseline attacks through the probe "
                             "(slow, requires GPU)")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Name for output files")
    parser.add_argument("--victim-model", type=str, default=None)

    args = parser.parse_args()

    output_name = args.output_name or os.path.basename(
        args.stealth_dir.rstrip("/")
    )

    # Load stealth experiment data
    print("=== Loading stealth experiment ===")
    stealth_metrics = load_metrics(args.stealth_dir)
    stealth_data = load_all_round_attacks(args.stealth_dir)

    # Score stealth attacks through probe
    print("\n=== Scoring stealth attacks through probe ===")
    stealth_scored = rescore_attacks(args.stealth_dir, args.probe_dir,
                                    victim_model=args.victim_model)

    # Load or score baseline
    baseline_metrics = []
    baseline_scored = []

    if args.baseline_dir:
        print("\n=== Loading baseline metrics ===")
        baseline_metrics = load_metrics(args.baseline_dir)

    if args.rescore_baseline:
        print("\n=== Scoring baseline attacks through probe ===")
        baseline_scored = rescore_attacks(args.rescore_baseline, args.probe_dir,
                                         victim_model=args.victim_model)
    elif args.baseline_dir:
        # Try to load pre-scored data if available
        baseline_scored = load_all_round_attacks(args.baseline_dir)
        # These won't have probe_score unless previously scored
        if baseline_scored and "probe_score" not in baseline_scored[0]:
            print("  [Note] Baseline attacks not scored. Use --rescore-baseline "
                  "for full comparison.")

    # --- Generate plots ---
    print("\n=== Generating plots ===")

    try:
        import matplotlib  # noqa: F401

        plot_stealth_trajectory(stealth_metrics, baseline_metrics, output_name)

        if stealth_scored:
            plot_evasion_by_round(stealth_scored, output_name)

        if stealth_scored and baseline_scored:
            has_scores = all("probe_score" in d for d in baseline_scored)
            if has_scores:
                plot_probe_distributions(stealth_scored, baseline_scored,
                                        output_name)

                stealth_win_stats = compute_text_stats(
                    [d for d in stealth_scored if d.get("unsafe")]
                )
                baseline_win_stats = compute_text_stats(
                    [d for d in baseline_scored if d.get("unsafe")]
                )
                plot_text_comparison(stealth_win_stats, baseline_win_stats,
                                    output_name)

    except ImportError:
        print("  [Warning] matplotlib not available — skipping plots.")

    # --- Qualitative comparison ---
    if stealth_scored and baseline_scored:
        has_scores = all("probe_score" in d for d in baseline_scored)
        if has_scores:
            print_qualitative_comparison(stealth_scored, baseline_scored)

    # --- JSON report ---
    output_dir = os.path.join("results", "stealth", output_name)
    report = generate_report(stealth_scored or stealth_data,
                             baseline_scored or [],
                             stealth_metrics, baseline_metrics,
                             output_dir)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    s = report["stealth"]
    b = report["baseline"]
    c = report["comparison"]
    print(f"  Stealth ASR: {s['overall_asr']:.1%}  |  "
          f"Baseline ASR: {b['overall_asr']:.1%}  |  "
          f"Delta: {c['asr_delta']:+.1%}")
    print(f"  Stealth probe (wins): {s['mean_probe_wins']:.3f}  |  "
          f"Baseline probe (wins): {b['mean_probe_wins']:.3f}  |  "
          f"Delta: {c['probe_score_delta']:+.3f}")
    print(f"  Stealth evasion: {s['evasion_rate']:.1%}  |  "
          f"Baseline evasion: {b['evasion_rate']:.1%}  |  "
          f"Delta: {c['evasion_rate_delta']:+.1%}")


if __name__ == "__main__":
    main()
