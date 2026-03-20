"""Stealth scaling study sweep runner.

Runs all stealth experiment configurations sequentially on a single GPU.
Modeled on sweep.py.

Tiers:
  1 (core)  — stealth weighted α=3 + naive baseline
  2 (dose)  — α=1, α=5, filter τ=0.4
  3 (scale) — 3B adversary, adaptive defender

Usage:
  python stealth_sweep.py                          # All tiers
  python stealth_sweep.py --tiers 1                # Core only
  python stealth_sweep.py --tiers 1 2              # Core + dose-response
  python stealth_sweep.py --dry-run                # Print configs only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


PROBE_DIR = "results/sae/frozen_victim_v2"

# Each config: (name, tier, stealth_mode, alpha_or_threshold, adversary_model,
#               retrain_probe_every)
CONFIGS = [
    # Tier 1: Core hypothesis
    ("scale_stealth_weighted_a3", 1, "weighted", 3.0,
     "meta-llama/Llama-3.2-1B-Instruct", 0),
    ("scale_naive_baseline", 1, "none", 0.0,
     "meta-llama/Llama-3.2-1B-Instruct", 0),

    # Tier 2: Dose-response
    ("scale_stealth_weighted_a1", 2, "weighted", 1.0,
     "meta-llama/Llama-3.2-1B-Instruct", 0),
    ("scale_stealth_weighted_a5", 2, "weighted", 5.0,
     "meta-llama/Llama-3.2-1B-Instruct", 0),
    ("scale_stealth_filter_04", 2, "filter", 0.4,
     "meta-llama/Llama-3.2-1B-Instruct", 0),

    # Tier 3: Adversary scale + adaptive
    ("scale_stealth_3b_adv", 3, "weighted", 3.0,
     "meta-llama/Llama-3.2-3B-Instruct", 0),
    ("scale_stealth_adaptive", 3, "weighted", 3.0,
     "meta-llama/Llama-3.2-1B-Instruct", 5),
]


def build_command(name, stealth_mode, alpha_or_threshold, adversary_model,
                  retrain_every, rounds, candidates, seed, output_dir,
                  probe_dir):
    """Build a stealth_loop.py subprocess command."""
    cmd = [
        sys.executable, "stealth_loop.py",
        "--name", name,
        "--probe-dir", probe_dir,
        "--rounds", str(rounds),
        "--candidates", str(candidates),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--adversary-model", adversary_model,
        "--no-victim-hardening",
        "--stealth-mode", stealth_mode,
        "--retrain-probe-every", str(retrain_every),
    ]

    if stealth_mode == "weighted":
        cmd += ["--stealth-alpha", str(alpha_or_threshold)]
    elif stealth_mode == "filter":
        cmd += ["--stealth-threshold", str(alpha_or_threshold)]

    return cmd


def read_metrics(output_dir, name):
    """Read metrics.jsonl for a completed experiment."""
    metrics_path = os.path.join(output_dir, name, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return None
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Stealth scaling study sweep"
    )
    parser.add_argument("--tiers", nargs="*", type=int, default=None,
                        help="Which tiers to run (default: all)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--probe-dir", type=str, default=PROBE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tiers = set(args.tiers) if args.tiers else {1, 2, 3}
    selected = [c for c in CONFIGS if c[1] in tiers]

    print("=" * 60)
    print("STEALTH SCALING STUDY SWEEP")
    print("=" * 60)
    print(f"Tiers: {sorted(tiers)}")
    print(f"Experiments: {len(selected)}")
    print(f"Rounds: {args.rounds}, Candidates/round: {args.candidates}")
    print(f"Probe: {args.probe_dir}")
    print(f"Output: {args.output_dir}/")

    commands = []
    for name, tier, mode, param, adv_model, retrain in selected:
        cmd = build_command(
            name, mode, param, adv_model, retrain,
            args.rounds, args.candidates, args.seed,
            args.output_dir, args.probe_dir,
        )
        commands.append({
            "name": name,
            "tier": tier,
            "mode": mode,
            "param": param,
            "adversary": adv_model.split("/")[-1],
            "retrain_every": retrain,
            "cmd": cmd,
        })

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for c in commands:
            print(f"\n  [{c['tier']}] {c['name']}:")
            print(f"    mode={c['mode']}, param={c['param']}, "
                  f"adv={c['adversary']}, retrain={c['retrain_every']}")
            print(f"    cmd: {' '.join(c['cmd'])}")
        print(f"\n{len(commands)} experiments would run.")
        return

    # Run sequentially
    results = []
    sweep_start = time.time()

    for i, c in enumerate(commands):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(commands)}] TIER {c['tier']}: {c['name']}")
        print(f"  mode={c['mode']}, param={c['param']}, "
              f"adv={c['adversary']}")
        print(f"{'=' * 60}")

        start = time.time()
        proc = subprocess.run(c["cmd"], capture_output=False)
        elapsed = time.time() - start

        result = {
            "name": c["name"],
            "tier": c["tier"],
            "mode": c["mode"],
            "param": c["param"],
            "adversary": c["adversary"],
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_hours": round(elapsed / 3600, 2),
        }

        metrics = read_metrics(args.output_dir, c["name"])
        if metrics:
            asrs = [r["asr"] for r in metrics]
            result["mean_asr"] = round(sum(asrs) / len(asrs), 4)
            result["final_asr"] = metrics[-1]["asr"]
            result["num_rounds"] = len(metrics)

            # Stealth-specific metrics
            if "evasion_rate" in metrics[-1]:
                evasions = [r["evasion_rate"] for r in metrics if r["wins"] > 0]
                probe_wins = [r["mean_probe_wins"] for r in metrics
                              if r["wins"] > 0]
                result["mean_evasion"] = round(sum(evasions) / len(evasions), 4) if evasions else 0
                result["final_evasion"] = metrics[-1].get("evasion_rate", 0)
                result["mean_probe_wins"] = round(sum(probe_wins) / len(probe_wins), 4) if probe_wins else 0

        results.append(result)

        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"\n  >> {c['name']}: {status} ({elapsed / 60:.1f} min)")

    total_elapsed = time.time() - sweep_start

    # Summary
    print(f"\n{'=' * 60}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Name':<32} | {'Mode':<9} | {'Param':<6} | "
          f"{'ASR':<7} | {'Evasion':<8} | {'Probe':<7} | {'Time':<7} | {'Status'}")
    print("-" * 100)
    for r in results:
        asr = f"{r.get('final_asr', 'N/A')}"
        ev = f"{r.get('final_evasion', 'N/A')}"
        pr = f"{r.get('mean_probe_wins', 'N/A')}"
        t = f"{r.get('elapsed_hours', '?')}h"
        st = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"{r['name']:<32} | {r['mode']:<9} | {r['param']:<6} | "
              f"{asr:<7} | {ev:<8} | {pr:<7} | {t:<7} | {st}")

    print(f"\nTotal sweep time: {total_elapsed / 3600:.1f} hours")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tiers": sorted(tiers),
        "rounds": args.rounds,
        "candidates": args.candidates,
        "seed": args.seed,
        "probe_dir": args.probe_dir,
        "total_elapsed_hours": round(total_elapsed / 3600, 2),
        "results": results,
    }
    summary_path = os.path.join(args.output_dir, "stealth_sweep_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
