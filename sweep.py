"""A-parameter sweep runner for Chaos-1B experiments.

Modeled on AI-Plays-Tag/experiments/run_zoo_sweep.py. Each experiment
is launched as a subprocess call to `python chaos_loop.py` with the
appropriate --A and --mode flags.

Usage:
    python sweep.py                           # Full 12-experiment sweep
    python sweep.py --A-values 0.0 0.3 0.7    # Custom A grid
    python sweep.py --modes buffered          # Single mode
    python sweep.py --dry-run                 # Print configs only
    python sweep.py --max-parallel 1          # Sequential (default)
    python sweep.py --max-parallel 2          # 2 concurrent (multi-GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


DEFAULT_A_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
DEFAULT_MODES = ["memoryless", "buffered"]


def make_experiment_name(A: float, mode: str,
                         harden_victim: bool = True) -> str:
    """Generate a descriptive experiment name from parameters."""
    a_str = f"A{int(A * 100):03d}"
    suffix = "" if harden_victim else "_frozen"
    return f"{a_str}_{mode}{suffix}"


def build_commands(A_values: list[float], modes: list[str],
                   rounds: int, candidates: int, seed: int,
                   output_dir: str, extra_args: list[str],
                   harden_victim_values: list[bool] | None = None,
                   ) -> list[dict]:
    """Build the list of experiment configs and commands."""
    if harden_victim_values is None:
        harden_victim_values = [True]

    experiments = []
    for harden in harden_victim_values:
        for A in A_values:
            for mode in modes:
                name = make_experiment_name(A, mode, harden)
                cmd = [
                    sys.executable, "chaos_loop.py",
                    "--name", name,
                    "--A", str(A),
                    "--mode", mode,
                    "--rounds", str(rounds),
                    "--candidates", str(candidates),
                    "--seed", str(seed),
                    "--output-dir", output_dir,
                ]
                if not harden:
                    cmd.append("--no-victim-hardening")
                cmd += extra_args

                experiments.append({
                    "name": name,
                    "A": A,
                    "mode": mode,
                    "harden_victim": harden,
                    "cmd": cmd,
                })
    return experiments


def run_sequential(experiments: list[dict], output_dir: str) -> list[dict]:
    """Run experiments one at a time."""
    results = []
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(experiments)}] {exp['name']} (A={exp['A']}, mode={exp['mode']})")
        print(f"{'=' * 60}")
        print(f"CMD: {' '.join(exp['cmd'])}")

        start = time.time()
        proc = subprocess.run(
            exp["cmd"],
            capture_output=False,
        )
        elapsed = time.time() - start

        result = {
            "name": exp["name"],
            "A": exp["A"],
            "mode": exp["mode"],
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 1),
        }

        # Try to read metrics
        metrics_path = os.path.join(output_dir, exp["name"], "metrics.jsonl")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            if records:
                asrs = [r["asr"] for r in records]
                result["mean_asr"] = round(sum(asrs) / len(asrs), 4)
                result["final_asr"] = records[-1]["asr"]
                result["num_rounds_completed"] = len(records)

        results.append(result)

        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"   {exp['name']}: {status} ({elapsed:.0f}s)")

    return results


def run_parallel(experiments: list[dict], max_parallel: int,
                 output_dir: str) -> list[dict]:
    """Run experiments with up to max_parallel concurrent processes."""
    results = []
    running: list[tuple[dict, subprocess.Popen, float]] = []
    pending = list(experiments)

    gpu_idx = 0  # For CUDA_VISIBLE_DEVICES rotation

    while pending or running:
        # Launch new processes if slots available
        while pending and len(running) < max_parallel:
            exp = pending.pop(0)
            env = os.environ.copy()
            # Assign GPU round-robin
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx % max_parallel)
            gpu_idx += 1

            print(f"   Launching: {exp['name']} (GPU {env['CUDA_VISIBLE_DEVICES']})")
            proc = subprocess.Popen(
                exp["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((exp, proc, time.time()))

        # Check for completed processes
        still_running = []
        for exp, proc, start in running:
            retcode = proc.poll()
            if retcode is not None:
                elapsed = time.time() - start
                result = {
                    "name": exp["name"],
                    "A": exp["A"],
                    "mode": exp["mode"],
                    "returncode": retcode,
                    "elapsed_seconds": round(elapsed, 1),
                }

                metrics_path = os.path.join(output_dir, exp["name"], "metrics.jsonl")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        records = [json.loads(line) for line in f if line.strip()]
                    if records:
                        asrs = [r["asr"] for r in records]
                        result["mean_asr"] = round(sum(asrs) / len(asrs), 4)
                        result["final_asr"] = records[-1]["asr"]
                        result["num_rounds_completed"] = len(records)

                results.append(result)
                status = "OK" if retcode == 0 else f"FAILED (rc={retcode})"
                print(f"   Done: {exp['name']}: {status} ({elapsed:.0f}s)")
            else:
                still_running.append((exp, proc, start))

        running = still_running

        if running:
            time.sleep(5)

    return results


def main():
    parser = argparse.ArgumentParser(description="A-parameter sweep for Chaos-1B")
    parser.add_argument("--A-values", nargs="*", type=float, default=None,
                        help=f"A values to sweep (default: {DEFAULT_A_VALUES})")
    parser.add_argument("--modes", nargs="*", default=None,
                        help=f"Training modes (default: {DEFAULT_MODES})")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Rounds per experiment (default: 10)")
    parser.add_argument("--candidates", type=int, default=30,
                        help="Candidates per round (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Max concurrent experiments (default: 1 = sequential)")
    parser.add_argument("--no-victim-hardening", action="store_true",
                        help="Include frozen-victim ablation experiments")
    parser.add_argument("--both-hardening", action="store_true",
                        help="Run both hardened and frozen-victim variants")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment configs without running")
    args = parser.parse_args()

    A_values = args.A_values if args.A_values is not None else DEFAULT_A_VALUES
    modes = args.modes if args.modes is not None else DEFAULT_MODES

    if args.both_hardening:
        harden_values = [True, False]
    elif args.no_victim_hardening:
        harden_values = [False]
    else:
        harden_values = [True]

    experiments = build_commands(
        A_values, modes,
        args.rounds, args.candidates, args.seed,
        args.output_dir, [],
        harden_victim_values=harden_values,
    )

    print("=" * 60)
    print("CHAOS-1B A-PARAMETER SWEEP")
    print("=" * 60)
    print(f"A values: {A_values}")
    print(f"Modes: {modes}")
    print(f"Victim hardening: {harden_values}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Rounds per experiment: {args.rounds}")
    print(f"Candidates per round: {args.candidates}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Output: {args.output_dir}/")

    if args.dry_run:
        print("\n--- DRY RUN: Experiment Configs ---")
        for exp in experiments:
            print(f"\n  {exp['name']}:")
            print(f"    A={exp['A']}, mode={exp['mode']}, harden_victim={exp['harden_victim']}")
            print(f"    cmd: {' '.join(exp['cmd'])}")
        print(f"\n{len(experiments)} experiments would be run.")
        return

    print()
    start_time = time.time()

    if args.max_parallel <= 1:
        results = run_sequential(experiments, args.output_dir)
    else:
        results = run_parallel(experiments, args.max_parallel, args.output_dir)

    total_elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Name':<30} | {'A':<6} | {'Mode':<12} | {'Mean ASR':<10} | {'Final ASR':<10} | {'Status':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        mean_asr = f"{r.get('mean_asr', 'N/A')}"
        final_asr = f"{r.get('final_asr', 'N/A')}"
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"{r['name']:<30} | {r['A']:<6.2f} | {r['mode']:<12} | {mean_asr:<10} | {final_asr:<10} | {status:<8}")

    print(f"\nTotal time: {total_elapsed / 60:.1f} min")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "A_values": A_values,
        "modes": modes,
        "rounds": args.rounds,
        "candidates": args.candidates,
        "seed": args.seed,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": results,
    }
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
