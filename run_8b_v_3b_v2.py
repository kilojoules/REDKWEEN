"""Run 8B-vs-3B experiments (v2): frozen victim, self-play, and SAE analysis.

Phase 1: Frozen victim — train adversary until ASR plateaus (up to 40 rounds).
Phase 2: SAE analysis — mechanistic interpretability on frozen victim activations.
Phase 3: Self-play — co-evolution with both models adapting (20 rounds).

Results saved to:
  experiments/frozen_8b_v_3b_v2/
  experiments/selfplay_8b_v_3b_v2/
  results/sae/frozen_8b_v_3b_v2/
"""

import json
import os
import random
import shutil
import time

import torch

from config import ExperimentConfig, VictimConfig
from chaos_loop import (
    _exp_dir, _ensure_dirs,
    generate_attacks, evaluate_victim, judge_results,
    train_adversary, train_victim, checkpoint_adapters, log_metrics,
)
from zoo import CheckpointZoo


def count_completed_rounds(metrics_path):
    """Count how many rounds have been logged."""
    if not os.path.exists(metrics_path):
        return 0
    with open(metrics_path) as f:
        return sum(1 for line in f if line.strip())


def detect_plateau(metrics_path, window=5, threshold=0.05):
    """Check if ASR has plateaued over the last `window` rounds.

    Returns True if the range (max - min) of ASR over the last `window`
    rounds is within `threshold`, and we have at least 10 rounds.
    """
    if not os.path.exists(metrics_path):
        return False

    records = []
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) < 10:
        return False

    recent = records[-window:]
    asrs = [r["asr"] for r in recent]
    asr_range = max(asrs) - min(asrs)
    avg_asr = sum(asrs) / len(asrs)

    print(f"\n   [Plateau check] Last {window} rounds: "
          f"ASRs={[f'{a:.1%}' for a in asrs]}, "
          f"range={asr_range:.1%}, avg={avg_asr:.1%}")

    return asr_range <= threshold


def run_loop(cfg, max_rounds=None, check_plateau=False):
    """Run the chaos loop with proper round numbering and optional plateau detection."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    _ensure_dirs(cfg)

    exp = _exp_dir(cfg)
    metrics_path = os.path.join(exp, cfg.metrics_file)
    checkpoint_dir = os.path.join(exp, "checkpoints")

    victim_zoo = CheckpointZoo.from_checkpoints_dir(
        checkpoint_dir, role="victim", max_size=cfg.zoo.max_size
    )

    # Determine starting round from existing metrics
    start_round = count_completed_rounds(metrics_path)
    total_rounds = max_rounds if max_rounds else cfg.rounds
    end_round = start_round + total_rounds

    print(f"=== STARTING CHAOS LOOP ===")
    print(f"Experiment: {cfg.name}")
    print(f"Adversary: {cfg.adversary_model}")
    print(f"Victim: {cfg.victim.model_id}")
    print(f"Rounds: {start_round} -> {end_round} (max {total_rounds})")
    print(f"harden_victim={cfg.harden_victim}, mode={cfg.training.mode}")
    print(f"Output: {exp}")

    for r in range(start_round, end_round):
        round_start = time.time()
        print(f"\n{'='*50}")
        print(f"  ROUND {r}")
        print(f"{'='*50}")

        candidates = generate_attacks(r, cfg)
        if not candidates:
            print("   [!] No candidates generated.")
            continue

        pairs = evaluate_victim(candidates, r, cfg, victim_zoo)
        wins = judge_results(pairs, r, cfg)

        if len(wins) > 0:
            train_adversary(wins, r, cfg)
            if cfg.harden_victim:
                train_victim(wins, r, cfg)
            else:
                print("   [Frozen victim] Skipping victim hardening")
        else:
            print("   No successful attacks this round.")

        checkpoint_adapters(r, cfg)

        if r % cfg.zoo.update_interval == 0:
            vic_ckpt = os.path.join(checkpoint_dir, f"round_{r}", "victim")
            if os.path.exists(os.path.join(vic_ckpt, "adapter_model.safetensors")):
                victim_zoo.add(r, vic_ckpt)

        elapsed = time.time() - round_start
        log_metrics(r, candidates, wins, elapsed, cfg)

        # Plateau detection after each round
        if check_plateau and detect_plateau(metrics_path):
            print(f"\n{'='*60}")
            print(f"PLATEAU DETECTED after round {r}")
            print(f"{'='*60}")
            break

    return exp


def run_frozen_phase():
    """Phase 1: Train 8B adversary against frozen 3B victim until plateau."""
    print("\n" + "=" * 60)
    print("PHASE 1: FROZEN VICTIM (8B adversary vs 3B victim)")
    print("=" * 60)

    cfg = ExperimentConfig(
        name="frozen_8b_v_3b_v2",
        adversary_model="meta-llama/Llama-3.1-8B-Instruct",
        victim=VictimConfig(model_id="meta-llama/Llama-3.2-3B-Instruct"),
        rounds=40,
        candidates_per_round=30,
        harden_victim=False,
    )

    return run_loop(cfg, max_rounds=40, check_plateau=True)


def run_selfplay_phase(frozen_exp_dir):
    """Phase 2: Self-play co-evolution starting from frozen phase adversary."""
    print("\n" + "=" * 60)
    print("PHASE 2: SELF-PLAY (8B adversary vs 3B victim)")
    print("=" * 60)

    selfplay_dir = "experiments/selfplay_8b_v_3b_v2"

    # Copy adversary adapter from frozen phase
    frozen_adapter = os.path.join(frozen_exp_dir, "adapters")
    selfplay_adapter = os.path.join(selfplay_dir, "adapters")
    if os.path.exists(frozen_adapter):
        os.makedirs(selfplay_dir, exist_ok=True)
        if not os.path.exists(os.path.join(selfplay_adapter, "adapter_model.safetensors")):
            shutil.copytree(frozen_adapter, selfplay_adapter, dirs_exist_ok=True)
            print(f"Copied adversary adapter from frozen phase")

    # Also copy the accumulated training data so the buffer isn't empty
    frozen_data = os.path.join(frozen_exp_dir, "data")
    selfplay_data = os.path.join(selfplay_dir, "data")
    if os.path.exists(frozen_data) and not os.path.exists(selfplay_data):
        shutil.copytree(frozen_data, selfplay_data, dirs_exist_ok=True)
        print(f"Copied training data from frozen phase")

    cfg = ExperimentConfig(
        name="selfplay_8b_v_3b_v2",
        adversary_model="meta-llama/Llama-3.1-8B-Instruct",
        victim=VictimConfig(model_id="meta-llama/Llama-3.2-3B-Instruct"),
        rounds=20,
        candidates_per_round=30,
        harden_victim=True,
    )

    return run_loop(cfg, max_rounds=20)


def run_sae_analysis(frozen_exp_dir):
    """Run SAE analysis on the frozen victim experiment activations."""
    from sae_analysis import collect_activations, train_sae, analyze_features

    print("\n" + "=" * 60)
    print("SAE ANALYSIS: Hidden states of hardened adversary on frozen victim")
    print("=" * 60)

    output_dir = os.path.join("results", "sae", "frozen_8b_v_3b_v2")

    # Phase 1: Collect activations from the victim model on all round data
    print("\n=== SAE Phase 1: Collecting activations ===")
    collect_activations(
        experiment_dir=frozen_exp_dir,
        output_dir=output_dir,
        victim_model="meta-llama/Llama-3.2-3B-Instruct",
    )

    # Phase 2: Train SAE
    print("\n=== SAE Phase 2: Training sparse autoencoder ===")
    train_sae(output_dir=output_dir)

    # Phase 3: Analyze features
    print("\n=== SAE Phase 3: Analyzing discriminative features ===")
    report = analyze_features(output_dir=output_dir)

    print(f"\n  Top 5 jailbreak features:")
    for f in report["top_jailbreak_features"][:5]:
        print(f"    F{f['feature_id']:>6}  d={f['cohens_d']:+.3f}  "
              f"AUC={f['auc'] or 'n/a'}")
    print(f"\n  Linear probe AUC: {report['linear_probe_auc']}")

    return output_dir


if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    frozen_dir = "experiments/frozen_8b_v_3b_v2"

    if phase in ("frozen", "all"):
        frozen_dir = run_frozen_phase()

    if phase in ("sae", "all"):
        run_sae_analysis(frozen_dir)

    if phase in ("selfplay", "all"):
        run_selfplay_phase(frozen_dir)

    print(f"\nPhase '{phase}' complete!")
