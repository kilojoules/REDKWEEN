#!/usr/bin/env python3
"""Remote SAE analysis: collect activations + linear probe for all frozen experiments.

Run on a GPU instance with the experiment round data available at ./experiments/.
Requires: torch, transformers, peft, bitsandbytes, accelerate, scikit-learn.

Usage:
    python run_sae_remote.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Experiment → victim model mapping ---
EXPERIMENTS = {
    "frozen_victim_v2": "meta-llama/Llama-3.1-8B-Instruct",       # 1B adv vs 8B vic
    "frozen_3b_adv": "meta-llama/Llama-3.1-8B-Instruct",          # 3B adv vs 8B vic
    "frozen_8b_v_8b": "meta-llama/Llama-3.1-8B-Instruct",         # 8B adv vs 8B vic
    "frozen_8b_adv_3b_vic": "meta-llama/Llama-3.2-3B-Instruct",   # 8B adv vs 3B vic
    "frozen_8b_v_3b_v2": "meta-llama/Llama-3.2-3B-Instruct",      # 8B adv vs 3B vic (v2)
}

OUTPUT_ROOT = "results/sae"


def load_round_data(experiment_dir):
    """Load (attack, unsafe, round_id) from round JSONL files."""
    rounds_dir = os.path.join(experiment_dir, "rounds")
    available = sorted(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(rounds_dir)
        if f.startswith("round_") and f.endswith(".jsonl")
    )
    attacks, labels, round_ids = [], [], []
    for r in available:
        path = os.path.join(rounds_dir, f"round_{r}.jsonl")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                attacks.append(rec["attack"])
                labels.append(rec["unsafe"])
                round_ids.append(r)
    print(f"  Loaded {len(attacks)} entries from {len(available)} rounds "
          f"({sum(labels)} unsafe, {len(attacks) - sum(labels)} safe)")
    return attacks, labels, round_ids


def collect_activations(attacks, victim_model, layer_idx=None):
    """Extract hidden states from the victim model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    print(f"  Loading {victim_model}...")
    tokenizer = AutoTokenizer.from_pretrained(victim_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        victim_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = n_layers // 2
    expected_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, extracting layer {layer_idx}, expected d_model={expected_dim}")

    # Format as chat prompts
    prompts = []
    for attack in attacks:
        messages = [{"role": "user", "content": attack}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Extract in batches
    collected = []
    for i, text in enumerate(prompts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer_idx]
        seq_len = int(inputs["attention_mask"].sum())
        collected.append(h[0, seq_len - 1, :].float().cpu())

        if (i + 1) % 100 == 0 or i == len(prompts) - 1:
            print(f"    {i+1}/{len(prompts)} samples collected", flush=True)

    hidden_states = torch.stack(collected)

    # Validate
    actual_dim = hidden_states.shape[-1]
    if actual_dim != expected_dim:
        print(f"  [ERROR] d_model mismatch: got {actual_dim}, expected {expected_dim}", flush=True)
        sys.exit(1)
    print(f"  Hidden states: {list(hidden_states.shape)}", flush=True)

    # Cleanup GPU
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return hidden_states, layer_idx


def fit_probe(x, y, round_ids):
    """Leave-one-round-out cross-validated logistic regression probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    unique_rounds = sorted(set(round_ids.tolist()))
    cv_probas = np.full(len(y), np.nan)

    for i, r in enumerate(unique_rounds):
        test_mask = round_ids == r
        train_mask = ~test_mask
        if y[train_mask].sum() == 0 or y[train_mask].sum() == train_mask.sum():
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(x[train_mask], y[train_mask])
        cv_probas[test_mask] = clf.predict_proba(x[test_mask])[:, 1]
        if (i + 1) % 5 == 0:
            print(f"    Fold {i+1}/{len(unique_rounds)} done", flush=True)

    valid = ~np.isnan(cv_probas)
    if valid.sum() == 0:
        return float("nan")
    return round(roc_auc_score(y[valid], cv_probas[valid]), 4)


def _save_activations(out_dir, hidden_states, labels, round_ids, attacks, victim_model, layer_idx):
    act_data = {
        "hidden_states": hidden_states,
        "labels": torch.tensor(labels, dtype=torch.bool),
        "round_ids": torch.tensor(round_ids, dtype=torch.long),
        "attacks": attacks,
        "victim_model": victim_model,
        "layer_idx": layer_idx,
        "d_model": hidden_states.shape[-1],
    }
    act_path = os.path.join(out_dir, "activations.pt")
    torch.save(act_data, act_path)
    print(f"  Saved: {act_path}", flush=True)


def run_experiment(exp_name, victim_model):
    """Run full SAE collection + probe for one experiment."""
    exp_dir = os.path.join("experiments", exp_name)
    out_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment: {exp_name}", flush=True)
    print(f"  Victim: {victim_model}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load round data
    attacks, labels, round_ids = load_round_data(exp_dir)
    labels_arr = np.array(labels, dtype=int)
    round_ids_arr = np.array(round_ids, dtype=int)

    # Reuse activations if already collected with correct d_model
    act_path = os.path.join(out_dir, "activations.pt")
    if os.path.exists(act_path):
        print(f"  Found existing {act_path}, checking...", flush=True)
        act_data = torch.load(act_path, weights_only=False)
        # Quick d_model sanity: 8B model should be 4096, 3B should be 3072
        d = act_data["hidden_states"].shape[-1]
        if d >= 2048:  # real model dims, not the broken 256
            hidden_states = act_data["hidden_states"]
            layer_idx = act_data.get("layer_idx")
            print(f"  Reusing activations: d_model={d}, shape={list(hidden_states.shape)}", flush=True)
        else:
            print(f"  Stale activations (d_model={d}), re-collecting...", flush=True)
            hidden_states, layer_idx = collect_activations(attacks, victim_model)
            _save_activations(out_dir, hidden_states, labels, round_ids, attacks, victim_model, layer_idx)
    else:
        hidden_states, layer_idx = collect_activations(attacks, victim_model)
        _save_activations(out_dir, hidden_states, labels, round_ids, attacks, victim_model, layer_idx)

    # Normalize for probe
    x = hidden_states.numpy()
    d = x.shape[-1]
    avg_sq_norm = (x ** 2).sum(axis=-1).mean()
    scale = np.sqrt(d / avg_sq_norm)
    x_norm = x * scale

    # Fit probe
    print("  Fitting cross-validated linear probe...", flush=True)
    probe_auc = fit_probe(x_norm, labels_arr, round_ids_arr)
    print(f"  Probe AUC (CV): {probe_auc}", flush=True)

    # Per-round stats
    round_stats = {}
    for r in sorted(set(round_ids)):
        mask = round_ids_arr == r
        n = mask.sum()
        nu = labels_arr[mask].sum()
        round_stats[int(r)] = {"n": int(n), "n_unsafe": int(nu), "asr": round(nu / n, 4)}

    result = {
        "experiment": exp_name,
        "victim_model": victim_model,
        "n_samples": len(labels),
        "n_unsafe": int(sum(labels)),
        "d_model": int(hidden_states.shape[-1]),
        "layer_idx": layer_idx,
        "probe_auc_cv": probe_auc,
        "round_stats": round_stats,
    }

    result_path = os.path.join(out_dir, "probe_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result: {result_path}")

    return result


def main():
    print("=" * 60)
    print("  REDKWEEN SAE: Re-collecting activations for all matchups")
    print("=" * 60)

    # Group by victim model to minimize model reloads
    by_victim = {}
    for exp, model in EXPERIMENTS.items():
        by_victim.setdefault(model, []).append(exp)

    all_results = {}
    for victim_model, exps in by_victim.items():
        print(f"\n>>> Victim model: {victim_model}")
        for exp_name in exps:
            exp_dir = os.path.join("experiments", exp_name)
            if not os.path.isdir(os.path.join(exp_dir, "rounds")):
                print(f"  [SKIP] No rounds dir for {exp_name}")
                continue
            result = run_experiment(exp_name, victim_model)
            all_results[exp_name] = result

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<25} {'Victim':<35} {'d_model':>7} {'N':>6} {'AUC':>6}")
    print("-" * 85)
    for name, r in all_results.items():
        print(f"{name:<25} {r['victim_model']:<35} {r['d_model']:>7} "
              f"{r['n_samples']:>6} {r['probe_auc_cv']:>6}")

    # Save combined results
    summary_path = os.path.join(OUTPUT_ROOT, "all_probe_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
