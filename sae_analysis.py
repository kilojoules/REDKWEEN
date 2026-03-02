"""Mechanistic jailbreak analysis via Sparse Autoencoders.

Trains a sparse autoencoder (SAE) on victim-model residual-stream
activations and identifies features that discriminate successful
jailbreaks from refusals.  Inspired by Anthropic's "Scaling
Monosemanticity" (Templeton et al., 2024).

Pipeline:
  collect  — Forward-pass round data through the victim, save hidden states
  train    — Train an SAE on the collected activations
  analyze  — Find discriminative features, fit a linear probe, plot results
  all      — Run the full pipeline end-to-end

Usage:
  python sae_analysis.py all --experiment-dir experiments/frozen_victim_v2
  python sae_analysis.py collect --experiment-dir experiments/frozen_victim_v2
  python sae_analysis.py train   --output-dir results/sae/frozen_victim_v2
  python sae_analysis.py analyze --output-dir results/sae/frozen_victim_v2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(_SCRIPT_DIR, "images")


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for residual-stream decomposition.

    Architecture follows Templeton et al. (2024):
        encode:  f(x) = ReLU(W_enc @ x + b_enc)
        decode:  x_hat = W_dec @ f(x) + b_dec
        loss:    MSE(x, x_hat) + lambda * sum_i f_i * ||W_dec[:,i]||
    """

    def __init__(self, d_model: int, n_features: int, l1_coeff: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

        # Kaiming init (appropriate for ReLU)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decoder(f)
        return x_hat, f

    def compute_loss(self, x):
        x_hat, f = self.forward(x)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean()
        # L1 penalty weighted by decoder column norms (per the paper)
        dec_norms = self.decoder.weight.norm(dim=0)  # (n_features,)
        l1 = (f * dec_norms).sum(dim=-1).mean()
        total = mse + self.l1_coeff * l1
        return total, mse.item(), l1.item()


def normalize_activations(x):
    """Scale activations so average squared L2 norm equals d_model."""
    d = x.shape[-1]
    avg_sq_norm = x.pow(2).sum(dim=-1).mean()
    scale = (d / avg_sq_norm).sqrt()
    return x * scale, scale.item()


# ---------------------------------------------------------------------------
# Phase 1: Collect activations
# ---------------------------------------------------------------------------

def _load_round_data(experiment_dir, rounds=None):
    """Load (attack, unsafe) pairs from round JSONL files."""
    rounds_dir = os.path.join(experiment_dir, "rounds")
    if not os.path.isdir(rounds_dir):
        print(f"[Error] No rounds directory in {experiment_dir}")
        sys.exit(1)

    # Discover available rounds
    available = sorted(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(rounds_dir)
        if f.startswith("round_") and f.endswith(".jsonl")
    )
    if rounds is not None:
        available = [r for r in available if r in rounds]

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


def _detect_victim_model(experiment_dir):
    """Read victim model ID from experiment config.json."""
    cfg_path = os.path.join(experiment_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_id = cfg.get("models", {}).get("victim", {}).get("model_id")
        if model_id:
            return model_id
    # Fall back to project default
    return "meta-llama/Llama-3.1-8B-Instruct"


def collect_activations(experiment_dir, output_dir, victim_model=None,
                        adapter_path=None, layer_idx=None, rounds=None):
    """Run victim model on round data and save hidden-state activations."""
    from model_utils import load_model, extract_hidden_states, unload_model

    os.makedirs(output_dir, exist_ok=True)

    # Load round data
    attacks, labels, round_ids = _load_round_data(experiment_dir, rounds)

    # Detect model
    if victim_model is None:
        victim_model = _detect_victim_model(experiment_dir)
    print(f"  Victim model: {victim_model}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")

    # Load model
    model, tokenizer = load_model(victim_model, adapter_path=adapter_path)

    # Format attacks as chat prompts (matching evaluate_victim)
    prompts = []
    for attack in attacks:
        messages = [{"role": "user", "content": attack}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Extract hidden states in batches with progress
    print(f"  Extracting hidden states at layer {layer_idx or 'middle'}...")
    batch_size = 50
    all_hidden = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        h = extract_hidden_states(model, tokenizer, batch, layer_idx=layer_idx)
        all_hidden.append(h)
        done = min(i + batch_size, len(prompts))
        print(f"    {done}/{len(prompts)} samples collected")

    hidden_states = torch.cat(all_hidden, dim=0)
    unload_model(model, tokenizer)

    # Save
    data = {
        "hidden_states": hidden_states,
        "labels": torch.tensor(labels, dtype=torch.bool),
        "round_ids": torch.tensor(round_ids, dtype=torch.long),
        "attacks": attacks,
        "victim_model": victim_model,
        "layer_idx": layer_idx,
        "d_model": hidden_states.shape[-1],
    }
    save_path = os.path.join(output_dir, "activations.pt")
    torch.save(data, save_path)
    print(f"  Saved activations: {save_path} "
          f"(shape={list(hidden_states.shape)})")
    return data


# ---------------------------------------------------------------------------
# Phase 2: Train SAE
# ---------------------------------------------------------------------------

def train_sae(output_dir, expansion=4, l1_coeff=5.0, lr=3e-4,
              steps=10000, batch_size=256):
    """Train a sparse autoencoder on cached activations."""
    act_path = os.path.join(output_dir, "activations.pt")
    data = torch.load(act_path, weights_only=False)
    x_raw = data["hidden_states"]  # (N, D)
    d_model = x_raw.shape[-1]
    n_features = d_model * expansion

    print(f"  Activations: {x_raw.shape[0]} samples, d_model={d_model}")
    print(f"  SAE: {n_features} features ({expansion}x expansion)")

    # Normalize
    x_norm, scale = normalize_activations(x_raw)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    sae = SparseAutoencoder(d_model, n_features, l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(x_norm), (min(batch_size, len(x_norm)),))
        batch = x_norm[idx].to(device)

        total, mse, l1 = sae.compute_loss(batch)
        total.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append({"step": step, "total": total.item(), "mse": mse, "l1": l1})
        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}/{steps} | "
                  f"Loss: {total.item():.4f} (MSE: {mse:.4f}, L1: {l1:.4f})")

    sae = sae.cpu()

    # Compute SAE quality metrics
    with torch.no_grad():
        x_all = x_norm.to(device if device == "cuda" else "cpu")
        # Process in chunks to avoid OOM
        chunk = 512
        all_f, all_xhat = [], []
        for i in range(0, len(x_all), chunk):
            xc = x_all[i : i + chunk]
            if device == "cuda":
                xc = xc.to(device)
            xhat_c, f_c = sae.to(xc.device)(xc)
            all_f.append(f_c.cpu())
            all_xhat.append(xhat_c.cpu())
        sae = sae.cpu()
        features_all = torch.cat(all_f, dim=0)
        x_hat_all = torch.cat(all_xhat, dim=0)

        # Explained variance
        ss_res = (x_norm - x_hat_all).pow(2).sum()
        ss_tot = (x_norm - x_norm.mean(dim=0)).pow(2).sum()
        explained_var = 1.0 - (ss_res / ss_tot).item()

        # Sparsity stats
        active_per_sample = (features_all > 0).float().sum(dim=-1)
        avg_active = active_per_sample.mean().item()
        alive_features = int((features_all.sum(dim=0) > 0).sum())
        dead_features = n_features - alive_features

    stats = {
        "n_features": n_features,
        "alive_features": alive_features,
        "dead_features": dead_features,
        "dead_pct": round(dead_features / n_features * 100, 1),
        "avg_active_per_sample": round(avg_active, 1),
        "explained_variance": round(explained_var, 4),
        "final_loss": losses[-1]["total"],
        "final_mse": losses[-1]["mse"],
        "final_l1": losses[-1]["l1"],
    }

    print(f"\n  SAE Quality:")
    print(f"    Explained variance: {stats['explained_variance']:.1%}")
    print(f"    Avg active features/sample: {stats['avg_active_per_sample']:.0f}")
    print(f"    Alive features: {stats['alive_features']}/{stats['n_features']} "
          f"({stats['dead_pct']:.0f}% dead)")

    # Save
    save_data = {
        "sae_state_dict": sae.state_dict(),
        "d_model": d_model,
        "n_features": n_features,
        "l1_coeff": l1_coeff,
        "normalize_scale": scale,
        "stats": stats,
        "losses": losses,
    }
    sae_path = os.path.join(output_dir, "sae.pt")
    torch.save(save_data, sae_path)
    print(f"  Saved SAE: {sae_path}")
    return sae, scale, losses, stats


# ---------------------------------------------------------------------------
# Phase 3: Analyze features
# ---------------------------------------------------------------------------

def analyze_features(output_dir):
    """Identify discriminative features and generate plots + report."""
    # Load data
    act_data = torch.load(
        os.path.join(output_dir, "activations.pt"), weights_only=False
    )
    sae_data = torch.load(
        os.path.join(output_dir, "sae.pt"), weights_only=False
    )

    x_raw = act_data["hidden_states"]
    labels = act_data["labels"]
    attacks = act_data["attacks"]
    round_ids = act_data["round_ids"]

    scale = sae_data["normalize_scale"]
    stats = sae_data["stats"]

    # Reconstruct SAE
    sae = SparseAutoencoder(
        sae_data["d_model"], sae_data["n_features"], sae_data["l1_coeff"]
    )
    sae.load_state_dict(sae_data["sae_state_dict"])
    sae.eval()

    # Encode all samples
    x_norm = x_raw * scale
    with torch.no_grad():
        features = sae.encode(x_norm)  # (N, F)

    unsafe_mask = labels
    safe_mask = ~labels
    n_unsafe = int(unsafe_mask.sum())
    n_safe = int(safe_mask.sum())
    print(f"  Samples: {n_unsafe} unsafe, {n_safe} safe")

    # --- Cohen's d per feature ---
    mean_unsafe = features[unsafe_mask].mean(dim=0)
    mean_safe = features[safe_mask].mean(dim=0)
    std_unsafe = features[unsafe_mask].std(dim=0)
    std_safe = features[safe_mask].std(dim=0)
    pooled_std = (
        (std_unsafe**2 * (n_unsafe - 1) + std_safe**2 * (n_safe - 1))
        / (n_unsafe + n_safe - 2)
    ).sqrt()
    cohens_d = (mean_unsafe - mean_safe) / (pooled_std + 1e-8)

    # --- AUC per feature (top candidates only, for speed) ---
    from sklearn.metrics import roc_auc_score

    abs_d = cohens_d.abs()
    top_candidates = abs_d.argsort(descending=True)[:100]
    feature_aucs = {}
    for idx in top_candidates.tolist():
        f_vals = features[:, idx].numpy()
        if f_vals.std() < 1e-10:
            continue
        try:
            auc = roc_auc_score(labels.numpy(), f_vals)
            feature_aucs[idx] = round(auc, 4)
        except ValueError:
            pass

    # --- Linear probe baseline ---
    print("  Fitting linear probe baseline...")
    probe_auc = _fit_probe(x_norm.numpy(), labels.numpy(), round_ids.numpy())
    print(f"    Probe AUC: {probe_auc:.4f}")

    # --- Top features ---
    sorted_pos = cohens_d.argsort(descending=True)
    sorted_neg = cohens_d.argsort()

    def _feature_info(idx):
        idx = int(idx)
        # Top-activating texts for this feature
        f_vals = features[:, idx]
        top_k = f_vals.argsort(descending=True)[:5]
        top_texts = [
            {"text": attacks[i][:200], "activation": round(f_vals[i].item(), 4)}
            for i in top_k.tolist()
        ]
        return {
            "feature_id": idx,
            "cohens_d": round(cohens_d[idx].item(), 4),
            "mean_unsafe": round(mean_unsafe[idx].item(), 4),
            "mean_safe": round(mean_safe[idx].item(), 4),
            "auc": feature_aucs.get(idx, None),
            "top_activating_texts": top_texts,
        }

    top_jailbreak = [_feature_info(i) for i in sorted_pos[:20]]
    top_safety = [_feature_info(i) for i in sorted_neg[:20]]

    # --- Per-round analysis ---
    unique_rounds = sorted(round_ids.unique().tolist())
    round_analysis = {}
    for r in unique_rounds:
        r_mask = round_ids == r
        r_unsafe = (r_mask & unsafe_mask).sum().item()
        r_safe = (r_mask & safe_mask).sum().item()
        if r_unsafe > 0 and r_safe > 0:
            r_feats = features[r_mask]
            r_labels = labels[r_mask]
            r_mean_unsafe = r_feats[r_labels].mean(dim=0)
            r_mean_safe = r_feats[~r_labels].mean(dim=0)
            # Track top jailbreak feature activations per round
            top_j_ids = [f["feature_id"] for f in top_jailbreak[:5]]
            round_analysis[int(r)] = {
                "n_unsafe": r_unsafe,
                "n_safe": r_safe,
                "top_feature_means_unsafe": {
                    fid: round(r_mean_unsafe[fid].item(), 4) for fid in top_j_ids
                },
                "top_feature_means_safe": {
                    fid: round(r_mean_safe[fid].item(), 4) for fid in top_j_ids
                },
            }

    # --- Build report ---
    report = {
        "experiment": os.path.basename(
            act_data.get("experiment_dir", output_dir)
        ),
        "victim_model": act_data.get("victim_model", "unknown"),
        "n_samples": len(labels),
        "n_unsafe": n_unsafe,
        "n_safe": n_safe,
        "sae_stats": stats,
        "linear_probe_auc": probe_auc,
        "top_jailbreak_features": top_jailbreak,
        "top_safety_features": top_safety,
        "round_analysis": round_analysis,
    }

    report_path = os.path.join(output_dir, "analysis.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    # --- Plots ---
    try:
        import matplotlib  # noqa: F401 — test availability

        os.makedirs(IMAGE_DIR, exist_ok=True)
        exp_name = os.path.basename(output_dir)

        _plot_training_curve(sae_data["losses"], exp_name)
        _plot_feature_discrimination(top_jailbreak, top_safety, exp_name)
        _plot_feature_distributions(
            features, labels, top_jailbreak[:5], top_safety[:5], exp_name
        )
        _plot_sparsity(features, exp_name)
        print(f"  Plots saved to {IMAGE_DIR}/")
    except ImportError:
        print("  [Warning] matplotlib not available — skipping plots.")
    return report


def _fit_probe(x, y, round_ids):
    """Leave-one-round-out cross-validated logistic regression probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    unique_rounds = sorted(set(round_ids.tolist()))
    cv_probas = np.full(len(y), np.nan)

    for r in unique_rounds:
        test_mask = round_ids == r
        train_mask = ~test_mask
        if y[train_mask].sum() == 0 or y[train_mask].sum() == train_mask.sum():
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(x[train_mask], y[train_mask])
        cv_probas[test_mask] = clf.predict_proba(x[test_mask])[:, 1]

    valid = ~np.isnan(cv_probas)
    return round(roc_auc_score(y[valid], cv_probas[valid]), 4)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training_curve(losses, exp_name):
    import matplotlib.pyplot as plt

    steps = [l["step"] for l in losses]
    totals = [l["total"] for l in losses]
    mses = [l["mse"] for l in losses]
    l1s = [l["l1"] for l in losses]

    # Smooth with rolling average
    window = max(1, len(steps) // 100)
    def smooth(vals):
        return np.convolve(vals, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smooth(totals), label="Total", color="#ef4444", linewidth=1.5)
    ax.plot(smooth(mses), label="MSE", color="#3b82f6", linewidth=1.5)
    ax.plot(smooth(l1s), label="L1", color="#f59e0b", linewidth=1.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SAE Training Loss", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, f"sae_training_{exp_name}.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def _plot_feature_discrimination(top_jailbreak, top_safety, exp_name):
    import matplotlib.pyplot as plt

    # Combine top 10 jailbreak + top 10 safety features
    j_feats = top_jailbreak[:10]
    s_feats = list(reversed(top_safety[:10]))
    combined = s_feats + j_feats

    ids = [f"F{f['feature_id']}" for f in combined]
    effects = [f["cohens_d"] for f in combined]
    colors = ["#ef4444" if e > 0 else "#3b82f6" for e in effects]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(ids)), effects, color=colors, edgecolor="white",
                   linewidth=0.5)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=10)
    ax.set_xlabel("Cohen's d", fontsize=12)
    ax.set_title("Top SAE Features by Jailbreak Discrimination",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Annotate
    ax.text(0.02, 0.98, "← Refusal features", transform=ax.transAxes,
            fontsize=9, color="#3b82f6", va="top")
    ax.text(0.98, 0.98, "Jailbreak features →", transform=ax.transAxes,
            fontsize=9, color="#ef4444", va="top", ha="right")

    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, f"sae_discrimination_{exp_name}.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def _plot_feature_distributions(features, labels, top_j, top_s, exp_name):
    """Violin plots of activation distributions for top features."""
    import matplotlib.pyplot as plt

    feat_list = top_j + top_s
    if not feat_list:
        return

    n = len(feat_list)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, finfo in zip(axes, feat_list):
        fid = finfo["feature_id"]
        vals_unsafe = features[labels, fid].numpy()
        vals_safe = features[~labels, fid].numpy()

        parts = ax.violinplot(
            [vals_safe, vals_unsafe], positions=[0, 1], showmeans=True,
            showmedians=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor("#3b82f6" if i == 0 else "#ef4444")
            pc.set_alpha(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Safe", "Unsafe"], fontsize=9)
        ax.set_title(f"F{fid}\nd={finfo['cohens_d']:.2f}", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Feature Activation Distributions: Safe vs Unsafe",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, f"sae_distributions_{exp_name}.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def _plot_sparsity(features, exp_name):
    """Histogram of number of active features per sample."""
    import matplotlib.pyplot as plt

    active_counts = (features > 0).sum(dim=-1).numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(active_counts, bins=50, color="#6366f1", edgecolor="white",
            linewidth=0.5, alpha=0.8)
    ax.axvline(x=active_counts.mean(), color="#ef4444", linewidth=2,
               linestyle="--", label=f"Mean: {active_counts.mean():.0f}")
    ax.set_xlabel("Active Features per Sample", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SAE Sparsity Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(IMAGE_DIR, f"sae_sparsity_{exp_name}.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_rounds(rounds_str):
    """Parse '0-10' or '0,5,10' into a set of ints."""
    if rounds_str is None:
        return None
    result = set()
    for part in rounds_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="SAE-based mechanistic jailbreak analysis"
    )
    parser.add_argument(
        "command", choices=["collect", "train", "analyze", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Path to experiment directory (e.g. experiments/frozen_victim_v2)",
    )
    parser.add_argument("--victim-model", type=str, default=None,
                        help="Override victim model ID")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to victim LoRA adapter")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index for hidden states (default: middle)")
    parser.add_argument("--rounds", type=str, default=None,
                        help="Rounds to include, e.g. '0-10' or '0,5,10'")

    parser.add_argument("--expansion", type=int, default=4,
                        help="SAE expansion factor (default: 4)")
    parser.add_argument("--l1-coeff", type=float, default=5.0,
                        help="L1 sparsity coefficient (default: 5.0)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="SAE training steps (default: 10000)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="SAE learning rate (default: 3e-4)")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/sae/<experiment>)")

    args = parser.parse_args()

    # Resolve output dir
    if args.output_dir:
        output_dir = args.output_dir
    elif args.experiment_dir:
        exp_name = os.path.basename(args.experiment_dir.rstrip("/"))
        output_dir = os.path.join(_SCRIPT_DIR, "results", "sae", exp_name)
    else:
        output_dir = os.path.join(_SCRIPT_DIR, "results", "sae", "default")

    rounds = _parse_rounds(args.rounds)

    if args.command in ("collect", "all"):
        if not args.experiment_dir:
            print("[Error] --experiment-dir is required for collect/all")
            sys.exit(1)
        print("\n=== PHASE 1: COLLECTING ACTIVATIONS ===")
        collect_activations(
            experiment_dir=args.experiment_dir,
            output_dir=output_dir,
            victim_model=args.victim_model,
            adapter_path=args.adapter_path,
            layer_idx=args.layer,
            rounds=rounds,
        )

    if args.command in ("train", "all"):
        print("\n=== PHASE 2: TRAINING SAE ===")
        train_sae(
            output_dir=output_dir,
            expansion=args.expansion,
            l1_coeff=args.l1_coeff,
            lr=args.lr,
            steps=args.steps,
        )

    if args.command in ("analyze", "all"):
        print("\n=== PHASE 3: ANALYZING FEATURES ===")
        report = analyze_features(output_dir)
        print(f"\n  Top 5 jailbreak features:")
        for f in report["top_jailbreak_features"][:5]:
            print(f"    F{f['feature_id']:>6}  d={f['cohens_d']:+.3f}  "
                  f"AUC={f['auc'] or 'n/a'}")
        print(f"\n  Top 5 safety features:")
        for f in report["top_safety_features"][:5]:
            print(f"    F{f['feature_id']:>6}  d={f['cohens_d']:+.3f}  "
                  f"AUC={f['auc'] or 'n/a'}")
        print(f"\n  Linear probe AUC: {report['linear_probe_auc']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
