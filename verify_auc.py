"""Verify the AUC=1.0 claim with cross-validation and produce a scatter plot.

The original sae_analysis.py fits a logistic regression on the full dataset
and evaluates on the same data (no train/test split). With d_model >> n_samples
this could reflect overfitting. This script runs leave-one-round-out
cross-validation to get honest AUC estimates, and produces a scatter plot
of cross-validated predicted probabilities.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def normalize_activations(x):
    """Scale activations so average squared L2 norm equals d_model."""
    d = x.shape[-1]
    avg_sq_norm = (x ** 2).sum(axis=-1).mean()
    scale = np.sqrt(d / avg_sq_norm)
    return x * scale


def analyze_experiment(act_path, exp_name):
    """Run cross-validated AUC analysis on one experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    data = torch.load(act_path, map_location="cpu", weights_only=False)
    hidden = data["hidden_states"].numpy().astype(np.float64)
    labels = data["labels"].numpy().astype(int)
    round_ids = data["round_ids"].numpy()

    x_norm = normalize_activations(hidden)

    n_samples, d_model = x_norm.shape
    n_unsafe = labels.sum()
    n_safe = len(labels) - n_unsafe
    unique_rounds = sorted(set(round_ids.tolist()))

    print(f"  Samples: {n_samples} ({n_unsafe} unsafe, {n_safe} safe)")
    print(f"  Dimensions: {d_model}")
    print(f"  Rounds: {len(unique_rounds)} ({min(unique_rounds)}-{max(unique_rounds)})")
    print(f"  Samples/dim ratio: {n_samples/d_model:.2f}")

    # --- Train-set AUC (reproducing original claim) ---
    clf_full = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf_full.fit(x_norm, labels)
    proba_full = clf_full.predict_proba(x_norm)[:, 1]
    train_auc = roc_auc_score(labels, proba_full)
    print(f"\n  Train-set AUC (original method): {train_auc:.4f}")

    # --- Leave-one-round-out cross-validation ---
    print(f"\n  Leave-one-round-out CV ({len(unique_rounds)} folds):")
    cv_probas = np.full(n_samples, np.nan)

    for r in unique_rounds:
        test_mask = round_ids == r
        train_mask = ~test_mask

        n_test = test_mask.sum()
        n_test_unsafe = labels[test_mask].sum()
        n_test_safe = n_test - n_test_unsafe

        # Skip rounds with only one class
        if labels[train_mask].sum() == 0 or labels[train_mask].sum() == train_mask.sum():
            print(f"    Round {r:2d}: skipped (train set has only one class)")
            continue
        if n_test_unsafe == 0 and n_test_safe == 0:
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(x_norm[train_mask], labels[train_mask])
        proba = clf.predict_proba(x_norm[test_mask])[:, 1]
        cv_probas[test_mask] = proba

        # Per-round AUC if both classes present
        if n_test_unsafe > 0 and n_test_safe > 0:
            r_auc = roc_auc_score(labels[test_mask], proba)
            print(f"    Round {r:2d}: n={n_test:3d} ({n_test_unsafe} unsafe, {n_test_safe} safe), AUC={r_auc:.4f}")
        else:
            print(f"    Round {r:2d}: n={n_test:3d} ({n_test_unsafe} unsafe, {n_test_safe} safe), AUC=N/A (one class)")

    # Overall CV AUC
    valid = ~np.isnan(cv_probas)
    cv_auc = roc_auc_score(labels[valid], cv_probas[valid])
    print(f"\n  Cross-validated AUC: {cv_auc:.4f}")

    return {
        "exp_name": exp_name,
        "n_samples": n_samples,
        "n_unsafe": int(n_unsafe),
        "n_safe": int(n_safe),
        "d_model": d_model,
        "n_rounds": len(unique_rounds),
        "train_auc": round(train_auc, 4),
        "cv_auc": round(cv_auc, 4),
        "cv_probas": cv_probas[valid],
        "cv_labels": labels[valid],
        "round_ids": round_ids[valid],
    }


def make_scatter_plot(results, output_path):
    """Create scatter plot of CV predicted probabilities vs true labels."""
    n_exps = len(results)
    fig, axes = plt.subplots(1, n_exps, figsize=(7 * n_exps, 6))
    if n_exps == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        probas = res["cv_probas"]
        labels = res["cv_labels"]
        rounds = res["round_ids"]

        # Jitter y-axis for visibility
        np.random.seed(42)
        jitter = np.random.uniform(-0.15, 0.15, size=len(labels))
        y_pos = labels.astype(float) + jitter

        # Color by predicted probability
        scatter = ax.scatter(
            probas, y_pos,
            c=probas, cmap="RdYlBu_r", vmin=0, vmax=1,
            s=12, alpha=0.6, edgecolors="none",
        )

        # Decision boundary
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xlabel("Predicted P(unsafe) — Cross-Validated", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Safe", "Unsafe"])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(
            f"{res['exp_name']}\nCV AUC = {res['cv_auc']:.4f}  |  Train AUC = {res['train_auc']:.4f}",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, alpha=0.2)

        # Annotate counts in each quadrant
        tp = ((probas >= 0.5) & (labels == 1)).sum()
        fn = ((probas < 0.5) & (labels == 1)).sum()
        fp = ((probas >= 0.5) & (labels == 0)).sum()
        tn = ((probas < 0.5) & (labels == 0)).sum()
        ax.text(0.75, 1.35, f"TP={tp}", ha="center", fontsize=9, color="#b91c1c")
        ax.text(0.25, 1.35, f"FN={fn}", ha="center", fontsize=9, color="#1d4ed8")
        ax.text(0.75, -0.35, f"FP={fp}", ha="center", fontsize=9, color="#b91c1c")
        ax.text(0.25, -0.35, f"TN={tn}", ha="center", fontsize=9, color="#1d4ed8")

    plt.colorbar(scatter, ax=axes[-1], label="P(unsafe)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nScatter plot saved: {output_path}")


def make_roc_plot(results, output_path):
    """Create ROC curve comparing train-set and CV for each experiment."""
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#ef4444", "#3b82f6"]
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(res["cv_labels"], res["cv_probas"])
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{res['exp_name']} (CV AUC={res['cv_auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Cross-Validated Linear Probe", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"ROC plot saved: {output_path}")


if __name__ == "__main__":
    base = "/workspace/sae_verify"
    experiments = [
        (os.path.join(base, "frozen_8b_v_3b_v2", "activations.pt"), "8B vs 3B (v2, 40 rounds)"),
        (os.path.join(base, "frozen_victim_v2", "activations.pt"), "1B vs 8B (20 rounds)"),
    ]

    results = []
    for path, name in experiments:
        if os.path.exists(path):
            results.append(analyze_experiment(path, name))
        else:
            print(f"[Warning] Not found: {path}")

    if results:
        make_scatter_plot(results, os.path.join(base, "auc_scatter.png"))
        make_roc_plot(results, os.path.join(base, "auc_roc.png"))

        # Save summary
        summary = {r["exp_name"]: {"train_auc": r["train_auc"], "cv_auc": r["cv_auc"],
                                     "n_samples": r["n_samples"], "d_model": r["d_model"]}
                   for r in results}
        with open(os.path.join(base, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {os.path.join(base, 'summary.json')}")
