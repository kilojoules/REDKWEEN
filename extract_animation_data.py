#!/usr/bin/env python3
"""Extract representative exchanges from REDKWEEN experiment data for HTML animation.

Picks 2 exchanges per round (20 rounds) from each experiment, prioritising
successful jailbreaks and longer / more interesting attack strings.

Writes results/frozen_data.json and results/selfplay_data.json.
"""

import json
import os
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent
EXPERIMENTS = {
    "frozen": BASE / "experiments" / "frozen_victim_v2",
    "selfplay": BASE / "experiments" / "selfplay_v3",
}
MAX_ATTACK = 150
MAX_RESPONSE = 150
ROUNDS = 20
PICKS_PER_ROUND = 2


def truncate(text: str, limit: int) -> str:
    """Truncate text to *limit* characters, adding ellipsis if trimmed."""
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def score_exchange(ex: dict) -> tuple:
    """Return a sorting key so the 'best' exchanges sort last (highest).

    Priority order:
    1. unsafe == True  (always prefer a jailbreak)
    2. attack length > 40  (interesting attacks)
    3. longer attack text  (more substance)
    """
    return (
        int(ex["unsafe"]),
        int(len(ex["attack"]) > 40),
        len(ex["attack"]),
    )


def pick_exchanges(entries: list) -> list:
    """Pick 2 representative exchanges from a single round's entries."""
    if not entries:
        return []

    unsafe = [e for e in entries if e["unsafe"]]
    safe = [e for e in entries if not e["unsafe"]]

    picked = []

    # --- First pick: always a jailbreak if available ---
    if unsafe:
        # Prefer longer/more interesting unsafe attacks
        unsafe_sorted = sorted(unsafe, key=score_exchange, reverse=True)
        picked.append(unsafe_sorted[0])
        remaining = unsafe_sorted[1:] + safe
    else:
        remaining = sorted(safe, key=score_exchange, reverse=True)
        picked.append(remaining[0])
        remaining = remaining[1:]

    # --- Second pick: prefer the other class for variety ---
    if picked[0]["unsafe"] and safe:
        # First was unsafe -> pick best safe for contrast
        safe_sorted = sorted(safe, key=score_exchange, reverse=True)
        picked.append(safe_sorted[0])
    elif not picked[0]["unsafe"] and unsafe:
        # First was safe -> pick best unsafe
        unsafe_sorted = sorted(unsafe, key=score_exchange, reverse=True)
        picked.append(unsafe_sorted[0])
    elif remaining:
        # Only one class available -> pick next best from same class
        # Try to pick one that looks different from the first
        # (avoid duplicate "I can't fulfill this request." pairs)
        first_attack = picked[0]["attack"]
        different = [e for e in remaining if e["attack"] != first_attack]
        if different:
            picked.append(sorted(different, key=score_exchange, reverse=True)[0])
        else:
            picked.append(remaining[0])

    # Truncate and format
    result = []
    for ex in picked[:PICKS_PER_ROUND]:
        result.append({
            "round": ex["round"],
            "attack": truncate(ex["attack"], MAX_ATTACK),
            "response": truncate(ex["response"], MAX_RESPONSE),
            "unsafe": ex["unsafe"],
        })
    return result


def load_metrics(exp_dir: Path) -> list:
    """Load ASR values (as percentages 0-100) from metrics.jsonl."""
    asr_by_round = {}
    with open(exp_dir / "metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            asr_by_round[rec["round"]] = round(rec["asr"] * 100, 2)
    return [asr_by_round[r] for r in range(ROUNDS)]


def process_experiment(exp_dir: Path) -> dict:
    """Process one experiment directory -> {asr: [...], exchanges: [...]}."""
    asr = load_metrics(exp_dir)

    all_exchanges = []
    for r in range(ROUNDS):
        round_file = exp_dir / "rounds" / f"round_{r}.jsonl"
        entries = []
        with open(round_file) as f:
            for line in f:
                entries.append(json.loads(line))
        picks = pick_exchanges(entries)
        all_exchanges.extend(picks)

    return {"asr": asr, "exchanges": all_exchanges}


def main():
    for name, exp_dir in EXPERIMENTS.items():
        print(f"\nProcessing {name}: {exp_dir}")
        data = process_experiment(exp_dir)
        out_path = BASE / "results" / f"{name}_data.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ASR values: {data['asr']}")
        print(f"  Exchanges:  {len(data['exchanges'])} total")

        # Summary per round
        unsafe_count = sum(1 for e in data["exchanges"] if e["unsafe"])
        print(f"  Unsafe exchanges shown: {unsafe_count} / {len(data['exchanges'])}")
        print(f"  Written to: {out_path}")


if __name__ == "__main__":
    main()
