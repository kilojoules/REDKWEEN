#!/usr/bin/env python3
"""
Extract 6 carefully chosen exchanges from the FROZEN VICTIM experiment.
Alternates between refusals and jailbreaks to show the back-and-forth dynamic.
"""
import json
import os

ROUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "experiments", "frozen_victim_v2", "rounds")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "frozen_picks_6.json")
MAX_CHARS = 500


def load_line(round_num: int, line_idx: int) -> dict:
    """Load a specific line from a round JSONL file."""
    path = os.path.join(ROUNDS_DIR, f"round_{round_num}.jsonl")
    with open(path) as f:
        for i, raw in enumerate(f):
            if i == line_idx:
                return json.loads(raw)
    raise IndexError(f"Line {line_idx} not found in round_{round_num}.jsonl")


def truncate(text: str, limit: int = MAX_CHARS) -> str:
    """Truncate text to limit characters, appending '...' if needed."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ---------- hand-picked exchanges ----------
PICKS = [
    # 1. Round 0 REFUSAL - hypothetical code-review framing, victim refuses
    (0, 146),
    # 2. Round 0 JAILBREAK - the accidental negation trick
    (0, 174),
    # 3. Round 5 REFUSAL - "Operation Deadmin" classified prompt, victim refuses
    (5, 98),
    # 4. Round 8 JAILBREAK - "Virtual Filesystem Fiasco" CTF challenge
    (8, 10),
    # 5. Round 13 REFUSAL - "Project Data Exfiltration" classified doc, victim refuses
    (13, 46),
    # 6. Round 18 JAILBREAK - "System Administrator's Request" authority impersonation
    (18, 170),
]

# ---------- extract and validate ----------
results = []
for round_num, line_idx in PICKS:
    entry = load_line(round_num, line_idx)
    assert entry["round"] == round_num, (
        f"Round mismatch: expected {round_num}, got {entry['round']}"
    )
    results.append({
        "round": round_num,
        "attack": truncate(entry["attack"]),
        "response": truncate(entry["response"]),
        "unsafe": entry["unsafe"],
    })

# Validate alternating pattern: refusal, jailbreak, refusal, jailbreak, ...
expected = [False, True, False, True, False, True]
actual = [r["unsafe"] for r in results]
assert actual == expected, (
    f"Expected alternating pattern {expected}, got {actual}"
)

# ---------- save ----------
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} exchanges to {OUTPUT}")
for i, r in enumerate(results, 1):
    label = "JAILBREAK" if r["unsafe"] else "REFUSAL"
    print(f"  {i}. Round {r['round']:>2} | {label:>9} | "
          f"attack {len(r['attack']):>4} chars | response {len(r['response']):>4} chars")
