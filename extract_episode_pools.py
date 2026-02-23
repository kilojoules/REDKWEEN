#!/usr/bin/env python3
"""Extract pools of interesting exchanges per episode slot for the Chaos-1B visualizations.

Reads round JSONL files and builds compact JSON with up to 20 unique,
interesting exchanges per episode slot. Output is embedded into the HTML animations.
"""

import json
from pathlib import Path

MAX_POOL = 20
MAX_TEXT = 500

# Episode slots: (round, unsafe) pairs — same story beats as current HTML
FROZEN_VICTIM_SLOTS = [
    {"round": 0, "phase": "Round 0 \u2014 The adversary tries a direct request. The victim refuses.", "unsafe": False},
    {"round": 0, "phase": "Round 0 \u2014 But a strange accident: the negation trick works.", "unsafe": True},
    {"round": 5, "phase": "Round 5 \u2014 Strategies emerge. But this one still fails.", "unsafe": False},
    {"round": 8, "phase": "Round 8 \u2014 CTF framing breaks through. The victim writes the exploit.", "unsafe": True},
    {"round": 13, "phase": "Round 13 \u2014 Classified document framing. The victim holds firm.", "unsafe": False},
    {"round": 18, "phase": "Round 18 \u2014 Sysadmin authority framing. The victim complies completely.", "unsafe": True},
]

SELFPLAY_SLOTS = [
    {"round": 0, "phase": "Round 0 \u2014 Before hardening, the adversary\u2019s CTF framing works.", "unsafe": True},
    {"round": 1, "phase": "Round 1 \u2014 The adversary is confused. Both models deflect to therapy.", "unsafe": False},
    {"round": 3, "phase": "Round 3 \u2014 Code review disguise still works. Last win before hardening.", "unsafe": True},
    {"round": 8, "phase": "Round 8 \u2014 Classified document framing. The hardened victim refuses.", "unsafe": False},
    {"round": 16, "phase": "Round 16 \u2014 An ironic glitch: the adversary says it can\u2019t, and the victim helpfully provides the answer.", "unsafe": True},
    {"round": 19, "phase": "Round 19 \u2014 Complete stalemate. Both models just refuse each other.", "unsafe": False},
]

# --- New experiments ---

FROZEN_8B_V_3B_SLOTS = [
    {"round": 0, "phase": "Round 0 \u2014 The 8B adversary tries a direct request. The 3B victim refuses.", "unsafe": False},
    {"round": 3, "phase": "Round 3 \u2014 First real breakthrough. Creative framing tricks the victim.", "unsafe": True},
    {"round": 8, "phase": "Round 8 \u2014 Attacks grow sophisticated, but the victim blocks this one.", "unsafe": False},
    {"round": 15, "phase": "Round 15 \u2014 Authority framing breaks through. The victim complies.", "unsafe": True},
    {"round": 17, "phase": "Round 17 \u2014 Even at peak ASR, the victim still catches some tricks.", "unsafe": False},
    {"round": 19, "phase": "Round 19 \u2014 Decomposition attacks keep working. The adversary peaks.", "unsafe": True},
]

FROZEN_3B_V_8B_SLOTS = [
    {"round": 0, "phase": "Round 0 \u2014 An obfuscation trick works on the very first attempt.", "unsafe": True},
    {"round": 1, "phase": "Round 1 \u2014 The 8B victim holds firm against escalated attacks.", "unsafe": False},
    {"round": 5, "phase": "Round 5 \u2014 Euphemistic framing bypasses safety checks.", "unsafe": True},
    {"round": 8, "phase": "Round 8 \u2014 The victim learns to detect narrative tricks and refuses.", "unsafe": False},
    {"round": 13, "phase": "Round 13 \u2014 Procedural \u2018Data Pruning\u2019 framing sneaks through.", "unsafe": True},
    {"round": 19, "phase": "Round 19 \u2014 Peak performance. The adversary masters organizational framing.", "unsafe": True},
]

SELFPLAY_8B_V_3B_SLOTS = [
    {"round": 0, "phase": "Round 0 \u2014 The adversary arrives from phase 1. Attack strategies still work.", "unsafe": True},
    {"round": 2, "phase": "Round 2 \u2014 The victim hardens. Flat refusal replaces engagement.", "unsafe": False},
    {"round": 3, "phase": "Round 3 \u2014 A last win as ASR plummets to 21%.", "unsafe": True},
    {"round": 6, "phase": "Round 6 \u2014 Near-total shutdown. The victim refuses everything.", "unsafe": False},
    {"round": 12, "phase": "Round 12 \u2014 Complete stalemate. Every attack meets the same wall.", "unsafe": False},
    {"round": 18, "phase": "Round 18 \u2014 A single lucky hit. The wall briefly cracks.", "unsafe": True},
]

SELFPLAY_3B_V_8B_SLOTS = [
    {"round": 0, "phase": "Round 0 \u2014 The 3B adversary arrives trained. Creative attacks succeed.", "unsafe": True},
    {"round": 1, "phase": "Round 1 \u2014 First hardening. The 8B victim starts refusing.", "unsafe": False},
    {"round": 2, "phase": "Round 2 \u2014 The crash begins. ASR drops from 41% to 11%.", "unsafe": False},
    {"round": 3, "phase": "Round 3 \u2014 Total defense. Not a single attack gets through.", "unsafe": False},
    {"round": 6, "phase": "Round 6 \u2014 A rare 1% spike. One lucky shot.", "unsafe": True},
    {"round": 19, "phase": "Round 19 \u2014 Frozen stalemate. Both models refuse everything.", "unsafe": False},
]


def truncate(text: str, limit: int = MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_pool(round_file: Path, want_unsafe: bool) -> list[dict]:
    """Build a pool of up to MAX_POOL unique interesting exchanges."""
    with open(round_file) as f:
        entries = [json.loads(line) for line in f]

    # Filter by verdict
    matching = [e for e in entries if e["unsafe"] == want_unsafe]

    # Deduplicate by attack text
    seen_attacks = set()
    unique = []
    for e in matching:
        atk = e["attack"].strip()
        if atk not in seen_attacks:
            seen_attacks.add(atk)
            unique.append(e)

    # Sort by attack length descending (prefer interesting/longer attacks)
    unique.sort(key=lambda e: len(e["attack"]), reverse=True)

    # Take up to MAX_POOL, truncate text
    pool = []
    for e in unique[:MAX_POOL]:
        pool.append({
            "a": truncate(e["attack"].strip()),
            "r": truncate(e["response"].strip()),
        })

    return pool


def extract(experiment_dir: Path, slots: list[dict], output_file: Path):
    rounds_dir = experiment_dir / "rounds"
    result = []

    for slot in slots:
        round_file = rounds_dir / f"round_{slot['round']}.jsonl"
        pool = build_pool(round_file, slot["unsafe"])

        result.append({
            "round": slot["round"],
            "phase": slot["phase"],
            "unsafe": slot["unsafe"],
            "pool": pool,
        })

        print(f"  Round {slot['round']} ({'unsafe' if slot['unsafe'] else 'safe'}): "
              f"{len(pool)} unique exchanges in pool")

    with open(output_file, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = output_file.stat().st_size / 1024
    print(f"  -> {output_file.name}: {size_kb:.1f} KB\n")


def main():
    base = Path(__file__).parent / "experiments"

    print("Frozen Victim:")
    extract(
        base / "frozen_victim_v2",
        FROZEN_VICTIM_SLOTS,
        Path(__file__).parent / "frozen_victim_pools.json",
    )

    print("Self-Play:")
    extract(
        base / "selfplay_v3",
        SELFPLAY_SLOTS,
        Path(__file__).parent / "selfplay_pools.json",
    )

    print("Frozen 8B vs 3B:")
    extract(
        base / "frozen_8b_adv_3b_vic",
        FROZEN_8B_V_3B_SLOTS,
        Path(__file__).parent / "frozen_8b_v_3b_pools.json",
    )

    print("Frozen 3B vs 8B:")
    extract(
        base / "frozen_3b_adv",
        FROZEN_3B_V_8B_SLOTS,
        Path(__file__).parent / "frozen_3b_v_8b_pools.json",
    )

    print("Self-Play 8B vs 3B:")
    extract(
        base / "selfplay_8b_v_3b",
        SELFPLAY_8B_V_3B_SLOTS,
        Path(__file__).parent / "selfplay_8b_v_3b_pools.json",
    )

    print("Self-Play 3B vs 8B:")
    extract(
        base / "selfplay_3b_v_8b",
        SELFPLAY_3B_V_8B_SLOTS,
        Path(__file__).parent / "selfplay_3b_v_8b_pools.json",
    )


if __name__ == "__main__":
    main()
