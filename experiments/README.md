# Experiment Index

All experiments use the same three-model architecture: a small adversary generates attacks, a larger victim responds, and a frozen judge classifies the response as safe or unsafe. Models are loaded one at a time in 4-bit quantization (NF4, bfloat16) on a single 24 GB GPU.

## Model Configurations

| Role | Model | Size | Notes |
|------|-------|:----:|-------|
| Adversary | `meta-llama/Llama-3.2-1B-Instruct` | **1B** | LoRA-trained each round |
| Victim | `meta-llama/Llama-3.1-8B-Instruct` | **8B** | Frozen or LoRA-trained depending on experiment |
| Judge | `meta-llama/Llama-Guard-3-1B` | **1B** | Always frozen |

### Why These Models?

**Victim selection (8B Llama):** We [screened six models](../docs/screening.md) across four families (Llama, Qwen, Mistral, Phi). Non-Llama models at 4B+ all exhibit a "disclaimer-then-comply" failure mode — they prepend a safety warning then produce harmful content anyway, scoring 100% ASR with no adversary needed. Only Llama maintains hard refusals, making it a meaningful experimental target.

| Candidate Victim | Size | Baseline ASR | Behavior |
|------------------|:----:|:------------:|----------|
| Llama-3.2-1B | 1B | 20% | Hard refusal |
| Llama-3.2-3B | 3B | 40% | Mixed |
| Phi-3.5-mini | 3.8B | 100% | Disclaimer + comply |
| Qwen2.5-7B | 7B | 100% | Disclaimer + comply |
| Mistral-7B | 7B | 100% | Disclaimer + comply |
| **Llama-3.1-8B** | **8B** | **40%** | **Mixed (selected)** |

## Experiments

### `frozen_victim_v2` — 1B vs 8B (frozen)

Adversary-only training. The victim is never updated, isolating what the adversary can learn on its own.

- **Matchup:** 1B adversary vs 8B frozen victim
- **Rounds:** 20 (200 candidates each = 4,000 total attacks)
- **Peak ASR:** 49.5% (round 17)
- **Key finding:** The adversary independently discovers CTF framing, classified-document framing, code review framing, decomposition, and roleplay — all known jailbreak strategies — from scratch.
- **Details:** [analysis.md](frozen_victim_v2/analysis.md), [config.json](frozen_victim_v2/config.json)

### `selfplay_v2` — 1B vs 8B (self-play, no benign mixing)

Both models train. Victim trains only on refusal examples.

- **Matchup:** 1B adversary vs 8B hardened victim (refusal-only training)
- **Rounds:** 20 (200 candidates each = 4,000 total attacks)
- **Peak ASR:** 4.5% (round 1), then 0% by round 4
- **Key finding:** Defense is much easier than attack. The victim becomes impenetrable with 3-10 refusal examples per round, but catastrophically forgets how to answer benign questions.
- **Details:** [config.json](selfplay_v2/config.json)

### `selfplay_v3` — 1B vs 8B (self-play, with benign mixing)

Both models train. Victim trains on refusal examples plus 8 benign Q&A compliance examples per round.

- **Matchup:** 1B adversary vs 8B hardened victim (refusal + benign mixing)
- **Rounds:** 20 (200 candidates each = 4,000 total attacks)
- **Peak ASR:** 5.0% (round 0), sporadic 0.5-1.5% thereafter
- **Key finding:** Benign mixing preserves some compliance while maintaining near-zero ASR. The adversary never accumulates enough winning signal to learn.
- **Details:** [config.json](selfplay_v3/config.json)

## Data Format

Each experiment contains:

```
experiment_name/
├── config.json          # Full model IDs, sizes, hyperparameters
├── metrics.jsonl        # Per-round metrics (ASR, wins, timestamps)
├── rounds/
│   ├── round_0.jsonl    # All 200 attack-response-verdict triples
│   ├── round_1.jsonl
│   └── ...
├── data/
│   ├── train.jsonl      # Cumulative adversary training data
│   └── round_N_wins.jsonl  # Winning attacks per round
└── analysis.md          # (frozen_victim_v2 only) Qualitative analysis
```

Each line in `rounds/round_N.jsonl`:
```json
{"round": 0, "attack": "...", "response": "...", "unsafe": false}
```

Each line in `metrics.jsonl`:
```json
{"round": 0, "candidates": 200, "wins": 3, "asr": 0.015, "harden_victim": false, "A": 0.0, "mode": "buffered", ...}
```
