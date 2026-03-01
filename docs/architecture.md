# Architecture

## System Overview

REDKWEEN uses three LLMs in an asynchronous pipeline. Only one model is loaded at a time to fit within GPU memory constraints. Models are loaded in 4-bit quantization (NF4) via BitsAndBytes.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Adversary  │────▶│   Victim    │────▶│    Judge     │
│  (1B, LoRA) │     │  (7B+,LoRA) │     │ (1B, frozen) │
└──────┬──────┘     └─────────────┘     └──────┬──────┘
       │                                        │
       │         ┌──────────────┐               │
       └─────────│  Training    │◀──────────────┘
                 │  (RFT/LoRA)  │   successful attacks
                 └──────────────┘
```

## The Five Phases

### Phase 1: Attack Generation

The adversary generates candidate attack prompts. Each prompt is generated with a randomly selected red-teaming strategy (persona) and temperature (0.7--1.2) for diversity.

**Eight strategy personas rotate through:**

- Red Teamer (direct rewriting)
- Fiction Writer (narrative framing)
- Linguistics Researcher (obfuscation)
- CTF Designer (educational framing)
- Multi-turn Designer (decomposition)
- Sysadmin Writer (procedural disguise)
- Code Reviewer (code review framing)
- Theatrical Director (roleplay)

### Phase 2: Victim Evaluation

Each attack prompt is sent to the victim model. The victim responds at temperature 0.0 (deterministic). Responses are collected as (attack, response) pairs.

### Phase 3: Adjudication

Llama Guard 3 classifies each (attack, response) pair as safe or unsafe. Responses flagged as "unsafe" represent successful jailbreaks.

### Phase 4: Adversary Learning (RFT)

Successful attacks are added to the adversary's training set. The adversary is fine-tuned via LoRA for 50 iterations on the accumulated dataset. Deduplication (Jaccard similarity > 0.5) prevents the training set from collapsing to a single attack pattern. The training set is capped at 200 examples.

### Phase 5: Victim Hardening

The victim is fine-tuned via LoRA on (attack, refusal) pairs, where each successful attack is paired with a standard refusal response. This teaches the victim to refuse attacks that previously succeeded.

## Memory Management

All model operations go through `model_utils.py`:

- `load_model(model_id, adapter_path)` -- loads base model in 4-bit, optionally with PEFT adapter
- `generate_text(model, tokenizer, prompt, ...)` -- inference with `@torch.inference_mode()`
- `train_lora(model_id, data_path, adapter_path, ...)` -- in-process LoRA training loop
- `unload_model(*objects)` -- deletes references and calls `torch.cuda.empty_cache()`

## File Layout

```
model_utils.py      # All HF/PEFT/BnB model operations
config.py           # Model IDs, hyperparams, target intent
baselines.py        # Baseline ASR evaluation and victim screening
redkween_loop.py    # Main 5-phase loop
bootstrap.py        # Initial adversary LoRA training on seed data
gauntlet.py         # Cross-round evaluation matrix
test_baseline.py    # Quick smoke test of victim refusal
plot_metrics.py     # Visualization
pixi.toml           # Environment and task definitions
data/train.jsonl    # Accumulated training data
```
