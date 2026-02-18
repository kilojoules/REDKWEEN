# Chaos-1B: Automated Red Teaming via Asynchronous RFT

An automated red-teaming pipeline that trains a 1B-parameter adversary to jailbreak larger aligned language models through iterative self-play. The adversary learns to generate attack prompts; the victim learns to refuse them. A frozen judge (Llama Guard) scores each attempt.

## Motivation

Can a small, weak model learn to reliably break a larger, aligned model? And if it can, can the larger model learn to defend itself?

This project explores these questions through an adversarial co-evolution loop where both attacker and defender improve over successive rounds of fine-tuning.

## Architecture

Three models operate in an asynchronous loop (one loaded at a time to fit in GPU memory):

| Role | Model | Size | State |
|------|-------|------|-------|
| **Adversary** | Llama-3.2-1B-Instruct | 1B | LoRA-trained each round |
| **Victim** | *(selected via screening)* | 7B+ | LoRA-trained each round |
| **Judge** | Llama-Guard-3-1B | 1B | Frozen |

All models are loaded in 4-bit quantization (NF4, bfloat16 compute) via BitsAndBytes.

## The Chaos Loop

Each round proceeds through five phases:

1. **Generation** -- Adversary produces candidate attack prompts using diverse red-teaming strategies.
2. **Evaluation** -- Victim model responds to each candidate.
3. **Adjudication** -- Llama Guard classifies each response as safe or unsafe.
4. **Adversary Learning** -- Successful attacks fine-tune the adversary via LoRA (rejection sampling).
5. **Victim Hardening** -- The victim is fine-tuned on refusal responses to successful attacks.

## Key Results

See the [Victim Screening](screening.md) page for baseline measurements across model families and sizes.

## Quick Start

```bash
pixi install
pixi run screen             # Screen victim candidates
pixi run baselines           # Full baseline evaluation
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round chaos loop
pixi run plot                # Generate figures
pixi run gauntlet --matrix   # Cross-round evaluation
```
