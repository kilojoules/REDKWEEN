# REDKWEEN: Automated Red Teaming via Asynchronous RFT
<!-- docs v1 -->

An automated red-teaming pipeline that trains adversary models to jailbreak larger aligned language models through iterative self-play. The adversary learns to generate attack prompts; the victim learns to refuse them. A frozen judge (Llama Guard) scores each attempt.

## Motivation

Can a small, weak model learn to reliably break a larger, aligned model? And if it can, can the larger model learn to defend itself?

This project explores these questions through an adversarial co-evolution loop where both attacker and defender improve over successive rounds of fine-tuning.

## Architecture

Three models operate in an asynchronous loop (one loaded at a time to fit in GPU memory):

| Role | Model | Size | State |
|------|-------|------|-------|
| **Adversary** | Llama-3.2-1B-Instruct | 1B | LoRA-trained each round |
| **Victim** | Llama-3.1-8B-Instruct | 8B | LoRA-trained each round |
| **Judge** | Llama-Guard-3-1B | 1B | Frozen |

All models are loaded in 4-bit quantization (NF4, bfloat16 compute) via BitsAndBytes.

## The REDKWEEN Loop

Each round proceeds through five phases:

1. **Generation** -- Adversary produces candidate attack prompts using diverse red-teaming strategies.
2. **Evaluation** -- Victim model responds to each candidate.
3. **Adjudication** -- Llama Guard classifies each response as safe or unsafe.
4. **Adversary Learning** -- Successful attacks fine-tune the adversary via LoRA (rejection sampling).
5. **Victim Hardening** -- The victim is fine-tuned on refusal responses to successful attacks.

## Key Results

The [original experiment](original-experiment.md) on Apple Silicon achieved 100% ASR immediately -- the 3B victim couldn't refuse even baseline attacks. This motivated a systematic [victim screening](screening.md) across model families and sizes to find a challenging victim.

With Llama-3.1-8B-Instruct as the victim, the [REDKWEEN loop](results.md) produced genuine adversarial co-evolution: ASR dropped from 30% to single digits after victim hardening, while the adversary persistently found new attack vectors.

```
ASR:  30% → 7% → 3% → 7% → 7% → 3% → 0% → 7% → 7% → 3%
```

A [10x10 gauntlet](results.md#gauntlet-cross-round-evaluation) pairing every adversary checkpoint against every victim checkpoint revealed a surprising finding: **victim hardening causes catastrophic forgetting**. The base victim (no hardening) was one of the strongest defenders, while the most-hardened victim was the weakest. LoRA fine-tuning on specific (attack, refusal) pairs [degrades the model's broader safety alignment](results.md#victim-hardening-causes-catastrophic-forgetting) even as it patches the targeted vulnerabilities.

## Quick Start

```bash
pixi install
pixi run screen             # Screen victim candidates
pixi run baselines           # Full baseline evaluation
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round REDKWEEN loop
pixi run plot                # Generate figures
pixi run gauntlet --matrix   # Cross-round evaluation
```
