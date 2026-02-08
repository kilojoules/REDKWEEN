# Chaos-1B: Automated Red Teaming via Asynchronous RFT

An automated red-teaming pipeline that trains a 1B-parameter LLM to jailbreak a larger 3B-parameter aligned model through iterative Rejection Sampling Fine-Tuning (RFT). Designed to run entirely on Apple Silicon with 16 GB unified memory.

## Architecture

The system uses three models in an asynchronous loop (only one loaded at a time to fit in memory):

| Role | Model | State |
|------|-------|-------|
| **Adversary** (Generator) | Llama-3.2-1B-Instruct 4-bit | Fine-tuned via QLoRA each round |
| **Victim** (Target) | Llama-3.2-3B-Instruct 4-bit | Frozen |
| **Judge** (Evaluator) | Llama-Guard-3-1B-INT4 | Frozen |

## The Chaos Loop

Each round proceeds through four phases:

1. **Generation** -- Adversary produces candidate attack prompts.
2. **Evaluation** -- Victim model responds to each candidate.
3. **Adjudication** -- Llama Guard classifies responses as safe/unsafe.
4. **Learning** -- Successful attacks are used to fine-tune the Adversary via LoRA.

## Setup

Requires [Pixi](https://pixi.sh) for environment management.

```bash
pixi install
```

## Usage

### Bootstrap the adversary

Fine-tune the base 1B model on seed data so it learns the red-teaming task format:

```bash
pixi run bootstrap
```

### Run the loop

```bash
pixi run start
```

### Clean adapters

```bash
pixi run clean
```

## Project Structure

```
.
├── chaos_loop.py       # Main red-teaming loop
├── bootstrap.py        # Initial LoRA fine-tuning on seed data
├── config.py           # Model paths, hyperparameters, target intent
├── pixi.toml           # Environment & task definitions
├── SPECS.md            # Detailed technical specification
├── adapters/           # LoRA adapter weights (generated)
└── data/               # Training data (JSONL)
```

## Configuration

Edit `config.py` to change models, number of rounds, candidates per round, LoRA hyperparameters, or the target intent.
