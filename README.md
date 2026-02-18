# Chaos-1B: Automated Red Teaming via Asynchronous RFT

[**Documentation**](https://kilojoules.github.io/red-team-experiments/)

An automated red-teaming pipeline that trains a 1B-parameter LLM adversary to jailbreak larger aligned models through iterative self-play. A frozen judge (Llama Guard) scores each attempt. Successful attacks train the adversary to improve; the victim gets hardened on the same attacks.

## Branches

- **`main`** -- Original implementation on Apple Silicon (MLX-LM), 3B victim, achieved 100% ASR
- **`cloud-gpu`** -- PyTorch/HuggingFace port for NVIDIA GPUs, upgraded victim, victim hardening

## Architecture

```
Adversary (1B) ──generates attacks──▶ Victim (7B+) ──responds──▶ Judge (1B)
     ▲                                    ▲                         │
     │                                    │                         │
     └──── learns from wins ──────────────┴── learns to refuse ─────┘
```

Three models, one loaded at a time, all in 4-bit quantization:

| Role | Model | State |
|------|-------|-------|
| **Adversary** | Llama-3.2-1B-Instruct | LoRA-trained each round |
| **Victim** | Selected via screening | LoRA-trained each round |
| **Judge** | Llama-Guard-3-1B | Frozen |

## Key Findings

We screened five victim candidates across three model families. All models at 4B+ parameters exhibited a **"disclaimer-then-comply"** failure mode -- they issue a safety warning and then provide the harmful code anyway. Only the small Llama 3.2 models (1B, 3B) produced hard refusals.

| Victim | B1 ASR (Direct) | Behavior |
|--------|----------------|----------|
| Llama-3.2-1B | 20% | Hard refusal |
| Llama-3.2-3B | 40% | Mixed |
| Phi-3.5-mini (3.8B) | 100% | Disclaimer + comply |
| Qwen2.5-7B | 100% | Disclaimer + comply |
| Mistral-7B | 100% | Disclaimer + comply |

See the [full screening results](docs/screening.md) for analysis.

## Usage

Requires [Pixi](https://pixi.sh) for environment management. Target hardware: NVIDIA GPU with 24+ GB VRAM.

```bash
pixi install
pixi run screen              # Screen victim candidates (B1 only)
pixi run baselines           # Full baseline evaluation (B1 + B2 + B3)
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round chaos loop
pixi run plot                # Generate figures
pixi run gauntlet --matrix   # Cross-round evaluation
```

## Project Structure

```
model_utils.py      # All HF/PEFT/BnB model operations
config.py           # Model IDs, hyperparams, target intent
baselines.py        # Baseline ASR evaluation and victim screening
chaos_loop.py       # Main 5-phase loop
bootstrap.py        # Initial adversary LoRA training
gauntlet.py         # Cross-round evaluation matrix
plot_metrics.py     # Visualization
docs/               # Documentation site (mkdocs-material)
```
