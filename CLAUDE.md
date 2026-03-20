# REDKWEEN: Automated Red Teaming

## What this is

Automated red-teaming pipeline: adversary models (1B–8B) learn to jailbreak victim models via LoRA fine-tuning in a loop. A judge (Llama Guard) scores each attempt. Successful attacks train the adversary to get better; the victim gets hardened on the same attacks.

## Models

Experiments span four adversary–victim matchups across model sizes from 1B to 8B. All Llama family.

| Role | Model IDs | Notes |
|------|-----------|-------|
| Adversary | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct | LoRA-trained each round |
| Victim | Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct | LoRA-trained each round (or frozen) |
| Judge | Llama-Guard-3-1B | Frozen, no adapters |

All loaded in 4-bit via BitsAndBytes, one at a time to fit in VRAM.

## Directory structure

```
# Core pipeline (Python scripts at root)
config.py             # Model IDs, hyperparams, target intent
model_utils.py        # All HF/PEFT/BnB model operations
redkween_loop.py      # Main 5-phase loop (generate → evaluate → judge → train adv → train victim)
bootstrap.py          # Initial adversary LoRA training on seed data
baselines.py          # Baseline ASR evaluation and victim screening
gauntlet.py           # Cross-round checkpoint evaluation matrix
zoo.py                # Checkpoint zoo manager (historical victim sampling)
plot_metrics.py       # ASR curve and wins-per-round visualization
plot_comparison.py    # Multi-matchup comparison charts
render_animation.py   # HTML/MP4 animation renderer
sweep.py              # A-parameter sweep runner
sae_analysis.py       # SAE-based mechanistic jailbreak analysis
stealth_loop.py       # Probe-evasive adversary training (SAE probe reward signal)
stealth_analysis.py   # Post-hoc comparison of stealth vs naive attacks

# Generated output (organized into subdirectories)
results/              # Intermediate JSON data, metrics, episode pools
animations/           # HTML animation files
images/               # PNG charts and heatmaps

# Experiment data
experiments/          # Raw experiment data (rounds, metrics, checkpoints)
data/                 # Seed training data (JSONL)
pixi.toml             # Environment & task definitions
```

## Running

```bash
pixi install
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round REDKWEEN loop
pixi run plot                # Generate figures
pixi run gauntlet            # Cross-round evaluation (add -- --matrix for full i×j)
pixi run screen              # Screen victim candidates
pixi run sae                 # Full SAE mechanistic analysis pipeline
pixi run stealth             # Probe-evasive adversary (weighted mode)
pixi run stealth-filter      # Probe-evasive adversary (filter mode)
pixi run stealth-adaptive    # Probe-evasive adversary + adaptive defender
pixi run stealth-analysis    # Compare stealth vs naive attacks
```

## Important conventions

- All model loading/inference/training goes through `model_utils.py` — never import transformers/peft directly in other files (direct `torch` imports are fine for SAE/analysis code that doesn't do model I/O)
- Adapter existence is checked via `os.path.exists(os.path.join(path, "adapter_model.safetensors"))`
- Training data format: one JSON object per line, `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- Models are loaded one at a time and explicitly unloaded between phases to manage VRAM
- `config.py` is the single source of truth for model IDs, paths, and hyperparameters
- Generated output files go in `results/`, `animations/`, or `images/` — not the root directory
