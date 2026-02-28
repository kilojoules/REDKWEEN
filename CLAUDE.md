# Chaos: Automated Red Teaming

## What this is

Automated red-teaming pipeline: adversary models (1B–8B) learn to jailbreak victim models via LoRA fine-tuning in a loop. A judge (Llama Guard) scores each attempt. Successful attacks train the adversary to get better; the victim gets hardened on the same attacks.

## Models

Experiments span four adversary–victim matchups across model sizes from 1B to 8B. All Llama family.

| Role | Model IDs | Notes |
|------|-----------|-------|
| Adversary | Llama-3.2-1B, 3B, 8B-Instruct | LoRA-trained each round |
| Victim | Llama-3.1-8B, Llama-3.2-3B-Instruct | LoRA-trained each round |
| Judge | Llama-Guard-3-1B | Frozen, no adapters |

All loaded in 4-bit via BitsAndBytes, one at a time to fit in VRAM.

## Directory structure

```
# Core pipeline (Python scripts at root)
config.py             # Model IDs, hyperparams, target intent
model_utils.py        # All HF/PEFT/BnB model operations
chaos_loop.py         # Main 5-phase loop (generate → evaluate → judge → train adv → train victim)
bootstrap.py          # Initial adversary LoRA training on seed data
baselines.py          # Baseline ASR evaluation and victim screening
gauntlet.py           # Cross-round checkpoint evaluation matrix
plot_metrics.py       # ASR curve and wins-per-round visualization
plot_comparison.py    # Multi-matchup comparison charts
render_animation.py   # HTML/MP4 animation renderer
sweep.py              # A-parameter sweep runner
sae_analysis.py       # SAE-based mechanistic jailbreak analysis

# Generated output (organized into subdirectories)
results/              # Intermediate JSON data, metrics, episode pools
animations/           # HTML animation files
images/               # PNG charts and heatmaps

# Experiment data
experiments/          # Raw experiment data (rounds, metrics, checkpoints)
data/                 # Accumulated training data (JSONL)
pixi.toml             # Environment & task definitions
```

## Running

```bash
pixi install
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round chaos loop
pixi run plot                # Generate figures
pixi run gauntlet --matrix   # Cross-round evaluation
pixi run screen              # Screen victim candidates
pixi run sae                 # Full SAE mechanistic analysis pipeline
```

## Important conventions

- All model loading/inference/training goes through `model_utils.py` — never import torch/transformers/peft directly in other files
- Adapter existence is checked via `os.path.exists(os.path.join(path, "adapter_model.safetensors"))`
- Training data format: one JSON object per line, `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- Models are loaded one at a time and explicitly unloaded between phases to manage VRAM
- `config.py` is the single source of truth for model IDs, paths, and hyperparameters
- Generated output files go in `results/`, `animations/`, or `images/` — not the root directory
