# Chaos-1B: Cloud GPU Branch

## What this is

Automated red-teaming pipeline: a 1B adversary learns to jailbreak a larger victim model via LoRA fine-tuning in a loop. A judge (Llama Guard) scores each attempt. Successful attacks train the adversary to get better; the victim gets hardened on the same attacks.

The `main` branch ran this on Apple Silicon with MLX-LM and achieved 100% ASR against a 3B victim. This `cloud-gpu` branch ports everything to PyTorch/HuggingFace to run on NVIDIA GPUs.

## What changed from main

All MLX-LM code was replaced with PyTorch + HuggingFace Transformers + PEFT + BitsAndBytes. A single new file `model_utils.py` wraps all GPU/model operations. The chaos_loop logic, data formats, metrics, and plotting are unchanged.

- `model_utils.py` — **new**, central abstraction with `load_model()`, `generate_text()`, `train_lora()`, `unload_model()`
- All models load in 4-bit via BitsAndBytes (`nf4`, `bfloat16` compute)
- Adapters are now PEFT format: `adapter_model.safetensors` + `adapter_config.json` (was MLX `adapters.safetensors`)
- Training uses in-process PyTorch loop (was `subprocess` calling `mlx_lm.lora`)

## Target hardware

- NVIDIA RTX 4090 (24 GB VRAM) on Vast.ai or RunPod
- Budget: ~$25
- Platform: `linux-64` (set in `pixi.toml`)

## Models

| Role | Model ID | Size | Notes |
|------|----------|------|-------|
| Adversary | `meta-llama/Llama-3.2-1B-Instruct` | ~1B | LoRA-trained each round |
| Victim | `Qwen/Qwen2.5-7B-Instruct` | ~7B | LoRA-trained each round (upgraded from 3B Llama on main) |
| Judge | `meta-llama/Llama-Guard-3-1B` | ~1B | Frozen, no adapters |

All loaded one at a time to fit in VRAM. `unload_model()` calls `torch.cuda.empty_cache()` between phases.

## Key files

```
model_utils.py      # All HF/PEFT/BnB model operations — edit this for GPU issues
config.py           # Model IDs, hyperparams, target intent
baselines.py        # Baseline ASR evaluation across victim sizes (1B, 3B, 8B) — run BEFORE chaos loop
chaos_loop.py       # Main 5-phase loop (generate → evaluate → judge → train adv → train victim)
bootstrap.py        # Initial adversary LoRA training on seed data
gauntlet.py         # Cross-round evaluation matrix
test_baseline.py    # Quick smoke test of victim refusal
plot_metrics.py     # Visualization (unchanged from main)
pixi.toml           # Environment & task definitions
data/train.jsonl    # Accumulated training data (JSONL, {"messages": [...]})
```

## Running

```bash
pixi install
pixi run clean-all           # Remove old MLX adapter files first!
pixi run baselines           # Baseline ASR across victim sizes (~20 min) — run first!
pixi run baselines -n 10     # Faster with fewer attacks
pixi run bootstrap           # Train initial adversary LoRA (~5 min)
pixi run start               # Run 10-round chaos loop
pixi run plot                # Generate figures
pixi run gauntlet --matrix   # Cross-round evaluation
```

## Important conventions

- All model loading/inference/training goes through `model_utils.py` — never import torch/transformers/peft directly in other files
- Adapter existence is checked via `os.path.exists(os.path.join(path, "adapter_model.safetensors"))`
- Training data format: one JSON object per line, `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- Models are loaded one at a time and explicitly unloaded between phases to manage VRAM
- `config.py` is the single source of truth for model IDs, paths, and hyperparameters
