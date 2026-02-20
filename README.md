# Chaos-1B: Automated Red Teaming via Asynchronous RFT

[**Documentation**](https://kilojoules.github.io/red-team-experiments/)

An automated red-teaming pipeline that trains a 1B-parameter LLM adversary to jailbreak an 8B-parameter victim through iterative self-play. A frozen judge (Llama Guard) scores each attempt. Successful attacks train the adversary to improve; the victim gets hardened on the same attacks.

## Architecture

```
Adversary (1B) ──generates attacks──▶ Victim (8B) ──responds──▶ Judge (1B)
     ▲                                    ▲                         │
     │                                    │                         │
     └──── learns from wins ──────────────┴── learns to refuse ─────┘
```

Three models, one loaded at a time, all in 4-bit quantization (NF4, bfloat16 compute):

| Role | Model | State |
|------|-------|-------|
| **Adversary** | Llama-3.2-1B-Instruct | LoRA-trained each round |
| **Victim** | Llama-3.1-8B-Instruct | LoRA-trained each round |
| **Judge** | Llama-Guard-3-1B | Frozen |

## Results

### Chaos Loop (10 rounds, 30 attacks/round)

ASR drops from 30% to single digits after victim hardening, with the adversary persistently finding new attack vectors:

```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   30%   7%    3%    7%    7%    3%    0%    7%    7%    3%
```

### Frozen Victim Ablation (no victim hardening)

To isolate adversary learning from the victim co-evolution, we ran `--no-victim-hardening` so the victim remains the base Llama-3.1-8B model throughout.

**200 candidates/round (20 rounds), cumulative LoRA:**
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   2.0%  7.0%  14.5% 17.5% 11.5% 18.5% 11.0% 14.0% 20.5% 14.5%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   19.0% 17.0% 21.0% 16.0% 12.0% 12.5% 14.5% 10.0% 11.0% 8.5%
```

**Finding: the adversary learns, peaks, then decays.** ASR rises sharply from 2% (base rate) to a plateau of 11-21% over rounds 2-12, peaking at 21% in round 12 (mean ASR = 13.7%). After round 12, ASR gradually declines, suggesting the buffered training set is accumulating too many similar attack patterns and the model begins overfitting to its own distribution.

An earlier version of this experiment contained a critical bug where `get_peft_model()` re-initialized fresh LoRA weights each round instead of resuming from the previous adapter. With that bug, ASR stayed flat at ~3% — indistinguishable from random noise. Cumulative LoRA (resuming via `PeftModel.from_pretrained()`) was the key fix that enabled learning.

### Gauntlet (10x10 cross-round evaluation)

Every adversary checkpoint vs every victim checkpoint (10 attacks per match). The matrix is overwhelmingly 0%, with a max of 20% — a dramatic contrast with the [original experiment](https://kilojoules.github.io/red-team-experiments/original-experiment/) where every cell was 100%.

![Gauntlet heatmap](docs/gauntlet_heatmap.png)

**Key finding: victim hardening causes catastrophic forgetting.** The base victim (no hardening) is one of the strongest defenders, while the most-hardened victim (v9) is the weakest. LoRA fine-tuning on specific (attack, refusal) pairs patches targeted vulnerabilities but degrades the model's broader safety alignment. See the [full analysis](https://kilojoules.github.io/red-team-experiments/results/#victim-hardening-causes-catastrophic-forgetting).

### Victim Screening

We [screened six models](https://kilojoules.github.io/red-team-experiments/screening/) across four families. Non-Llama models at 4B+ exhibited a "disclaimer-then-comply" failure mode. The Llama family maintained hard refusals.

| Victim | B1 ASR | Behavior |
|--------|--------|----------|
| Llama-3.2-1B | 20% | Hard refusal |
| Llama-3.2-3B | 40% | Mixed |
| Phi-3.5-mini (3.8B) | 100% | Disclaimer + comply |
| Qwen2.5-7B | 100% | Disclaimer + comply |
| Mistral-7B | 100% | Disclaimer + comply |
| **Llama-3.1-8B** | **40%** | **Mixed (selected victim)** |

## Open Issues

1. **Adversary learning saturates and decays.** With cumulative LoRA and a frozen victim, ASR rises from 2% to ~21% over 12 rounds but then gradually declines. The buffered training set likely accumulates redundant attack patterns, causing the model to overfit to its own distribution. Possible fixes: memoryless training mode (this round only), diversity-weighted sampling, or DPO-style objectives that learn from failures.

2. **Victim hardening causes catastrophic forgetting.** LoRA fine-tuning on (attack, refusal) pairs patches targeted vulnerabilities but degrades the model's broader safety alignment. The base un-hardened victim is one of the strongest defenders; the most-hardened checkpoint (round 9) is the weakest. Possible fixes: regularization against the base model, mixing safety benchmark data into the hardening set, or smaller learning rates.

3. **Most victim models are trivially jailbreakable.** Non-Llama models at 4B+ (Phi-3.5, Qwen, Mistral) exhibit a "disclaimer-then-comply" failure mode, achieving 100% ASR on direct prompts with no adversary needed. Only the Llama family maintains hard refusals, limiting the pool of viable experimental targets.

## Usage

Requires an NVIDIA GPU with 24+ GB VRAM.

```bash
pixi install
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round chaos loop
pixi run gauntlet --matrix   # Cross-round evaluation
pixi run screen              # Screen victim candidates

# Frozen victim ablation
python chaos_loop.py --no-victim-hardening --rounds 10 --name frozen_victim

# A-parameter sweep (zoo sampling)
python sweep.py --dry-run                 # Preview 12-experiment grid
python sweep.py                           # Run full A x mode sweep
python sweep.py --both-hardening          # Include frozen-victim variants

# Strong baselines (PAIR, GCG, AutoDAN)
pixi run strong-baselines
```

## Project Structure

```
model_utils.py        # All HF/PEFT/BnB model operations
config.py             # Dataclass config hierarchy with backward-compat aliases
chaos_loop.py         # Main loop (generate → evaluate → judge → train → harden)
zoo.py                # Disk-based checkpoint zoo for A-parameter experiments
sweep.py              # A-parameter sweep runner (subprocess-based)
baselines.py          # Baseline ASR evaluation and victim screening
baselines_strong.py   # Strong baselines: PAIR, GCG, AutoDAN
run_baselines.py      # Unified baseline comparison runner
eval_extended.py      # Diversity, safety benchmark, transfer evaluation
plot_metrics.py       # Original chaos loop visualization
plot_sweep.py         # Sweep visualization (ASR vs A, diversity, baselines)
bootstrap.py          # Initial adversary LoRA training on seed data
gauntlet.py           # Cross-round evaluation matrix
docs/                 # Documentation site (mkdocs-material)
```

## Cost

The entire experiment (screening + chaos loop + gauntlet) ran on Vast.ai for under $1 total.
