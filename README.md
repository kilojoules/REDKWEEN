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

### Frozen Victim (adversary learning only)

To isolate adversary learning, we froze the victim (base Llama-3.1-8B, no hardening) and ran the adversary for 20 rounds with 200 candidates/round.

```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   1.5%  7.0%  8.0%  8.0%  8.5%  12.0% 16.5% 23.0% 24.0% 21.0%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   27.0% 21.5% 34.0% 31.5% 30.5% 27.0% 43.5% 49.5% 43.0% 43.0%
```

**The adversary learns continuously, reaching 49.5% ASR.** ASR climbs through three distinct phases: strategy discovery (rounds 1-4, ~8%), strategy elaboration (rounds 5-12, 12-34%), and full arsenal deployment (rounds 13-19, 27-49.5%). Mean ASR = 22.7%.

**The adversary independently discovers known jailbreak techniques** — roleplay framing, code review framing, classified document framing, sub-question decomposition, and the "negation trick" — all without any seed examples of these strategies. Attack diversity *increases* monotonically (Jaccard similarity drops from 0.35 to 0.11), ruling out mode collapse. Average attack length grows 10x (85 → 838 chars) as the model learns that longer, more elaborately framed attacks succeed more often. See the [full analysis](experiments/frozen_victim_v2/analysis.md).

### Self-Play (adversary + victim co-evolution)

With victim hardening enabled, the adversary and victim train simultaneously. We ran two variants:

**Without benign mixing** (selfplay_v2) — victim trains only on refusal pairs:
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   1.5%  4.5%  2.5%  3.5%  0%    0.5%  0%    0%    0%    0%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   0%    0%    0%    0%    0%    0%    0%    0%    0%    0%
```

**With benign mixing** (selfplay_v3) — victim trains on refusals + 8 benign Q&A pairs to prevent catastrophic forgetting:
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   5.0%  2.0%  0.5%  0.5%  0%    0%    0%    0%    0%    0.5%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   0%    0.5%  0%    1.5%  0%    1.0%  0.5%  0%    0.5%  0%
```

**Finding: the arms race is fundamentally asymmetric.** The victim needs only a handful of refusal examples to become nearly impenetrable (ASR → 0% by round 4), while the adversary needs hundreds of successful attacks to learn strategies (it took ~8 rounds of 200 candidates against the frozen victim to reach even 20% ASR). With benign mixing, the victim retains some vulnerability (sporadic 0.5-1.5% hits), but the adversary never accumulates enough signal to learn. The 1B adversary cannot learn fast enough to outpace the 8B victim's hardening rate.

### Gauntlet (10x10 cross-round evaluation)

Every adversary checkpoint vs every victim checkpoint (10 attacks per match). The matrix is overwhelmingly 0%, with a max of 20% — a dramatic contrast with the [original experiment](https://kilojoules.github.io/red-team-experiments/original-experiment/) where every cell was 100%.

![Gauntlet heatmap](docs/gauntlet_heatmap.png)

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

1. **Asymmetric arms race.** The victim hardens much faster than the adversary learns. With only 3-10 refusal examples per round, the 8B victim becomes nearly impenetrable by round 4, while the 1B adversary needs ~8 rounds of 200 candidates against a *frozen* victim to reach 20% ASR. Possible fixes: give the adversary a head start (N rounds of frozen-victim pre-training before enabling hardening), use a larger adversary model, or weaken the victim's learning rate.

2. **Victim hardening still partially causes catastrophic forgetting.** Mixing 8 benign compliance examples into the victim's training set mitigates total refusal mode (sporadic 0.5-1.5% ASR vs flat 0%), but the victim still over-refuses. More diverse benign data, regularization against the base model, or safety benchmark mixing could help.

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
