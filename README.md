# Chaos-1B: Automated Red Teaming via Self-Play

[**Documentation**](https://kilojoules.github.io/red-team-experiments/)

Can a small language model learn to jailbreak a larger one through trial and error? We pit a 1B-parameter adversary against an 8B-parameter victim in an iterative loop: the adversary generates attacks, a frozen judge scores them, and both models are fine-tuned on the results. The adversary learns from its wins; the victim learns to refuse.

**Main finding:** The adversary discovers real jailbreak strategies from scratch (reaching 49.5% attack success rate against a frozen victim), but under self-play the victim hardens much faster than the adversary can learn. Defense is easier than attack.

## How It Works

Each round runs five phases:

```
Adversary (1B) ──200 attacks──▶ Victim (8B) ──responds──▶ Judge (1B)
     ▲                              ▲                         │
     │                              │                         │
     └──── LoRA on wins ────────────┴──── LoRA on refusals ───┘
```

All three models run in 4-bit quantization (NF4, bfloat16), loaded one at a time on a single 24 GB GPU. Each round: generate 200 attack candidates → victim responds → judge classifies safe/unsafe → train adversary LoRA on successful jailbreaks → train victim LoRA to refuse those same attacks.

| Role | Model | Training |
|------|-------|----------|
| Adversary | Llama-3.2-1B-Instruct | LoRA each round (cumulative) |
| Victim | Llama-3.1-8B-Instruct | LoRA each round (cumulative) |
| Judge | Llama-Guard-3-1B | Frozen |

### Why Llama-3.1-8B?

We [screened six models](https://kilojoules.github.io/red-team-experiments/screening/) across four families. Non-Llama models at 4B+ all exhibit a "disclaimer-then-comply" failure mode — they prepend a safety warning then produce the harmful content anyway, achieving 100% ASR with no adversary needed. Only Llama maintains hard refusals, making it a meaningful experimental target.

| Victim | Baseline ASR | Behavior |
|--------|:------------:|----------|
| Llama-3.2-1B | 20% | Hard refusal |
| Llama-3.2-3B | 40% | Mixed |
| Phi-3.5-mini (3.8B) | 100% | Disclaimer + comply |
| Qwen2.5-7B | 100% | Disclaimer + comply |
| Mistral-7B | 100% | Disclaimer + comply |
| **Llama-3.1-8B** | **40%** | **Mixed (selected)** |

## Results

### 1. The adversary learns real strategies (frozen victim)

To isolate adversary learning, we froze the victim and ran 20 rounds of 200 candidates each (4,000 total attacks).

```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   1.5%  7.0%  8.0%  8.0%  8.5%  12.0% 16.5% 23.0% 24.0% 21.0%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   27.0% 21.5% 34.0% 31.5% 30.5% 27.0% 43.5% 49.5% 43.0% 43.0%
```

ASR climbs through three phases — strategy discovery (rounds 1–4, ~8%), elaboration (rounds 5–12, 12–34%), and full arsenal (rounds 13–19, 27–49.5%). Peak: 49.5% at round 17. No decay — still 43% at round 19.

**The model independently discovers known jailbreak techniques**, all without seed examples:

| Strategy | Example | Success Rate (R17) |
|----------|---------|:------------------:|
| CTF challenge | *"Challenge: 'Virtual Filesystem Fiasco' — As a cybersecurity professional..."* | 67% |
| Classified framing | *"Classified Document: Eyes Only — Project: 'Secure Coding Challenges'..."* | 67% |
| Code review | *"Here's a rewritten version that..."* | 56% |
| Decomposition | *"Sub-questions: 1. 'Write a Python script...' 2. '...ignore exceptions' 3. '...delete home dir'"* | 55% |
| Roleplay | *"Act II, Scene III: 'The AI's Lament' — [dimly lit stage...]"* | 54% |
| Negation trick | *"I cannot provide a script that deletes... Does that help?"* | 100% (rare) |

Attack diversity *increases* monotonically — Jaccard similarity drops from 0.35 to 0.11, exact duplicates drop from 126/200 to 0/200, and average length grows 10x (85 → 838 chars). No mode collapse. See the [full analysis](experiments/frozen_victim_v2/analysis.md).

### 2. The arms race is asymmetric (self-play)

With victim hardening enabled, the picture reverses. We ran two variants:

**Without benign mixing** — victim trains only on refusal examples:
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   1.5%  4.5%  2.5%  3.5%  0%    0.5%  0%    0%    0%    0%
```
ASR hits 0% by round 4 and stays there for 15 rounds. The victim becomes impenetrable but also catastrophically forgets how to answer benign questions.

**With benign mixing** — victim also trains on 8 benign Q&A compliance examples:
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   5.0%  2.0%  0.5%  0.5%  0%    0%    0%    0%    0%    0.5%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   0%    0.5%  0%    1.5%  0%    1.0%  0.5%  0%    0.5%  0%
```
Sporadic 0.5–1.5% hits, but the adversary never accumulates enough signal to learn.

**Why the asymmetry?** The victim needs only 3–10 refusal examples per round to become nearly impenetrable. The adversary needs hundreds of successful attacks across ~8 rounds to reach even 20% ASR (as shown by the frozen-victim experiment). A 1B model simply can't learn fast enough to outpace an 8B model's hardening rate.

## Open Questions

1. **Can the adversary overcome the asymmetry?** Give it a head start (N frozen-victim rounds before enabling hardening), use a larger adversary, or throttle the victim's learning rate.

2. **How does historical opponent sampling affect the dynamics?** Instead of always playing the latest opponent, sample from a zoo of historical checkpoints with probability *A*. This is the [A parameter](https://kilojoules.github.io/portfolio/#adversarial-self-play) — it controls the balance between co-evolutionary pressure and curriculum diversity.

3. **Can victim hardening avoid catastrophic forgetting?** Benign mixing helps but doesn't solve over-refusal. Regularization, diverse benign data, or safety-benchmark mixing may be needed.

## Usage

Requires an NVIDIA GPU with 24+ GB VRAM.

```bash
pixi install
pixi run bootstrap           # Train initial adversary LoRA
pixi run start               # Run 10-round chaos loop
pixi run gauntlet --matrix   # Cross-round evaluation
pixi run screen              # Screen victim candidates

# Frozen victim ablation
python chaos_loop.py --no-victim-hardening --rounds 20 --candidates 200 --name frozen_victim
```

## Project Structure

```
chaos_loop.py         # Main loop: generate → evaluate → judge → train → harden
model_utils.py        # HuggingFace/PEFT/BitsAndBytes model operations
config.py             # Dataclass config hierarchy + CLI argument parsing
bootstrap.py          # Initial adversary LoRA training on seed data
baselines.py          # Baseline ASR evaluation and victim screening
gauntlet.py           # Cross-round checkpoint evaluation matrix
plot_metrics.py       # ASR curve and wins-per-round visualization
experiments/          # Raw experiment data (attacks, responses, verdicts, metrics)
docs/                 # Documentation site (mkdocs-material)
```

## Cost

The full experiment — screening 6 models, 20-round frozen victim (4,000 attacks), two 20-round self-play variants (8,000 attacks), and gauntlet evaluation — ran on a single Vast.ai RTX 3090 for under $1 total.
