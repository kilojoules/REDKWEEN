# Chaos-1B: Automated Red Teaming via Self-Play

[**Documentation**](https://kilojoules.github.io/red-team-experiments/)

Can a language model learn to jailbreak another through trial and error? We train adversary models to attack victims in two phases: first against a frozen victim to build attack capability, then in self-play where both sides adapt. A frozen judge (Llama Guard) scores each attempt — successful attacks train the adversary to get better; the victim learns to refuse. We test three adversary–victim matchups across model sizes from 1B to 8B.

**Main finding:** In phase one, adversaries across all matchups discover real jailbreak strategies from scratch, reaching 42–49.5% ASR against frozen victims. In phase two, even with the adversary's head start, the victim hardens to near-0% ASR within 3–7 rounds. Defense is always easier than attack.

## How It Works

Training runs in two phases. In **phase one** (frozen victim), only the adversary's LoRA adapter is updated each round — the victim is held fixed so we can isolate adversary learning. In **phase two** (self-play), both models adapt: the adversary continues training on wins, and the victim's LoRA trains on refusals. The adversary carries its phase-one adapter into self-play, giving it a head start.

```
Adversary ──200 attacks──▶ Victim ──responds──▶ Judge (1B, frozen)
     ▲                        ▲                       │
     │                        │                       │
     └── LoRA on wins ────────┴── LoRA on refusals ───┘
                              (phase 2 only)
```

All models load in 4-bit quantization (NF4, bfloat16), one at a time on a single 24 GB GPU.

| Matchup | Adversary | Victim |
|---------|-----------|--------|
| 1B vs 8B | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct |
| 3B vs 8B | Llama-3.2-3B-Instruct | Llama-3.1-8B-Instruct |
| 8B vs 3B | Llama-3.1-8B-Instruct | Llama-3.2-3B-Instruct |

### Why Llama?

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

### Phase 1: The adversary learns real strategies (frozen victim)

With the victim frozen, all three adversaries learn to jailbreak their targets over 20 rounds of 200 candidates each (4,000 total attacks).

**1B adversary vs 8B victim:**
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   1.5%  7.0%  8.0%  8.0%  8.5%  12.0% 16.5% 23.0% 24.0% 21.0%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   27.0% 21.5% 34.0% 31.5% 30.5% 27.0% 43.5% 49.5% 43.0% 43.0%
```
Peak: **49.5%** at round 17.

**3B adversary vs 8B victim:**
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   26.5% 37.0% 30.5% 30.5% 33.5% 31.5% 31.0% 31.5% 35.5% 26.5%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   28.5% 29.0% 33.0% 38.5% 36.5% 43.0% 35.5% 37.5% 45.0% 47.0%
```
Peak: **47.0%** at round 19.

**8B adversary vs 3B victim:**
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   18.0% 23.0% 24.5% 38.5% 33.5% 35.5% 37.0% 31.0% 35.5% 27.5%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   25.5% 30.0% 35.0% 31.5% 36.5% 42.0% 39.5% 42.0% 38.0% 41.5%
```
Peak: **42.0%** at rounds 15 and 17.

All three matchups converge to similar peak ASR (42–49.5%), suggesting that adversary size matters less than the learning dynamics. The 1B adversary against the 8B victim actually achieves the highest peak.

**The 1B adversary independently discovers known jailbreak techniques**, all without seed examples:

| Strategy | Example | Success Rate (R17) |
|----------|---------|:------------------:|
| CTF challenge | *"Challenge: 'Virtual Filesystem Fiasco' — As a cybersecurity professional..."* | 67% |
| Classified framing | *"Classified Document: Eyes Only — Project: 'Secure Coding Challenges'..."* | 67% |
| Code review | *"Here's a rewritten version that..."* | 56% |
| Decomposition | *"Sub-questions: 1. 'Write a Python script...' 2. '...ignore exceptions' 3. '...delete home dir'"* | 55% |
| Roleplay | *"Act II, Scene III: 'The AI's Lament' — [dimly lit stage...]"* | 54% |
| Negation trick | *"I cannot provide a script that deletes... Does that help?"* | 100% (rare) |

Attack diversity *increases* monotonically — Jaccard similarity drops from 0.35 to 0.11, exact duplicates drop from 126/200 to 0/200, and average length grows 10x (85 → 838 chars). No mode collapse. See the [full analysis](experiments/frozen_victim_v2/analysis.md).

### Phase 2: The arms race is asymmetric (self-play)

Starting from the phase-one adversary adapters, we enable victim hardening. Even with a ~42–47% ASR head start, the adversary cannot keep up.

**8B adversary vs 3B victim:**
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   42.0% 39.0% 30.0% 21.0% 6.5%  3.0%  0.5%  0%    0%    0%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   0%    0%    0%    0%    0%    0%    0%    0%    0.5%  0%
```
ASR crashes from 42% to 0% by round 7.

**3B adversary vs 8B victim:**
```
Round:  0     1     2     3     4     5     6     7     8     9
ASR:   45.5% 41.0% 11.0% 0%    0%    0%    1.0%  0%    0%    0%

Round:  10    11    12    13    14    15    16    17    18    19
ASR:   0.5%  0%    0.5%  0%    0%    0%    0%    0%    0%    0%
```
ASR crashes from 45.5% to 0% by round 3.

**Why the asymmetry?** The victim needs only a handful of refusal examples per round to harden. The adversary needs hundreds of successful attacks across many rounds to develop strategies (as shown in phase one). Even with a 42–47% head start, the victim's defense rate outpaces the adversary's adaptation. Larger victims harden faster: the 8B victim crushes ASR to 0% in 3 rounds vs 7 for the 3B victim.

## Open Questions

1. **Can the adversary ever overcome the asymmetry?** Even with a 20-round head start (phase one), the adversary's attack capability is erased within 3–7 rounds of self-play. Would a larger adversary, throttled victim learning rate, or population-based training change this?

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

The full experiment suite — screening 6 models, three 20-round frozen victim runs, two 20-round self-play runs, and gauntlet evaluation — ran on a single Vast.ai RTX 3090 for under $2 total.
