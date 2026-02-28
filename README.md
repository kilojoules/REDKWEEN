# Chaos: Automated Red Teaming via Self-Play

[**Documentation**](https://kilojoules.github.io/red-team-experiments/)

Can a language model learn to jailbreak another through trial and error? We train adversary models to attack victims in two phases: first against a frozen victim to build attack capability, then in self-play where both sides adapt. A frozen judge (Llama Guard) scores each attempt — successful attacks train the adversary to get better; the victim learns to refuse. We test four adversary–victim matchups across model sizes from 1B to 8B.

**Main finding:** In phase one, adversaries across all matchups discover real jailbreak strategies from scratch, reaching 40–50% ASR against frozen victims. In phase two, even with the adversary's head start, the victim hardens to near-0% ASR within 3–7 rounds — regardless of whether the adversary is smaller, equal, or larger. Defense is always easier than attack. SAE analysis on the victim's hidden states shows that jailbreaks are perfectly linearly separable (AUC=1.0) — the model "knows" it's being attacked even when it complies.

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

| Matchup | Adversary | Victim | Frozen Rounds | Self-Play Rounds |
|---------|-----------|--------|:---:|:---:|
| 1B vs 8B | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | 20 | — |
| 3B vs 8B | Llama-3.2-3B-Instruct | Llama-3.1-8B-Instruct | 20 | 20 |
| 8B vs 8B | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct | 20 | 20 |
| 8B vs 3B | Llama-3.1-8B-Instruct | Llama-3.2-3B-Instruct | 20 | 20 |
| **8B vs 3B (v2)** | **Llama-3.1-8B-Instruct** | **Llama-3.2-3B-Instruct** | **40** | **20** |

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

With the victim frozen, all four adversaries learn to jailbreak their targets over 20 rounds of 200 candidates each (4,000 total attacks).

![Phase 1: Frozen Victim ASR](images/frozen_asr.png)

All matchups converge to a similar 40–50% ASR ceiling regardless of model size. The 1B adversary against the 8B victim actually achieves the highest peak (49.5%), suggesting that adversary capacity matters less than the learning dynamics.

**Extended run (8B vs 3B v2):** We ran the 8B adversary against the 3B victim for 40 rounds (vs. 20 in v1) to test whether ASR continues climbing or plateaus. ASR rose from ~20% to ~48% average over the last 10 rounds, peaking at 70% (round 24) and 66.7% (round 32). The adversary never truly plateaued — high-variance oscillation continued throughout, suggesting the frozen victim has a soft ceiling rather than a hard one.

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

Starting from the phase-one adversary adapters, we enable victim hardening. Even with a ~32–45% ASR head start, the adversary cannot keep up.

![Phase 2: Self-Play ASR](images/selfplay_asr.png)

The collapse speed scales with the victim's capacity advantage: the 3B victim (facing an 8B adversary) takes 7 rounds to harden, the 8B victim facing an equal-capacity adversary takes 4 rounds, and the 8B victim facing a smaller 3B adversary takes just 3 rounds. In all cases, ASR drops to near-0% and stays there.

**Even a 40-round head start doesn't help.** In the extended 8B vs 3B v2 run, the adversary entered self-play with 40 rounds of frozen-victim training (vs. 20 in v1) and an initial ASR of 70%. The 3B victim still crushed it — ASR dropped from 70% to 0% in 6 rounds and stayed there for the remaining 14.

**Why the asymmetry?** The victim needs only a handful of refusal examples per round to harden. The adversary needs hundreds of successful attacks across many rounds to develop strategies (as shown in phase one). Even with a 40–70% head start, the victim's defense rate outpaces the adversary's adaptation.

### Mechanistic Analysis: What the victim "sees" (SAE)

We trained a sparse autoencoder on the 3B victim's middle-layer hidden states across all 40 frozen-victim rounds (1,200 attack/response pairs, 440 unsafe). The goal: understand what internal features distinguish successful jailbreaks from refused attacks.

**Key findings:**

| Metric | Value |
|--------|-------|
| Explained variance | 92% |
| Linear probe AUC | **1.0** |
| Alive features | 295 / 12,288 (2.4%) |
| Avg active per sample | 3.4 |

A linear probe on raw activations achieves perfect separation (AUC=1.0) — the victim's residual stream *always* encodes whether an attack will succeed. The model "knows" it's being jailbroken, even when it complies.

The SAE decomposes this into interpretable features:

![SAE Feature Discrimination](images/sae_discrimination_frozen_8b_v_3b_v2.png)

**Top jailbreak features** (red, positive Cohen's d) fire on CTF-style exercise framing — the adversary's primary learned strategy:
- **F5926** (d=+0.39): "Home Directory Cleanup Utility" exercises
- **F11111** (d=+0.39): "Mysterious File Organizer" CTF challenges
- **F11811** (d=+0.36): sysadmin troubleshooting scenarios

**Top refusal features** (blue, negative Cohen's d) fire on direct/explicit harmful requests — attacks the victim easily blocks.

The low individual feature AUCs (~0.55) despite perfect linear probe AUC show that no single feature is decisive — jailbreak detection is distributed across the representation, requiring a linear combination to achieve perfect classification.

## Open Questions

1. **Can the adversary ever overcome the asymmetry?** Even with a 40-round head start and 70% initial ASR (8B vs 3B v2), the adversary's attack capability is erased within 6 rounds of self-play. We tested adversary sizes from 1B to 8B (including equal capacity) — capacity alone doesn't break the pattern. Would throttled victim learning rate or population-based training change this?

2. **How does historical opponent sampling affect the dynamics?** Instead of always playing the latest opponent, sample from a zoo of historical checkpoints with probability *A*. This is the [A parameter](https://kilojoules.github.io/portfolio/#adversarial-self-play) — it controls the balance between co-evolutionary pressure and curriculum diversity.

3. **Can victim hardening avoid catastrophic forgetting?** Benign mixing helps but doesn't solve over-refusal. Regularization, diverse benign data, or safety-benchmark mixing may be needed.

4. **Can we use the SAE features to build a better defense?** The victim's hidden states perfectly separate jailbreaks from safe requests (AUC=1.0). A lightweight classifier on these features could detect attacks *before* generating a response, potentially without any LoRA fine-tuning.

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
model_utils.py        # HuggingFace/PEFT/BitsAndBytes model operations + hidden state extraction
config.py             # Dataclass config hierarchy + CLI argument parsing
bootstrap.py          # Initial adversary LoRA training on seed data
baselines.py          # Baseline ASR evaluation and victim screening
gauntlet.py           # Cross-round checkpoint evaluation matrix
sae_analysis.py       # SAE-based mechanistic jailbreak analysis
plot_metrics.py       # ASR curve and wins-per-round visualization
experiments/          # Raw experiment data (attacks, responses, verdicts, metrics)
results/sae/          # SAE analysis outputs (activations, trained SAE, reports)
docs/                 # Documentation site (mkdocs-material)
```

## Cost

The full experiment suite — screening 6 models, four frozen victim runs (20–40 rounds each), four self-play runs, gauntlet evaluation, and SAE analysis — ran on Vast.ai RTX 4090 instances for under $15 total.
