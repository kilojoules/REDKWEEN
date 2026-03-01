"""Render MP4 animations of the REDKWEEN experiments.

Produces two videos:
  1. frozen_victim_animation.mp4  — adversary learning against a static target
  2. selfplay_animation.mp4      — adversary vs hardening victim

Each video shows:
  - Two animated blobs (adversary, victim) with speech bubbles
  - Real attack text, victim responses, and judge verdicts
  - A live ASR curve drawn as rounds progress

Usage:
  python render_animation.py                          # Both animations
  python render_animation.py --experiment frozen      # Frozen victim only
  python render_animation.py --experiment selfplay    # Self-play only
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BG_COLOR = "#0f0f1a"
PANEL_COLOR = "#1a1a2e"
ADV_COLOR = "#ef4444"
ADV_GLOW = "#ff6b6b"
VIC_COLOR = "#3b82f6"
VIC_GLOW = "#60a5fa"
JUDGE_SAFE = "#22c55e"
JUDGE_UNSAFE = "#ef4444"
TEXT_COLOR = "#e2e8f0"
DIM_TEXT = "#64748b"
GRID_COLOR = "#2a2a3e"
ASR_LINE = "#ef4444"
ASR_FILL = "#ef444433"
BUBBLE_ADV = "#2a1a1a"
BUBBLE_VIC = "#1a1a2a"

FPS = 30
ROUND_SECONDS = 4.0  # seconds per round
INTRO_SECONDS = 2.0
OUTRO_SECONDS = 3.0
FRAMES_PER_ROUND = int(FPS * ROUND_SECONDS)
INTRO_FRAMES = int(FPS * INTRO_SECONDS)
OUTRO_FRAMES = int(FPS * OUTRO_SECONDS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "images")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment(exp_dir, num_rounds=20):
    """Load round data and metrics from an experiment directory."""
    rounds_data = []
    for r in range(num_rounds):
        path = os.path.join(exp_dir, "rounds", f"round_{r}.jsonl")
        entries = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        rounds_data.append(entries)

    metrics = []
    metrics_path = os.path.join(exp_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
    metrics.sort(key=lambda m: m["round"])

    return rounds_data, metrics


def pick_exchanges(round_data, count=3):
    """Pick representative exchanges from a round.

    Prioritizes successful jailbreaks, then picks diverse failures.
    Returns list of (attack, response, is_unsafe) tuples.
    """
    wins = [e for e in round_data if e.get("unsafe", False)]
    losses = [e for e in round_data if not e.get("unsafe", False)]

    # Filter out boring bare-refusal attacks (< 40 chars)
    interesting_losses = [e for e in losses if len(e["attack"]) > 40]

    picks = []
    # Always show a win first if available
    for w in wins[:2]:
        picks.append((w["attack"], w["response"], True))
    # Fill remaining with interesting losses
    for l in interesting_losses[:count - len(picks)]:
        picks.append((l["attack"], l["response"], False))
    # If still short, use boring losses
    for l in losses[:count - len(picks)]:
        picks.append((l["attack"], l["response"], False))

    return picks[:count]


def truncate(text, max_chars=120):
    """Truncate text and add ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "\u2026"


def wrap_text(text, width=45):
    """Wrap text for display in bubbles."""
    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_blob(ax, x, y, radius, color, glow_color, pulse=0.0, label=None):
    """Draw an animated blob with glow effect."""
    r = radius * (1.0 + 0.08 * np.sin(pulse * 2 * np.pi))
    # Outer glow
    for i in range(3):
        glow_r = r * (1.3 + i * 0.15)
        alpha = 0.08 - i * 0.02
        circle = plt.Circle((x, y), glow_r, color=glow_color, alpha=alpha,
                             transform=ax.transData, zorder=2)
        ax.add_patch(circle)
    # Main blob
    circle = plt.Circle((x, y), r, color=color, alpha=0.9, zorder=3)
    ax.add_patch(circle)
    # Label
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground=color)])


def draw_speech_bubble(ax, x, y, text, color, bg_color, align="left",
                       alpha=1.0, fontsize=7.5, max_width=42):
    """Draw a speech bubble with wrapped text."""
    wrapped = wrap_text(truncate(text, max_chars=160), width=max_width)
    lines = wrapped.count("\n") + 1
    box_h = lines * 0.038 + 0.02
    box_w = 0.38

    if align == "left":
        box_x = x
    else:
        box_x = x - box_w

    bbox = dict(boxstyle="round,pad=0.015", facecolor=bg_color,
                edgecolor=color, alpha=alpha * 0.85, linewidth=1.5)
    ax.text(box_x + box_w / 2, y, wrapped, fontsize=fontsize, color=TEXT_COLOR,
            ha="center", va="center", family="monospace",
            transform=ax.transAxes, bbox=bbox, alpha=alpha, zorder=5)


def draw_judge_verdict(ax, x, y, is_unsafe, alpha=1.0):
    """Draw the judge verdict badge."""
    if is_unsafe:
        text = "UNSAFE"
        color = JUDGE_UNSAFE
        bg = "#2a1a1a"
    else:
        text = "SAFE"
        color = JUDGE_SAFE
        bg = "#1a2a1a"
    bbox = dict(boxstyle="round,pad=0.008", facecolor=bg,
                edgecolor=color, alpha=alpha * 0.9, linewidth=2)
    ax.text(x, y, f"  Judge: {text}  ", fontsize=9, fontweight="bold",
            color=color, ha="center", va="center",
            transform=ax.transAxes, bbox=bbox, alpha=alpha, zorder=6)


# ---------------------------------------------------------------------------
# Main animation
# ---------------------------------------------------------------------------

def render(exp_dir, output_path, title, subtitle, num_rounds=20):
    """Render the full animation for one experiment."""
    rounds_data, metrics = load_experiment(exp_dir, num_rounds)
    asr_values = [m["asr"] * 100 for m in metrics[:num_rounds]]

    # Pre-pick exchanges for each round
    exchanges_per_round = []
    for r in range(num_rounds):
        exchanges_per_round.append(pick_exchanges(rounds_data[r], count=2))

    total_frames = INTRO_FRAMES + num_rounds * FRAMES_PER_ROUND + OUTRO_FRAMES

    # --- Figure setup ---
    fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08)

    # Two axes: conversation area (top) and ASR chart (bottom)
    ax_conv = fig.add_axes([0.04, 0.38, 0.92, 0.58], facecolor=PANEL_COLOR)
    ax_asr = fig.add_axes([0.08, 0.06, 0.84, 0.28], facecolor=PANEL_COLOR)

    # ASR axis setup
    ax_asr.set_xlim(-0.5, num_rounds - 0.5)
    ax_asr.set_ylim(0, max(max(asr_values) * 1.15, 10))
    ax_asr.set_xlabel("Round", fontsize=11, color=TEXT_COLOR, labelpad=8)
    ax_asr.set_ylabel("ASR %", fontsize=11, color=TEXT_COLOR, labelpad=8)
    ax_asr.tick_params(colors=DIM_TEXT, labelsize=9)
    ax_asr.set_xticks(range(0, num_rounds, 2))
    ax_asr.grid(True, alpha=0.2, color=GRID_COLOR)
    for spine in ax_asr.spines.values():
        spine.set_color(GRID_COLOR)

    # Conversation axis — no ticks
    ax_conv.set_xlim(0, 1)
    ax_conv.set_ylim(0, 1)
    ax_conv.set_xticks([])
    ax_conv.set_yticks([])
    for spine in ax_conv.spines.values():
        spine.set_color(GRID_COLOR)

    # Persistent ASR line data
    asr_xs = []
    asr_ys = []
    asr_line, = ax_asr.plot([], [], "o-", color=ASR_LINE, linewidth=2.5,
                             markersize=7, zorder=5, markeredgecolor="white",
                             markeredgewidth=1)
    asr_fill = None

    # Title
    title_text = fig.text(0.5, 0.96, title, ha="center", va="center",
                          fontsize=20, fontweight="bold", color=TEXT_COLOR,
                          family="sans-serif")

    def animate(frame):
        nonlocal asr_fill

        # Clear conversation area each frame
        ax_conv.clear()
        ax_conv.set_xlim(0, 1)
        ax_conv.set_ylim(0, 1)
        ax_conv.set_xticks([])
        ax_conv.set_yticks([])
        ax_conv.set_facecolor(PANEL_COLOR)
        for spine in ax_conv.spines.values():
            spine.set_color(GRID_COLOR)

        # --- Intro ---
        if frame < INTRO_FRAMES:
            t = frame / INTRO_FRAMES
            alpha = min(t * 2, 1.0)
            ax_conv.text(0.5, 0.6, title, ha="center", va="center",
                         fontsize=22, fontweight="bold", color=TEXT_COLOR,
                         alpha=alpha, transform=ax_conv.transAxes)
            ax_conv.text(0.5, 0.45, subtitle, ha="center", va="center",
                         fontsize=13, color=DIM_TEXT, alpha=alpha,
                         transform=ax_conv.transAxes)
            # Draw idle blobs
            pulse = frame / FPS
            draw_blob(ax_conv, 0.18, 0.2, 0.06, ADV_COLOR, ADV_GLOW,
                      pulse=pulse, label="ADV\n1B")
            draw_blob(ax_conv, 0.82, 0.2, 0.08, VIC_COLOR, VIC_GLOW,
                      pulse=pulse + 0.5, label="VIC\n8B")
            return

        # --- Outro ---
        if frame >= INTRO_FRAMES + num_rounds * FRAMES_PER_ROUND:
            outro_f = frame - (INTRO_FRAMES + num_rounds * FRAMES_PER_ROUND)
            t = outro_f / OUTRO_FRAMES
            alpha = max(1.0 - t * 1.5, 0.0)

            # Final stats
            final_asr = asr_values[-1]
            mean_asr = np.mean(asr_values)
            peak_asr = max(asr_values)
            peak_round = asr_values.index(peak_asr)

            ax_conv.text(0.5, 0.75, f"Final ASR: {final_asr:.1f}%",
                         ha="center", va="center", fontsize=24,
                         fontweight="bold", color=ADV_COLOR, alpha=alpha,
                         transform=ax_conv.transAxes)
            ax_conv.text(0.5, 0.58,
                         f"Peak: {peak_asr:.1f}% (round {peak_round})  |  "
                         f"Mean: {mean_asr:.1f}%",
                         ha="center", va="center", fontsize=14,
                         color=TEXT_COLOR, alpha=alpha,
                         transform=ax_conv.transAxes)
            ax_conv.text(0.5, 0.42,
                         f"{num_rounds} rounds  \u00d7  "
                         f"{len(rounds_data[0])} candidates  =  "
                         f"{num_rounds * len(rounds_data[0]):,} attacks",
                         ha="center", va="center", fontsize=12,
                         color=DIM_TEXT, alpha=alpha,
                         transform=ax_conv.transAxes)

            pulse = frame / FPS
            draw_blob(ax_conv, 0.18, 0.15, 0.06, ADV_COLOR, ADV_GLOW,
                      pulse=pulse, label="ADV\n1B")
            draw_blob(ax_conv, 0.82, 0.15, 0.08, VIC_COLOR, VIC_GLOW,
                      pulse=pulse + 0.5, label="VIC\n8B")
            return

        # --- Main round animation ---
        round_frame = frame - INTRO_FRAMES
        current_round = round_frame // FRAMES_PER_ROUND
        frame_in_round = round_frame % FRAMES_PER_ROUND
        t = frame_in_round / FRAMES_PER_ROUND  # 0..1 progress through round

        if current_round >= num_rounds:
            return

        # Pulse based on time
        pulse = frame / FPS

        # Round info
        asr = asr_values[current_round]
        wins = sum(1 for e in rounds_data[current_round] if e.get("unsafe"))
        total = len(rounds_data[current_round])
        exchanges = exchanges_per_round[current_round]

        # --- Draw blobs ---
        # Adversary pulses stronger when it has wins
        adv_pulse_amp = 1.0 + (wins / max(total, 1)) * 2
        draw_blob(ax_conv, 0.12, 0.25, 0.055, ADV_COLOR, ADV_GLOW,
                  pulse=pulse * adv_pulse_amp, label="ADV\n1B")
        draw_blob(ax_conv, 0.88, 0.25, 0.07, VIC_COLOR, VIC_GLOW,
                  pulse=pulse * 0.7 + 0.5, label="VIC\n8B")

        # --- Round header ---
        ax_conv.text(0.5, 0.95,
                     f"Round {current_round}",
                     ha="center", va="center", fontsize=16, fontweight="bold",
                     color=TEXT_COLOR, transform=ax_conv.transAxes)
        ax_conv.text(0.5, 0.88,
                     f"{wins}/{total} jailbreaks  \u2022  ASR: {asr:.1f}%",
                     ha="center", va="center", fontsize=11,
                     color=ADV_COLOR if wins > 0 else DIM_TEXT,
                     transform=ax_conv.transAxes)

        # --- Speech bubbles ---
        # Phase the exchanges across the round duration
        num_ex = len(exchanges)
        if num_ex > 0:
            # Each exchange gets a slice of the round
            ex_duration = 0.85 / num_ex  # leave room at start/end

            for i, (attack, response, is_unsafe) in enumerate(exchanges):
                ex_start = 0.08 + i * ex_duration
                ex_end = ex_start + ex_duration

                if t < ex_start:
                    continue

                # Fade in/out within this exchange's window
                local_t = (t - ex_start) / ex_duration
                if local_t < 0.15:
                    alpha = local_t / 0.15
                elif local_t > 0.85:
                    alpha = (1.0 - local_t) / 0.15
                else:
                    alpha = 1.0
                alpha = max(0, min(1, alpha))

                # Vertical position for this exchange
                y_base = 0.68 - i * 0.28

                # Attack bubble (left side)
                if local_t > 0.0:
                    atk_alpha = alpha
                    draw_speech_bubble(ax_conv, 0.20, y_base, attack,
                                       ADV_COLOR, BUBBLE_ADV, align="left",
                                       alpha=atk_alpha)
                    ax_conv.text(0.20, y_base + 0.07, "Attack:",
                                fontsize=8, color=ADV_COLOR, alpha=atk_alpha,
                                fontweight="bold", transform=ax_conv.transAxes)

                # Response bubble (right side)
                if local_t > 0.25:
                    resp_alpha = alpha * min((local_t - 0.25) / 0.15, 1.0)
                    draw_speech_bubble(ax_conv, 0.98, y_base - 0.13, response,
                                       VIC_COLOR, BUBBLE_VIC, align="right",
                                       alpha=resp_alpha)
                    ax_conv.text(0.80, y_base - 0.06, "Response:",
                                fontsize=8, color=VIC_COLOR, alpha=resp_alpha,
                                fontweight="bold", transform=ax_conv.transAxes)

                # Judge verdict
                if local_t > 0.5:
                    v_alpha = alpha * min((local_t - 0.5) / 0.15, 1.0)
                    draw_judge_verdict(ax_conv, 0.5, y_base - 0.22,
                                       is_unsafe, alpha=v_alpha)

                    # Flash effect on unsafe
                    if is_unsafe and 0.5 < local_t < 0.65:
                        flash = (0.65 - local_t) / 0.15
                        ax_conv.axhspan(0, 1, alpha=flash * 0.08,
                                        color=JUDGE_UNSAFE, zorder=1,
                                        transform=ax_conv.transAxes)

        # --- Update ASR chart ---
        # Add point at the end of each round
        if frame_in_round == FRAMES_PER_ROUND - 1:
            asr_xs.append(current_round)
            asr_ys.append(asr)
        # Or if this round was already added, just update line
        if asr_xs:
            asr_line.set_data(asr_xs, asr_ys)
            # Update fill
            if asr_fill is not None:
                asr_fill.remove()
            asr_fill = ax_asr.fill_between(asr_xs, asr_ys, alpha=0.15,
                                            color=ASR_LINE, zorder=2)
            # Vertical marker for current round
            ax_asr.axvline(current_round, color=TEXT_COLOR, alpha=0.3,
                           linestyle="--", linewidth=0.8, zorder=1)

        return

    print(f"Rendering {total_frames} frames ({total_frames / FPS:.0f}s) ...")

    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                    interval=1000 / FPS, blit=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render REDKWEEN experiment animations")
    parser.add_argument("--experiment", type=str, default="both",
                        choices=["frozen", "selfplay", "both"],
                        help="Which experiment to render")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    exp_base = os.path.join(SCRIPT_DIR, "experiments")

    if args.experiment in ("frozen", "both"):
        render(
            exp_dir=os.path.join(exp_base, "frozen_victim_v2"),
            output_path=os.path.join(args.output_dir, "redkween_frozen_victim.mp4"),
            title="REDKWEEN: Frozen Victim",
            subtitle="Adversary learns to jailbreak a static 8B victim over 20 rounds",
            num_rounds=args.rounds,
        )

    if args.experiment in ("selfplay", "both"):
        # Use selfplay_v3 (with benign mixing) as it's more interesting
        render(
            exp_dir=os.path.join(exp_base, "selfplay_v3"),
            output_path=os.path.join(args.output_dir, "redkween_selfplay.mp4"),
            title="REDKWEEN: Self-Play",
            subtitle="Adversary vs hardening victim \u2014 the defender wins",
            num_rounds=args.rounds,
        )


if __name__ == "__main__":
    main()
