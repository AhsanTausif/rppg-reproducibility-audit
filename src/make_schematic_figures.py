"""
Three additional report illustrations requested in the revision pass:
  fig9_research_questions.png   - 1.3 Problem Statement, replaces the RQ table
  fig10_literature_landscape.png - Section 2, replaces the 2.7 Synthesis prose
  fig11_implementation_workflow.png - Section 4 opener, a six-step flowchart
All light-theme, matched to make_figures.py's existing report palette.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
GOLD = "#c98500"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Manrope", "Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# ============================================================
# Figure: Research questions (three cards, replaces RQ table)
# ============================================================
def fig_research_questions():
    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=220)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    cards = [
        (BLUE, "RQ1", "Reproduction",
         "Can CHROM, POS, and a pretrained\nPhysNet be reproduced to published\naccuracy on a subject-independent split?"),
        (GREEN, "RQ2", "Robustness",
         "Does accuracy hold across a real\ncondition axis (resting vs. post-exercise),\nor is any gap a pipeline artifact?"),
        (ORANGE, "RQ3", "Failure Diagnosis",
         "When a method fails on one subject, is\nthe failure explainable, and does a fix\ngeneralize or just relocate the failure?"),
    ]

    for i, (color, tag, title, body) in enumerate(cards):
        cx = 0.5 + i
        # connecting arrow
        if i > 0:
            ax.annotate("", xy=(cx - 0.5 + 0.06, 0.5), xytext=(cx - 1 + 0.5 - 0.06, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=BASELINE, lw=1.6, shrinkA=0, shrinkB=0))
        box = FancyBboxPatch((cx - 0.44, 0.08), 0.88, 0.84,
                              boxstyle="round,pad=0.02,rounding_size=0.05",
                              linewidth=1.4, edgecolor=color, facecolor="white", zorder=2)
        ax.add_patch(box)
        circ = Circle((cx, 0.72), 0.11, facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(circ)
        ax.text(cx, 0.72, tag, ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)
        ax.text(cx, 0.50, title, ha="center", va="center", color=INK, fontsize=13.5, fontweight="bold")
        ax.text(cx, 0.26, body, ha="center", va="center", color=INK2, fontsize=9.6, linespacing=1.55)

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig9_research_questions.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure: Literature landscape quadrant (replaces 2.7 Synthesis)
# ============================================================
def fig_literature_landscape():
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # quadrant shading: top-right = the unaudited gap
    ax.add_patch(Rectangle((5, 5), 5, 5, facecolor="#fbeceb", edgecolor="none", zorder=0))
    ax.text(9.7, 9.55, "UNAUDITED\nGAP", ha="right", va="top", color="#c14a42",
            fontsize=11.5, fontweight="bold", linespacing=1.3)

    ax.axvline(5, color=GRID, lw=1.2, zorder=1)
    ax.axhline(5, color=GRID, lw=1.2, zorder=1)

    pts = [
        (2.0, 1.6, BLUE, "Wang et al. [1]\n(CHROM/POS model)", (0.15, -0.55)),
        (7.6, 1.9, ORANGE, "Tulyakov [4]\nYu et al. [5]", (0.2, -0.6)),
        (6.6, 3.3, GREEN, "Sun & Li [2, 6]\n(Contrast-Phys)", (0.25, 0.35)),
        (1.7, 7.7, GOLD, "Bondarenko et al. [3]\n(bias meta-analysis)", (0.25, 0.4)),
    ]
    for x, y, color, label, off in pts:
        ax.scatter([x], [y], s=170, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(label, (x, y), xytext=(x + off[0], y + off[1]), fontsize=9.3, color=INK2,
                    ha="left", va="center", linespacing=1.35)

    # this project: robustness tested, condition-axis (not the original fairness axis) -> plotted near the boundary
    ax.scatter([7.3], [5.0], s=230, marker="*", color="#c14a42", edgecolor="white", linewidth=1.2, zorder=4)
    ax.annotate("This project\n(condition-robustness axis)", (7.3, 5.0), xytext=(6.7, 6.35),
                fontsize=9.3, color="#c14a42", fontweight="bold", ha="left", va="center", linespacing=1.35)

    ax.set_xlabel("Robustness tested (motion, compression, conditions)  \u2192", color=INK2, fontsize=10.5)
    ax.set_ylabel("Fairness audited (skin tone, demographics)  \u2192", color=INK2, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASELINE)

    plt.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig10_literature_landscape.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure: Implementation workflow (Section 4 opener)
# ============================================================
def fig_implementation_workflow():
    fig, ax = plt.subplots(figsize=(10.6, 2.6), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("1", "Environment", "Versioned setup", BASELINE, INK2),
        ("2", "Preprocess", "Face ROI, GT align", BASELINE, INK2),
        ("3", "Split", "Subject-independent", BASELINE, INK2),
        ("4", "Reproduce", "Hard gate vs. published", BLUE, "white"),
        ("5", "Core Eval", "Resting vs. exercise", BLUE, "white"),
        ("6", "Diagnose", "Failure + mitigation", BLUE, "white"),
    ]
    for i, (num, title, sub, fc, tc) in enumerate(steps):
        cx = 0.5 + i
        if i > 0:
            ax.annotate("", xy=(cx - 0.5 + 0.05, 0.5), xytext=(cx - 1 + 0.5 - 0.05, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=BASELINE, lw=1.6, shrinkA=0, shrinkB=0))
        box = FancyBboxPatch((cx - 0.45, 0.12), 0.9, 0.76,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.3 if fc == BLUE else 1.0,
                              edgecolor=BLUE if fc == BLUE else BASELINE,
                              facecolor=fc if fc == BLUE else "white", zorder=2)
        ax.add_patch(box)
        ax.text(cx, 0.66, num, ha="center", va="center", color=tc if fc == BLUE else MUTED,
                fontsize=15, fontweight="bold")
        ax.text(cx, 0.42, title, ha="center", va="center", color=tc if fc == BLUE else INK, fontsize=10.3, fontweight="bold")
        ax.text(cx, 0.24, sub, ha="center", va="center", color=tc if fc == BLUE else MUTED, fontsize=7.8)

    ax.text(3.5, -0.1, "Step 4 is a hard gate: the pipeline does not advance to Steps 5-6 until baseline reproduction passes.",
            ha="center", va="top", color=INK2, fontsize=9, style="italic")

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig11_implementation_workflow.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_research_questions()
    fig_literature_landscape()
    fig_implementation_workflow()
    print("done")
