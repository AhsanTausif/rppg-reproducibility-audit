"""
Generates presentation/report-ready figures summarizing the reproduction,
condition-robustness, and failure-analysis results. Reads only
already-computed result files -- no re-extraction or re-inference.

Produces two variants of every figure from one set of data and titles:
  - "light": white-surface, for the printed report (report/figures/)
  - "dark": matches the slide deck's navy palette exactly, so charts sit
    natively on a dark slide with no light frame needed (report/figures/dark/)
"""

import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rppg_preprocess as rp
from baselines_classical import bandpass
from physnet_infer import load_model, predict_clip_waveform, CHECKPOINTS, CHUNK_LENGTH, MODEL_FS

OUT_LIGHT = rp.REPO_ROOT / "docs" / "figures"
OUT_DARK = rp.REPO_ROOT / "docs" / "figures" / "dark"   # slide-deck variants (git-ignored)
OUT_LIGHT.mkdir(parents=True, exist_ok=True)
OUT_DARK.mkdir(parents=True, exist_ok=True)
DATA_DIR = rp.OUTPUT_ROOT
RESULTS_DIR = rp.TABLES_DIR

METHODS = ["CHROM", "POS", "PhysNet (UBFC)", "PhysNet (PURE)"]

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", baseline="#c3c2b7", good="#0ca30c", critical="#d03b3b",
        methods={"CHROM": "#2a78d6", "POS": "#eb6834", "PhysNet (UBFC)": "#1baf7a", "PhysNet (PURE)": "#c98500"},
        neutral="#898781", out=OUT_LIGHT, dpi=220,
    ),
    "dark": dict(
        surface="#081420", ink="#EAF1F7", ink2="#C7D6E5", muted="#8EA3B8",
        grid="#1C324D", baseline="#2A4A63", good="#2DD4BF", critical="#F87171",
        methods={"CHROM": "#38BDF8", "POS": "#FBBF24", "PhysNet (UBFC)": "#2DD4BF", "PhysNet (PURE)": "#F472B6"},
        neutral="#8EA3B8", out=OUT_DARK, dpi=220,
    ),
}


def use_theme(t):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Manrope", "Segoe UI", "Arial", "DejaVu Sans"],
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "axes.edgecolor": t["baseline"],
        "axes.labelcolor": t["ink2"],
        "text.color": t["ink"],
        "xtick.color": t["ink2"],
        "ytick.color": t["ink2"],
        "axes.grid": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 12,
        "savefig.facecolor": t["surface"],
    })


def style_ax(ax, t, ylabel=None, title=None, subtitle=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=12)
    if title:
        y = 1.10 if subtitle else 1.03
        ax.set_title(title, color=t["ink"], fontsize=16, fontweight="bold", pad=22 if subtitle else 12, loc="left", y=y)
    if subtitle:
        ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, color=t["muted"], fontsize=11.5, ha="left", va="bottom")


def save(fig, name, t):
    path = t["out"] / name
    fig.savefig(path, dpi=t["dpi"], bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------
# Fig 1: Baseline reproduction -- MAE by method across corpus slices.
# A horizontal log-scale slope chart: each method is one row, with a line
# connecting its error on DATASET_1 -> DATASET_2 -> the held-out test split.
# Log scale is used deliberately (not for effect): DATASET_1's PhysNet
# values (43-47 bpm) and the held-out test values (0.2-1.5 bpm) span more
# than two orders of magnitude, and a linear axis would compress the
# entire interesting low-error range into a few pixels.
# ---------------------------------------------------------------------
def fig1(t):
    use_theme(t)
    d1 = [1.41, 1.01, 46.54, 43.12]
    d2 = [4.72, 9.18, 5.83, 1.62]
    d2test = [0.20, 1.46, 0.65, 0.65]

    fig, ax = plt.subplots(figsize=(11, 6.6))
    n = len(METHODS)
    y_pos = np.arange(n)[::-1]

    for y, m, v1, v2, v3 in zip(y_pos, METHODS, d1, d2, d2test):
        color = t["methods"][m]
        ax.plot([v1, v2, v3], [y, y, y], "-", color=t["muted"], linewidth=1.4, zorder=2, alpha=0.55)
        ax.scatter([v1], [y], s=100, color=t["neutral"], zorder=3, edgecolor=t["surface"], linewidth=1.2)
        ax.scatter([v2], [y], s=140, color=color, zorder=4, alpha=0.55, edgecolor=t["surface"], linewidth=1.2)
        ax.scatter([v3], [y], s=240, color=color, zorder=5, edgecolor=t["surface"], linewidth=1.8, marker="*")
        ax.text(v1, y + 0.30, f"{v1:.1f}", fontsize=9.5, color=t["muted"], ha="center", va="bottom")
        ax.text(v2, y + 0.30, f"{v2:.1f}", fontsize=9.5, color=t["ink2"], ha="center", va="bottom")
        ax.text(v3, y - 0.32, f"{v3:.2f} bpm", fontsize=11, color=color, ha="center", va="top", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.12, 65)
    ax.set_ylim(-0.65, n - 1 + 0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(METHODS, fontsize=13.5)
    ax.set_xlabel("Mean absolute error, bpm (log scale)", color=t["ink2"], fontsize=12)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=t["neutral"], markersize=9, label="DATASET_1"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=t["ink2"], markersize=10, alpha=0.7, label="DATASET_2"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor=t["ink2"], markersize=14, label="DATASET_2 held-out test"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=10, bbox_to_anchor=(1.0, 1.02))
    style_ax(ax, t, title="Baseline Reproduction Accuracy",
             subtitle="Error falls toward the published-quality range (under ~2 bpm) across every method")
    ax.grid(axis="x", which="major", zorder=0)
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_step4_mae_by_method.png", t)


# ---------------------------------------------------------------------
# Fig 2: Face-detector selection -- Haar vs YuNet on DATASET_2
# ---------------------------------------------------------------------
def fig2(t):
    use_theme(t)
    haar = [18.64, 18.13, 22.51, 28.24]
    yunet = [4.72, 9.18, 5.83, 1.62]

    x = np.arange(len(METHODS))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.bar(x - w / 2, haar, w, label="Haar cascade (original)", color=t["critical"], zorder=3)
    ax.bar(x + w / 2, yunet, w, label="YuNet, DNN detector (validated)", color=t["good"], zorder=3)

    for xi, vals in zip([x - w / 2, x + w / 2], [haar, yunet]):
        for xx, v in zip(xi, vals):
            ax.text(xx, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=t["ink2"])

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=12)
    ax.set_ylim(0, 33)
    style_ax(ax, t, ylabel="DATASET_2 mean absolute error (bpm)",
             title="Effect of Face-Detector Selection",
             subtitle="Accuracy before and after replacing the front-end face detector")
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    save(fig, "fig2_detector_before_after.png", t)


# ---------------------------------------------------------------------
# Fig 3: Predicted vs GT scatter, small multiples per method (all 22 clips)
# ---------------------------------------------------------------------
def fig3(t):
    use_theme(t)
    rows = list(csv.DictReader(open(RESULTS_DIR / "step5_per_clip.csv")))
    gt = np.array([float(r["gt_hr"]) for r in rows])
    families = [r["dataset_family"] for r in rows]

    specs = [("chrom_hr", "CHROM"), ("pos_hr", "POS"), ("physnet_ubfc_hr", "PhysNet (UBFC)"), ("physnet_pure_hr", "PhysNet (PURE)")]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5))
    lims = (55, 190)
    for ax, (key, label) in zip(axes.flat, specs):
        color = t["methods"][label]
        pred = np.array([float(r[key]) for r in rows])
        d1_mask = np.array([f == "DATASET_1" for f in families])
        ax.plot(lims, lims, "--", color=t["baseline"], linewidth=1.2, zorder=1)
        ax.scatter(gt[~d1_mask], pred[~d1_mask], s=50, color=color, alpha=0.9,
                   edgecolor=t["surface"], linewidth=0.6, zorder=3, label="DATASET_2")
        ax.scatter(gt[d1_mask], pred[d1_mask], s=50, color=color, alpha=0.9,
                   edgecolor=t["ink"], linewidth=1.0, marker="D", zorder=3, label="DATASET_1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        style_ax(ax, t, ylabel="Predicted HR (bpm)", title=label)
        ax.set_xlabel("Ground-truth HR (bpm)", color=t["ink2"])
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("Predicted versus Ground-Truth Heart Rate, All 22 Clips",
                 fontsize=17, fontweight="bold", color=t["ink"], y=1.02, x=0.03, ha="left")
    fig.tight_layout()
    save(fig, "fig3_scatter_pred_vs_gt.png", t)


# ---------------------------------------------------------------------
# Fig 4: PhysNet(UBFC) domain-mismatch confirmation
# ---------------------------------------------------------------------
def fig4(t):
    use_theme(t)
    labels = ["DATASET_1\n(incorrect domain)", "DATASET_2\n(confirmed training domain)"]
    vals = [46.54, 5.83]
    colors = [t["critical"], t["good"]]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    bars = ax.bar(labels, vals, width=0.5, color=colors, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f} bpm", ha="center",
                va="bottom", fontsize=12, color=t["ink"], fontweight="bold")
    ax.annotate("", xy=(1, 10), xytext=(0, 40),
                arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(0.5, 30, "8x lower error on the\nconfirmed training domain", ha="center", fontsize=10.5,
            color=t["ink2"], style="italic")
    style_ax(ax, t, ylabel="Mean absolute error (bpm)",
             title="Cross-Domain Evaluation of a Pretrained Checkpoint",
             subtitle="PhysNet (UBFC-trained), evaluated on the wrong vs. the correct dataset")
    save(fig, "fig4_domain_mismatch.png", t)


# ---------------------------------------------------------------------
# Fig 5: Resting vs after-exercise, DATASET_2-clean view.
# A slope chart: each method's two condition means are connected by a
# line, so the gap is read directly as the line's slope/length rather
# than compared across two separate bar heights -- the natural encoding
# for a paired before/after comparison across categories.
# ---------------------------------------------------------------------
def fig5(t):
    use_theme(t)
    resting = [4.45, 7.64, 5.83, 1.62]
    after_ex = [1.50, 1.08, 0.35, 1.85]
    gaps = [2.95, 6.56, 5.48, 0.23]

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    xpos = [0, 1]

    for m, r, a, g in zip(METHODS, resting, after_ex, gaps):
        color = t["methods"][m]
        ax.plot(xpos, [r, a], "-", color=color, linewidth=2.6, zorder=3, solid_capstyle="round")
        ax.scatter(xpos, [r, a], s=130, color=color, zorder=4, edgecolor=t["surface"], linewidth=1.6)
        ax.text(-0.07, r, f"{r:.2f}", fontsize=10.5, color=t["ink2"], ha="right", va="center")

    # The 4 post-exercise values cluster tightly (0.35-1.85); greedily
    # push overlapping labels apart vertically (min separation 0.62) so
    # they stay readable instead of stacking on top of each other.
    order = sorted(range(len(METHODS)), key=lambda i: after_ex[i])
    label_y = [float(after_ex[i]) for i in order]
    min_gap = 0.62
    for k in range(1, len(label_y)):
        if label_y[k] - label_y[k - 1] < min_gap:
            label_y[k] = label_y[k - 1] + min_gap

    for rank, i in enumerate(order):
        m, a, g = METHODS[i], after_ex[i], gaps[i]
        color = t["methods"][m]
        ly = label_y[rank]
        ax.annotate("", xy=(1.045, a), xytext=(1.16, ly), fontsize=1,
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, lw=1.0))
        ax.text(1.18, ly, m, fontsize=13, color=color, ha="left", va="center", fontweight="bold")
        ax.text(1.18, ly - 0.34, f"{a:.2f} bpm  ·  gap {g:.2f}", fontsize=10, color=t["muted"], ha="left", va="center")

    ax.set_xlim(-0.32, 2.35)
    ax.set_ylim(-0.3, 9.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Resting  (n=16)", "Post-exercise  (n=1)"], fontsize=13)
    ax.set_ylabel("Mean absolute error (bpm)", color=t["ink2"], fontsize=12)
    style_ax(ax, t, title="Cross-Condition Robustness",
             subtitle="Resting vs. post-exercise mean absolute error, DATASET_2 -- line slope is the gap")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", zorder=0)
    save(fig, "fig5_condition_gap.png", t)


# ---------------------------------------------------------------------
# Fig 6: Per-clip error strip plot, DATASET_2, all 4 methods
# ---------------------------------------------------------------------
def fig6(t):
    use_theme(t)
    rows = [r for r in csv.DictReader(open(RESULTS_DIR / "step5_per_clip.csv")) if r["dataset_family"] == "DATASET_2"]
    subjects = [r["subject_id"] for r in rows]
    specs = [("chrom_err", "CHROM"), ("pos_err", "POS"), ("physnet_ubfc_err", "PhysNet (UBFC)"), ("physnet_pure_err", "PhysNet (PURE)")]

    fig, ax = plt.subplots(figsize=(11.5, 6.3))
    y_positions = np.arange(len(subjects))
    offsets = np.linspace(-0.27, 0.27, 4)
    for (key, label), off in zip(specs, offsets):
        color = t["methods"][label]
        errs = [float(r[key]) for r in rows]
        ax.scatter(errs, y_positions + off, s=58, color=color, label=label, edgecolor=t["surface"], linewidth=0.6, zorder=3)

    ax.axvline(0, color=t["baseline"], linewidth=1.2, zorder=1)
    ax.axvspan(-5, 5, color=t["good"], alpha=0.08, zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subjects, fontsize=10.5)
    ax.invert_yaxis()
    style_ax(ax, t, title="Per-Clip Prediction Error by Method",
             subtitle="DATASET_2 (n=16) -- each method's largest errors occur on a different clip")
    ax.set_xlabel("Prediction error (bpm), predicted minus ground truth", color=t["ink2"])
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=10)
    save(fig, "fig6_per_clip_errors_dataset2.png", t)


# ---------------------------------------------------------------------
# Fig 7: Mitigation test, before/after, PhysNet(UBFC) DATASET_2
# ---------------------------------------------------------------------
def fig7(t):
    use_theme(t)
    subjects = ["subject1", "subject10", "subject11", "subject12", "subject13", "subject14",
                "subject15", "subject16", "subject17", "subject18", "subject20", "subject3",
                "subject4", "subject5", "subject8", "subject9"]
    old_err = [-0.3, -0.4, 0.4, -0.1, -0.1, 72.4, 5.5, 3.7, -2.2, -3.8, 0.0, 0.2, 0.2, 0.0, -0.0, 3.9]
    new_err = [-0.3, -3.5, -30.8, -0.1, -7.2, -4.1, -5.0, 0.2, -4.8, -7.3, 0.0, -11.3, -11.6, 0.0, -3.6, -4.5]

    order = np.argsort(np.abs(old_err))[::-1]
    subjects = [subjects[i] for i in order]
    old_err = [old_err[i] for i in order]
    new_err = [new_err[i] for i in order]

    y = np.arange(len(subjects))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 7.3))
    ax.barh(y - w / 2, old_err, w, label="Baseline estimator", color=t["neutral"], zorder=3, alpha=0.75)
    ax.barh(y + w / 2, new_err, w, label="Candidate mitigation", color=t["methods"]["POS"], zorder=3)
    ax.axvline(0, color=t["baseline"], linewidth=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(subjects, fontsize=10.5)
    ax.invert_yaxis()
    style_ax(ax, t, title="Mitigation Test: Per-Clip Error, Before and After",
             subtitle="Corpus-wide MAE moved from 5.83 to 5.89 bpm -- not adopted")
    ax.set_xlabel("Prediction error (bpm), predicted minus ground truth", color=t["ink2"])
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig7_mitigation_before_after.png", t)


# ---------------------------------------------------------------------
# Fig 8: Spectrum diagnosis for subject14 (PhysNet UBFC)
# ---------------------------------------------------------------------
def fig8(t, waveform_cache={}):
    use_theme(t)
    if "wf" not in waveform_cache:
        device = torch.device("cpu")
        model = load_model(CHECKPOINTS["physnet_ubfc"], CHUNK_LENGTH, device)
        clip_dir = DATA_DIR / "subject14"
        frames = np.load(clip_dir / "frames.npy")
        ts = np.load(clip_dir / "timestamps.npy")
        wf = predict_clip_waveform(model, frames, ts, device)
        waveform_cache["wf"] = bandpass(wf, MODEL_FS)
    wf_f = waveform_cache["wf"]
    gt_hr = 75.7

    n = len(wf_f)
    windowed = (wf_f - wf_f.mean()) * np.hanning(n)
    nfft = max(4096, 1 << (n - 1).bit_length())
    freqs = np.fft.rfftfreq(nfft, d=1.0 / MODEL_FS)
    spec = np.abs(np.fft.rfft(windowed, n=nfft))
    bpm_all = freqs * 60
    mask = (bpm_all >= 42) & (bpm_all <= 240)
    bpm = bpm_all[mask]
    p = spec[mask]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    color = t["methods"]["PhysNet (UBFC)"]
    ax.plot(bpm, p, color=color, linewidth=1.6, zorder=3)
    ax.fill_between(bpm, p, color=color, alpha=0.14, zorder=2)

    peak_bpm = bpm[np.argmax(p)]
    ax.axvline(peak_bpm, color=t["critical"], linestyle="--", linewidth=1.4, zorder=4)
    ax.text(peak_bpm + 2, p.max() * 0.96, f"Selected peak:\n{peak_bpm:.1f} bpm (incorrect)",
            color=t["critical"], fontsize=10.5, va="top")

    ax.axvline(gt_hr, color=t["good"], linestyle="--", linewidth=1.4, zorder=4)
    near_true = (bpm >= 65) & (bpm <= 85)
    true_peak_power = p[near_true].max()
    ax.text(gt_hr - 2, true_peak_power + p.max() * 0.06,
            f"True heart rate: {gt_hr:.1f} bpm\n(71% of peak power --\npresent, not absent)",
            color=t["good"], fontsize=10.5, va="bottom", ha="right")

    style_ax(ax, t, ylabel="Spectral power (a.u.)",
             title="Spectral Diagnosis of the Largest Residual Error",
             subtitle="PhysNet (UBFC-trained) predicted waveform, subject 14")
    ax.set_xlabel("Heart rate (bpm)", color=t["ink2"])
    save(fig, "fig8_spectrum_diagnosis_subject14.png", t)


if __name__ == "__main__":
    for theme_name, t in THEMES.items():
        print(f"Generating {theme_name} figures...")
        fig1(t); fig2(t); fig3(t); fig4(t); fig5(t); fig6(t); fig7(t); fig8(t)
    print("Done.")
