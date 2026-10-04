"""
Step 5: cross-condition robustness/fairness-gap analysis -- pooled
DATASET_1 + DATASET_2 (22 clips), re-run after step 4's YuNet face-detector
fix.

Scope (chosen given this dataset's actual labels): this data has no
skin-tone or lighting metadata, so the subgroup axis we can honestly test
is resting vs. after-exercise -- a real physiological/motion condition we
do have ground truth for. CHROM, POS, and BOTH PhysNet checkpoints are
included now that step 4 validated all four methods on DATASET_2.

Important confound, carried over from step 4: PhysNet (both checkpoints)
regressed on DATASET_1 with the new face detector (crop-tightness
mismatch, not yet root-caused). Since "after-exercise" is itself a
DATASET_1 clip, and DATASET_1 supplies 5 of the pooled "resting" clips too,
PhysNet's resting-vs-exercise numbers are confounded by dataset family, not
purely by condition. Reported both ways: the full pooled comparison, and a
DATASET_2-only comparison that avoids the confound. CHROM/POS are
unaffected by this issue and their pooled numbers can be read directly.

The fairness-gap metric is |MAE_condition_A - MAE_condition_B| per method.
With only 1 after-exercise clip this is always a single data point, not a
statistically powered estimate -- reported as such, not smoothed over.
"""

import csv

import numpy as np
import torch

import rppg_preprocess as rp
from baselines_classical import (
    resample_uniform, chrom_dehaan, pos_wang, bandpass, estimate_hr_fft, mae_rmse,
)
from physnet_infer import load_model, predict_clip_waveform, CHECKPOINTS, CHUNK_LENGTH, MODEL_FS

RESULTS_DIR = rp.TABLES_DIR


def per_clip_predictions():
    subjects = rp.discover_subjects(rp.DATASET_ROOT)
    device = torch.device("cpu")
    models = {name: load_model(path, CHUNK_LENGTH, device) for name, path in CHECKPOINTS.items()}

    rows = []
    for s in subjects:
        sid = s["subject_id"]
        clip_dir = rp.OUTPUT_ROOT / sid
        rgb = np.load(clip_dir / "rgb_trace.npy")
        ts = np.load(clip_dir / "timestamps.npy")
        frames = np.load(clip_dir / "frames.npy")
        gt = np.load(clip_dir / "gt.npy")
        gt_time = np.load(clip_dir / "gt_time.npy")
        gt_fs = len(gt_time) / (gt_time[-1] - gt_time[0])
        gt_filt = bandpass(gt, gt_fs)
        gt_hr = estimate_hr_fft(gt_filt, gt_fs)

        fps = len(ts) / (ts[-1] - ts[0])
        rgb_u, _ = resample_uniform(rgb, ts, fps)
        chrom_hr = estimate_hr_fft(bandpass(chrom_dehaan(rgb_u, fps), fps), fps)
        pos_hr = estimate_hr_fft(bandpass(pos_wang(rgb_u, fps), fps), fps)

        row = {
            "subject_id": sid,
            "dataset_family": s["dataset_family"],
            "condition": "after-exercise" if sid == "after-exercise" else "resting",
            "gt_hr": gt_hr,
            "chrom_hr": chrom_hr, "chrom_err": chrom_hr - gt_hr,
            "pos_hr": pos_hr, "pos_err": pos_hr - gt_hr,
        }
        for name, model in models.items():
            wf = predict_clip_waveform(model, frames, ts, device)
            hr = estimate_hr_fft(bandpass(wf, MODEL_FS), MODEL_FS) if len(wf) > MODEL_FS * 2 else np.nan
            row[f"{name}_hr"] = hr
            row[f"{name}_err"] = hr - gt_hr

        rows.append(row)
        print(f"  [{row['dataset_family']}] {sid} ({row['condition']}): GT={gt_hr:.1f}  "
              f"CHROM={chrom_hr:.1f}  POS={pos_hr:.1f}  "
              f"PhysNet(UBFC)={row['physnet_ubfc_hr']:.1f}  PhysNet(PURE)={row['physnet_pure_hr']:.1f}")
    return rows


def summarize(rows, methods, filter_fn, label):
    print(f"\n=== {label} ===")
    result = {"label": label}
    for method in methods:
        pairs_by_cond = {"resting": [], "after-exercise": []}
        for r in rows:
            if not filter_fn(r):
                continue
            pairs_by_cond[r["condition"]].append((r[f"{method}_hr"], r["gt_hr"]))

        mae_rest, rmse_rest = mae_rmse(pairs_by_cond["resting"])
        mae_ex, rmse_ex = mae_rmse(pairs_by_cond["after-exercise"])
        gap = abs(mae_ex - mae_rest) if not (np.isnan(mae_ex) or np.isnan(mae_rest)) else np.nan
        print(f"[{method}] resting: n={len(pairs_by_cond['resting'])} MAE={mae_rest:.2f} RMSE={rmse_rest:.2f}")
        print(f"[{method}] after-exercise: n={len(pairs_by_cond['after-exercise'])} MAE={mae_ex:.2f} RMSE={rmse_ex:.2f}")
        print(f"[{method}] fairness gap |MAE_ex - MAE_rest| = {gap:.2f} bpm")
        result.update({
            f"{method}_resting_n": len(pairs_by_cond["resting"]), f"{method}_resting_mae": mae_rest,
            f"{method}_after_ex_n": len(pairs_by_cond["after-exercise"]), f"{method}_after_ex_mae": mae_ex,
            f"{method}_gap": gap,
        })
    return result


def main():
    rows = per_clip_predictions()
    with open(RESULTS_DIR / "step5_per_clip.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    all_methods = ["chrom", "pos", "physnet_ubfc", "physnet_pure"]
    summary = [
        summarize(rows, all_methods, lambda r: True, "ALL clips pooled (DATASET_1+2)"),
        summarize(rows, ["chrom", "pos", "physnet_ubfc", "physnet_pure"],
                   lambda r: r["dataset_family"] == "DATASET_2" or r["subject_id"] == "after-exercise",
                   "DATASET_2 resting + DATASET_1 after-exercise (avoids PhysNet's DATASET_1 crop confound)"),
    ]

    with open(RESULTS_DIR / "step5_group_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nWritten to {RESULTS_DIR / 'step5_per_clip.csv'} and step5_group_summary.csv")


if __name__ == "__main__":
    main()
