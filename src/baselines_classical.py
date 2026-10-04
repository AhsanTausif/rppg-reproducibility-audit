"""
CHROM and POS classical (training-free) rPPG baselines, evaluated against
the BVP ground truth for every clip preprocessed by rppg_preprocess.py.

Both algorithms turn a per-frame mean-RGB trace over the face ROI into a
single reconstructed pulse waveform; heart rate is then read off as the
dominant frequency in the cardiac band (42-240 bpm) per 30s window.

Since neither method is trained/fit on data (they are closed-form signal
processing), they are evaluated on the FULL dataset as the primary, most
stable number, and also broken out by the same subject-independent split
used for the deep baseline so the two are directly comparable on the same
held-out test people. "after-exercise" is also broken out separately since
it's physiologically distinct (elevated resting HR) from the other clips.

References:
  CHROM: de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG",
         IEEE TBME 2013.
  POS:   Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy import signal

import rppg_preprocess as rp

OUTPUT_ROOT = rp.OUTPUT_ROOT
LOW_BPM, HIGH_BPM = 42.0, 240.0
WINDOW_SEC = 30.0


def bandpass(x: np.ndarray, fs: float, low_hz=0.7, high_hz=4.0, order=3) -> np.ndarray:
    nyq = fs / 2.0
    b, a = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)


def resample_uniform(x: np.ndarray, timestamps: np.ndarray, fs: float, n_channels=None):
    """Resample onto a uniform time grid before FFT/filtering (handles the
    small timing jitter/gaps in both the frame timestamps and the GT
    device's own timestamps)."""
    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    if x.ndim == 1:
        return np.interp(t_uniform, timestamps, x), t_uniform
    x_uniform = np.stack([np.interp(t_uniform, timestamps, x[:, c]) for c in range(x.shape[1])], axis=1)
    return x_uniform, t_uniform


def chrom_dehaan(rgb: np.ndarray, fs: float) -> np.ndarray:
    win_len = int(round(1.6 * fs))
    if win_len % 2:
        win_len += 1
    step = win_len // 2
    n = rgb.shape[0]
    bvp = np.zeros(n)
    win_start = 0
    while win_start + win_len <= n:
        seg = rgb[win_start:win_start + win_len]
        base = seg.mean(axis=0)
        base[base == 0] = 1e-8
        norm = seg / base
        xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        alpha = np.std(xs) / (np.std(ys) + 1e-8)
        s = xs - alpha * ys
        # RGBNorm has mean ~1 per channel by construction (seg/seg.mean()),
        # so xs/ys carry a per-window DC offset; demeaning before the Hann
        # taper avoids injecting that offset as a spurious low-frequency
        # artifact into the overlap-add (POS avoids this the same way).
        s = s - s.mean()
        s = s * np.hanning(win_len)
        bvp[win_start:win_start + win_len] += s
        win_start += step
    return bvp


def pos_wang(rgb: np.ndarray, fs: float) -> np.ndarray:
    win_len = math.ceil(1.6 * fs)
    n = rgb.shape[0]
    h = np.zeros(n)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for end in range(win_len, n + 1):
        start = end - win_len
        seg = rgb[start:end]
        base = seg.mean(axis=0)
        base[base == 0] = 1e-8
        cn = (seg / base).T  # 3 x win_len
        s = proj @ cn        # 2 x win_len
        alpha = np.std(s[0]) / (np.std(s[1]) + 1e-8)
        seg_h = s[0] + alpha * s[1]
        h[start:end] += seg_h - seg_h.mean()
    return h


def estimate_hr_fft(x: np.ndarray, fs: float, low_bpm=LOW_BPM, high_bpm=HIGH_BPM):
    """Single whole-clip FFT peak. Kept for reference/debugging -- see
    estimate_hr_welch for why this isn't used as the default anymore."""
    n = len(x)
    if n < 8:
        return np.nan
    windowed = (x - x.mean()) * np.hanning(n)
    nfft = max(4096, 1 << (n - 1).bit_length())
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(windowed, n=nfft))
    mask = (freqs * 60 >= low_bpm) & (freqs * 60 <= high_bpm)
    if not mask.any():
        return np.nan
    return freqs[mask][np.argmax(spec[mask])] * 60.0


def estimate_hr_welch(x: np.ndarray, fs: float, low_bpm=LOW_BPM, high_bpm=HIGH_BPM, nperseg_sec=15.0):
    """
    Welch's method (averaged periodogram over overlapping sub-windows),
    tried as a candidate replacement for the whole-clip FFT peak after a
    handful of DATASET_2 clips (subject1/10/11/16/17) showed large errors.
    NOT used by default -- kept here, and the negative result documented,
    rather than silently discarded.

    Tested head-to-head against plain FFT on all 22 clips, correctly
    pairing each estimator with its OWN ground-truth estimate (Welch-GT vs.
    Welch-pred, FFT-GT vs. FFT-pred) rather than mixing them. Result: not a
    net improvement. It fixed subject1/16/3 outright (was 35-47 bpm error,
    became <1 bpm) but broke several clips plain FFT already had right --
    subject13 (CHROM: -0.4 -> -56.2 bpm error), subject14 (-3.8 -> -24.0),
    subject15/POS (+18.7 -> -40.2), and after-exercise (-1.9 -> -7.9, likely
    because Welch's shorter sub-windows interact badly with that clip's
    genuinely non-stationary, still-declining post-exercise HR). Whichever
    estimator you pick trades one set of failures for a different set; it
    is not a strict improvement, so switching the default would just be
    picking whichever number looks better on this specific small test set --
    exactly the kind of post-hoc tuning that shouldn't happen. Plain FFT
    (whole_clip_hr_pairs) remains the default.
    """
    n = len(x)
    if n < 8:
        return np.nan
    nperseg = min(n, int(nperseg_sec * fs))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs * 60 >= low_bpm) & (freqs * 60 <= high_bpm)
    if not mask.any():
        return np.nan
    return freqs[mask][np.argmax(psd[mask])] * 60.0


def whole_clip_hr_pairs(pred_signal, pred_fs, gt_signal, gt_fs):
    """
    One HR estimate for the entire clip, rather than several fixed 30s
    sub-windows. UBFC-rPPG clips here are short (43-118s) and a single
    whole-clip estimate matches how UBFC-rPPG results are usually reported
    (one HR per video). Uses a plain FFT peak (see estimate_hr_fft) -- an
    alternative (Welch's method, estimate_hr_welch) was tested and found
    not to be a net improvement; see its docstring for the head-to-head
    numbers.
    """
    p_hr = estimate_hr_fft(pred_signal, pred_fs)
    g_hr = estimate_hr_fft(gt_signal, gt_fs)
    if np.isnan(p_hr) or np.isnan(g_hr):
        return []
    return [(p_hr, g_hr)]


def windowed_hr_pairs(pred_signal, pred_fs, gt_signal, gt_fs, window_sec=WINDOW_SEC):
    """Slice both signals into aligned non-overlapping time windows and
    return (pred_hr, gt_hr) for every full window."""
    duration = min(len(pred_signal) / pred_fs, len(gt_signal) / gt_fs)
    n_windows = int(duration // window_sec)
    pairs = []
    for w in range(n_windows):
        t0, t1 = w * window_sec, (w + 1) * window_sec
        p_seg = pred_signal[int(t0 * pred_fs):int(t1 * pred_fs)]
        g_seg = gt_signal[int(t0 * gt_fs):int(t1 * gt_fs)]
        p_hr = estimate_hr_fft(p_seg, pred_fs)
        g_hr = estimate_hr_fft(g_seg, gt_fs)
        if not (np.isnan(p_hr) or np.isnan(g_hr)):
            pairs.append((p_hr, g_hr))
    return pairs


def mae_rmse(pairs):
    if not pairs:
        return np.nan, np.nan
    errs = np.array([p - g for p, g in pairs])
    return float(np.mean(np.abs(errs))), float(np.sqrt(np.mean(errs ** 2)))


def load_split():
    split_by_id = {}
    with open(rp.TABLES_DIR / "split_manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            split_by_id[row["subject_id"]] = {
                "split": row["split"],
                "person_id": row["person_id"],
                "dataset_family": row.get("dataset_family", "DATASET_1"),
            }
    return split_by_id


def evaluate_clip(subject_id: str, video_fps: float):
    clip_dir = OUTPUT_ROOT / subject_id
    rgb = np.load(clip_dir / "rgb_trace.npy")
    ts = np.load(clip_dir / "timestamps.npy")
    gt = np.load(clip_dir / "gt.npy")
    gt_time = np.load(clip_dir / "gt_time.npy")

    rgb_u, _ = resample_uniform(rgb, ts, video_fps)
    gt_fs = len(gt_time) / (gt_time[-1] - gt_time[0])
    gt_u, _ = resample_uniform(gt, gt_time, gt_fs)

    chrom_sig = bandpass(chrom_dehaan(rgb_u, video_fps), video_fps)
    pos_sig = bandpass(pos_wang(rgb_u, video_fps), video_fps)
    gt_filt = bandpass(gt_u, gt_fs)

    chrom_pairs = whole_clip_hr_pairs(chrom_sig, video_fps, gt_filt, gt_fs)
    pos_pairs = whole_clip_hr_pairs(pos_sig, video_fps, gt_filt, gt_fs)
    return chrom_pairs, pos_pairs


def main():
    split_by_id = load_split()
    subjects = rp.discover_subjects(rp.DATASET_ROOT)
    fps_by_id = {}
    for s in subjects:
        clip_dir = OUTPUT_ROOT / s["subject_id"]
        ts_path = clip_dir / "timestamps.npy"
        if ts_path.exists():
            ts = np.load(ts_path)
            fps_by_id[s["subject_id"]] = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 28.7

    results = {}  # subject_id -> (chrom_pairs, pos_pairs)
    for subject_id, fps in fps_by_id.items():
        try:
            results[subject_id] = evaluate_clip(subject_id, fps)
            n_c, n_p = len(results[subject_id][0]), len(results[subject_id][1])
            print(f"  {subject_id}: {n_c} CHROM windows, {n_p} POS windows")
        except Exception as e:
            print(f"  {subject_id}: FAILED ({e})")

    def aggregate(filter_fn, label):
        chrom_all, pos_all = [], []
        for sid, (c, p) in results.items():
            if filter_fn(sid):
                chrom_all += c
                pos_all += p
        c_mae, c_rmse = mae_rmse(chrom_all)
        p_mae, p_rmse = mae_rmse(pos_all)
        print(f"\n[{label}] n_windows={len(chrom_all)}")
        print(f"  CHROM  MAE={c_mae:.2f} bpm  RMSE={c_rmse:.2f} bpm")
        print(f"  POS    MAE={p_mae:.2f} bpm  RMSE={p_rmse:.2f} bpm")
        return {"label": label, "n_windows": len(chrom_all),
                "chrom_mae": c_mae, "chrom_rmse": c_rmse,
                "pos_mae": p_mae, "pos_rmse": p_rmse}

    summary = []
    summary.append(aggregate(lambda sid: True, "ALL clips (full pooled dataset, training-free)"))
    summary.append(aggregate(lambda sid: split_by_id[sid]["split"] == "test", "TEST split only (subject-independent)"))
    summary.append(aggregate(lambda sid: sid != "after-exercise", "resting clips only (excl. after-exercise)"))
    summary.append(aggregate(lambda sid: sid == "after-exercise", "after-exercise clip only"))
    summary.append(aggregate(lambda sid: split_by_id[sid]["dataset_family"] == "DATASET_1", "DATASET_1 only"))
    summary.append(aggregate(lambda sid: split_by_id[sid]["dataset_family"] == "DATASET_2", "DATASET_2 only"))
    summary.append(aggregate(lambda sid: split_by_id[sid]["dataset_family"] == "DATASET_2" and split_by_id[sid]["split"] == "test", "DATASET_2 + TEST split"))

    with open(rp.TABLES_DIR / "classical_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "n_windows", "chrom_mae", "chrom_rmse", "pos_mae", "pos_rmse"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nWritten to {rp.TABLES_DIR / 'classical_results.csv'}")


if __name__ == "__main__":
    main()
