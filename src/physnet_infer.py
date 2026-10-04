"""
Inference-only evaluation of the pretrained PhysNet checkpoints shipped by
rPPG-Toolbox (ubicomplab/rPPG-Toolbox, final_model_release/) against our
preprocessed clips. No training happens here -- see baselines_classical.py
for the training-free CHROM/POS methods and rppg_preprocess.py for how the
face-ROI frame stacks were produced.

Why the fps correction: PhysNet_padding_Encoder_Decoder_MAX is a fixed-
window spatiotemporal CNN. It was trained on 128-frame chunks sampled at
30 fps (~4.27s/chunk), so a full cardiac cycle spans a specific number of
frames in its receptive field. UBFC-rPPG's own capture is ~28.7 fps (close
to, but not exactly, 30), so we still nearest-neighbor resample each
clip's extracted frames onto a uniform 30 fps grid (using the true
per-frame timestamps saved by extract_roi_frames) before chunking, so the
network sees the frame rate it was trained on -- the correction is small
here, unlike the earlier grayscale dataset's ~20 fps capture.

Preprocessing (matches rPPG-Toolbox's DiffNormalized config for PhysNet):
  - frames resized to 72x72, RGB order
  - per-chunk discrete time difference, normalized by the chunk's std
    (diff_normalize_data, reproduced from rPPG-Toolbox's BaseLoader)
  - CHUNK_LENGTH = 128 frames per chunk, matching the training config

Chunking overlap: naive non-overlapping 128-frame chunks concatenated
end-to-end produced sharp amplitude/phase discontinuities at every chunk
boundary (~every 4.3s), which injected a comb of spurious low-frequency
artifacts whose harmonics land inside the cardiac band and dominate the
FFT peak -- verified by comparing per-chunk boundary values and by the
fact the artifact's fundamental (~1/chunk_length) harmonics line up with
the bogus HR estimates observed. Fixed with 50%-overlap Hann-windowed
overlap-add (same fix CHROM needed -- see baselines_classical.py), which
tapers each chunk to zero at its own edges instead of splicing raw values.
The first/last half-window (where overlap-add coverage is incomplete) is
then trimmed off.

HR estimation: uses whole_clip_hr_pairs (one HR per clip) rather than
fixed 30s sub-windows, for the same reason given in baselines_classical.py.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

import rppg_preprocess as rp
from baselines_classical import bandpass, whole_clip_hr_pairs, mae_rmse, load_split

PHYSNET_DIR = rp.REPO_ROOT / "models"   # populate with scripts/download_models.py
CHECKPOINTS = {
    "physnet_ubfc": PHYSNET_DIR / "UBFC-rPPG_PhysNet_DiffNormalized.pth",
    "physnet_pure": PHYSNET_DIR / "PURE_PhysNet_DiffNormalized.pth",
}
MODEL_FS = 30.0          # training frame rate the checkpoints expect
MODEL_RES = 72           # training spatial resolution
CHUNK_LENGTH = 128        # frames per chunk, matches training config

import sys
sys.path.insert(0, str(PHYSNET_DIR))
from PhysNet import PhysNet_padding_Encoder_Decoder_MAX  # noqa: E402


def diff_normalize_data(data: np.ndarray) -> np.ndarray:
    """Reproduces rPPG-Toolbox's DiffNormalized preprocessing: discrete
    time-difference per pixel, normalized by the chunk's overall std."""
    n, h, w, c = data.shape
    diff_len = n - 1
    out = np.zeros((diff_len, h, w, c), dtype=np.float32)
    for j in range(diff_len):
        out[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
    std = np.std(out)
    out = out / std if std > 1e-8 else out
    out = np.append(out, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
    out[np.isnan(out)] = 0
    return out


def resample_frames_to_fps(frames: np.ndarray, timestamps: np.ndarray, target_fps: float):
    duration = timestamps[-1] - timestamps[0]
    n_out = int(duration * target_fps)
    t_out = timestamps[0] + np.arange(n_out) / target_fps
    idx = np.searchsorted(timestamps, t_out)
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return frames[idx]


def load_model(ckpt_path: Path, chunk_len: int, device):
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=chunk_len)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_clip_waveform(model, frames_bgr: np.ndarray, timestamps: np.ndarray, device):
    """50%-overlap Hann-windowed overlap-add inference -- see module
    docstring for why naive non-overlapping chunk concatenation fails."""
    frames_rgb = frames_bgr[..., ::-1]
    frames_30 = resample_frames_to_fps(frames_rgb, timestamps, MODEL_FS)
    n = frames_30.shape[0]
    if n < CHUNK_LENGTH:
        return np.array([])

    hop = CHUNK_LENGTH // 2
    win = np.hanning(CHUNK_LENGTH)
    acc = np.zeros(n)
    wsum = np.zeros(n)
    start = 0
    with torch.no_grad():
        while start + CHUNK_LENGTH <= n:
            chunk = np.stack([
                cv2.resize(frames_30[start + i], (MODEL_RES, MODEL_RES)) for i in range(CHUNK_LENGTH)
            ]).astype(np.float32)
            chunk = diff_normalize_data(chunk)               # [T,H,W,C]
            chunk = np.transpose(chunk, (3, 0, 1, 2))          # [C,T,H,W]
            x = torch.from_numpy(chunk).unsqueeze(0).float().to(device)  # [1,C,T,H,W]
            rppg, *_ = model(x)
            p = rppg.squeeze(0).cpu().numpy()
            p = p - p.mean()  # per-chunk DC offset, same issue CHROM had
            acc[start:start + CHUNK_LENGTH] += p * win
            wsum[start:start + CHUNK_LENGTH] += win
            start += hop

    valid_start, valid_end = hop, n - hop  # trim incomplete-coverage edges
    if valid_end <= valid_start:
        return acc / np.maximum(wsum, 1e-8)
    return acc[valid_start:valid_end] / np.maximum(wsum[valid_start:valid_end], 1e-8)


def main():
    device = torch.device("cpu")
    split_by_id = load_split()
    subjects = rp.discover_subjects(rp.DATASET_ROOT)

    all_results = {name: {} for name in CHECKPOINTS}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== {ckpt_name} ({ckpt_path.name}) ===")
        model = load_model(ckpt_path, CHUNK_LENGTH, device)
        for s in subjects:
            sid = s["subject_id"]
            clip_dir = rp.OUTPUT_ROOT / sid
            frames_path = clip_dir / "frames.npy"
            ts_path = clip_dir / "timestamps.npy"
            gt_path = clip_dir / "gt.npy"
            if not (frames_path.exists() and ts_path.exists() and gt_path.exists()):
                continue
            frames = np.load(frames_path)
            timestamps = np.load(ts_path)
            gt = np.load(gt_path)
            gt_time = np.load(clip_dir / "gt_time.npy")
            if len(frames) < CHUNK_LENGTH:
                print(f"  {sid}: too short, skipped")
                continue

            waveform = predict_clip_waveform(model, frames, timestamps, device)
            if len(waveform) < MODEL_FS * 2:
                print(f"  {sid}: prediction too short, skipped")
                continue

            gt_fs = len(gt_time) / (gt_time[-1] - gt_time[0])
            pred_filt = bandpass(waveform, MODEL_FS)
            gt_filt = bandpass(gt, gt_fs)
            pairs = whole_clip_hr_pairs(pred_filt, MODEL_FS, gt_filt, gt_fs)
            all_results[ckpt_name][sid] = pairs
            print(f"  {sid}: {len(pairs)} windows")

    for ckpt_name, results in all_results.items():
        def aggregate(filter_fn, label):
            pairs_all = []
            for sid, pairs in results.items():
                if filter_fn(sid):
                    pairs_all += pairs
            mae, rmse = mae_rmse(pairs_all)
            print(f"[{ckpt_name}] [{label}] n_windows={len(pairs_all)}  MAE={mae:.2f} bpm  RMSE={rmse:.2f} bpm")
            return {"checkpoint": ckpt_name, "label": label, "n_windows": len(pairs_all), "mae": mae, "rmse": rmse}

        summary = [
            aggregate(lambda sid: True, "ALL clips"),
            aggregate(lambda sid: split_by_id[sid]["split"] == "test", "TEST split only"),
            aggregate(lambda sid: sid != "after-exercise", "resting clips only (excl. after-exercise)"),
            aggregate(lambda sid: sid == "after-exercise", "after-exercise clip only"),
            aggregate(lambda sid: split_by_id[sid]["dataset_family"] == "DATASET_1", "DATASET_1 only"),
            aggregate(lambda sid: split_by_id[sid]["dataset_family"] == "DATASET_2", "DATASET_2 only"),
            aggregate(lambda sid: split_by_id[sid]["dataset_family"] == "DATASET_2" and split_by_id[sid]["split"] == "test", "DATASET_2 + TEST split"),
        ]
        out_csv = rp.TABLES_DIR / f"{ckpt_name}_results.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "label", "n_windows", "mae", "rmse"])
            writer.writeheader()
            writer.writerows(summary)
        print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
