"""
rPPG dataset preprocessing pipeline -- UBFC-rPPG (DATASET_1 + DATASET_2, pooled).

Dataset layout (confirmed by inspection of UBFC_DATASET/):

  DATASET_1 (original, 8-subject set):
    <DATASET_ROOT>/<N-gt>/vid*.avi   (vid.avi / vid-12.avi / vid-007.avi)
    <DATASET_ROOT>/<N-gt>/gtdump.xmp
    ~640x480, ~28.7 fps. gtdump.xmp is a plain CSV (no header despite the
    .xmp extension), 4 columns: time_ms, hr_bpm(device), spo2, ppg_trace,
    sampled at ~62.1 Hz. Confirmed against ubfcrppg_data_processor.py.

  DATASET_2 (public 42-subject benchmark):
    <DATASET_ROOT>/subject<N>/vid.avi
    <DATASET_ROOT>/subject<N>/ground_truth.txt
    ~640x480, ~29.3-29.6 fps. ground_truth.txt is whitespace-separated, 3
    rows: ppg_trace, hr_bpm(device), time_seconds (already in seconds,
    unlike DATASET_1's ms). Confirmed against ubfcrppg_data_processor.py
    and rPPG-Toolbox's own UBFCrPPGLoader (which only recognizes this
    subject*/ground_truth.txt layout -- i.e. any "UBFC-rPPG"-named
    pretrained checkpoint from that toolbox was trained on DATASET_2, not
    DATASET_1; this matters when interpreting cross-checkpoint results).

  The two families are pooled into one subject-independent split (by
  request), but every subject is tagged with its `dataset_family` so
  results can still be broken out per-family -- pooling numbers from two
  different acquisitions without that tag would make comparisons to
  published (DATASET_2-only) numbers misleading.

Covers the three things that matter most for this task:
  1. Face/skin-ROI extraction per frame
  2. Ground-truth pulse signal alignment with video frames (GT resampled
     onto each frame's timestamp using the GT file's own recorded times)
  3. Subject-independent train/val/test splitting by subject id
"""

import csv
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
# Raw UBFC-rPPG videos (not redistributed). Override with the UBFC_ROOT env var.
DATASET_ROOT = Path(os.environ.get("UBFC_ROOT", REPO_ROOT / "data" / "raw" / "UBFC_DATASET"))
# Heavy intermediate artifacts (face-crop stacks, RGB traces) -- git-ignored.
OUTPUT_ROOT = REPO_ROOT / "data" / "processed"
# Small, version-controlled result tables (split manifest, per-method metrics).
TABLES_DIR = REPO_ROOT / "results" / "tables"
FRAME_SIZE = (128, 128)          # standard small ROI size used in most rPPG models
FACE_DETECTOR_MARGIN = 0.15      # extra padding around detected face box, as a fraction
DETECT_SCALE_WIDTH = 480         # downscale to this width for face detection speed only

# 7-gt/vid-007.avi and 12-gt/vid-12.avi are byte-for-byte identical files
# (verified via MD5) despite having different ground truth (82.5s/94.7bpm
# vs 83.6s/95.1bpm) -- one subject's video is mislabeled/duplicated in the
# downloaded dataset. Excluded until the correct file is re-sourced, so
# reported numbers aren't built on a known-wrong video/GT pairing.
EXCLUDED_SUBJECTS = {"7-gt", "12-gt"}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Haar cascade (the original choice) produced visibly loose/unstable crops
# on a subset of DATASET_2 subjects (confirmed by inspection -- e.g.
# subject1's crop was dominated by patterned clothing, not skin), which
# measurably degraded every downstream method (CHROM/POS/PhysNet) on those
# clips. YuNet is a small (230KB), modern DNN detector (bundled with
# OpenCV's objdetect module since 4.5.4) that's substantially more robust
# to pose/expression/lighting -- switched in after that finding.
YUNET_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL_PATH), "", (320, 320), score_threshold=0.6)


def discover_subjects(dataset_root: Path):
    """
    One folder per subject/clip directly under dataset_root, auto-detecting
    which of the two UBFC-rPPG formats it is by which ground-truth file is
    present (gtdump.xmp -> DATASET_1, ground_truth.txt -> DATASET_2).
    """
    subjects = []
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name in EXCLUDED_SUBJECTS:
            continue

        gt1_path = subject_dir / "gtdump.xmp"
        gt2_path = subject_dir / "ground_truth.txt"
        if gt1_path.exists():
            dataset_family = "DATASET_1"
            gt_path = gt1_path
        elif gt2_path.exists():
            dataset_family = "DATASET_2"
            gt_path = gt2_path
        else:
            continue  # no recognized ground truth, skip

        video_candidates = sorted(subject_dir.glob("vid*.avi"))
        if not video_candidates:
            continue

        subjects.append({
            "subject_id": subject_dir.name,
            "person_id": subject_dir.name,   # one clip per person -- no condition split
            "dataset_family": dataset_family,
            "video_path": video_candidates[0],
            "gt_path": gt_path,
        })
    return subjects


def extract_roi_frames(video_path: Path, out_dir: Path):
    """
    Extract face ROI from every frame, resize, save as .npy stack.
    Also records per-frame timestamps and the mean RGB over the ROI (the
    raw trace classical methods like CHROM/POS operate on).
    Detection runs on a downscaled copy for speed; the crop itself is taken
    from the full-resolution frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    rgb_trace = []
    timestamps = []
    last_box = None
    frame_idx = 0
    detect_size_set = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h_full, w_full = frame.shape[:2]
        scale = DETECT_SCALE_WIDTH / w_full if w_full > DETECT_SCALE_WIDTH else 1.0
        small = cv2.resize(frame, (int(w_full * scale), int(h_full * scale))) if scale != 1.0 else frame
        small_h, small_w = small.shape[:2]
        if detect_size_set != (small_w, small_h):
            face_detector.setInputSize((small_w, small_h))
            detect_size_set = (small_w, small_h)
        _, detections = face_detector.detect(small)

        if detections is not None and len(detections) > 0:
            best = max(detections, key=lambda d: d[14])  # highest confidence score
            x, y, w, h = best[0], best[1], best[2], best[3]
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            mx, my = int(w * FACE_DETECTOR_MARGIN), int(h * FACE_DETECTOR_MARGIN)
            x, y = max(0, x - mx), max(0, y - my)
            w, h = w + 2 * mx, h + 2 * my
            last_box = (x, y, w, h)
        elif last_box is not None:
            x, y, w, h = last_box
        else:
            frame_idx += 1
            continue

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            frame_idx += 1
            continue

        rgb_trace.append(roi.reshape(-1, 3).mean(axis=0)[::-1])  # BGR -> RGB mean
        roi = cv2.resize(roi, FRAME_SIZE)
        frames.append(roi)
        timestamps.append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "frames.npy", np.array(frames, dtype=np.uint8))
    np.save(out_dir / "rgb_trace.npy", np.array(rgb_trace, dtype=np.float64))
    np.save(out_dir / "timestamps.npy", np.array(timestamps, dtype=np.float64))
    return fps, len(frames)


def load_ground_truth(gt_path: Path, dataset_family: str):
    """
    DATASET_1 (gtdump.xmp): plain CSV, no header, 4 columns: time_ms,
    hr_bpm(device), spo2, ppg_trace.
    DATASET_2 (ground_truth.txt): whitespace-separated, 3 rows: ppg_trace,
    hr_bpm(device), time_seconds (already in seconds).
    Both confirmed against ubfcrppg_data_processor.py. Returns
    (trace, time_seconds) -- we use each file's own timestamps rather than
    assuming a nominal rate, since both formats give them directly and
    DATASET_2's sample spacing is irregular (not a clean fixed rate).
    """
    if gt_path is None:
        return None, None
    if dataset_family == "DATASET_1":
        data = pd.read_csv(gt_path, header=None).values
        trace = data[:, 3].astype(np.float64)
        time_s = data[:, 0].astype(np.float64) / 1000.0
    else:
        data = np.loadtxt(gt_path)
        trace = data[0, :].astype(np.float64)
        time_s = data[2, :].astype(np.float64)
    return trace, time_s


def resample_gt_to_frames(gt: np.ndarray, gt_time: np.ndarray, frame_timestamps: np.ndarray) -> np.ndarray:
    """Linearly interpolate the GT trace (at its own timestamps) onto each extracted frame's timestamp."""
    return np.interp(frame_timestamps, gt_time, gt)


def subject_independent_split(subjects, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split by subject id -- the same person's clip never appears in more
    than one split.
    """
    rng = np.random.RandomState(seed)
    person_ids = sorted({s["person_id"] for s in subjects})
    rng.shuffle(person_ids)

    n = len(person_ids)
    n_train = max(1, int(round(n * train_frac)))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n - 1) if n > 1 else n_train  # leave at least 1 for test if possible

    person_split = {
        "train": set(person_ids[:n_train]),
        "val": set(person_ids[n_train:n_train + n_val]),
        "test": set(person_ids[n_train + n_val:]),
    }

    split = {"train": [], "val": [], "test": []}
    for s in subjects:
        for split_name, people in person_split.items():
            if s["person_id"] in people:
                split[split_name].append(s["subject_id"])
    return split


def main():
    subjects = discover_subjects(DATASET_ROOT)
    n_d1 = sum(1 for s in subjects if s["dataset_family"] == "DATASET_1")
    n_d2 = sum(1 for s in subjects if s["dataset_family"] == "DATASET_2")
    print(f"Found {len(subjects)} clips ({n_d1} DATASET_1, {n_d2} DATASET_2).")

    manifest_rows = []
    for s in subjects:
        out_dir = OUTPUT_ROOT / s["subject_id"]
        # Always re-extract (no skip-if-exists): every clip must use the
        # same detector for the pooled analysis to be internally consistent,
        # and the switch from Haar to YuNet invalidates all prior crops.
        fps, n_frames = extract_roi_frames(s["video_path"], out_dir)
        gt, gt_time = load_ground_truth(s["gt_path"], s["dataset_family"])
        if gt is not None:
            np.save(out_dir / "gt.npy", gt)
            np.save(out_dir / "gt_time.npy", gt_time)
            timestamps = np.load(out_dir / "timestamps.npy")
            if n_frames > 0:
                gt_aligned = resample_gt_to_frames(gt, gt_time, timestamps)
                np.save(out_dir / "gt_aligned.npy", gt_aligned)

        manifest_rows.append({
            "subject_id": s["subject_id"],
            "person_id": s["person_id"],
            "dataset_family": s["dataset_family"],
            "n_frames": n_frames,
            "fps": fps,
            "has_gt": gt is not None,
        })
        print(f"  [{s['dataset_family']}] {s['subject_id']}: {n_frames} frames @ {fps:.2f}fps, "
              f"gt={'yes (' + str(len(gt)) + ' samples)' if gt is not None else 'MISSING'}")

    split = subject_independent_split(subjects, train_frac=0.7, val_frac=0.15)
    with open(TABLES_DIR / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "person_id", "dataset_family", "split", "n_frames", "fps"])
        row_by_id = {r["subject_id"]: r for r in manifest_rows}
        for split_name, ids in split.items():
            for sid in ids:
                r = row_by_id[sid]
                writer.writerow([sid, r["person_id"], r["dataset_family"], split_name, r["n_frames"], f"{r['fps']:.3f}"])

    print(f"\nSplit sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    print(f"Manifest written to {TABLES_DIR / 'split_manifest.csv'}")


if __name__ == "__main__":
    main()
