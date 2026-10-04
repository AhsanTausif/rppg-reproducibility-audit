# data/

Raw data and heavy intermediates live here locally and are **git-ignored**. The UBFC-rPPG videos contain identifiable participants and are not redistributed.

## Expected layout

```
data/
├── raw/
│   └── UBFC_DATASET/              # obtain from the UBFC-rPPG authors
│       ├── 5-gt/ 6-gt/ ...        # DATASET_1: vid*.avi + gtdump.xmp
│       └── subject1/ subject3/ ...  # DATASET_2: vid.avi + ground_truth.txt
└── processed/                     # created by src/rppg_preprocess.py
    └── <subject_id>/              # frames.npy, rgb_trace.npy, timestamps.npy, gt.npy, gt_time.npy
```

The pipeline auto-detects both layouts and pools them. If the dataset lives elsewhere, set the `UBFC_ROOT` environment variable instead of moving files:

```bash
# bash
export UBFC_ROOT=/path/to/UBFC_DATASET
# PowerShell
$env:UBFC_ROOT = "D:\datasets\UBFC_DATASET"
```

Two DATASET_1 folders (`7-gt`, `12-gt`) contain byte-identical videos despite different ground truth and are excluded automatically (see `EXCLUDED_SUBJECTS` in `src/rppg_preprocess.py`).

## Citation

If you use the dataset, please cite: S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, *Unsupervised skin tissue segmentation for remote photoplethysmography*, Pattern Recognition Letters, 2017.
