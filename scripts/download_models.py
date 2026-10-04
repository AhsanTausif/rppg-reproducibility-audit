"""
Fetch the third-party model files the pipeline needs into ./models/.

  * PhysNet architecture + two pretrained checkpoints (UBFC-rPPG, PURE),
    published by rPPG-Toolbox (ubicomplab/rPPG-Toolbox).
  * YuNet face detector (ONNX), published by the OpenCV Zoo.

These files are third-party artifacts distributed under their own licenses,
so they are intentionally NOT committed to this repository.

Usage:
    python scripts/download_models.py [--dest models]
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

TOOLBOX = "https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet"

# (filename, url, expected MD5 or None). The checkpoint hashes are the values
# we verified during the audit to prove the two checkpoints are distinct files.
FILES = [
    ("PhysNet.py", f"{TOOLBOX}/neural_methods/model/PhysNet.py", None),
    ("UBFC-rPPG_PhysNet_DiffNormalized.pth",
     f"{TOOLBOX}/final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth",
     "a8ad4f0a026a93284eb150dc8cb87088"),
    ("PURE_PhysNet_DiffNormalized.pth",
     f"{TOOLBOX}/final_model_release/PURE_PhysNet_DiffNormalized.pth",
     "f12ce5b4b88cefa20d78913d416305d8"),
    ("face_detection_yunet_2023mar.onnx", f"{ZOO}/face_detection_yunet_2023mar.onnx", None),
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"))
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    failed = False
    for name, url, expected in FILES:
        target = dest / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, target)
        else:
            print(f"found       {name}")
        digest = md5(target)
        if expected and digest != expected:
            print(f"  !! MD5 mismatch for {name}: got {digest}, expected {expected}")
            failed = True
        else:
            print(f"  ok  md5={digest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
