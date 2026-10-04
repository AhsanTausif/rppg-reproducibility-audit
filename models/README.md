# models/

Third-party model files used by the pipeline. They are **not** committed (see `.gitignore`); fetch them with:

```bash
python scripts/download_models.py
```

| File | Source | Used for |
|---|---|---|
| `PhysNet.py` | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (`neural_methods/model/PhysNet.py`) | PhysNet architecture definition |
| `UBFC-rPPG_PhysNet_DiffNormalized.pth` | rPPG-Toolbox (`final_model_release/`) | Pretrained PhysNet, trained on UBFC-rPPG DATASET_2 |
| `PURE_PhysNet_DiffNormalized.pth` | rPPG-Toolbox (`final_model_release/`) | Pretrained PhysNet, trained on PURE |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | YuNet face detector |

The checkpoints are MD5-verified by the download script (`a8ad4f0a026a93284eb150dc8cb87088` for the UBFC checkpoint, `f12ce5b4b88cefa20d78913d416305d8` for the PURE checkpoint), which is how the audit established that the two files are distinct. Please consult each project's license before redistributing.
