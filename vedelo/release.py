import shutil
import os
from pathlib import Path

from ultralytics import YOLO

from vedelo.common import device_available, create_zip


MODEL = "Vedelo-V1S"
CHECKPOINT = "epoch40"
DEVICE = device_available()
EXPORT_FORMATS = ("torchscript", "onnx")
ZIP_DATASET = False
ZIP_MODEL = False

if __name__ == "__main__":
    model = YOLO(f"artifacts/{MODEL}/weights/{CHECKPOINT}.pt")
    for format in EXPORT_FORMATS:
        model.export(format=format, half=True, dynamic=True, device=DEVICE)

    os.makedirs("release", exist_ok=True)
    if Path("dataset").exists() and ZIP_DATASET:
        create_zip(
            "release/vedelo2.zip",
            "dataset/images",
            "dataset/labels",
            "dataset/test",
        )

    weights_dir = Path(f"artifacts/{MODEL}/weights")
    if weights_dir.exists() and ZIP_MODEL:
        export_files = tuple(
            (f"{weights_dir}/{f}", f"{weights_dir}/{MODEL}.{f.split('.')[-1]}")
            for f in os.listdir(weights_dir)
            if CHECKPOINT in f
        )
        for src, dest in export_files:
            shutil.copy(src, dest)
        create_zip("release/vedelo-v1s.zip", *(f[1] for f in export_files))

        for _, f in export_files:
            os.remove(f)
