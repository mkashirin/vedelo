import shutil
import os
from pathlib import Path

from ultralytics import YOLO

from vedelo.common import device_available


MODEL = "vedelo-v1s"
CHECKPOINT = "epoch40"
DEVICE = device_available()
EXPORT_FORMATS = ("torchscript", "onnx")

if __name__ == "__main__":
    model = YOLO(f"artifacts/{MODEL}/weights/{CHECKPOINT}.pt")
    for format in EXPORT_FORMATS:
        model.export(format=format, half=True, dynamic=True, device=DEVICE)

    weights_dir = Path(f"artifacts/{MODEL}/weights")
    if weights_dir.exists():
        export_files = tuple(
            (f"{weights_dir}/{f}", f"{weights_dir}/{MODEL}.{f.split('.')[-1]}")
            for f in os.listdir(weights_dir)
            if CHECKPOINT in f
        )
        for src, dest in export_files:
            shutil.copy(src, dest)

        for _, src in export_files:
            shutil.move(src, f"exported/{src.split('/')[-1]}")
