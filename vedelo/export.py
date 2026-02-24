import shutil
import os
from pathlib import Path

from ultralytics import YOLO

from vedelo.common import device_available


MODEL = "vedelo-v1s"
CHECKPOINT = "epoch20"
DEVICE = device_available()
EXPORT_FORMATS = ("torchscript", "onnx")

if __name__ == "__main__":
    model = YOLO(f"artifacts/{MODEL}/weights/{CHECKPOINT}.pt")
    export_name = f"{MODEL}-{CHECKPOINT}-{DEVICE}"
    for format in EXPORT_FORMATS:
        if not Path(f"exported/{export_name}.{format}").exists():
            model.export(format=format, half=True, dynamic=True, device=DEVICE)

    weights_dir = Path(f"artifacts/{MODEL}/weights")
    if weights_dir.exists():
        export_files = []
        for filename in os.listdir(weights_dir):
            if filename.split(".")[0] == CHECKPOINT:
                src = f"{weights_dir}/{filename}"
                dest = f"{weights_dir}/{export_name}.{filename.split('.')[-1]}"
                export_files.append((src, dest))

        for src, dest in export_files:
            shutil.copy(src, dest)

        for _, src in export_files:
            shutil.move(src, f"exported/{src.split('/')[-1]}")
