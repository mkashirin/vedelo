import shutil
import os
from pathlib import Path

from ultralytics import YOLO

from vedelo.common import torch_device_available


ROOT_DIR = "training-stage/artifacts"
PROJECT = f"{ROOT_DIR}/v1n_batch8_imgsz1280"
EXPORT_PATH = f"{ROOT_DIR}/exported"
NAME = "v1n"
CHECKPOINT = "best"
DEVICE = torch_device_available()
EXPORT_FORMATS = ("torchscript", "onnx")

if __name__ == "__main__":
    model = YOLO(f"{PROJECT}/weights/{CHECKPOINT}.pt")
    export_name = f"{NAME}_{CHECKPOINT}_{DEVICE}"
    for format in EXPORT_FORMATS:
        if not Path(f"{EXPORT_PATH}/{export_name}.{format}").exists():
            model.export(format=format, half=True, dynamic=True, device=DEVICE)

    weights_dir = Path(f"{PROJECT}/weights")
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
            shutil.move(src, f"{EXPORT_PATH}/{src.split('/')[-1]}")
