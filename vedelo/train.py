import os
import warnings
from pathlib import Path

from ultralytics import YOLO

from vedelo import ROOT, torch_device_available, snapshot_download_dataset_v1


DATA = "training-stage/dataset/data.yaml"
DEVICE: str = torch_device_available()
MODE = "train"
ARGS = dict(
    data=DATA,
    epochs=60,
    patience=20,
    batch=8,
    imgsz=1280,
    save_period=20,
    device=DEVICE,
    workers=0,
    project=ROOT / "training-stage/artifacts",
    name="v1s_batch8_imgsz1280",
    exist_ok=True,
    rect=True,
    cos_lr=True,
    amp=True,
    compile=True,
)


if __name__ == "__main__":
    dataset_path = Path("training-stage/dataset")
    if not dataset_path.exists():
        snapshot_download_dataset_v1(dataset_path)
    data = f"""path: {dataset_path}

train: images/train
val: images/val
nc: 2
names:
  0: car
  1: truck
    """
    with open(DATA, "w") as f:
        f.write(data)

    model = YOLO("training-stage/artifacts/base/yolo26s.pt")
    warnings.simplefilter("ignore")
    getattr(model, MODE)(**ARGS)
    os.remove("yolo26n.pt")
