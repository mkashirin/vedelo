import os
import warnings
from pathlib import Path

from ultralytics import YOLO

from vedelo import ROOT, device_available, get_dataset


DATA = "dataset/data.yaml"
DEVICE: str = device_available()
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
    project=ROOT / "artifacts",
    name="Vedelo-V1S",
    exist_ok=True,
    rect=True,
    cos_lr=True,
    amp=True,
    compile=True,
)


if __name__ == "__main__":
    if not Path("dataset").exists():
        get_dataset(Path("dataset.zip"))
    data_path: Path = ROOT / "dataset"
    data = f"""path: {data_path}

train: images/train
val: images/val
nc: 2
names:
  0: car
  1: truck
    """
    with open(DATA, "w") as f:
        f.write(data)

    model = YOLO("artifacts/pretrained/yolo26s.pt")
    warnings.simplefilter("ignore")
    getattr(model, MODE)(**ARGS)
    os.remove("yolo26n.pt")
