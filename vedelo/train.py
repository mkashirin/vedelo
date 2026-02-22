import warnings
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from vedelo import ROOT, device_available, get_dataset


DATA = "dataset/data.yaml"
DEVICE: str = device_available()
MODE = "train"
ARGS = dict(
    data=DATA,
    epochs=100,
    patience=5,
    batch=4,
    imgsz=1280,
    save_period=10,
    device=DEVICE,
    workers=0,
    project="vedelo",
    name="Vedelo-V1",
    exist_ok=True,
    optimizer="MuSGD",
    rect=True,
    cos_lr=True,
    amp=True,
    compile=True,
)
EXPORT_FORMAT = "torchscript"


if __name__ == "__main__":
    if not (ROOT / "dataset").exists():
        get_dataset(ROOT / "dataset.zip")
    data_path: Path = ROOT / "dataset/images+labels"
    data = f"""path: {data_path}

train: images/train
val: images/val
nc: 2
names:
  0: car
  1: truck
    """
    with open(ROOT / Path(DATA), "w") as f:
        f.write(data)

    model = YOLO("pretrained/yolo26m.pt")
    warnings.simplefilter("ignore")
    getattr(model, MODE)(**ARGS)
    model.export(format=EXPORT_FORMAT)
