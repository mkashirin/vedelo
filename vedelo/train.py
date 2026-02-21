import warnings
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from vedelo import device_available, get_dataset


_DEPTH: int = 1
ROOT: Path = Path(__file__).resolve().parents[_DEPTH]
DATA = "bundle/dataset/data.yaml"
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
    name="Vedelo-1",
    exist_ok=True,
    optimizer="MuSGD",
    rect=True,
    cos_lr=True,
    amp=True,
    compile=True,
)


if __name__ == "__main__":
    get_dataset(ROOT / "dataset.zip")
    data_path: Path = ROOT / "dataset/imgs+labels"
    data = f"""path: {data_path}

train: imgs/train
val: imgs/val
nc: 2
names:
  0: car
  1: truck
    """
    with open(ROOT / Path(DATA), "w") as f:
        f.write(data)

    model = YOLO(".yolo/yolo26m.pt")
    warnings.simplefilter("ignore")
    getattr(model, MODE)(**ARGS)
    model.export(format="torchscript")
