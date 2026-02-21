import warnings
from pathlib import Path

from ultralytics import YOLO

from vedelo import device


ROOT = Path(__file__).resolve()
DATA = "dataset/data.yaml"
DEVICE = device()
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
    data_path = ROOT / Path("dataset/imgs+labels")
    data = f"""path: {ROOT}

train: imgs/train
val: imgs/val
nc: 2
names:
  0: car
  1: truck
    """
    with open(ROOT / Path(DATA), "w") as f:
        f.write(data)

    model = YOLO("models/yolo26m.pt")
    warnings.simplefilter("ignore")
    getattr(model, MODE)(**ARGS)
