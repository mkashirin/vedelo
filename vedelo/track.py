import os
import random
from pathlib import Path
from typing import List

from ultralytics import YOLO

from vedelo import ROOT


VAL_PATH = Path("dataset/images/val")
SOURCE = "dataset/test/video_test.mp4"
TRACKER = "botsort.yaml"
MODE = "track"
ARGS = dict(
    source=SOURCE,
    stream=True,
    imgsz=1280,
    conf=0.5,
    iou=0.2,
    save=True,
    tracker=TRACKER,
    project=ROOT / "artifacts",
    name="vedelo-v1s_track",
)

if __name__ == "__main__":
    model = YOLO("artifacts/vedelo-v1s/weights/best.pt")

    print(getattr(model.model, "names"))
    print(getattr(model.model, "nc"))
    print("Testing one val image detection")
    val_dir: List[str] = os.listdir(VAL_PATH)
    model.predict(
        VAL_PATH / val_dir[random.randint(0, len(val_dir))],
        conf=0.5,
        iou=0.2,
        save=True,
        project=ROOT / "artifacts",
        name="vedelo-v1s_predict",
    )

    results = getattr(model, MODE)(**ARGS)

    unique_vehicle_ids = set()
    for result in results:
        print(result.boxes.conf)
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            unique_vehicle_ids.update(ids)

    unique_vehicles = len(unique_vehicle_ids)
    print(f"Total unique vehicles detected in video: {unique_vehicles}")
