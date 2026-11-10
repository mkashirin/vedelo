from typing import Literal
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import torch as pt


DATASET_REPO_ID = "mkashirin/vedelo-dataset-v1"
MODEL_REPO_ID = "mkashirin/vedelo-v1"


def torch_device_available() -> Literal["cuda", "mps", "cpu"]:
    """Prefer CUDA, then available Apple MPS, with CPU as the fallback."""
    if pt.cuda.is_available():
        return "cuda"
    if pt.backends.mps.is_available():
        return "mps"
    return "cpu"


def hf_hub_download_model_v1(
    size: Literal["n", "s"] = "s",
    checkpoint: Literal["epoch20", "epoch40", "epoch60", "best"] = "best",
    batch: Literal[8, 12] = 8,
    imgsz: int = 1280,
    format: Literal["pt", "torchscript", "onnx"] = "torchscript",
    local_dir: Path = Path("models"),
) -> None:
    filename = f"v1{size}_batch{batch}_imgsz{imgsz}_{checkpoint}_cuda.{format}"
    if not local_dir.exists():
        os.mkdir("models")
    hf_hub_download(
        MODEL_REPO_ID, filename, subfolder=format, local_dir=local_dir
    )


def snapshot_download_dataset_v1(
    local_dir: Path = Path("training-stage/dataset"),
) -> None:
    if not local_dir.exists():
        snapshot_download(
            DATASET_REPO_ID, repo_type="dataset", local_dir=local_dir
        )
