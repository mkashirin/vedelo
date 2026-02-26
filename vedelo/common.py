from typing import Literal
import shutil
import os
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import torch as pt


DATASET_REPO_ID = "mkashirin/vedelo-dataset-v1"
MODEL_REPO_ID = "mkashirin/vedelo-v1"


def torch_device_available() -> str:
    using: str
    if pt.cuda.is_available():
        using = "cuda"
    else:
        using = "cpu"
    return using


def hf_hub_download_model_v1(
    size: Literal["s", "m"] = "s",
    checkpoint: Literal["epoch20", "epoch40", "epoch60", "best"] = "epoch40",
    format: Literal["pt", "torchscript", "onnx"] = "torchscript",
    subfolder: Literal[
        "n-batch8", "n-batch12", "s-batch8", "s-batch12"
    ] = "s-batch8",
    local_dir: Path = Path("models"),
) -> None:
    filename = f"vedelo-v1{size}-{checkpoint}-cuda.{format}"
    if not local_dir.exists():
        os.mkdir("finetuned")
        hf_hub_download(
            MODEL_REPO_ID, filename, subfolder=subfolder, local_dir=local_dir
        )


def snapshot_download_dataset_v1(
    local_dir: Path = Path("training-stage/dataset")
) -> None:
    if not local_dir.exists():
        snapshot_download(
            DATASET_REPO_ID, repo_type="dataset", local_dir=local_dir
        )
