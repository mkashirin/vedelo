import shutil
import os
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import torch as pt


# _BASE_URL = "https://storage.yandexcloud.net/lab-storage"
# MODEL_URL = f"{_BASE_URL}/vedelo-v1s.zip"
# DATASET_URL = f"{_BASE_URL}/vedelo2.zip"
DATASET_REPO_ID = "mkashirin/vedelo-dataset-v1"
MODEL_REPO_ID = "mkashirin/vedelo-v1s"


def device_available() -> str:
    using: str
    if pt.cuda.is_available():
        using = "cuda"
    elif pt.mps.is_available():
        using = "mps"
    else:
        using = "cpu"
    return using


def download_torchscript(output_path: Path) -> None:
    if not output_path.exists():
        os.mkdir("finetuned")
        hf_hub_download(
            MODEL_REPO_ID, "vedelo-v1s.torchscript", local_dir=output_path
        )


def download_dataset(output_path: Path) -> None:
    if not output_path.exists():
        snapshot_download(
            DATASET_REPO_ID, repo_type="dataset", local_dir=output_path
        )
