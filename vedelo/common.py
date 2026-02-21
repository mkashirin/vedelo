import os
import urllib.request
import zipfile
from pathlib import Path

import torch as pt


_BASE_URL = "https://storage.yandexcloud.net/lab-storage"
MODEL_URL = f"{_BASE_URL}/vedelo_model.zip"
DATASET_URL = f"{_BASE_URL}/vedelo_dataset.zip"


def device_available() -> str:
    using: str
    if pt.cuda.is_available():
        using = "cuda"
    elif pt.mps.is_available():
        using = "mps"
    else:
        using = "cpu"
    return using


def get_model(output_path: Path) -> None:
    _get_unpacked(MODEL_URL, output_path)


def get_dataset(output_path: Path) -> None:
    _get_unpacked(DATASET_URL, output_path)


def _get_unpacked(
    url: str, output_path: Path, remove_zip: bool = True
) -> None:
    print(f"Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Download failed with status {response.status}"
                )
            output_path.write_bytes(response.read())
    except Exception as e:
        raise RuntimeError(f"Error downloading file: {e}")
    print(f"Saved to {output_path}")

    extract_to = output_path.parent
    print(f"Extracting {output_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        raise RuntimeError("Downloaded file is not a valid ZIP archive")
    print("Extraction complete.")

    if remove_zip:
        os.remove(output_path)
