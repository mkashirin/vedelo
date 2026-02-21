import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch as pt


BUNDLE_URL = "https://storage.yandexcloud.net/lab-storage/vedelo_bundle.zip"


def device_available() -> str:
    using: str
    if pt.cuda.is_available():
        using = "cuda"
    elif pt.mps.is_available():
        using = "mps"
    else:
        using = "cpu"
    return using


def download_bundle(output_path: Path) -> None:
    print(f"Downloading {BUNDLE_URL}...")
    try:
        with urllib.request.urlopen(BUNDLE_URL) as response:
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
