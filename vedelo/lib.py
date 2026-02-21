import torch as pt


DATASET_URL = ""
MODEL_URL = ""


def device() -> str:
    using: str
    if pt.cuda.is_available():
        using = "cuda"
    elif pt.mps.is_available():
        using = "mps"
    else:
        using = "cpu"
    return using


def download_dataset() -> None: ...


def download_model() -> None: ...
