# Vedelo

Vedelo (Vehicle Detection with YOLO).

To build the Rust app:
```shell
# If you DO NOT have a Python virtual environment set up yet:
uv venv && uv sync

# When you do:
source .venv/bin/activate
export LIBTORCH_USE_PYTORCH=1
cargo build
```

To download and extract the Vedelo-V1 (`source .venv/bin/activate` first):
```shell
python -c "from vedelo import *; get_model(ROOT / 'finetuned.zip')"
```
