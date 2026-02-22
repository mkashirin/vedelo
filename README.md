# Vedelo

Vedelo (Vehicle Detection with YOLO).

To build the Rust app:
```shell
# If you DO NOT have a Python virtual environment set up yet:
uv venv && uv sync

# When you do:
source .venv/bin/activate
export LIBTROCH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")
export DEP_TCH_LIBTORCH_LIB=$LIBTORCH
export LD_LIBRARY_PATH=$LIBTORCH:$LD_LIBRARY_PATH
LIBTORCH_USE_PYTORCH=1 cargo build
```

To run the app built:
```shell
cargo run --release
```

To download and extract the Vedelo-V1 (`source .venv/bin/activate` first):
```shell
python -c "from vedelo import *; get_model(ROOT / 'finetuned.zip')"
```
