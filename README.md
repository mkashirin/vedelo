# Vedelo

Vedelo (Vehicle Detection with YOLO).

Dataset name: vedelo2

To build the Rust app:
```shell
# If you DO NOT have a Python virtual environment set up yet:
uv venv && uv sync

# If you do:
source .venv/bin/activate
DEP_TCH_LIBTORCH_LIB=$(python -c "from vedelo import *; print(PYTORCH)")/lib \
LIBTORCH_USE_PYTORCH=1 \
cargo build --release
```

To download and extract the Vedelo-V1 (`source .venv/bin/activate` first):
```shell
python -c "from vedelo import *; get_model(ROOT / 'finetuned.zip')"
```

To run the app built (exports are necessary, tch-rs needs to know, where to
search for LibTorch's shared library files):
```shell
LIBTROCH=$(python -c "from vedelo import *; print(PYTORCH)")/lib \
LD_LIBRARY_PATH=$LIBTORCH:$LD_LIBRARY_PATH \
./target/release/vedelo \
    -m artifacts/Vedelo-V1S/weights/epoch40.torchscript \
    -s dataset/test/video_test.mp4 \
    -d video_test_pred.mp4 \
    --conf 0.2 \
    --iou-thresh 0.15 \
```

