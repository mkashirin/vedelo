set shell := ["bash", "-cu"]

if os_family() != "unix" { error("Windows build is not supported") } else {}
if os() != "linux" { error("Builds on linux only") }

default: build

@venv:
    #!/usr/bin/env sh
    if [ ! -d .venv ]; then
        echo "Creating virtual environment..."
        uv venv && sync
    fi
    source .venv/bin/activate

sync:
    uv sync

export LIBTORCH :=
    `source .venv/bin/activate && \
        python -c "from vedelo import *; print(PYTORCH)"`/lib
export DEP_TCH_LIBTORCH_LIB := $LIBTORCH
export LIBTORCH_USE_PYTORCH := 1
export LD_LIBRARY_PATH := $LIBTORCH:${LD_LIBRARY_PATH

build: venv
    cargo build --release
    ln -s target/release/vedelo vedelo

build-dev: venv
    cargo build

run: build
    python -c "from vedelo import *; get_model('finetuned.zip')"
    vedelo \
        -m finetuned/Vedelo-V1S.torchscript \
        -s dataset/test/video_test.mp4 \
        -d static/video_track.mp4 \
        --conf 0.5 \
        --iou-thresh 0.2 \
        --max-age 90

clean:
    cargo clean

rebuild: clean build
rebuild-dev: clean build-dev
