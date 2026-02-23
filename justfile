set shell := ["bash", "-cu"]

_os_family := if os_family() != "unix" {
    error("Windows build is not supported")
} else { "" }
_os := if os() != "linux" { error("Builds on linux only") } else { "" }

default: build

venv:
    #!/usr/bin/env sh
    if [ ! -d .venv ]; then
        echo "Creating virtual environment..."
        uv venv && uv sync
    fi

sync:
    uv sync

export LIBTORCH := ```
        uv run python -c "from vedelo import *; print(PYTORCH)"
    ``` + "/lib"
export DEP_TCH_LIBTORCH_LIB := LIBTORCH
export LIBTORCH_USE_PYTORCH := "1"
export LD_LIBRARY_PATH := LIBTORCH

build: venv
    @echo $LD_LIBRARY_PATH
    source .venv/bin/activate
    cargo build --release
    ln -sf target/release/vedelo vedelo-cli

build-dev: venv
    cargo build

run: build
    uv run python -c "from vedelo import *; get_model('finetuned.zip')"
    vedelo-cli \
        -m exported/vedelo-v1s.torchscript \
        -s dataset/test/video_test.mp4 \
        -d static/video_track.mp4 \
        --conf 0.5 \
        --iou-thresh 0.2 \
        --max-age 90

clean:
    cargo clean

train:
    uv run python -m vedelo.train

test:
    uv run python -m vedelo.test

download-finetuned:
    uv run python -c "from vedelo import *; download_torchscript('finetuned')"

download-dataset:
    uv run python -c "from vedelo import *; download_dataset('dataset')"

clean-build: clean build
clean-build-dev: clean build-dev
