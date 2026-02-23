set shell := ["bash", "-cu"]

default: build

# Detect OS once
os := `uname`

venv:
    @if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv
        uv sync
    fi

sync:
    uv sync

libtorch :=
    `source .venv/bin/activate && \
        python -c "from vedelo import *; print(PYTORCH)"`/lib

build: venv
    @echo "Building on {{os}}"
    source .venv/bin/activate
    export DEP_TCH_LIBTORCH_LIB="{{libtorch}}"
    export LIBTORCH_USE_PYTORCH=1
    if [[ "{{os}}" == "Darwin" ]]; then
        export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:${DYLD_LIBRARY_PATH:-}"
    fi
    cargo build --release

build-dev: venv
    @echo "Building (dev) on {{os}}"
    source .venv/bin/activate
    export DEP_TCH_LIBTORCH_LIB="{{libtorch}}"
    export LIBTORCH_USE_PYTORCH=1
    if [[ "{{os}}" == "Darwin" ]]; then
        export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:${DYLD_LIBRARY_PATH:-}"
    fi
    cargo build

run: build
    @echo "Running on {{os}}"
    source .venv/bin/activate
    python -c 'from vedelo import *; get_model("finetuned.zip")'
    export LIBTORCH="{{libtorch}}"
    if [[ "{{os}}" == "Darwin" ]]; then
        export DYLD_LIBRARY_PATH="$LIBTORCH:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="$LIBTORCH:${LD_LIBRARY_PATH:-}"
    fi
    ./target/release/vedelo \
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
