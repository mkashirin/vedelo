set shell := ["bash", "-cu"]

_os := if os() != "linux" { error("Builds on Linux only") } else { "" }

default: build

export FFMPEG_INSTALL_DIR := (
    invocation_directory() + "/runtime/third-party/install"
)
export DEP_FFMPEG_LIB := FFMPEG_INSTALL_DIR+ "/lib"
export LIBTORCH := ```
        uv run python -c "from vedelo import *; print(PYTORCH)"
    ```
export PKG_CONFIG_PATH := DEP_FFMPEG_LIB + "/pkgconfig"
export PKG_CONFIG_LIBDIR := PKG_CONFIG_PATH
export DEP_TCH_LIBTORCH_LIB := LIBTORCH + "/lib"
export LD_LIBRARY_PATH := LIBTORCH + "/lib" 

[group("vedelo")]
venv:
    #!/usr/bin/env sh
    if [ ! -d .vedelo-pyenv ]; then
        echo "Creating virtual environment..."
        uv venv .vedelo-pyenv --prompt vedelo && uv sync --all-groups
    fi

[group("vedelo")]
sync:
    uv sync --all-groups

[group("runtime")]
build-deps:
    git submodule update --init --recursive
    @echo "$FFMPEG_INSTALL_DIR"
    @echo "$PKG_CONFIG_LIBDIR"
    pkg-config --list-all
    mkdir -p "$FFMPEG_INSTALL_DIR"

    cd runtime/third-party/nv-codec-headers && \
    make PREFIX="$FFMPEG_INSTALL_DIR" install

    cd runtime/third-party/x264 && \
    ./configure \
        --prefix="$FFMPEG_INSTALL_DIR" \
        --enable-static \
        --enable-pic \
        --disable-cli && \
    make -j$(nproc) && make install

    cd runtime/third-party/FFmpeg && \
    ./configure \
        --prefix="$FFMPEG_INSTALL_DIR" \
        --extra-cflags="-I$FFMPEG_INSTALL_DIR/include" \
        --extra-ldflags="-I$FFMPEG_INSTALL_DIR/lib" \
        \
        --disable-shared \
        --enable-static \
        --enable-pic \
        \
        --disable-everything \
        --disable-programs \
        --disable-doc \
        --disable-network \
        --disable-indevs \
        --disable-outdevs \
        \
        --enable-gpl \
        --enable-libx264 \
        \
        --enable-zlib \
        --enable-protocol=file \
        --enable-demuxer=mov,h264,m4v,image2 \
        --enable-muxer=mp4,h264,image2 \
        --enable-decoder=h264,mpeg4,mjpeg,png \
        --enable-encoder=libx264,mpeg4,png \
        --enable-filter=scale,transpose,format \
        --enable-swscale \
        --enable-cuda-llvm \
        --enable-ffnvcodec \
        --enable-encoder=h264_nvenc \
        --enable-hwaccel=h264_nvdec && \
    make -j$(nproc) && make install
    

[group("runtime")]
build: venv
    source .venv/bin/activate
    cd runtime && cargo build --release && \
    ln -sf runtime/target/release/vedelo-rt vedelo-rt

[group("runtime")]
build-dev: venv
    source .venv/bin/activate
    cd runtime && cargo build

[group("runtime")]
run-v1n:
    ./vedelo-rt \
        -m training-stage/artifacts/exported/v1n_batch8_imgsz1280_best_cuda.torchscript \
        -s training-stage/dataset/test/video_test.mp4 \
        -d assets/video_track_by_v1n_batch12_imgsz1280_best_cuda.mp4 \
        --conf 0.6 \
        --iou-thresh 0.1 \
        --max-age 90

[group("runtime")]
run-v1s:
    ./vedelo-rt \
        -m training-stage/artifacts/exported/v1s_batch12_imgsz1280_best_cuda.torchscript \
        -s training-stage/dataset/test/video_test.mp4 \
        -d assets/video_track_by_v1s_batch12_imgsz1280_best_cuda.mp4 \
        --conf 0.7 \
        --iou-thresh 0.1 \
        --max-age 90

[group("runtime")]
clean:
    cargo clean

[group("runtime")]
clean-build: clean build
[group("runtime")]
clean-build-dev: clean build-dev

[group("vedelo")]
download-model:
    uv run python -c "from vedelo import *; hf_hub_download_model_v1()"

[group("vedelo")]
download-dataset:
    uv run python -c "from vedelo import *; snapshot_download_dataset_v1()"

[group("vedelo")]
export:
    uv run python -m vedelo.export

[group("vedelo")]
train:
    uv run python -m vedelo.train

[group("vedelo")]
track:
    uv run python -m vedelo.track
