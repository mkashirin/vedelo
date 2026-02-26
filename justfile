set shell := ["bash", "-cu"]

_os := if os() != "linux" { error("Builds on Linux only") } else { "" }

default: build

[group("vedelo")]
venv:
    #!/usr/bin/env sh
    if [ ! -d .venv ]; then
        echo "Creating virtual environment..."
        uv venv && uv sync
    fi

[group("vedelo")]
sync:
    uv sync

export FFMPEG_INSTALL_DIR := (
    invocation_directory() + "/runtime/third-party/FFmpeg-n6.1.1-install"
)
export DEP_FFMPEG_LIB := FFMPEG_INSTALL_DIR + "/lib"
export LIBTORCH := ```
        uv run python -c "from vedelo import *; print(PYTORCH)"
    ```
export DEP_TCH_LIBTORCH_LIB := LIBTORCH + "/lib"
export LD_LIBRARY_PATH := LIBTORCH + "/lib" 
export PKG_CONFIG_PATH := DEP_FFMPEG_LIB + "/pkgconfig"
export PKG_CONFIG_LIBDIR := PKG_CONFIG_PATH

[group("runtime")]
build-deps:
    git submodule update --init --recursive
    mkdir -p "$FFMPEG_INSTALL_DIR"

    cd runtime/third-party/nv-codec-headers-n12.0.16.1 && \
    make PREFIX="$FFMPEG_INSTALL_DIR" install

    cd runtime/third-party/x264-stable && make distclean && \
    ./configure \
        --prefix="$FFMPEG_INSTALL_DIR" \
        --enable-static \
        --enable-pic \
        --disable-cli && \
    make -j$(nproc) && make install

    cd runtime/third-party/FFmpeg-n6.1.1 && make distclean && \
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
run:
    ./vedelo-rt \
        -m training-stage/artifacts/exported/vedelo-v1s-best-cuda.torchscript \
        -s training-stage/dataset/test/video_test.mp4 \
        -d showcase/video_track_exp.mp4 \
        --conf 0.4 \
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
