set shell := ["bash", "-cu"]

_os := if os() == "linux" { "" } else if os() == "macos" { "" } else {
    error("Linux or macOS required")
}
python := env("VEDELO_PYTHON", justfile_directory() + "/.venv/bin/python")
jobs := `getconf _NPROCESSORS_ONLN`
cuda_flags := if os() == "linux" {
    (
        "--enable-cuda-llvm --enable-ffnvcodec "
        + "--enable-encoder=h264_nvenc --enable-hwaccel=h264_nvdec"
    )
} else { "" }

# Resolve Apple's tools/SDK only when configuring native dependencies on macOS.
mac_cc_env := if os() == "macos" {
    'CC="${CC:-$(xcrun --find clang)}"'
} else { "" }
mac_cc_flag := if os() == "macos" {
    '--cc="${CC:-$(xcrun --find clang)}"'
} else { "" }
mac_sysroot := if os() == "macos" {
    '--sysroot="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"'
} else { "" }
ffmpeg_system := env("FFMPEG_SYSTEM", "0")
exported_models := "training-stage/artifacts/exported"

default: build

export FFMPEG_INSTALL_DIR := (
    justfile_directory() + "/runtime/third-party/install"
)
export DEP_FFMPEG_LIB := FFMPEG_INSTALL_DIR + "/lib"
# System mode leaves pkg-config's defaults and user overrides intact.
# Pinned mode uses repository-built libraries, not an unrelated system install.
pkg_env := if ffmpeg_system == "1" { "" } else {
    (
        'export PKG_CONFIG_PATH="$FFMPEG_INSTALL_DIR/lib/pkgconfig"; '
        + 'export PKG_CONFIG_LIBDIR="'
        + '${PKG_CONFIG_LIBDIR-$FFMPEG_INSTALL_DIR/lib/pkgconfig}";'
    )
}

[group("vedelo")]
venv:
    #!/usr/bin/env sh
    if [ ! -d .venv ]; then
        uv venv .venv --prompt vedelo || exit $?
    fi
    uv sync --all-groups

[group("vedelo")]
sync:
    uv sync --all-groups

[group("runtime")]
build-deps:
    git submodule update --init --recursive
    @echo "$FFMPEG_INSTALL_DIR"
    {{pkg_env}} pkg-config --list-all
    mkdir -p "$FFMPEG_INSTALL_DIR"

    if [ "{{os()}}" = linux ]; then \
        cd runtime/third-party/nv-codec-headers && \
        make PREFIX="$FFMPEG_INSTALL_DIR" install; \
    fi

    cd runtime/third-party/x264 && \
    {{mac_cc_env}} ./configure \
        {{mac_sysroot}} \
        --prefix="$FFMPEG_INSTALL_DIR" \
        --enable-static \
        --enable-pic \
        --disable-cli && \
    make -j{{jobs}} && make install

    {{pkg_env}} cd runtime/third-party/FFmpeg && \
    ./configure \
        {{mac_cc_flag}} \
        {{mac_sysroot}} \
        --prefix="$FFMPEG_INSTALL_DIR" \
        --extra-cflags="-I$FFMPEG_INSTALL_DIR/include" \
        --extra-ldflags="-L$FFMPEG_INSTALL_DIR/lib" \
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
        {{cuda_flags}} && \
    make -j{{jobs}} && make install

[group("runtime")]
check-ffmpeg:
    command -v pkg-config >/dev/null || { \
        printf '%s %s\n' \
            'pkg-config is required; build the pinned native dependencies' \
            'with just build-deps' >&2; \
        exit 1; \
    }
    # Discovery only; the Rust bindings detect the installed FFmpeg API.
    {{pkg_env}} \
    for name in libavcodec libavformat libavutil libswscale; do \
        version=$(pkg-config --print-errors --modversion "$name") || exit 1; \
        printf '%s %s\n' "$name" "$version"; \
    done

[group("runtime")]
build: check-ffmpeg
    {{pkg_env}} \
    LIBTORCH="$("{{python}}" -c \
        'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" \
        && export LIBTORCH \
        && cd runtime \
        && cargo build --locked --release
    ln -sf runtime/target/release/vedelo-rt vedelo-rt

[group("runtime")]
build-dev: check-ffmpeg
    {{pkg_env}} \
    LIBTORCH="$("{{python}}" -c \
        'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')" \
        && export LIBTORCH \
        && cd runtime \
        && cargo build --locked

[group("runtime")]
run-v1n:
    ./vedelo-rt \
        -m {{exported_models}}/v1n_batch8_imgsz1280_best_cuda.torchscript \
        -s training-stage/dataset/test/video_test.mp4 \
        -d assets/video_track_by_v1n_batch8_imgsz1280_best_cuda.mp4 \
        --conf 0.6 \
        --iou-thresh 0.1 \
        --max-age 90 \
        --batch-size 8 \
        --profile

[group("runtime")]
run-v1s:
    ./vedelo-rt \
        -m {{exported_models}}/v1s_batch8_imgsz1280_best_cuda.torchscript \
        -s training-stage/dataset/test/video_test.mp4 \
        -d assets/video_track_by_v1s_batch8_imgsz1280_best_cuda.mp4 \
        --conf 0.7 \
        --iou-thresh 0.1 \
        --max-age 90 \
        --batch-size 8 \
        --profile

[group("runtime")]
run-downloaded-v1s:
    ./vedelo-rt \
        -m models/torchscript/v1s_batch8_imgsz1280_best_cuda.torchscript \
        -s training-stage/dataset/test/video_test.mp4 \
        -d assets/video_track_by_v1s_batch8_imgsz1280_best_cuda.mp4 \
        --device mps \
        --conf 0.7 \
        --iou-thresh 0.1 \
        --max-age 90 \
        --batch-size 8 \
        --profile

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
