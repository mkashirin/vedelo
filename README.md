# Vedelo

Vedelo is a GPU-accelerated vehicle-detection and multi-object-tracking
pipeline for video. It fine-tunes and exports YOLO models on the
[Vedelo Dataset V1](https://huggingface.co/datasets/mkashirin/vedelo-dataset-v1),
then runs exported CUDA TorchScript models through a Rust runtime. The runtime
decodes video, performs batched inference, assigns stable IDs with SORT, draws
annotations, and writes an MP4 result.

The repository contains both stages:

- `vedelo/` — Python utilities for training, exporting, and model/data downloads.
- `runtime/` — the Rust inference and tracking executable, `vedelo-rt`.
- `training-stage/` — expected location for datasets and exported model artifacts.

## Requirements

The runtime targets Linux (including NVIDIA CUDA) and macOS CPU/Apple MPS
using tch/libtorch, not MLX. The macOS debug build and project unit tests have
passed against the pinned FFmpeg build; actual exported-model MPS compatibility
and end-to-end video behavior still require separate validation. Linux CUDA
builds require:

- NVIDIA driver and CUDA support appropriate for the installed PyTorch build.
- `build-essential`, `pkg-config`, `git`, `make`, and a C compiler toolchain.
- [Rust](https://rustup.rs/), [uv](https://docs.astral.sh/uv/), and
  [just](https://just.systems/).
- Python 3.14 or newer.

The build compiles pinned FFmpeg, x264, and NVIDIA codec headers from the Git
submodules. It uses those libraries for H.264 decode/encode and NVIDIA codec
support on Linux. macOS builds omit NVIDIA headers and CUDA configuration.
The pinned FFmpeg source is **6.1.1**. Alternatively set `FFMPEG_SYSTEM=1`
to use system development libraries compatible with the Rust wrappers; an
executable alone is insufficient. `ffmpeg-next` 8.0.0 supports multiple native
FFmpeg versions and does not require native FFmpeg 8. The preflight prints the
versions of the four core video libraries found by pkg-config; it is a discovery
check, not proof of API, linker, codec or runtime compatibility.

### macOS development

Use Xcode command-line tools, Rust, just, pkg-config, and an existing Python
PyTorch **2.10** installation matching tch 0.23. Set `VEDELO_PYTHON` to its
Python executable (default `.venv/bin/python`). Only build/build-dev recipes discover libtorch from
that interpreter, without running uv sync; bootstrap/list/dry-run do not import
Torch. The runtime discovers its own library rpath, not transitive Cargo metadata.
Use `just build-deps` to build the pinned minimal FFmpeg/x264 configuration
with portable job counts and CPU codecs, rather than replacing it to match the
Rust crate's version number. Supply matching headers, libraries and pkg-config
files if using system mode. This source recipe
initializes submodules and therefore requires network access if absent; do not
run it in offline/no-install validation sessions.

With prerequisites already present:

```bash
FFMPEG_SYSTEM=1 just build-dev
# Direct Cargo does not read FFMPEG_SYSTEM or VEDELO_PYTHON:
# System pkg-config mode: retain its compiled defaults; optionally export
# PKG_CONFIG_PATH=/your/ffmpeg/lib/pkgconfig (and any intentional LIBDIR override).
export LIBTORCH="$("${VEDELO_PYTHON:-.venv/bin/python}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
unset LIBTORCH_USE_PYTORCH LIBTORCH_LIB
# Instead, for pinned native libraries, before cd runtime:
# export PKG_CONFIG_PATH="$PWD/runtime/third-party/install/lib/pkgconfig"
# export PKG_CONFIG_LIBDIR="$PKG_CONFIG_PATH"
cd runtime
cargo check --locked --offline
cargo test --locked --offline
```

Direct Cargo supports `LIBTORCH=<root>` with libraries in `<root>/lib`, or
`LIBTORCH_USE_PYTORCH=1`. The latter matches pinned torch-sys 0.23: it runs
`python` when `VIRTUAL_ENV` is set, otherwise `python3`, through `PATH`; it does
**not** honor `PYTHON` or `VEDELO_PYTHON`. Activate the intended environment or
adjust PATH consistently. Unset `LIBTORCH_USE_PYTORCH` to use an explicit root.
Implicit download/system discovery and `LIBTORCH_LIB` split overrides are
explicitly rejected: pinned torch-sys appends `/lib` to that override, so treating
it as an actual library directory would produce inconsistent linking. No new
Cargo dependency or synthetic `DEP_TCH_*` export is used.

CPU tensor/policy tests do not establish MPS model compatibility. Separately
exercise a real TorchScript model on `--device cpu` and `--device mps` with CPU
codecs, including partial batches and output frame/timestamp checks. MPS uses
FP32; archive remapping cannot fix hard-coded CUDA operators or unsupported MPS
operations. NVIDIA acceptance remains a separate Linux CUDA/NVENC/NVDEC check;
macOS success does not validate it.

## Build

Clone the repository with its submodules, create the Python environment, build
the native video dependencies, and compile the release runtime:

```bash
git clone --recurse-submodules <repository-url> vedelo
cd vedelo
just venv
just build-deps
just build
```

`just venv` creates `.venv` if missing and always runs `uv sync --all-groups`;
`just sync` also synchronizes that standard environment. Legacy `.vedelo-env`
users can explicitly set `VEDELO_PYTHON="$PWD/.vedelo-env/bin/python"` for native
build recipes; bootstrap intentionally uses `.venv`, not the legacy directory.
System FFmpeg recipes preserve pkg-config defaults (including Linux multiarch)
and explicit user `PKG_CONFIG_LIBDIR`; pinned recipes isolate `.pc` lookup to
the install prefix unless LIBDIR is explicitly overridden. The discovery
preflight does not enforce a native FFmpeg major version; actual compatibility
is checked by the Rust bindings during compilation and linking.

`just build` creates the release binary at `runtime/target/release/vedelo-rt`
and symlinks it to `./vedelo-rt`. If the repository was cloned without
submodules, run `git submodule update --init --recursive` before `just build-deps`.

## Run inference

The runtime accepts an exported TorchScript model, an input video, and a
destination video. `--device auto|cpu|cuda|mps` controls model inference only.
Auto prefers CUDA, then available MPS, then CPU. Explicit unavailable devices
fail instead of falling back. The selected device is logged; CUDA uses FP16,
CPU/MPS FP32.

```bash
./vedelo-rt \
  --model training-stage/artifacts/exported/v1n_batch8_imgsz1280_best_cuda.torchscript \
  --source training-stage/dataset/test/video_test.mp4 \
  --destination assets/tracked.mp4 \
  --conf 0.6 \
  --iou-thresh 0.1 \
  --max-age 90 \
  --batch-size 8 \
  --profile
```

Key options:

- `--conf` is the minimum detection confidence.
- `--iou-thresh` and `--max-age` tune SORT track association and lifetime.
- `--batch-size` is the number of decoded frames evaluated in one model call;
  increase it only while GPU memory permits. It must be at least 1.
- `--imgsz` defaults to 1280 and must match the intended model preprocessing.
- `--profile` prints total elapsed time and end-to-end FPS after encoding.
- `--decode-backend cpu|cuda|auto` and `--encode-backend cpu|cuda|auto`
  independently select video codecs (both default to `cpu`; `nvidia` aliases
  `cuda`). These do not change the model device. For example, add
  `--decode-backend auto --encode-backend cuda` to the command above.

CPU encoding explicitly opens `libx264`; CUDA encoding explicitly opens
`h264_nvenc`, with separate fixed quality policies: libx264 uses preset medium,
CRF 23; NVENC uses preset p4, VBR CQ 23 with bitrate and maxrate explicitly
zero (unconstrained quality targeting, not the implicit 2 Mb/s default). These
quality scales are not identical and do not promise equal visual quality or file
size. The selected policy is logged. Nominal frame rate is set independently of
the 1/90000 timestamp timebase; original Time values are rationally rescaled for
PTS. Nonfinite, nonpositive, or unrepresentable source rates are rejected before
opening the output (conversion uses micro-fps precision, range 0.000001 to about
2147 fps). `auto` attempts hardware initialization
and logs a startup fallback to CPU; explicit `cuda` fails if initialization fails.
The selected encoder is opened before creating the output file. Later decode,
encode, and mux errors fail the run without switching codecs; failed runs can
leave partial output. The adapter targets H.264 MP4 output without audio.

CUDA decoding requests video-rs 0.11's CUDA device context. A successful
initialization is logged, **not** proof of per-frame NVDEC use: video-rs may
accept software frames and exposes no per-frame hardware verification. Hardware
frames are downloaded and converted to RGB host memory. Annotated RGB storage
is moved into ndarray without an application full-frame clone, but codecs still
copy/convert data; this is not a zero-copy GPU pipeline.

Decode, ordered inference/tracking/drawing, and encode overlap through two
capacity-one batch queues. Including active batches, at most five batches of
full-resolution host frames are held by these stages (roughly
`5 * batch_size * width * height * 3` bytes), plus codec buffers, conversion
frames and model/preprocessing scratch/VRAM. Backpressure bounds queues; the
final partial batch and original timestamps are retained. Workers disconnect
channels before joining on errors. Panic propagation applies to unwind builds;
release builds use `panic=abort` and terminate on panic. No throughput gain is
claimed without fresh measurements.

### Verification prerequisites

The codec adapter and pipeline require fresh verification on supported Linux
with initialized/built native dependencies, matching libtorch, a real NVIDIA
GPU/driver and the exported model. Run locked compilation and tests sequentially
(`cargo check --locked`, `cargo test --locked` in `runtime/`) after parallel
source edits finish. Pure-std pipeline tests can also be compiled separately
from `runtime/src/pipeline.rs`. Test CPU/CUDA/auto initialization and fallback,
missing NVENC/device failures, decode errors, frame counts 0/1/B-1/B/B+1,
irregular timestamps, and `ffprobe` output PTS/DTS/frame counts after flush.
CUDA decoder initialization alone must not be reported as NVDEC verification.
These integration checks have not been run on the macOS implementation host.

For the repository’s standard test video and exported models, use the bundled
recipes:

```bash
just run-v1n
just run-v1s
```

They write annotated videos under `assets/`. Ensure the corresponding model
artifacts and `training-stage/dataset/test/video_test.mp4` are present first.

## Measured inference performance

The following end-to-end measurements use the bundled 1,800 × 1,100, 20 FPS
test video (202 frames), batch size 8, CUDA TorchScript models, and include
decode, preprocessing, inference, tracking, drawing, and output encoding.

Host: NVIDIA GeForce RTX 5060 8 GB and AMD Ryzen 5 5600X.

| Model                             | Confidence  | Elapsed | End-to-end throughput | Unique IDs  |
| --------------------------------- | ----------: | ------: | --------------------: | ----------: |
| `v1n_batch8_imgsz1280_best_cuda`  |         0.6 | 10.112s |             19.98 FPS |           4 |
| `v1s_batch8_imgsz1280_best_cuda`  |         0.7 | 11.307s |             17.87 FPS |           4 |

These historical results predate the codec/pipeline changes above and do not
validate their performance. They are workload-specific end-to-end results, not raw model-only benchmark
figures. Resolution, codec, storage, GPU driver, model settings, and tracker
parameters can change the observed FPS.

## Train and export

After setting up the environment, the primary development recipes are:

```bash
just download-dataset
just train
just export
```

Python training and tracking prefer CUDA, then available MPS, then CPU.
Training enables AMP and `torch.compile` only on CUDA; CPU/MPS use eager FP32
execution. Exports use CUDA/FP16 on CUDA hosts and CPU/FP32 otherwise, including
MPS hosts, to produce portable TorchScript/ONNX artifacts rather than trace on
MPS. The Rust runtime can explicitly load a CPU TorchScript archive onto MPS;
actual model/operator compatibility must still be checked.

`just export` produces TorchScript and ONNX exports from the configured training
artifact. Review the constants in `vedelo/train.py` and `vedelo/export.py`
before launching a new training run or export.

Run Python device-policy tests and the small, availability-gated MPS training
smoke test without downloading models or datasets:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
