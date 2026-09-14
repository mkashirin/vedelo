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

The supported runtime target is x86_64 Debian-based Linux with an NVIDIA GPU
and a CUDA-capable PyTorch/libtorch installation. Building requires:

- NVIDIA driver and CUDA support appropriate for the installed PyTorch build.
- `build-essential`, `pkg-config`, `git`, `make`, and a C compiler toolchain.
- [Rust](https://rustup.rs/), [uv](https://docs.astral.sh/uv/), and
  [just](https://just.systems/).
- Python 3.14 or newer.

The build compiles pinned FFmpeg, x264, and NVIDIA codec headers from the Git
submodules. It uses those libraries for H.264 decode/encode and NVIDIA codec
support; a system FFmpeg development package is not used by the `just` build.

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

`just build` creates the release binary at `runtime/target/release/vedelo-rt`
and symlinks it to `./vedelo-rt`. If the repository was cloned without
submodules, run `git submodule update --init --recursive` before `just build-deps`.

## Run inference

The runtime accepts an exported TorchScript model, an input video, and a
destination video. CUDA is selected automatically when available; otherwise it
runs on the CPU.

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

These are workload-specific end-to-end results, not raw model-only benchmark
figures. Resolution, codec, storage, GPU driver, model settings, and tracker
parameters can change the observed FPS.

## Train and export

After setting up the environment, the primary development recipes are:

```bash
just download-dataset
just train
just export
```

`just export` produces TorchScript and ONNX exports from the configured training
artifact. Review the constants in `vedelo/train.py` and `vedelo/export.py`
before launching a new training run or export.
