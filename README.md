# Vedelo

Vehicle detection and tracking with YOLO and SORT. Python utilities train and
export models on the [Vedelo Dataset V1](https://huggingface.co/datasets/mkashirin/vedelo-dataset-v1).
The Rust runtime processes video with TorchScript and writes annotated MP4 output.
Runs on Linux (CPU/CUDA) and macOS (CPU/Apple MPS).

## Requirements

- Python 3.14+, Rust, [uv](https://docs.astral.sh/uv/),
  [just](https://just.systems/), Git, pkg-config, and a C/C++ build toolchain.
- Linux: NVIDIA drivers for CUDA inference and hardware codecs.
- macOS: Xcode command-line tools; MPS requires a supported Apple GPU.

PyTorch **2.10.0** and tch **0.23.0** must stay compatible. The build uses the
pinned minimal **FFmpeg 6.1.1** and x264 submodules, not system FFmpeg libraries.

## Build and test

From the repository root:

```bash
just venv         # Create/sync .venv using locked dependencies
just build-deps   # Initialize submodules and build native libraries
just build        # Build the release runtime and ./vedelo-rt symlink
just check
just test
```

Use `just build-dev` for a debug build, or `just sync` to refresh dependencies.
Build/check/test recipes locate libtorch through `.venv/bin/python`; override
`VEDELO_PYTHON` to use another compatible Python environment.

## Run inference

Download the model and test dataset:

```bash
just download-model
just download-dataset
```

Run a compatible TorchScript export:

```bash
./vedelo-rt \
  --model path/to/model.torchscript \
  --source input.mp4 \
  --destination tracked.mp4 \
  --device auto --imgsz 1280 --batch-size 8 --profile
```

- `--device auto|cpu|cuda|mps`: auto prefers CUDA, then MPS, then CPU.
- `--conf`: detection confidence threshold.
- `--iou-thresh` and `--max-age`: SORT association and track lifetime.
- `--imgsz`: model input size; match the export settings.
- `--batch-size`: frames per inference call; tune to available memory.
- `--profile`: processing time and end-to-end FPS, excluding startup.
- `--decode-backend cpu|cuda|auto` and `--encode-backend cpu|cuda|auto`:
  codec selection, independent of inference; both default to CPU.

CPU encoding uses libx264; NVIDIA encoding uses h264_nvenc. NVENC uses no
B-frames to avoid FFmpeg 6.1.1's reorder-delay timestamp issue with the
runtime's 90 kHz clock. This preserves source presentation timestamps,
but may increase bitrate at the same CQ quality setting. On macOS, use CPU
codecs even with MPS inference. CUDA-traced device operations are not made
portable by changing `--device`.

The bundled `just run-v1n-cuda` and `just run-v1s-cuda` recipes use CUDA exports;
`just run-hf-v1s-mps` uses the downloaded CPU export with MPS. Paths and
tracking settings are defined in `justfile`.

## Profiling

```bash
just profile          # Build release and profile available devices/codecs
just profile --help
```

Linux profiles CPU/CUDA combinations; macOS profiles CPU/MPS with CPU codecs.
Detailed logs, metadata and summaries go to `training-stage/artifacts/profiles/`.

### Results: Ryzen 5 5600X / RTX 5060, WSL2

v1s at 1280, batch 8; 202 frames, 1800 × 1100 at 20 FPS. Medians of three
full-video runs, excluding startup. CPU uses FP32; CUDA uses FP16.

| Inference | Decode | Encode | Time (s) | FPS |
|---|---|---|---:|---:|
| CPU | CPU | libx264 | 114.156 | 1.77 |
| CPU | CPU | NVENC | 109.629 | 1.84 |
| CPU | CUDA | libx264 | 113.867 | 1.77 |
| CPU | CUDA | NVENC | 109.919 | 1.84 |
| CUDA | CPU | libx264 | 6.118 | 33.02 |
| CUDA | CPU | NVENC | 5.719 | 35.32 |
| CUDA | CUDA | libx264 | 6.262 | 32.26 |
| CUDA | CUDA | NVENC | 5.917 | 34.14 |

Codec settings are not quality-equivalent; CUDA decode does not verify per-frame
NVDEC use. Mac results are pending; run the same `just profile` command there.

## Train and export

```bash
just train
just export
just track
```

Review model paths and settings in `vedelo/train.py`, `vedelo/export.py`, and
`vedelo/track.py` first. Training and Python tracking prefer CUDA, then MPS,
then CPU.

The Rust runtime expects end-to-end detections shaped `[batch, detections, 6]`:
`[x1, y1, x2, y2, confidence, class_id]`. Export with `nms=False` when the
checkpoint supports it, using FP32 for CPU/MPS or FP16 for CUDA. Raw prediction
exports are not supported. Existing export files are skipped; regenerate them
after changing export settings.
