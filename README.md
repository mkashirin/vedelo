# Vedelo

**Vedelo** (Vehicle Detection with YOLO) is trained on the
[Vedelo Dataset V1](https://huggingface.co/datasets/mkashirin/vedelo-dataset-v1).

## System Requirements

To build `vedelo-cli` and run Vedelo TorchScript models, ensure you have:
* An **x86_64** machine running a **Debian-based Linux distribution** with a CUDA-compatible GPU;
* [uv](https://docs.astral.sh/uv/) — Python package and project manager;
* [OpenCV](https://opencv.org/) installed via APT (built and tested with v4.10.0);
* [`just`](https://just.systems/) command runner.

## Build and Run

Build the Rust crate:
```shell
just build
```

Download and extract the Vedelo V1S TorchScript model:
```shell
just download-torchscript
```

Run the application:
```shell
just run
```

## Training and Evaluation

Train (fine-tune) a base YOLO model:
```shell
just train
```

Run prediction and tracking tests:
```shell
just test
```

Export the trained model:
```shell
just export
```
