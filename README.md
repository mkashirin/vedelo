# Vedelo

Vedelo (Vehicle Detection with YOLO) model is trained on [Vedelo Dataset V1](
    https://huggingface.co/datasets/mkashirin/vedelo-dataset-v1
)

## System requirements

To build vedelo-cli and use it to run Vedelo models in TorchScript format, you will need the
following points checked:
    * x86_64 machine with Debian-based Linux distribution with CUDA-compatible device;
    * [uv](https://docs.astral.sh/uv/) Python package and project manager;
    * [OpenCV](https://opencv.org/) installed from APT Debian repositries (built and run, rocking
        version 4.10.0).
    * [`just`](https://just.systems/) command runner.

## Building and running

To build the Rust crate:
```shell
just build
```

To download and extract the Vedelo V1S in TorchScript format:
```shell
just download-torchscript
```

To run the app built:
```shell
just run
```

## Training, testing and exporting

To tune a base YOLO model:
```shell
just train
```

To predict and track to test the Vedelo model:
```
just test
```

To export the model tuned:
```
just export
```
