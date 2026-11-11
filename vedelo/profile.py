"""Profile the Rust runtime on CPU, CUDA and Apple MPS."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch as pt

from vedelo import ROOT

type Device = Literal["cpu", "cuda", "mps"]
type Backend = Literal["cpu", "cuda"]
type Configuration = tuple[Device, Backend, Backend]
type Timing = tuple[int, float, float]

MODEL_DIRS = (
    ROOT / "training-stage/artifacts/exported",
    ROOT / "models/torchscript",
)
PROFILE = re.compile(
    r"^Profile: (\d+) frames in ([\d.]+)s \(([\d.]+) FPS\)$", re.MULTILINE
)
NOTE = (
    "Runtime timing includes first inference and encoder flush, not startup. "
    "No warmup. CPU/MPS use FP32; CUDA uses FP16. CUDA decode confirms a "
    "device context, not per-frame NVDEC use."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=NOTE)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda", "mps"))
    for kind in ("decode", "encode"):
        parser.add_argument(
            f"--{kind}-backends", nargs="+", choices=("cpu", "cuda")
        )
    parser.add_argument("--variant", choices=("n", "s"), default="s")
    for name, default in (
        ("imgsz", 1280),
        ("batch", 8),
        ("repeats", 3),
        ("max-age", 90),
        ("log-every", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--iou", type=float, default=0.1)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "training-stage/dataset/test/video_test.mp4",
    )
    parser.add_argument("--binary", type=Path, default=ROOT / "vedelo-rt")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="otherwise search exported models, then models/torchscript",
    )
    parser.add_argument(
        "--cpu-model", type=Path, help="CPU archive, also used for MPS"
    )
    parser.add_argument("--cuda-model", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new directory; defaults to training-stage/artifacts/profiles/<OS>-<UTC>",
    )
    return parser.parse_args()


def query(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError, subprocess.TimeoutExpired:
        return None


def sha256(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def host_metadata(system: str) -> dict[str, object]:
    cpu, memory = platform.processor(), None
    if system == "Linux":
        try:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
            memory = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            )
        except OSError, StopIteration:
            pass
    elif system == "Darwin":
        cpu = query("sysctl", "-n", "machdep.cpu.brand_string") or cpu
        memory = query("sysctl", "-n", "hw.memsize")
    return {
        "platform": platform.platform(),
        "system": system,
        "machine": platform.machine(),
        "cpu": cpu,
        "memory": memory,
        "python": sys.version,
        "torch": str(pt.__version__),
        "mac_model": query("sysctl", "-n", "hw.model")
        if system == "Darwin"
        else None,
        "torch_cuda": pt.version.cuda,
        "cuda_available": pt.cuda.is_available(),
        "mps_available": pt.backends.mps.is_available(),
        "mps_built": pt.backends.mps.is_built(),
        "nvidia_smi": query(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
        },
    }


def configuration(
    args: argparse.Namespace,
    system: str,
) -> tuple[list[Configuration], dict[Device, Path]]:
    available: list[Device] = ["cpu"]
    if system == "Linux" and pt.cuda.is_available():
        available.append("cuda")
    if system == "Darwin" and pt.backends.mps.is_available():
        available.append("mps")
    codecs: list[Backend] = ["cpu", "cuda"] if "cuda" in available else ["cpu"]
    devices: list[Device] = list(dict.fromkeys(args.devices or available))
    decode: list[Backend] = list(dict.fromkeys(args.decode_backends or codecs))
    encode: list[Backend] = list(dict.fromkeys(args.encode_backends or codecs))
    if set(devices) - set(available) or (set(decode) | set(encode)) - set(
        codecs
    ):
        raise ValueError(f"available devices={available}, codecs={codecs}")
    models: dict[Device, Path] = {}
    for device in devices:
        kind = "cuda" if device == "cuda" else "cpu"
        filename = f"v1{args.variant}_batch{args.batch}_imgsz{args.imgsz}_best_{kind}.torchscript"
        directories = [args.model_dir] if args.model_dir else MODEL_DIRS
        model = getattr(args, f"{kind}_model") or next(
            (
                directory / filename
                for directory in directories
                if (directory / filename).is_file()
            ),
            None,
        )
        if model is None or not model.is_file():
            raise ValueError(f"missing {kind} model; supply --{kind}-model")
        models[device] = model.resolve()
    return list(itertools.product(devices, decode, encode)), models


def parse_sample(
    config: Configuration,
    video: Path,
    log: str,
    expected_frames: int | None,
    sample: dict[str, object],
) -> Timing:
    device, decode, encode = config
    matches = PROFILE.findall(log)
    if len(matches) != 1:
        raise ValueError("expected exactly one runtime Profile line")
    frames, seconds, fps = (
        int(matches[0][0]),
        float(matches[0][1]),
        float(matches[0][2]),
    )
    sample.update(frames=frames, seconds=seconds, fps=fps)
    expected = (
        (
            "device",
            r"^Detector running on (.+)$",
            {"cpu": "Cpu", "cuda": "Cuda(0)", "mps": "Mps"}[device],
        ),
        (
            "decoder",
            r"^Decoder initialized: (.+)$",
            "CPU"
            if decode == "cpu"
            else "CUDA device context requested successfully; per-frame NVDEC use is not observable (host RGB output)",
        ),
        (
            "encoder",
            r"^Encoder selected and opened: ([^;]+)",
            "libx264" if encode == "cpu" else "h264_nvenc",
        ),
    )
    selected = {
        name: re.findall(pattern, log, re.MULTILINE)
        for name, pattern, _ in expected
    }
    sample["selected"] = selected
    if any(selected[name] != [value] for name, _, value in expected):
        raise ValueError("runtime selected a different device or codec")
    if re.search(r"fall(?:ing)?[ -]?back", log, re.IGNORECASE):
        raise ValueError("runtime reported a fallback")
    if min(frames, seconds, fps) <= 0:
        raise ValueError("runtime reported empty video or nonpositive timing")
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(
            f"frame count mismatch: {frames} != {expected_frames}"
        )
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError("runtime did not produce a nonempty output video")
    return frames, seconds, fps


def save_results(
    output: Path,
    results: dict[str, object],
    timings: dict[Configuration, list[Timing]],
    repeats: int,
) -> None:
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output / "results.json")
    fields = [
        "device",
        "precision",
        "decoder",
        "encoder",
        "runs",
        "expected_runs",
        "frames",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "median_fps",
    ]
    rows: list[dict[str, str | int | float]] = []
    for (device, decode, encode), samples in timings.items():
        row: dict[str, str | int | float] = dict.fromkeys(fields, "")
        row.update(
            device=device,
            precision="FP16" if device == "cuda" else "FP32",
            decoder="CPU"
            if decode == "cpu"
            else "CUDA context (NVDEC unverified)",
            encoder="libx264" if encode == "cpu" else "h264_nvenc",
            runs=len(samples),
            expected_runs=repeats,
        )
        if samples:
            seconds = [sample[1] for sample in samples]
            row.update(
                frames=samples[0][0],
                median_seconds=statistics.median(seconds),
                min_seconds=min(seconds),
                max_seconds=max(seconds),
                median_fps=statistics.median(sample[2] for sample in samples),
            )
        rows.append(row)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Runtime profile",
        "",
        f"Complete: {results['complete']}",
        "",
        NOTE,
        "",
        "Only successful samples enter the summary; blank cells mean no measurement.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = [
            f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in row.values()
        ]
        lines.append("| " + " | ".join(values) + " |")
    if results.get("error"):
        lines.extend(["", f"Failure: {results['error']}"])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    if (
        min(args.imgsz, args.batch, args.repeats) <= 0
        or min(args.max_age, args.log_every) < 0
    ):
        raise ValueError(
            "imgsz/batch/repeats must be positive; max-age/log-every nonnegative"
        )
    if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
        raise ValueError("conf and iou must be between 0 and 1")
    system = platform.system()
    if system not in ("Linux", "Darwin"):
        raise ValueError("profiling supports Linux and macOS only")
    configs, models = configuration(args, system)
    args.source, args.binary = args.source.resolve(), args.binary.resolve()
    for path in (args.source, args.binary):
        if not path.is_file():
            raise ValueError(f"missing file: {path}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (
        args.output_dir
        or ROOT / "training-stage/artifacts/profiles" / f"{system}-{stamp}"
    ).absolute()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, object]] = []
    timings: dict[Configuration, list[Timing]] = {
        config: [] for config in configs
    }
    results: dict[str, object] = {
        "complete": False,
        "started_utc": stamp,
        "notes": NOTE,
        "repeats": args.repeats,
        "expected_runs": len(configs) * args.repeats,
        "samples": samples,
        "configurations": [
            {
                "device": device,
                "precision": "FP16" if device == "cuda" else "FP32",
                "decode": decode,
                "encode": encode,
            }
            for device, decode, encode in configs
        ],
        "workload": {
            name: getattr(args, name)
            for name in (
                "variant",
                "imgsz",
                "batch",
                "conf",
                "iou",
                "max_age",
                "log_every",
            )
        },
    }
    sample: dict[str, object] | None = None
    expected_frames: int | None = None
    try:
        dirty = query(
            "git", "status", "--porcelain", "--untracked-files=normal"
        )
        results["git"] = {
            "commit": query("git", "rev-parse", "HEAD"),
            "dirty": None if dirty is None else bool(dirty),
            "status": dirty,
        }
        results["host"] = host_metadata(system)
        results["binary"] = sha256(args.binary)
        results["source"] = sha256(args.source)
        results["models"] = {
            device: sha256(model) for device, model in models.items()
        }
        for repeat in range(1, args.repeats + 1):
            for config in configs:
                device, decode, encode = config
                name = f"{device}-decode-{decode}-encode-{encode}-{repeat}"
                video, log_path = (
                    output / f"{name}.mp4",
                    output / f"{name}.log",
                )
                command = [
                    str(args.binary),
                    "--model",
                    str(models[device]),
                    "--source",
                    str(args.source),
                    "--destination",
                    str(video),
                    "--device",
                    device,
                    "--decode-backend",
                    decode,
                    "--encode-backend",
                    encode,
                    "--profile",
                ]
                for flag, value in (
                    ("imgsz", args.imgsz),
                    ("batch-size", args.batch),
                    ("conf", args.conf),
                    ("iou-thresh", args.iou),
                    ("max-age", args.max_age),
                    ("log-every", args.log_every),
                ):
                    command.extend([f"--{flag}", str(value)])
                sample = {
                    "device": device,
                    "precision": "FP16" if device == "cuda" else "FP32",
                    "decode": decode,
                    "encode": encode,
                    "repeat": repeat,
                    "command": command,
                    "cwd": str(ROOT),
                    "video": str(video),
                    "log": str(log_path),
                    "status": "running",
                }
                samples.append(sample)
                (output / f"{name}.command.txt").write_text(
                    shlex.join(command) + "\n"
                )
                save_results(output, results, timings, args.repeats)
                print(
                    f"[{len(samples)}/{len(configs) * args.repeats}] {name}",
                    flush=True,
                )
                started = time.perf_counter()
                try:
                    with log_path.open("w") as log:
                        process = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    sample["returncode"] = process.returncode
                finally:
                    sample["process_wall_seconds"] = (
                        time.perf_counter() - started
                    )
                if process.returncode:
                    raise RuntimeError(
                        f"runtime exited with status {process.returncode}; see {log_path}"
                    )
                timing = parse_sample(
                    config,
                    video,
                    log_path.read_text(errors="replace"),
                    expected_frames,
                    sample,
                )
                frames, seconds, fps = timing
                expected_frames = frames
                sample.update(
                    frames=frames, seconds=seconds, fps=fps, status="complete"
                )
                timings[config].append(timing)
                print(
                    f"  {frames} frames in {seconds:.3f}s ({fps:.2f} FPS)",
                    flush=True,
                )
                save_results(output, results, timings, args.repeats)
        results["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        message = str(error) or type(error).__name__
        results["error"] = message
        if sample is not None and sample["status"] == "running":
            sample.update(status="failed", error=message)
        (output / "failure.log").write_text(message + "\n")
        raise
    finally:
        save_results(output, results, timings, args.repeats)
        status = "complete" if results["complete"] else "INCOMPLETE"
        print(f"Results ({status}): {output}", flush=True)


def main():
    try:
        run(parse_args())
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        print(
            f"Profiling failed: {str(error) or 'interrupted'}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
