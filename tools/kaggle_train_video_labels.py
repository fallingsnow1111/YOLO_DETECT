from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "cv2": "opencv-python-headless",
    "easydict": "easydict",
    "git": "GitPython",
    "hydra": "hydra-core",
    "IPython": "IPython",
    "thop": "thop",
}


def kaggle_path(name: str) -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / name
    return Path(name)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build the runtime vehicle dataset from video-label zips and train YOLOv8."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "ultralytics" / "yolo" / "v8" / "dataset",
        help="Compact dataset directory with mp4/ and labels/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=kaggle_path("traffic_vehicle_dataset_runtime"),
        help="Temporary YOLO dataset output directory.",
    )
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=kaggle_path("runs"))
    parser.add_argument("--name", default="traffic_vehicle_yolov8m_video_labels")
    parser.add_argument("--artifacts-output", type=Path, default=kaggle_path("paper_artifacts"))
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--no-overwrite", action="store_true", help="Do not rebuild output if it exists.")
    parser.add_argument("--skip-install-deps", action="store_true", help="Skip missing dependency installation.")
    parser.add_argument("--skip-export-artifacts", action="store_true", help="Skip paper artifact export.")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def install_missing_deps() -> None:
    missing = [
        package
        for module_name, package in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        run([sys.executable, "-m", "pip", "install", "-q", *missing])


def check_cuda_device(device: str) -> None:
    if device.lower() == "cpu":
        return
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. In Kaggle, enable GPU T4 x2, then rerun this script.")

    device_id = int(device.split(",", maxsplit=1)[0])
    name = torch.cuda.get_device_name(device_id)
    major, minor = torch.cuda.get_device_capability(device_id)
    print(f"CUDA device {device_id}: {name}, capability sm_{major}{minor}", flush=True)
    if major < 7:
        raise SystemExit(
            "This Kaggle PyTorch build does not support this GPU architecture. "
            "Use Kaggle Accelerator: GPU T4 x2 instead of GPU P100."
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_yaml = args.output / "traffic_vehicle.yaml"

    if not args.skip_install_deps:
        install_missing_deps()

    check_cuda_device(args.device)

    build_command = [
        sys.executable,
        str(repo_root / "tools" / "build_dataset_from_videos.py"),
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--jpg-quality",
        str(args.jpg_quality),
    ]
    if not args.no_overwrite:
        build_command.append("--overwrite")
    run(build_command)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not old_pythonpath else str(repo_root) + os.pathsep + old_pythonpath

    train_command = [
        sys.executable,
        str(repo_root / "ultralytics" / "yolo" / "v8" / "detect" / "train.py"),
        f"model={args.model}",
        f"data={dataset_yaml}",
        f"epochs={args.epochs}",
        f"patience={args.patience}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        "cache=False",
        f"device={args.device}",
        f"project={args.project}",
        f"name={args.name}",
    ]
    run(train_command, env=env)

    run_dir = args.project / args.name
    if not args.skip_export_artifacts:
        export_command = [
            sys.executable,
            str(repo_root / "tools" / "export_paper_artifacts.py"),
            "--dataset",
            str(args.output),
            "--run",
            str(run_dir),
            "--output",
            str(args.artifacts_output),
        ]
        run(export_command, env=env)

    weights = args.project / args.name / "weights" / "best.pt"
    print("\nDone.")
    print(f"Best weights: {weights}")
    print(f"Runtime dataset: {args.output}")
    if not args.skip_export_artifacts:
        print(f"Paper artifacts: {args.artifacts_output}")


if __name__ == "__main__":
    main()
