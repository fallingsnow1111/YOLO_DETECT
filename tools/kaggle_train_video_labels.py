from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=kaggle_path("runs"))
    parser.add_argument("--name", default="traffic_vehicle_yolov8m_video_labels")
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--no-overwrite", action="store_true", help="Do not rebuild output if it exists.")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_yaml = args.output / "traffic_vehicle.yaml"

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
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        "cache=False",
        f"device={args.device}",
        f"project={args.project}",
        f"name={args.name}",
    ]
    run(train_command, env=env)

    weights = args.project / args.name / "weights" / "best.pt"
    print("\nDone.")
    print(f"Best weights: {weights}")
    print(f"Runtime dataset: {args.output}")


if __name__ == "__main__":
    main()
