from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2


SPLITS = ("train", "val", "test")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
OLD_LABEL_RE = re.compile(r"^d(?P<video_id>[^_]+)_frame_(?P<frame>\d+)\.txt$")
FRAME_RE = re.compile(r"frame_(?P<frame>\d+)", re.IGNORECASE)
NAMES = ["bus", "car", "two_wheeler"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a temporary YOLO dataset from compact video + CVAT YOLO labels."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory with mp4/ + labels/*.zip, or legacy videos/ + labels/{train,val,test}.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--jpg-quality", type=int, default=90, help="JPEG quality for extracted frames.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output directory before rebuilding.")
    return parser.parse_args()


def find_video(videos_dir: Path, video_id: str) -> Path:
    candidates = [video_id]
    if video_id.isdigit():
        candidates.extend([str(int(video_id)), f"{int(video_id):02d}"])

    seen = set()
    for stem in candidates:
        if stem in seen:
            continue
        seen.add(stem)
        for ext in VIDEO_EXTS:
            candidate = videos_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Missing video for id {video_id}: {videos_dir / (video_id + '.mp4')}")


def parse_old_label_name(label_path: Path) -> tuple[str, int]:
    match = OLD_LABEL_RE.match(label_path.name)
    if not match:
        raise ValueError(f"Label name must look like d7_frame_000123.txt: {label_path.name}")
    return match.group("video_id"), int(match.group("frame"))


def parse_frame_id(path_text: str) -> int | None:
    match = FRAME_RE.search(Path(path_text).name)
    if not match:
        return None
    return int(match.group("frame"))


def validate_label_text(text: str, source_name: str) -> tuple[str, Counter]:
    counts: Counter[int] = Counter()
    lines = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{source_name}:{line_no} should have 5 columns, got {len(parts)}")
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= len(NAMES):
            raise ValueError(f"{source_name}:{line_no} class id {cls_id} is outside 0-{len(NAMES) - 1}")
        coords = [float(v) for v in parts[1:]]
        if any(v < 0 or v > 1 for v in coords):
            raise ValueError(f"{source_name}:{line_no} bbox values must be normalized to 0-1")
        counts[cls_id] += 1
        lines.append(" ".join([str(cls_id), *parts[1:]]))

    return (("\n".join(lines) + "\n") if lines else ""), counts


def validate_and_copy_label(src: Path, dst: Path) -> Counter:
    normalized, counts = validate_label_text(src.read_text(encoding="utf-8"), str(src))
    dst.write_text(normalized, encoding="utf-8")
    return counts


class VideoReaderCache:
    def __init__(self, videos_dir: Path):
        self.videos_dir = videos_dir
        self.captures: dict[str, cv2.VideoCapture] = {}

    def read_frame(self, video_id: str, frame_idx: int):
        if video_id not in self.captures:
            video_path = find_video(self.videos_dir, video_id)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            self.captures[video_id] = cap

        cap = self.captures[video_id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from video {video_id}")
        return frame

    def close(self) -> None:
        for cap in self.captures.values():
            cap.release()


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists, use --overwrite to rebuild: {output}")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(output: Path) -> None:
    yaml_text = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES!r}\n"
    )
    (output / "traffic_vehicle.yaml").write_text(yaml_text, encoding="utf-8")


def split_for_zip(zip_stem: str) -> str:
    lowered = zip_stem.lower()
    if lowered == "val":
        return "val"
    if lowered == "test":
        return "test"
    return "train"


def read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8-sig")


def collect_zip_labels(zf: zipfile.ZipFile) -> dict[int, str]:
    labels: dict[int, str] = {}
    for name in zf.namelist():
        normalized = name.replace("\\", "/")
        if not normalized.lower().endswith(".txt"):
            continue
        if Path(normalized).name.lower() == "train.txt":
            continue
        frame_idx = parse_frame_id(normalized)
        if frame_idx is None:
            continue
        labels[frame_idx] = name
    return labels


def collect_zip_frames(zf: zipfile.ZipFile, label_entries: dict[int, str]) -> list[int]:
    train_list_name = next((name for name in zf.namelist() if Path(name).name.lower() == "train.txt"), None)
    frames: set[int] = set()
    if train_list_name:
        for raw in read_zip_text(zf, train_list_name).splitlines():
            frame_idx = parse_frame_id(raw.strip())
            if frame_idx is not None:
                frames.add(frame_idx)

    if not frames:
        frames.update(label_entries.keys())

    return sorted(frames)


def write_frame_image(frame, image_path: Path, jpg_quality: int) -> None:
    ok = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    if not ok:
        raise RuntimeError(f"Could not write image: {image_path}")


def build_from_zip_dataset(input_root: Path, output_root: Path, jpg_quality: int) -> tuple[Counter, Counter, Counter]:
    videos_dir = input_root / "mp4"
    labels_dir = input_root / "labels"
    if not videos_dir.is_dir():
        raise SystemExit(f"Missing mp4 directory: {videos_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labels directory: {labels_dir}")

    zip_paths = sorted(labels_dir.glob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No label zip files found in: {labels_dir}")

    reader = VideoReaderCache(videos_dir)
    image_counts = Counter()
    empty_counts = Counter()
    box_counts: Counter[int] = Counter()

    try:
        for zip_path in zip_paths:
            video_id = zip_path.stem
            split = split_for_zip(video_id)
            with zipfile.ZipFile(zip_path) as zf:
                label_entries = collect_zip_labels(zf)
                frame_indices = collect_zip_frames(zf, label_entries)

                for frame_idx in frame_indices:
                    out_stem = f"{video_id}_frame_{frame_idx:06d}"
                    image_path = output_root / "images" / split / f"{out_stem}.jpg"
                    out_label_path = output_root / "labels" / split / f"{out_stem}.txt"

                    frame = reader.read_frame(video_id, frame_idx)
                    write_frame_image(frame, image_path, jpg_quality)

                    label_entry = label_entries.get(frame_idx)
                    if label_entry is None:
                        out_label_path.write_text("", encoding="utf-8")
                        empty_counts[split] += 1
                    else:
                        text = read_zip_text(zf, label_entry)
                        normalized, counts = validate_label_text(text, f"{zip_path.name}:{label_entry}")
                        out_label_path.write_text(normalized, encoding="utf-8")
                        box_counts.update(counts)
                        if not normalized.strip():
                            empty_counts[split] += 1

                    image_counts[split] += 1
    finally:
        reader.close()

    return image_counts, empty_counts, box_counts


def build_from_legacy_dataset(input_root: Path, output_root: Path, jpg_quality: int) -> tuple[Counter, Counter, Counter]:
    videos_dir = input_root / "videos"
    labels_dir = input_root / "labels"
    if not videos_dir.is_dir():
        raise SystemExit(f"Missing videos directory: {videos_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labels directory: {labels_dir}")

    reader = VideoReaderCache(videos_dir)
    image_counts = Counter()
    empty_counts = Counter()
    box_counts: Counter[int] = Counter()

    try:
        for split in SPLITS:
            split_labels = labels_dir / split
            if not split_labels.is_dir():
                continue

            for label_path in sorted(split_labels.glob("*.txt")):
                video_id, frame_idx = parse_old_label_name(label_path)
                image_path = output_root / "images" / split / f"{label_path.stem}.jpg"
                out_label_path = output_root / "labels" / split / label_path.name

                frame = reader.read_frame(video_id, frame_idx)
                write_frame_image(frame, image_path, jpg_quality)

                counts = validate_and_copy_label(label_path, out_label_path)
                box_counts.update(counts)
                image_counts[split] += 1
                if not out_label_path.read_text(encoding="utf-8").strip():
                    empty_counts[split] += 1
    finally:
        reader.close()

    return image_counts, empty_counts, box_counts


def main() -> None:
    args = parse_args()
    input_root = args.input
    output_root = args.output

    prepare_output(output_root, args.overwrite)

    if (input_root / "mp4").is_dir() and (input_root / "labels").is_dir():
        image_counts, empty_counts, box_counts = build_from_zip_dataset(
            input_root, output_root, args.jpg_quality
        )
    else:
        image_counts, empty_counts, box_counts = build_from_legacy_dataset(
            input_root, output_root, args.jpg_quality
        )

    write_yaml(output_root)
    print("Dataset prepared:", output_root)
    print("Images:", dict(image_counts))
    print("Empty labels:", dict(empty_counts))
    print("Boxes:", {NAMES[i]: box_counts[i] for i in range(len(NAMES))})
    print("YAML:", output_root / "traffic_vehicle.yaml")


if __name__ == "__main__":
    main()
