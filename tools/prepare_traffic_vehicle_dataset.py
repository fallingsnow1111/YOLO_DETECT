from __future__ import annotations

import shutil
import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "dataset"

VARIANTS = {
    "baseline": {
        "output": ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "traffic_vehicle_dataset_3cls",
        "splits": {
            "train": ["1", "2", "3", "4", "5"],
            "val": ["6"],
            "test": ["7"],
        },
    },
    "improved": {
        "output": ROOT / "ultralytics" / "yolo" / "v8" / "detect" / "traffic_vehicle_dataset_improved_3cls",
        "splits": {
            "train": ["1", "2", "3", "4", "5", "7"],
            "val": ["6"],
            "test": ["6"],
        },
    },
}

# CVAT export IDs: 0 bus, 1 car, 2 truck, 3 two_wheeler.
# The course dataset does not keep truck, so two_wheeler is remapped from 3 to 2.
CLASS_REMAP = {
    0: 0,
    1: 1,
    3: 2,
}

NAMES = ["bus", "car", "two_wheeler"]


def convert_label(src: Path, dst: Path) -> Counter:
    counts: Counter[int] = Counter()
    lines: list[str] = []

    if src.exists():
        for raw in src.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            class_id = int(float(parts[0]))
            if class_id not in CLASS_REMAP:
                continue
            new_id = CLASS_REMAP[class_id]
            lines.append(" ".join([str(new_id), *parts[1:5]]))
            counts[new_id] += 1

    dst.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return counts


def prepare_dirs(output_root: Path, splits: dict[str, list[str]], overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists, use --overwrite to rebuild: {output_root}")
        shutil.rmtree(output_root)

    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS.keys(), default="baseline")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = VARIANTS[args.variant]["output"]
    splits = VARIANTS[args.variant]["splits"]

    prepare_dirs(output_root, splits, args.overwrite)
    total_images = Counter()
    total_boxes = Counter()
    empty_labels = Counter()

    for split, dataset_ids in splits.items():
        for dataset_id in dataset_ids:
            image_dir = SOURCE_ROOT / dataset_id / "images" / "train"
            label_dir = SOURCE_ROOT / dataset_id / "labels" / "train"

            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                stem = f"d{dataset_id}_{image_path.stem}"
                dst_image = output_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
                dst_label = output_root / "labels" / split / f"{stem}.txt"
                src_label = label_dir / f"{image_path.stem}.txt"

                shutil.copy2(image_path, dst_image)
                counts = convert_label(src_label, dst_label)

                total_images[split] += 1
                total_boxes.update(counts)
                if not dst_label.read_text(encoding="utf-8").strip():
                    empty_labels[split] += 1

    yaml_text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        f"names: {NAMES!r}\n"
    )
    (output_root / "traffic_vehicle_3cls.yaml").write_text(yaml_text, encoding="utf-8")

    print("Dataset prepared:", output_root)
    print("Images:", dict(total_images))
    print("Empty labels:", dict(empty_labels))
    print("Boxes:", {NAMES[k]: total_boxes[k] for k in range(len(NAMES))})


if __name__ == "__main__":
    main()
