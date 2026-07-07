from __future__ import annotations

import argparse
import ast
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_NAMES = ["bus", "car", "two_wheeler"]
COLORS = [(66, 133, 244), (52, 168, 83), (251, 188, 5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready figures from a YOLO runtime dataset/run.")
    parser.add_argument("--dataset", required=True, type=Path, help="YOLO dataset directory.")
    parser.add_argument("--run", required=True, type=Path, help="YOLO training run directory.")
    parser.add_argument("--output", required=True, type=Path, help="Artifact output directory.")
    parser.add_argument("--samples-per-split", type=int, default=6)
    return parser.parse_args()


def read_names(dataset: Path) -> list[str]:
    yaml_path = dataset / "traffic_vehicle.yaml"
    if not yaml_path.exists():
        return DEFAULT_NAMES
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            try:
                return list(ast.literal_eval(line.split(":", maxsplit=1)[1].strip()))
            except (SyntaxError, ValueError):
                return DEFAULT_NAMES
    return DEFAULT_NAMES


def read_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        boxes.append((cls_id, *(float(v) for v in parts[1:])))
    return boxes


def collect_stats(dataset: Path, names: list[str]) -> tuple[list[dict[str, int]], Counter]:
    rows = []
    class_counts: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        images = sorted(image_dir.glob("*.jpg"))
        labeled_images = 0
        empty_images = 0
        split_boxes = 0
        split_counts: Counter[int] = Counter()
        for image_path in images:
            boxes = read_label(label_dir / f"{image_path.stem}.txt")
            if boxes:
                labeled_images += 1
            else:
                empty_images += 1
            for cls_id, *_ in boxes:
                split_counts[cls_id] += 1
                class_counts[cls_id] += 1
                split_boxes += 1

        row = {
            "split": split,
            "images": len(images),
            "labeled_images": labeled_images,
            "empty_images": empty_images,
            "boxes": split_boxes,
        }
        for idx, name in enumerate(names):
            row[name] = split_counts[idx]
        rows.append(row)
    return rows, class_counts


def write_csv(path: Path, rows: list[dict[str, int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[int]) -> None:
    width, height = 1100, 650
    margin_left, margin_right, margin_top, margin_bottom = 110, 60, 90, 120
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (margin_left, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    bar_w = max(40, int(plot_w / max(len(values), 1) * 0.55))

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_h), (120, 120, 120), 2)
    cv2.line(canvas, (margin_left, margin_top + plot_h), (width - margin_right, margin_top + plot_h), (120, 120, 120), 2)

    for i, (label, value) in enumerate(zip(labels, values)):
        cx = margin_left + int((i + 0.5) * plot_w / len(values))
        bar_h = int(value / max_value * plot_h)
        x1, x2 = cx - bar_w // 2, cx + bar_w // 2
        y1, y2 = margin_top + plot_h - bar_h, margin_top + plot_h
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.putText(canvas, str(value), (x1, max(70, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
        cv2.putText(canvas, label, (x1, y2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def draw_boxes(image, boxes: list[tuple[int, float, float, float, float]], names: list[str]):
    h, w = image.shape[:2]
    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        cv2.putText(image, label, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return image


def make_sample_grid(dataset: Path, output: Path, split: str, names: list[str], samples_per_split: int) -> None:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    images = sorted(image_dir.glob("*.jpg"))
    labeled = [p for p in images if read_label(label_dir / f"{p.stem}.txt")]
    selected = (labeled + [p for p in images if p not in set(labeled)])[:samples_per_split]
    if not selected:
        return

    thumbs = []
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = read_label(label_dir / f"{image_path.stem}.txt")
        image = draw_boxes(image, boxes, names)
        scale = 420 / image.shape[1]
        thumb = cv2.resize(image, (420, int(image.shape[0] * scale)))
        label_strip = np.full((38, thumb.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(label_strip, image_path.stem, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        thumbs.append(np.vstack([thumb, label_strip]))

    if not thumbs:
        return

    cols = 2
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = max(t.shape[1] for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * cell_h : r * cell_h + thumb.shape[0], c * cell_w : c * cell_w + thumb.shape[1]] = thumb

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)


def copy_training_outputs(run_dir: Path, output: Path) -> list[str]:
    copied = []
    output.mkdir(parents=True, exist_ok=True)
    wanted_suffixes = {".png", ".jpg", ".jpeg", ".csv"}
    for src in sorted(run_dir.glob("*")):
        if src.is_file() and src.suffix.lower() in wanted_suffixes:
            dst = output / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)
    return copied


def main() -> None:
    args = parse_args()
    names = read_names(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)

    split_rows, class_counts = collect_stats(args.dataset, names)
    write_csv(args.output / "dataset_summary.csv", split_rows)
    write_csv(
        args.output / "class_counts.csv",
        [{"class": names[i], "boxes": class_counts[i]} for i in range(len(names))],
    )

    draw_bar_chart(
        args.output / "split_image_counts.png",
        "Dataset Images by Split",
        [row["split"] for row in split_rows],
        [row["images"] for row in split_rows],
    )
    draw_bar_chart(
        args.output / "class_box_counts.png",
        "Bounding Boxes by Class",
        names,
        [class_counts[i] for i in range(len(names))],
    )

    for split in ("train", "val", "test"):
        make_sample_grid(args.dataset, args.output / f"sample_{split}_annotations.jpg", split, names, args.samples_per_split)

    copied = copy_training_outputs(args.run, args.output / "training_outputs")
    readme = [
        "# Paper Artifacts",
        "",
        "- `dataset_summary.csv`: split-level image, label, and box counts.",
        "- `class_counts.csv`: class-level box counts.",
        "- `split_image_counts.png`: dataset split figure.",
        "- `class_box_counts.png`: class distribution figure.",
        "- `sample_*_annotations.jpg`: annotated examples from each split.",
        "- `training_outputs/`: YOLO training curves, confusion matrix, and batch visualizations when generated.",
        "",
        f"Copied training outputs: {', '.join(copied) if copied else 'none'}",
    ]
    (args.output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Paper artifacts: {args.output}")


if __name__ == "__main__":
    main()
