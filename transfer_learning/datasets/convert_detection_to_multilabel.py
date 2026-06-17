from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rshs_dataset import IMAGE_EXTENSIONS, _read_yaml_names, resolve_data_root, split_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO detection labels to image-level multi-label CSV files.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def convert_split(root: Path, split: str, class_names: list[str], output_root: Path) -> None:
    base = split_dir(root, split)
    image_root = base / "images" if (base / "images").exists() else root / "images" / split
    label_root = base / "labels" if (base / "labels").exists() else root / "labels" / split
    if not image_root.exists() or not label_root.exists():
        raise FileNotFoundError(f"Expected YOLO images and labels for split '{split}' under {base} or {root}")
    rows = []
    for image_path in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        labels = [0] * len(class_names)
        label_path = label_root / image_path.with_suffix(".txt").name
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")
                class_id = int(float(parts[0]))
                if class_id < 0 or class_id >= len(class_names):
                    raise ValueError(f"Class id {class_id} in {label_path} is outside configured classes")
                labels[class_id] = 1
        rows.append({"filename": image_path.name, **{name: labels[i] for i, name in enumerate(class_names)}})
    with (output_root / f"{split}_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", *class_names])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = resolve_data_root(args.data_root, args.data_yaml)
    output_root = args.output_root or root
    output_root.mkdir(parents=True, exist_ok=True)
    class_names = [name for name in _read_yaml_names(args.data_yaml or root / "data.yaml") if name != "normal"]
    if not class_names:
        raise ValueError("Could not determine class names. Provide --data-yaml with a YOLO-style names section.")
    for split in ["train", "val", "test"]:
        convert_split(root, split, class_names, output_root)


if __name__ == "__main__":
    main()
