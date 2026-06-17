from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"val": ["val", "valid", "validation"], "valid": ["val", "valid", "validation"]}


def find_data_yaml(data_root: str | Path, data_yaml: str | Path | None = None) -> Path | None:
    candidates = []
    if data_yaml:
        candidates.append(Path(data_yaml))
    root = Path(data_root)
    candidates.extend([root / "data.yaml", root / root.name / "data.yaml"])
    if root.exists():
        candidates.extend(root.glob("*/data.yaml"))
    return next((path for path in candidates if path.exists()), None)


def _read_yaml_names(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(x) for x in names]
    except Exception:
        pass
    names: list[str] = []
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip().isdigit():
                names.append(value.strip().strip("'\""))
            elif names:
                break
    return names


def resolve_data_root(data_root: str | Path, data_yaml: str | Path | None = None) -> Path:
    root = Path(data_root)
    if data_yaml:
        yaml_parent = Path(data_yaml).parent
        if yaml_parent.exists() and any((yaml_parent / s).exists() for s in ["train", "val", "valid", "test"]):
            return yaml_parent
    if any((root / s).exists() for s in ["train", "val", "valid", "test"]):
        return root
    children = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    for child in children:
        if any((child / s).exists() for s in ["train", "val", "valid", "test"]):
            return child
    return root


def normalize_split_name(split: str) -> str:
    return "val" if split in {"valid", "validation"} else split


def split_dir(root: Path, split: str) -> Path:
    names = SPLIT_ALIASES.get(split, [split])
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find split directory for '{split}' under {root}")


def discover_classes(data_root: str | Path, data_yaml: str | Path | None = None) -> list[str]:
    root = resolve_data_root(data_root, data_yaml)
    names = _read_yaml_names(find_data_yaml(data_root, data_yaml) or root / "data.yaml")
    existing = set()
    for split in ["train", "val", "test"]:
        try:
            base = split_dir(root, split)
        except FileNotFoundError:
            continue
        existing.update(p.name for p in base.iterdir() if p.is_dir())
    if names:
        return [name for name in names if name != "normal"]
    return sorted(name for name in existing if name != "normal")


def create_transforms(image_size: int, train: bool = False) -> Callable:
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend([transforms.RandomRotation(5), transforms.ColorJitter(brightness=0.1, contrast=0.1)])
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def _image_key(path: Path) -> str:
    return path.name.lower()


class RSHSMultiLabelDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        class_names: list[str] | None = None,
        data_yaml: str | Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.root = resolve_data_root(data_root, data_yaml)
        self.split = normalize_split_name(split)
        self.class_names = class_names or discover_classes(self.root, data_yaml)
        self.transform = transform
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"No images found for split '{split}' under {self.root}")

    def _load_samples(self) -> list[dict]:
        csv_path = self.root / f"{self.split}_labels.csv"
        if self.split == "val" and not csv_path.exists():
            csv_path = self.root / "valid_labels.csv"
        if csv_path.exists():
            return self._load_csv_samples(csv_path)
        return self._load_folder_samples()

    def _load_csv_samples(self, csv_path: Path) -> list[dict]:
        image_dirs = [self.root / self.split / "images", self.root / self.split, self.root / "images" / self.split]
        samples: list[dict] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{csv_path} has no header")
            filename_col = reader.fieldnames[0]
            for row in reader:
                filename = row[filename_col]
                path = next((d / filename for d in image_dirs if (d / filename).exists()), image_dirs[0] / filename)
                labels = [float(row.get(name, 0) or 0) for name in self.class_names]
                samples.append({"path": path, "labels": labels, "filename": filename})
        return samples

    def _load_folder_samples(self) -> list[dict]:
        base = split_dir(self.root, self.split)
        by_file: dict[str, dict] = {}
        class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            class_name = class_dir.name
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                key = _image_key(image_path)
                entry = by_file.setdefault(
                    key,
                    {"path": image_path, "labels": [0.0] * len(self.class_names), "filename": image_path.name},
                )
                if class_name != "normal" and class_name in class_to_idx:
                    entry["labels"][class_to_idx[class_name]] = 1.0
        return list(by_file.values())

    def class_distribution(self) -> list[dict]:
        labels = torch.stack([torch.tensor(sample["labels"], dtype=torch.float32) for sample in self.samples])
        positives = labels.sum(dim=0)
        total = labels.shape[0]
        rows = []
        for idx, name in enumerate(self.class_names):
            pos = int(positives[idx].item())
            neg = int(total - pos)
            rows.append({"class_name": name, "positive_count": pos, "negative_count": neg, "pos_weight": neg / max(pos, 1)})
        return rows

    def save_class_distribution(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = self.class_distribution()
        with (output / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class_name", "positive_count", "negative_count", "pos_weight"])
            writer.writeheader()
            writer.writerows(rows)
        (output / "class_distribution.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
            "path": str(sample["path"]),
            "filename": sample["filename"],
        }
