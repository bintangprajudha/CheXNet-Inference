#!/usr/bin/env python3
"""Run CheXNet-style inference on train/valid/test image folders.

Defaults are aligned with this workspace:
- Checkpoint: model.pth.tar
- Dataset root: Splitted dataset
- Splits: train, valid, test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_CHEXNET14_CLASS_NAMES = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
]

DATASET_NAME_TO_CANONICAL = {
    "kavitas": "cavity",
    "infiltrat": "infiltration",
    "limfadenopati": "lymphadenopathy",
    "tuberkuloma": "tuberculoma",
    "bronkiektasis": "bronchiectasis",
    "pneumothorax": "pneumothorax",
    "efusi pleura": "effusion",
    "atelektasis": "atelectasis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on train/valid/test splits using model.pth.tar")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model.pth.tar"),
        help="Path to checkpoint (.pth/.pth.tar).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Splitted dataset"),
        help="Dataset root containing train/valid/test class folders.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Path to dataset YAML with class names. Defaults to <data-root>/data.yaml.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Deprecated: path to one test directory. If set, only this directory is processed as split 'test'.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to process from data-root.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("inference") / "inference_test_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Deprecated: labels are now read directly from class subdirectories.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("inference") / "inference_test_metrics.json",
        help="Output JSON path for evaluation metrics.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (square).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for multi-label predictions.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    parser.add_argument(
        "--generate-heatmaps",
        action="store_true",
        help="Generate Grad-CAM heatmaps and bounding boxes for inferred images.",
    )
    parser.add_argument(
        "--heatmap-output-dir",
        type=Path,
        default=Path("inference") / "heatmaps",
        help="Output directory for Grad-CAM heatmaps, masks, and bounding-box overlays.",
    )
    parser.add_argument(
        "--heatmap-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing Grad-CAM heatmaps before connected-component boxes.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=25,
        help="Minimum component area in heatmap pixels.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Maximum bounding-box components to keep per image/class.",
    )
    parser.add_argument(
        "--heatmap-target-classes",
        choices=["predicted", "top1", "all"],
        default="predicted",
        help="Classes to generate heatmaps for.",
    )
    parser.add_argument(
        "--roi-mask-root",
        type=Path,
        default=Path("lung segmentation") / "output_masks",
        help="Root directory containing lung ROI masks by split. Used for model inference.",
    )
    parser.add_argument(
        "--no-roi-mask",
        action="store_true",
        help="Disable lung ROI masking for model inference.",
    )
    return parser.parse_args()


def resolve_data_root(data_root: Path) -> Path:
    """Resolve plain or accidentally nested dataset roots."""
    if (data_root / "data.yaml").exists():
        return data_root

    nested_root = data_root / data_root.name
    if (nested_root / "data.yaml").exists():
        return nested_root

    return data_root


def resolve_data_yaml(data_root: Path, data_yaml: Path | None) -> Path:
    if data_yaml is not None:
        return data_yaml
    return data_root / "data.yaml"


def read_class_names(data_yaml: Path) -> List[str]:
    """Read class names from a YOLO-style YAML file without hard dependency on PyYAML."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    text = data_yaml.read_text(encoding="utf-8")

    # Try PyYAML first when available.
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        names_obj = parsed.get("names", {}) if isinstance(parsed, dict) else {}
        if isinstance(names_obj, dict):
            items = sorted((int(k), str(v)) for k, v in names_obj.items())
            return [name for _, name in items]
        if isinstance(names_obj, list):
            return [str(x) for x in names_obj]
    except Exception:
        pass

    # Fallback parser for simple `names:` mappings.
    names: Dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            in_names = True
            continue

        if in_names:
            if raw.startswith(" ") or raw.startswith("\t"):
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    try:
                        idx = int(k.strip())
                    except ValueError:
                        continue
                    names[idx] = v.strip().strip('"\'')
            else:
                break

    if not names:
        raise ValueError(f"Could not parse class names from: {data_yaml}")

    return [name for _, name in sorted(names.items(), key=lambda kv: kv[0])]


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    return model


def pick_state_dict(checkpoint_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state = checkpoint_obj.get(key)
            if isinstance(state, dict):
                return state
        # Sometimes checkpoint is already a raw state_dict.
        if all(isinstance(k, str) for k in checkpoint_obj.keys()):
            if any("weight" in k or "bias" in k for k in checkpoint_obj.keys()):
                return checkpoint_obj  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format; expected state_dict-like object.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def normalize_densenet_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common legacy CheXNet/DenseNet checkpoint key formats."""
    normalized: Dict[str, torch.Tensor] = {}

    replacements: Tuple[Tuple[str, str], ...] = (
        (".norm.1.", ".norm1."),
        (".relu.1.", ".relu1."),
        (".conv.1.", ".conv1."),
        (".norm.2.", ".norm2."),
        (".relu.2.", ".relu2."),
        (".conv.2.", ".conv2."),
    )

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("densenet121."):
            new_key = new_key.replace("densenet121.", "", 1)

        for old, new in replacements:
            new_key = new_key.replace(old, new)

        if new_key.startswith("classifier.0."):
            new_key = new_key.replace("classifier.0.", "classifier.", 1)

        normalized[new_key] = value

    return normalized


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int | None:
    weight = state_dict.get("classifier.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    bias = state_dict.get("classifier.bias")
    if isinstance(bias, torch.Tensor) and bias.ndim == 1:
        return int(bias.shape[0])
    return None


def default_model_class_names(num_classes: int) -> List[str]:
    if num_classes == len(DEFAULT_CHEXNET14_CLASS_NAMES):
        return list(DEFAULT_CHEXNET14_CLASS_NAMES)
    return [f"class_{i}" for i in range(num_classes)]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


def canonicalize_dataset_name(name: str) -> str:
    normalized = normalize_name(name)
    mapped = DATASET_NAME_TO_CANONICAL.get(normalized, normalized)
    return mapped.replace(" ", "_")


def canonicalize_model_name(name: str) -> str:
    return normalize_name(name).replace(" ", "_")


def build_intersection_map(
    dataset_class_names: List[str],
    model_class_names: List[str],
) -> List[Tuple[int, str, str]]:
    """Return tuples of (model_index, dataset_display_name, canonical_name)."""
    model_index_by_canonical = {
        canonicalize_model_name(model_name): idx for idx, model_name in enumerate(model_class_names)
    }

    intersection: List[Tuple[int, str, str]] = []
    for dataset_name in dataset_class_names:
        canonical = canonicalize_dataset_name(dataset_name)
        if canonical in model_index_by_canonical:
            intersection.append((model_index_by_canonical[canonical], dataset_name, canonical))

    return intersection


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_images_from_split(split_dir: Path, split_name: str) -> List[Tuple[str, str, Path]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_records: List[Tuple[str, str, Path]] = []
    for p in sorted(split_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            sample_id = p.relative_to(split_dir).as_posix()
            image_records.append((split_name, sample_id, p))

    if not image_records:
        raise ValueError(f"No images found in: {split_dir}")

    return image_records


def collect_images(data_root: Path, splits: List[str]) -> List[Tuple[str, str, Path]]:
    image_records: List[Tuple[str, str, Path]] = []
    for split in splits:
        image_records.extend(collect_images_from_split(data_root / split, split))
    return image_records


def load_ground_truth_vectors(
    data_root: Path,
    splits: List[str],
    intersect_map: List[Tuple[int, str, str]],
) -> Dict[str, List[int]]:
    """Build ground truth vectors for all unique images by scanning split/class directories."""
    dataset_name_to_local_idx = {
        dataset_display_name: local_idx
        for local_idx, (_, dataset_display_name, _) in enumerate(intersect_map)
    }

    class_count = len(intersect_map)
    gt_vectors: Dict[str, List[int]] = {}

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            local_idx = dataset_name_to_local_idx.get(class_name)

            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    sample_id = img_path.relative_to(split_dir).as_posix()
                    image_key = f"{split}/{sample_id}"
                    if image_key not in gt_vectors:
                        gt_vectors[image_key] = [0] * class_count

                    if local_idx is not None:
                        gt_vectors[image_key][local_idx] = 1

    return gt_vectors


def compute_metrics(
    pred_vectors: Dict[str, List[int]],
    gt_vectors: Dict[str, List[int]],
    class_names: List[str],
) -> Dict[str, object]:
    image_names = [name for name in pred_vectors.keys() if name in gt_vectors]
    n = len(image_names)
    if n == 0:
        raise ValueError("No overlapping images between predictions and ground-truth labels.")

    class_metrics: Dict[str, Dict[str, float | int]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    exact_match = 0

    for image_name in image_names:
        if pred_vectors[image_name] == gt_vectors[image_name]:
            exact_match += 1

    for class_idx, class_name in enumerate(class_names):
        tp = fp = tn = fn = 0
        for image_name in image_names:
            p = pred_vectors[image_name][class_idx]
            t = gt_vectors[image_name][class_idx]
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = tp + fn

        class_metrics[class_name] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    class_count = len(class_names)
    macro_precision /= class_count
    macro_recall /= class_count
    macro_f1 /= class_count

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "num_images": n,
        "exact_match_accuracy": round(exact_match / n, 6),
        "micro": {
            "precision": round(micro_precision, 6),
            "recall": round(micro_recall, 6),
            "f1": round(micro_f1, 6),
        },
        "macro": {
            "precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(macro_f1, 6),
        },
        "per_class": class_metrics,
    }


def save_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def compute_split_metrics(
    pred_vectors: Dict[str, List[int]],
    gt_vectors: Dict[str, List[int]],
    class_names: List[str],
    splits: List[str],
) -> Dict[str, object]:
    by_split: Dict[str, object] = {}

    for split in splits:
        prefix = f"{split}/"
        split_pred_vectors = {
            key: value for key, value in pred_vectors.items()
            if key.startswith(prefix)
        }
        split_gt_vectors = {
            key: value for key, value in gt_vectors.items()
            if key.startswith(prefix)
        }
        by_split[split] = compute_metrics(
            pred_vectors=split_pred_vectors,
            gt_vectors=split_gt_vectors,
            class_names=class_names,
        )

    return {
        "overall": compute_metrics(
            pred_vectors=pred_vectors,
            gt_vectors=gt_vectors,
            class_names=class_names,
        ),
        "by_split": by_split,
    }


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[:, class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_np -= float(cam_np.min())
        cam_np /= max(float(cam_np.max()), 1e-8)
        return cam_np


def denormalize_tensor(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = torch.clamp(tensor.detach().cpu() * std + mean, 0, 1)
    return transforms.ToPILImage()(image)


def overlay_heatmap(image: Image.Image, cam: np.ndarray) -> Image.Image:
    image_rgb = image.convert("RGB")
    heatmap = apply_jet_colormap(cam)
    heatmap = heatmap.resize(image_rgb.size, Image.Resampling.BILINEAR)
    return Image.blend(image_rgb, heatmap, alpha=0.45)


def apply_jet_colormap(cam: np.ndarray) -> Image.Image:
    values = np.clip(cam.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(np.uint8(rgb * 255), mode="RGB")


def threshold_heatmap(cam: np.ndarray, threshold: float) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--heatmap-threshold must be between 0.0 and 1.0")
    return (cam >= threshold).astype(np.uint8)


def load_roi_mask(mask_root: Path | None, split: str, image_name: str, size: Tuple[int, int]) -> Tuple[np.ndarray, str]:
    if mask_root is None:
        return np.ones((size[1], size[0]), dtype=np.float32), ""

    mask_path = mask_root / split / image_name
    if not mask_path.exists():
        return np.ones((size[1], size[0]), dtype=np.float32), ""

    with Image.open(mask_path) as mask_img:
        mask_img = mask_img.convert("L").resize(size, Image.Resampling.NEAREST)
        mask = (np.array(mask_img) > 0).astype(np.float32)

    return mask, mask_path.as_posix()


def apply_roi_to_tensor(tensor: torch.Tensor, roi_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    mask_tensor = torch.from_numpy(roi_mask).to(device=device, dtype=tensor.dtype).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=tensor.dtype)[:, None, None]
    image = torch.clamp(tensor * std + mean, 0, 1)
    masked_image = image * mask_tensor
    return (masked_image - mean) / std


def connected_components(mask: np.ndarray, min_area: int, max_components: int) -> List[Dict[str, float | int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Dict[str, float | int]] = []

    for start_y in range(height):
        for start_x in range(width):
            if mask[start_y, start_x] == 0 or visited[start_y, start_x]:
                continue

            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            pixels: List[Tuple[int, int]] = []

            while stack:
                x, y = stack.pop()
                pixels.append((x, y))

                for next_x in (x - 1, x, x + 1):
                    for next_y in (y - 1, y, y + 1):
                        if next_x == x and next_y == y:
                            continue
                        if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                            continue
                        if visited[next_y, next_x] or mask[next_y, next_x] == 0:
                            continue
                        visited[next_y, next_x] = True
                        stack.append((next_x, next_y))

            area = len(pixels)
            if area < min_area:
                continue

            xs = [x for x, _ in pixels]
            ys = [y for _, y in pixels]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)
            components.append(
                {
                    "x": int(x_min),
                    "y": int(y_min),
                    "width": int(x_max - x_min + 1),
                    "height": int(y_max - y_min + 1),
                    "area": int(area),
                    "centroid_x": round(float(sum(xs) / area), 4),
                    "centroid_y": round(float(sum(ys) / area), 4),
                }
            )

    components.sort(key=lambda item: int(item["area"]), reverse=True)
    return components[:max_components]


def overlay_bounding_boxes(
    image: Image.Image,
    cam: np.ndarray,
    components: List[Dict[str, float | int]],
    class_name: str,
    probability: float,
) -> Image.Image:
    overlay = overlay_heatmap(image, cam)
    draw = ImageDraw.Draw(overlay)
    for component in components:
        x = int(component["x"])
        y = int(component["y"])
        width = int(component["width"])
        height = int(component["height"])
        x2 = x + width
        y2 = y + height
        label = f"{class_name} {probability:.2f}"

        draw.rectangle((x, y, x2, y2), outline=(255, 255, 0), width=2)
        text_bbox = draw.textbbox((x, y), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, y - text_height - 4)
        draw.rectangle((x, label_y, x + text_width + 6, label_y + text_height + 4), fill=(255, 255, 0))
        draw.text((x + 3, label_y + 2), label, fill=(0, 0, 0))
    return overlay


def component_to_xywh(component: Dict[str, float | int]) -> str:
    return f"{component['x']},{component['y']},{component['width']},{component['height']}"


def component_to_yolo(component: Dict[str, float | int], image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    x = float(component["x"])
    y = float(component["y"])
    width = float(component["width"])
    height = float(component["height"])
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    return x_center, y_center, width / image_width, height / image_height


def slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def select_heatmap_targets(
    target_mode: str,
    row: Dict[str, str],
    intersect_map: List[Tuple[int, str, str]],
) -> List[Tuple[int, str]]:
    if target_mode == "all":
        return [(model_idx, dataset_name) for model_idx, dataset_name, _ in intersect_map]

    if target_mode == "top1":
        top1_label = row["top1_label"]
        return [
            (model_idx, dataset_name)
            for model_idx, dataset_name, _ in intersect_map
            if dataset_name == top1_label
        ]

    predicted_labels = {label for label in row["predicted_labels"].split("|") if label}
    selected = [
        (model_idx, dataset_name)
        for model_idx, dataset_name, _ in intersect_map
        if dataset_name in predicted_labels
    ]
    if selected:
        return selected

    top1_label = row["top1_label"]
    return [
        (model_idx, dataset_name)
        for model_idx, dataset_name, _ in intersect_map
        if dataset_name == top1_label
    ]


def row_probability(row: Dict[str, str], class_name: str) -> float:
    return float(row.get(f"prob_{class_name}", "0") or 0.0)


def save_heatmap_outputs(
    model: torch.nn.Module,
    image_records: List[Tuple[str, str, Path]],
    rows: List[Dict[str, str]],
    intersect_map: List[Tuple[int, str, str]],
    transform: transforms.Compose,
    device: torch.device,
    threshold: float,
    output_dir: Path,
    heatmap_threshold: float,
    min_component_area: int,
    max_components: int,
    target_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"
    bboxes_dir = output_dir / "bboxes"
    yolo_labels_dir = output_dir / "yolo_labels"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    bboxes_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    cam_generator = GradCAM(model, model.features.denseblock4)
    index_rows: List[Dict[str, str]] = []
    component_rows: List[Dict[str, str | int | float]] = []
    row_by_key = {f"{row['split']}/{row['sample_id']}": row for row in rows}

    model.eval()
    with torch.enable_grad():
        for image_index, (split, sample_id, image_path) in enumerate(image_records):
            image_key = f"{split}/{sample_id}"
            row = row_by_key[image_key]

            with Image.open(image_path) as img:
                tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
            original = denormalize_tensor(tensor[0])

            logits = model(tensor)
            _ = torch.sigmoid(logits).squeeze(0).detach().cpu()
            targets = select_heatmap_targets(target_mode, row, intersect_map)

            heatmap_files: List[str] = []
            bbox_values: List[str] = []
            yolo_bbox_values: List[str] = []
            yolo_label_lines: List[str] = []

            for model_idx, class_name in targets:
                cam = cam_generator(tensor, model_idx)
                local_class_idx = next(
                    local_idx
                    for local_idx, (candidate_model_idx, _, _) in enumerate(intersect_map)
                    if candidate_model_idx == model_idx
                )
                probability = row_probability(row, class_name)

                split_slug = slugify(split)
                stem_slug = slugify(Path(sample_id).stem)
                sample_slug = slugify(sample_id.replace("/", "_"))
                class_slug = slugify(class_name)
                file_stem = f"{image_index:04d}_{split_slug}_{sample_slug}_{class_slug}"

                mask = threshold_heatmap(cam, heatmap_threshold)
                components = connected_components(mask, min_component_area, max_components)

                overlay_path = overlays_dir / f"{file_stem}_heatmap.jpg"
                mask_path = masks_dir / f"{file_stem}_mask.png"
                bbox_path = bboxes_dir / f"{file_stem}_bbox.jpg"

                overlay_heatmap(original, cam).save(overlay_path, quality=95)
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)
                overlay_bounding_boxes(original, cam, components, class_name, probability).save(bbox_path, quality=95)

                relative_overlay = overlay_path.relative_to(output_dir).as_posix()
                relative_mask = mask_path.relative_to(output_dir).as_posix()
                relative_bbox = bbox_path.relative_to(output_dir).as_posix()
                heatmap_files.append(relative_bbox)

                largest = components[0] if components else {}
                image_width, image_height = original.size
                component_bboxes_xywh: List[str] = []
                component_bboxes_yolo: List[str] = []

                for component_rank, component in enumerate(components, start=1):
                    bbox_xywh = component_to_xywh(component)
                    x_center, y_center, yolo_width, yolo_height = component_to_yolo(
                        component,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    bbox_yolo = (
                        f"{local_class_idx} {x_center:.6f} {y_center:.6f} "
                        f"{yolo_width:.6f} {yolo_height:.6f}"
                    )
                    component_bboxes_xywh.append(bbox_xywh)
                    component_bboxes_yolo.append(bbox_yolo)
                    bbox_values.append(f"{class_name}#{component_rank}:{bbox_xywh}")
                    yolo_bbox_values.append(f"{class_name}#{component_rank}:{bbox_yolo}")
                    yolo_label_lines.append(bbox_yolo)

                    component_rows.append(
                        {
                            "split": split,
                            "image": image_path.name,
                            "sample_id": sample_id,
                            "class_id": local_class_idx,
                            "class_name": class_name,
                            "probability": f"{probability:.6f}",
                            "component_rank": component_rank,
                            "heatmap_threshold": f"{heatmap_threshold:.4f}",
                            "bbox_xywh": bbox_xywh,
                            "yolo_bbox": bbox_yolo,
                            "yolo_x_center": f"{x_center:.6f}",
                            "yolo_y_center": f"{y_center:.6f}",
                            "yolo_width": f"{yolo_width:.6f}",
                            "yolo_height": f"{yolo_height:.6f}",
                            **component,
                        }
                    )

                largest_bbox = component_to_xywh(largest) if largest else ""
                index_rows.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "sample_id": sample_id,
                        "class_id": str(local_class_idx),
                        "class_name": class_name,
                        "probability": f"{probability:.6f}",
                        "target_mode": target_mode,
                        "heatmap_threshold": f"{heatmap_threshold:.4f}",
                        "overlay_file": relative_overlay,
                        "mask_file": relative_mask,
                        "bbox_file": relative_bbox,
                        "component_count": str(len(components)),
                        "largest_component_bbox_xywh": largest_bbox,
                        "component_bboxes_xywh": "|".join(component_bboxes_xywh),
                        "component_bboxes_yolo": "|".join(component_bboxes_yolo),
                    }
                )

            yolo_split_dir = yolo_labels_dir / slugify(split)
            yolo_split_dir.mkdir(parents=True, exist_ok=True)
            yolo_label_path = yolo_split_dir / f"{slugify(sample_id.replace('/', '_'))}.txt"
            yolo_label_path.write_text("\n".join(yolo_label_lines) + ("\n" if yolo_label_lines else ""), encoding="utf-8")

            row["heatmap_files"] = "|".join(heatmap_files)
            row["bbox_xywh"] = "|".join(bbox_values)
            row["bbox_yolo"] = "|".join(yolo_bbox_values)
            row["yolo_label_file"] = yolo_label_path.relative_to(output_dir).as_posix()

    for row in rows:
        row.setdefault("heatmap_files", "")
        row.setdefault("bbox_xywh", "")
        row.setdefault("bbox_yolo", "")
        row.setdefault("yolo_label_file", "")

    with (output_dir / "heatmap_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "sample_id",
            "image",
            "class_id",
            "class_name",
            "probability",
            "target_mode",
            "heatmap_threshold",
            "overlay_file",
            "mask_file",
            "bbox_file",
            "component_count",
            "largest_component_bbox_xywh",
            "component_bboxes_xywh",
            "component_bboxes_yolo",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    with (output_dir / "heatmap_components.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "sample_id",
            "image",
            "class_id",
            "class_name",
            "probability",
            "component_rank",
            "heatmap_threshold",
            "bbox_xywh",
            "yolo_bbox",
            "yolo_x_center",
            "yolo_y_center",
            "yolo_width",
            "yolo_height",
            "x",
            "y",
            "width",
            "height",
            "area",
            "centroid_x",
            "centroid_y",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(component_rows)


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    image_records: List[Tuple[str, str, Path]],
    intersect_map: List[Tuple[int, str, str]],
    transform: transforms.Compose,
    device: torch.device,
    threshold: float,
    roi_mask_root: Path | None,
) -> Tuple[List[Dict[str, str]], Dict[str, List[int]]]:
    if not intersect_map:
        raise ValueError("No intersected classes found between dataset names and model classes.")

    results: List[Dict[str, str]] = []
    pred_vectors: Dict[str, List[int]] = {}
    model.eval()

    for split, sample_id, image_path in image_records:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

        roi_mask, roi_mask_file = load_roi_mask(roi_mask_root, split, image_path.name, (tensor.shape[-1], tensor.shape[-2]))
        tensor = apply_roi_to_tensor(tensor.squeeze(0), roi_mask, device).unsqueeze(0)

        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu()

        intersect_scores = [float(probs[model_idx].item()) for model_idx, _, _ in intersect_map]
        top_local_index = int(torch.tensor(intersect_scores).argmax().item())
        top_label = intersect_map[top_local_index][1]
        top_score = intersect_scores[top_local_index]

        positive_labels = [
            dataset_display_name
            for (model_idx, dataset_display_name, _) in intersect_map
            if float(probs[model_idx].item()) >= threshold
        ]

        pred_vector = [
            1 if float(probs[model_idx].item()) >= threshold else 0
            for (model_idx, _, _) in intersect_map
        ]

        row: Dict[str, str] = {
            "split": split,
            "sample_id": sample_id,
            "image": image_path.name,
            "top1_label": top_label,
            "top1_score": f"{top_score:.6f}",
            "predicted_labels": "|".join(positive_labels),
            "roi_mask_file": roi_mask_file,
        }

        for model_idx, dataset_display_name, _ in intersect_map:
            row[f"prob_{dataset_display_name}"] = f"{float(probs[model_idx]):.6f}"

        results.append(row)
        pred_vectors[f"{split}/{sample_id}"] = pred_vector

    return results, pred_vectors


def save_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["split", "sample_id", "image", "top1_label", "top1_score", "predicted_labels"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    data_root = resolve_data_root(args.data_root)
    data_yaml = resolve_data_yaml(data_root, args.data_yaml)
    dataset_class_names = read_class_names(data_yaml)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    checkpoint_obj = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(pick_state_dict(checkpoint_obj))
    state_dict = normalize_densenet_keys(state_dict)

    inferred_num_classes = infer_num_classes_from_state_dict(state_dict)
    if inferred_num_classes is None:
        raise ValueError("Could not infer classifier output size from checkpoint.")

    num_classes = inferred_num_classes
    model_class_names = default_model_class_names(num_classes)
    intersect_map = build_intersection_map(dataset_class_names, model_class_names)

    if not intersect_map:
        raise ValueError(
            "No class name intersection between dataset YAML and model classes. "
            "Update DATASET_NAME_TO_CANONICAL mapping in inference/inference.py."
        )

    kept_dataset_names = [dataset_name for _, dataset_name, _ in intersect_map]
    print(f"Using intersect classes ({len(kept_dataset_names)}): {kept_dataset_names}")

    model = build_model(num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: missing keys in checkpoint load ({len(missing)}): {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(
            f"Warning: unexpected keys in checkpoint load ({len(unexpected)}): "
            f"{unexpected[:6]}{'...' if len(unexpected) > 6 else ''}"
        )

    if args.test_dir is not None:
        splits = ["test"]
        image_records = collect_images_from_split(args.test_dir, "test")
        gt_root = args.test_dir.parent
    else:
        splits = args.splits
        image_records = collect_images(data_root, splits)
        gt_root = data_root

    transform = get_transform(args.image_size)

    rows, pred_vectors = run_inference(
        model=model,
        image_records=image_records,
        intersect_map=intersect_map,
        transform=transform,
        device=device,
        threshold=args.threshold,
        roi_mask_root=None if args.no_roi_mask else args.roi_mask_root,
    )

    gt_vectors = load_ground_truth_vectors(
        data_root=gt_root,
        splits=splits,
        intersect_map=intersect_map,
    )

    metric_class_names = [dataset_display_name for _, dataset_display_name, _ in intersect_map]
    metrics = compute_split_metrics(
        pred_vectors=pred_vectors,
        gt_vectors=gt_vectors,
        class_names=metric_class_names,
        splits=splits,
    )

    if args.generate_heatmaps:
        save_heatmap_outputs(
            model=model,
            image_records=image_records,
            rows=rows,
            intersect_map=intersect_map,
            transform=transform,
            device=device,
            threshold=args.threshold,
            output_dir=args.heatmap_output_dir,
            heatmap_threshold=args.heatmap_threshold,
            min_component_area=args.min_component_area,
            max_components=args.max_components,
            target_mode=args.heatmap_target_classes,
        )

    save_csv(rows, args.output_csv)
    save_metrics(metrics, args.metrics_json)

    print(f"Processed {len(rows)} images from splits {splits} under: {gt_root}")
    print(f"Saved predictions to: {args.output_csv}")
    print(f"Saved evaluation metrics to: {args.metrics_json}")
    if args.generate_heatmaps:
        print(f"Saved heatmaps and bounding-box overlays to: {args.heatmap_output_dir}")
    overall_metrics = metrics["overall"]
    print(
        "Metrics summary - "
        f"ExactMatch: {overall_metrics['exact_match_accuracy']}, "
        f"Micro-F1: {overall_metrics['micro']['f1']}, "
        f"Macro-F1: {overall_metrics['macro']['f1']}"
    )


if __name__ == "__main__":
    main()
