from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from transfer_learning.datasets.rshs_dataset import RSHSMultiLabelDataset, create_transforms, discover_classes
from transfer_learning.models.chexnet import build_chexnet
from transfer_learning.train_transfer_learning import choose_device, collate
from transfer_learning.utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a fine-tuned CheXNet model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transfer_learning/gradcam"))
    parser.add_argument("--split", choices=["train", "val", "valid", "test"], default="test")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-threshold", type=float, default=0.5, help="Threshold for binarizing Grad-CAM heatmap before CCA.")
    parser.add_argument("--min-component-area", type=int, default=25, help="Minimum CCA component area in pixels.")
    parser.add_argument("--max-components", type=int, default=3, help="Maximum CCA components to draw per image.")
    parser.add_argument(
        "--target-classes",
        choices=["predicted", "positive", "all"],
        default="predicted",
        help="Classes to generate Grad-CAM for: predicted threshold-positive classes, ground-truth positive classes, or all classes.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
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
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam


def denormalize(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = torch.clamp(tensor.cpu() * std + mean, 0, 1)
    return transforms.ToPILImage()(image)


def overlay_heatmap(image: Image.Image, cam: np.ndarray) -> Image.Image:
    image = image.convert("RGBA")
    heat = Image.fromarray(np.uint8(cam * 255), mode="L").resize(image.size)
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(heat.point(lambda x: int(x * 0.45)))
    return Image.alpha_composite(image, red).convert("RGB")


def threshold_heatmap(cam: np.ndarray, threshold: float) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--heatmap-threshold must be between 0.0 and 1.0")
    return (cam >= threshold).astype(np.uint8)


def connected_components(mask: np.ndarray, min_area: int, max_components: int) -> list[dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label_id in range(1, num_labels):
        x, y, width, height, area = stats[label_id]
        if int(area) < min_area:
            continue
        cx, cy = centroids[label_id]
        components.append(
            {
                "label_id": int(label_id),
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
            }
        )
    components.sort(key=lambda item: item["area"], reverse=True)
    return components[:max_components]


def overlay_cca(image: Image.Image, cam: np.ndarray, mask: np.ndarray, components: list[dict]) -> Image.Image:
    overlay = np.array(overlay_heatmap(image, cam).convert("RGB"))
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    for component in components:
        x, y, width, height = component["x"], component["y"], component["width"], component["height"]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
    return Image.fromarray(overlay)


def select_target_classes(target_mode: str, probabilities: np.ndarray, thresholds: np.ndarray, labels: torch.Tensor) -> list[int]:
    if target_mode == "all":
        return list(range(len(probabilities)))
    if target_mode == "positive":
        positives = [idx for idx, value in enumerate(labels.cpu().numpy().astype(int)) if value == 1]
        return positives or [int(probabilities.argmax())]
    predicted = np.where(probabilities >= thresholds)[0].tolist()
    return predicted or [int(probabilities.argmax())]


def main() -> None:
    args = parse_args()
    device = choose_device(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names") or discover_classes(args.data_root, args.data_yaml)
    model = build_chexnet(len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.eval()
    threshold_info = load_json(args.thresholds)
    thresholds = np.array([threshold_info[name]["threshold"] if isinstance(threshold_info[name], dict) else threshold_info[name] for name in class_names])
    dataset = RSHSMultiLabelDataset(args.data_root, args.split, class_names, args.data_yaml, create_transforms(args.image_size, False))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    cam_generator = GradCAM(model, model.features.denseblock4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    cca_dir = args.output_dir / "cca"
    mask_dir.mkdir(parents=True, exist_ok=True)
    cca_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    component_rows = []
    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        image = batch["image"].to(device)
        logits = model(image)
        prob = torch.sigmoid(logits)[0].detach().cpu().numpy()
        original = denormalize(batch["image"][0])
        stem = Path(batch["filename"][0]).stem
        true_labels = [name for j, name in enumerate(class_names) if int(batch["labels"][0, j].item()) == 1]
        target_indices = select_target_classes(args.target_classes, prob, thresholds, batch["labels"][0])
        for class_idx in target_indices:
            cam = cam_generator(image, class_idx)
            overlay = overlay_heatmap(original, cam)
            mask = threshold_heatmap(cam, args.heatmap_threshold)
            components = connected_components(mask, args.min_component_area, args.max_components)
            cca_overlay = overlay_cca(original, cam, mask, components)
            class_slug = class_names[class_idx].replace(" ", "_")
            out_path = args.output_dir / f"{idx:04d}_{stem}_{class_slug}.jpg"
            mask_path = mask_dir / f"{idx:04d}_{stem}_{class_slug}_mask_t{args.heatmap_threshold:.2f}.png"
            cca_path = cca_dir / f"{idx:04d}_{stem}_{class_slug}_cca.jpg"
            overlay.save(out_path)
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)
            cca_overlay.save(cca_path)
            largest = components[0] if components else {}
            rows.append(
                {
                    "output_file": out_path.name,
                    "mask_file": str(mask_path.relative_to(args.output_dir)),
                    "cca_file": str(cca_path.relative_to(args.output_dir)),
                    "filename": batch["filename"][0],
                    "predicted_class": class_names[class_idx],
                    "probability": f"{prob[class_idx]:.6f}",
                    "class_threshold": f"{thresholds[class_idx]:.4f}",
                    "heatmap_threshold": f"{args.heatmap_threshold:.4f}",
                    "target_mode": args.target_classes,
                    "component_count": len(components),
                    "largest_component_area": largest.get("area", ""),
                    "largest_component_bbox_xywh": (
                        f"{largest.get('x')},{largest.get('y')},{largest.get('width')},{largest.get('height')}" if largest else ""
                    ),
                    "ground_truth_labels": "|".join(true_labels),
                }
            )
            for component_rank, component in enumerate(components, start=1):
                component_rows.append(
                    {
                        "filename": batch["filename"][0],
                        "predicted_class": class_names[class_idx],
                        "component_rank": component_rank,
                        "heatmap_threshold": f"{args.heatmap_threshold:.4f}",
                        **component,
                    }
                )
    with (args.output_dir / "gradcam_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with (args.output_dir / "gradcam_components.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "filename",
            "predicted_class",
            "component_rank",
            "heatmap_threshold",
            "label_id",
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
    print(
        "Grad-CAM overlays, threshold masks, and CCA overlays saved. "
        "These heatmaps/components are explainability outputs, not final object detection boxes."
    )


if __name__ == "__main__":
    main()
