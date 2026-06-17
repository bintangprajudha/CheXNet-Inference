from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
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
    rows = []
    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        image = batch["image"].to(device)
        logits = model(image)
        prob = torch.sigmoid(logits)[0].detach().cpu().numpy()
        predicted = np.where(prob >= thresholds)[0]
        class_idx = int(predicted[0] if len(predicted) else prob.argmax())
        cam = cam_generator(image, class_idx)
        original = denormalize(batch["image"][0])
        overlay = overlay_heatmap(original, cam)
        stem = Path(batch["filename"][0]).stem
        out_path = args.output_dir / f"{idx:04d}_{stem}_{class_names[class_idx].replace(' ', '_')}.jpg"
        overlay.save(out_path)
        true_labels = [name for j, name in enumerate(class_names) if int(batch["labels"][0, j].item()) == 1]
        rows.append(f"{out_path.name},{batch['filename'][0]},{class_names[class_idx]},{prob[class_idx]:.6f},{'|'.join(true_labels)}")
    (args.output_dir / "gradcam_index.csv").write_text(
        "output_file,filename,predicted_class,probability,ground_truth_labels\n" + "\n".join(rows),
        encoding="utf-8",
    )
    print("Grad-CAM overlays saved. These heatmaps are weak localization/explainability outputs, not object detection boxes.")


if __name__ == "__main__":
    main()
