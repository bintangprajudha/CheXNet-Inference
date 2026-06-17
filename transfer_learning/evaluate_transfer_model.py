from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from transfer_learning.datasets.rshs_dataset import RSHSMultiLabelDataset, create_transforms, discover_classes
from transfer_learning.models.chexnet import build_chexnet
from transfer_learning.train_transfer_learning import collate, collect_predictions, choose_device
from transfer_learning.utils.io import load_json, save_json
from transfer_learning.utils.metrics import multilabel_metrics, predictions_to_rows, save_predictions_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CheXNet multi-label model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "valid", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transfer_learning"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names") or discover_classes(args.data_root, args.data_yaml)
    model = build_chexnet(len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    dataset = RSHSMultiLabelDataset(args.data_root, args.split, class_names, args.data_yaml, create_transforms(args.image_size, False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    threshold_info = load_json(args.thresholds)
    thresholds = np.array([threshold_info[name]["threshold"] if isinstance(threshold_info[name], dict) else threshold_info[name] for name in class_names])
    y_true, y_prob, filenames, paths = collect_predictions(model, loader, device)
    metrics = multilabel_metrics(y_true, y_prob, thresholds, class_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output_dir / f"{args.split}_metrics.json")
    save_predictions_csv(
        predictions_to_rows(filenames, paths, y_true, y_prob, thresholds, class_names),
        args.output_dir / f"{args.split}_predictions.csv",
    )
    print(f"Evaluation complete. Metrics saved to {args.output_dir / f'{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
