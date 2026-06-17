from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from transfer_learning.datasets.rshs_dataset import RSHSMultiLabelDataset, create_transforms, discover_classes
from transfer_learning.models.chexnet import build_chexnet, set_stage_a_trainable, set_stage_b_trainable
from transfer_learning.utils.io import save_json
from transfer_learning.utils.metrics import multilabel_metrics, predictions_to_rows, save_predictions_csv
from transfer_learning.utils.seed import seed_everything
from transfer_learning.utils.thresholds import tune_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning for CheXNet/DenseNet121 on RSHS multi-label data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth.tar"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transfer_learning"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stage-a-epochs", type=int, default=10)
    parser.add_argument("--stage-b-epochs", type=int, default=30)
    parser.add_argument("--stage-a-lr", type=float, default=1e-3)
    parser.add_argument("--stage-b-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--stage-b-classifier-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and run one train and validation batch.")
    return parser.parse_args()


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.cpu or args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "filename": [b["filename"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_count += images.size(0)
        if train and getattr(loader, "dry_run", False):
            break
    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    labels, probs, filenames, paths = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        filenames.extend(batch["filename"])
        paths.extend(batch["path"])
    return np.concatenate(labels), np.concatenate(probs), filenames, paths


def make_optimizer_stage_b(model, args):
    return torch.optim.AdamW(
        [
            {"params": model.features.denseblock4.parameters(), "lr": args.stage_b_backbone_lr},
            {"params": model.features.norm5.parameters(), "lr": args.stage_b_backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.stage_b_classifier_lr},
        ]
    )


def save_checkpoint(model, path: Path, class_names: list[str], thresholds: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_names": class_names, "thresholds": thresholds or {}}, path)


def save_config(config: dict, path: Path) -> None:
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    try:
        import yaml

        path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gradcam").mkdir(parents=True, exist_ok=True)
    class_names = discover_classes(args.data_root, args.data_yaml)
    train_ds = RSHSMultiLabelDataset(args.data_root, "train", class_names, args.data_yaml, create_transforms(args.image_size, True))
    val_ds = RSHSMultiLabelDataset(args.data_root, "val", class_names, args.data_yaml, create_transforms(args.image_size, False))
    test_ds = RSHSMultiLabelDataset(args.data_root, "test", class_names, args.data_yaml, create_transforms(args.image_size, False))
    train_ds.save_class_distribution(args.output_dir)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate),
    }
    pos = torch.tensor([row["positive_count"] for row in train_ds.class_distribution()], dtype=torch.float32)
    neg = torch.tensor([row["negative_count"] for row in train_ds.class_distribution()], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(neg / torch.clamp(pos, min=1)).to(device))
    model = build_chexnet(len(class_names), args.checkpoint if args.checkpoint.exists() else None, freeze_backbone=True).to(device)
    config = vars(args).copy()
    config.update({"class_names": class_names, "device_used": str(device)})
    save_config(config, args.output_dir / "config_used.yaml")
    if args.dry_run:
        loaders["train"].dry_run = True
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.stage_a_lr)
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        print(f"Dry run complete. train_loss={train_loss:.4f} val_loss={val_loss:.4f} classes={class_names}")
        return
    log_rows = []
    best_a = float("inf")
    set_stage_a_trainable(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.stage_a_lr)
    for epoch in range(1, args.stage_a_epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        if val_loss < best_a:
            best_a = val_loss
            save_checkpoint(model, args.output_dir / "best_model_stage_a.pth", class_names)
        log_rows.append({"stage": "A", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Stage A epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    best_b = float("inf")
    stale = 0
    set_stage_b_trainable(model)
    optimizer = make_optimizer_stage_b(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    for epoch in range(1, args.stage_b_epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step(val_loss)
        if val_loss < best_b:
            best_b = val_loss
            stale = 0
            save_checkpoint(model, args.output_dir / "best_model_stage_b.pth", class_names)
        else:
            stale += 1
        log_rows.append({"stage": "B", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Stage B epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if stale >= args.patience:
            break
    with (args.output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    best_path = args.output_dir / "best_model_stage_b.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    y_val, p_val, _, _ = collect_predictions(model, loaders["val"], device)
    threshold_info = tune_thresholds(y_val, p_val, class_names)
    thresholds = np.array([threshold_info[name]["threshold"] for name in class_names])
    save_json(threshold_info, args.output_dir / "thresholds.json")
    save_json(multilabel_metrics(y_val, p_val, thresholds, class_names), args.output_dir / "validation_metrics.json")
    y_test, p_test, filenames, paths = collect_predictions(model, loaders["test"], device)
    save_json(multilabel_metrics(y_test, p_test, thresholds, class_names), args.output_dir / "test_metrics.json")
    save_predictions_csv(predictions_to_rows(filenames, paths, y_test, p_test, thresholds, class_names), args.output_dir / "test_predictions.csv")
    save_checkpoint(model, args.output_dir / "final_model.pth", class_names, threshold_info)
    print(f"Training complete. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
