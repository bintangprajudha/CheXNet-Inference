# Transfer Learning Pipeline

This folder contains the CheXNet/DenseNet121 transfer learning workflow for the RSHS chest X-ray dataset. It is separate from the existing root inference workflow.

## Dataset Formats

The loader supports multi-label CSV files named `train_labels.csv`, `val_labels.csv`, and `test_labels.csv`, where the first column is the filename and remaining columns are binary class labels.

It also supports folder-per-class splits such as `train/kavitas`, `train/infiltrat`, and `train/normal`. If the same filename appears in multiple class folders, the loader creates a multi-hot label vector. The `normal` folder is treated as all-zero labels.

If the dataset is nested as `Splitted dataset/Splitted dataset`, the scripts resolve that automatically. `valid` is accepted as an alias for `val`.

## Training

```bash
python transfer_learning/train_transfer_learning.py \
  --data-root "Splitted dataset" \
  --data-yaml "Splitted dataset/data.yaml" \
  --checkpoint model.pth.tar \
  --output-dir outputs/transfer_learning \
  --batch-size 8 \
  --image-size 224 \
  --stage-a-epochs 10 \
  --stage-b-epochs 30 \
  --device auto
```

The model uses `BCEWithLogitsLoss`, class `pos_weight`, sigmoid only during evaluation, and validation-tuned per-class thresholds. Stage A freezes the DenseNet backbone and trains the classifier. Stage B unfreezes `denseblock4`, `norm5`, and the classifier.

Use `--dry-run` to verify data loading, checkpoint loading, and one train/validation batch without running full training.

## Evaluation

```bash
python transfer_learning/evaluate_transfer_model.py \
  --data-root "Splitted dataset" \
  --data-yaml "Splitted dataset/data.yaml" \
  --checkpoint outputs/transfer_learning/final_model.pth \
  --thresholds outputs/transfer_learning/thresholds.json \
  --split test \
  --output-dir outputs/transfer_learning
```

## Grad-CAM

```bash
python transfer_learning/gradcam.py \
  --data-root "Splitted dataset" \
  --data-yaml "Splitted dataset/data.yaml" \
  --checkpoint outputs/transfer_learning/final_model.pth \
  --thresholds outputs/transfer_learning/thresholds.json \
  --output-dir outputs/transfer_learning/gradcam \
  --num-samples 20
```

Grad-CAM heatmaps are weak localization and explainability outputs. They are not final bounding-box detections.

## Detection Conversion

For YOLO-style detection data with `images/<split>` and `labels/<split>`, generate multi-label CSV files with:

```bash
python transfer_learning/datasets/convert_detection_to_multilabel.py \
  --data-root "Splitted dataset" \
  --data-yaml "Splitted dataset/data.yaml"
```

## Outputs

Training writes to `outputs/transfer_learning/`:

- `best_model_stage_a.pth`
- `best_model_stage_b.pth`
- `final_model.pth`
- `training_log.csv`
- `class_distribution.csv`
- `class_distribution.json`
- `thresholds.json`
- `validation_metrics.json`
- `test_metrics.json`
- `test_predictions.csv`
- `config_used.yaml`

Grad-CAM writes overlays and `gradcam_index.csv` under `outputs/transfer_learning/gradcam/`.
