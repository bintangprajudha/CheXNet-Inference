# Inference Pipeline

This document explains the runtime flow of `inference/inference.py`.

## 1. Inputs

The script starts from these inputs:

- `model.pth.tar` checkpoint
- `Splitted dataset/data.yaml` or `<data-root>/data.yaml`
- dataset splits: `train`, `valid`, `test` by default
- lung ROI masks from `lung segmentation/output_masks/<split>/`

The script also supports a legacy `--test-dir` mode, but the default workflow uses the split folders under `Splitted dataset/`.

## 2. Dataset Discovery

The loader scans every requested split and collects image files from class folders.

For each sample, it creates a stable `sample_id` from the path relative to the split folder. This avoids collisions when the same filename appears in multiple class folders.

Example:

```text
train/atelektasis/CXR015_001.jpg
train/kavitas/CXR015_001.jpg
```

These are treated as separate samples.

## 3. Model Setup

The script:

1. reads class names from `data.yaml`
2. loads the CheXNet/DenseNet121 checkpoint
3. normalizes legacy DenseNet key formats
4. infers the output class count from the checkpoint
5. builds the model head to match the checkpoint

## 4. ROI Masking For Inference

Before the forward pass, the script:

1. loads the lung mask for the sample
2. converts the mask to a binary ROI
3. applies the ROI to the input tensor
4. runs the model on the masked tensor

This affects prediction scores and metrics.

If a mask is missing, the script falls back to the original image for that sample.

## 5. Prediction Generation

The model outputs logits for all classes.

The script then:

- applies sigmoid to get probabilities
- picks the top-1 class
- selects all classes above the threshold for multi-label predictions
- stores per-class probabilities in the CSV

Each output row includes:

- `split`
- `sample_id`
- `image`
- `top1_label`
- `top1_score`
- `predicted_labels`
- `roi_mask_file`
- `prob_<class>`

## 6. Metrics

Metrics are computed from the predicted vectors and the labels reconstructed from the split/class folders.

The JSON output contains:

- `overall`
- `by_split`

The reported metrics include exact match accuracy, micro precision/recall/F1, macro precision/recall/F1, and per-class scores.

## 7. Heatmap And Bounding-Box Generation

When `--generate-heatmaps` is enabled, the script runs a Grad-CAM pass for the selected target classes.

Target class selection follows:

- `predicted` default: all predicted classes above threshold, fallback to top-1
- `top1`: only the top-1 class
- `all`: every intersected class

Important behavior:

- ROI is used for inference
- Grad-CAM is computed from the original image input
- heatmaps use a jet-style colormap
- bounding boxes are derived from thresholded Grad-CAM components

The heatmap output is written to:

```text
inference/heatmaps/
```

## 8. Heatmap Outputs

The heatmap directory contains:

- `overlays/`
- `masks/`
- `bboxes/`
- `yolo_labels/`
- `heatmap_index.csv`
- `heatmap_components.csv`

The CSV outputs are:

- `heatmap_index.csv`: one row per image/class heatmap
- `heatmap_components.csv`: one row per connected component

The YOLO label files use:

```text
class_id x_center y_center width height
```

## 9. Execution Summary

The end-to-end flow is:

```text
discover splits -> load checkpoint -> apply ROI to input -> run inference
-> compute metrics -> optionally run Grad-CAM -> save CSV/JSON/heatmaps
```

## 10. Useful Commands

Basic inference:

```bash
python inference/inference.py --cpu
```

Heatmap generation:

```bash
python inference/inference.py --cpu --generate-heatmaps
```

Test split only:

```bash
python inference/inference.py --cpu --splits test
```
