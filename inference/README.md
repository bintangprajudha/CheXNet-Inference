# Inference Workflow

This folder contains the CheXNet/DenseNet121 inference workflow for the RSHS chest X-ray dataset.

`inference.py` loads `model.pth.tar`, runs multi-label inference on the selected dataset splits, writes prediction and metric files, and can optionally generate Grad-CAM heatmaps with weak-localization bounding boxes.

## Basic Inference

Run from the repository root:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset"
```

By default, the script processes:

```text
train valid test
```

To process only specific splits:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --splits test
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --splits train valid
```

The script automatically resolves the nested dataset layout if the dataset is stored as:

```text
Splitted dataset/Splitted dataset
```

## Default Outputs

Prediction CSV:

```text
inference/inference_test_results.csv
```

Metrics JSON:

```text
inference/inference_test_metrics.json
```

The CSV includes one row per image with:

```text
split, image, top1_label, top1_score, predicted_labels, prob_<class>
```

The metrics JSON includes:

```text
overall
by_split
```

## Heatmaps And Bounding Boxes

To generate Grad-CAM heatmaps and weak-localization bounding boxes:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --generate-heatmaps
```

By default, heatmaps and boxes are restricted to lung ROI masks from:

```text
lung segmentation/output_masks/
```

Disable ROI masking with:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --generate-heatmaps --no-roi-mask
```

Heatmap outputs are written to:

```text
inference/heatmaps/
```

Output structure:

```text
inference/heatmaps/
  overlays/
  masks/
  bboxes/
  yolo_labels/
  heatmap_index.csv
  heatmap_components.csv
```

## Multi-Label Behavior

This is a multi-label classification workflow. For heatmaps, the default target mode is:

```text
predicted
```

That means one image can produce heatmaps and bounding boxes for multiple predicted labels. If no label passes the sigmoid threshold, the script falls back to the top-1 label.

Other target modes:

```bash
--heatmap-target-classes top1
--heatmap-target-classes all
```

## YOLO-Style BBox Output

The script writes YOLO-style normalized label files under:

```text
inference/heatmaps/yolo_labels/<split>/
```

Each line uses:

```text
class_id x_center y_center width height
```

The inference CSV also includes:

```text
bbox_xywh
bbox_yolo
yolo_label_file
```

Important: these boxes are generated from Grad-CAM heatmaps and lung ROI masks. They are weak-localization/explainability boxes, not final object-detection ground truth.

## Useful Options

```bash
--threshold 0.5
--image-size 224
--cpu
--heatmap-threshold 0.5
--min-component-area 25
--max-components 3
--roi-mask-root "lung segmentation/output_masks"
```

## Validation

Compile check:

```bash
python -m py_compile inference/inference.py
```

Runtime smoke test on test split:

```bash
.venv\Scripts\python.exe inference\inference.py --cpu --splits test
```

Runtime smoke test with heatmaps:

```bash
.venv\Scripts\python.exe inference\inference.py --cpu --splits test --generate-heatmaps
```
