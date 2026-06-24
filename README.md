## Step-by-step transfer learning

1. **Load dataset:**
   - Load dataset with name `Splitted dataset`. Refer to Miro for dataset, with the name dataset RSHS.

2. **Train model**
   - The chexnet model is in `model.pth.tar`
   - Continue training as usual 

## Transfer learning pipeline

The transfer learning implementation is kept in [`transfer_learning/README.md`](transfer_learning/README.md). It adds a separate CheXNet/DenseNet121 multi-label training, evaluation, threshold tuning, and Grad-CAM workflow.

## Inference

The inference script is kept in [`inference/inference.py`](inference/inference.py).

Run from the repository root:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset"
```

By default, inference loads `train`, `valid`, and `test`. To process specific splits only:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --splits test
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --splits train valid
```

Default inference outputs are written to:

```text
inference/inference_test_results.csv
inference/inference_test_metrics.json
```

To also generate Grad-CAM heatmaps and weak-localization bounding boxes for inferred images:

```bash
python inference/inference.py --checkpoint model.pth.tar --data-root "Splitted dataset" --generate-heatmaps
```

By default, inference loads `train`, `valid`, and `test`, and applies lung ROI masks before the model forward pass from:

```text
lung segmentation/output_masks/
```

Use `--no-roi-mask` to disable ROI masking for inference. Grad-CAM heatmaps and bounding boxes are generated from the original image, using the inferred classes as targets.

Heatmap outputs are written to:

```text
inference/heatmaps/
```

The heatmap output includes overlay images, mask images, bounding-box overlay images, CSV metadata, and YOLO-style label files:

```text
inference/heatmaps/
inference/heatmaps/yolo_labels/
```

YOLO bbox values are normalized as:

```text
class_id x_center y_center width height
```

## Dataset Utilities

Dataset conversion and bounding-box heatmap utilities are kept in `transfer_learning/datasets/`.
Generated dataset heatmaps are written under `outputs/`.
