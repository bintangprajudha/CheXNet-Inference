# Transfer Learning Instructions for CheXNet-Inference

## Project Context

This repository is used for CheXNet/DenseNet121 inference on the RSHS chest X-ray dataset. The current goal is to extend the repository with a complete transfer learning pipeline so the existing CheXNet checkpoint can be fine-tuned on the RSHS dataset.

The existing inference workflow must not be removed or broken. Add new training, evaluation, dataset, metrics, and Grad-CAM utilities while preserving backward compatibility with the existing inference script.

Repository:

```text
https://github.com/bintangprajudha/CheXNet-Inference/tree/main
```

Main checkpoint:

```text
model.pth.tar
```

Expected dataset root:

```text
Splitted dataset/
```

## Main Objective

Implement a transfer learning pipeline for CheXNet/DenseNet121 using the RSHS dataset.

The pipeline must support:

1. Loading a pretrained CheXNet/DenseNet121 checkpoint.
2. Replacing the final classifier according to the number of RSHS classes.
3. Training using multi-label classification.
4. Fine-tuning the final DenseNet block.
5. Evaluating on the test set after training.
6. Saving metrics, predictions, thresholds, training logs, and trained model files.
7. Generating Grad-CAM heatmaps for explainability.

## Important Task Definition

Use multi-label classification, not single-label softmax classification.

Rationale:

The RSHS dataset may originate from detection or bounding-box annotations. A single image may contain more than one abnormality. Therefore, each image should be represented as a multi-hot label vector.

Use:

```python
torch.nn.BCEWithLogitsLoss
```

Do not use:

```python
torch.nn.CrossEntropyLoss
```

During training, the model should output logits. Do not apply sigmoid inside the model during training because `BCEWithLogitsLoss` expects raw logits.

Apply sigmoid only during evaluation and inference.

## Dataset Formats to Support

The implementation should be flexible and support at least the following formats.

### Format A: CSV Multi-label Format

Expected structure:

```text
Splitted dataset/
├── train/
│   └── images/
├── val/
│   └── images/
├── test/
│   └── images/
├── train_labels.csv
├── val_labels.csv
├── test_labels.csv
└── data.yaml
```

Example CSV:

```csv
filename,kavitas,infiltrat,limfadenopati,tuberkuloma,bronkiektasis,pneumothorax,efusi_pleura,atelektasis
img001.jpg,0,1,0,0,0,0,1,0
img002.jpg,1,0,0,0,0,0,0,0
```

The first column must contain the image filename. The remaining columns are class labels with binary values.

### Format B: Folder-per-class Format

Expected structure:

```text
Splitted dataset/
├── train/
│   ├── atelektasis/
│   ├── efusi pleura/
│   └── pneumothorax/
├── val/
└── test/
```

If the same image appears in multiple class folders, create a multi-hot image-level label.

### Format C: Detection/YOLO Format

If YOLO-style detection labels are found, add a conversion utility:

```text
datasets/convert_detection_to_multilabel.py
```

The script should read bounding-box labels, extract class IDs or class names, and generate:

```text
train_labels.csv
val_labels.csv
test_labels.csv
```

If the label format is unclear, raise a clear error message and document the expected format.

## Preprocessing

Use preprocessing compatible with DenseNet/CheXNet.

For all splits:

```text
Resize image to 224x224
Convert to RGB
ToTensor
Normalize with ImageNet statistics:
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

For training, use only light augmentation:

```text
RandomRotation up to 5 degrees
Small brightness/contrast ColorJitter
```

Avoid aggressive augmentation that may damage medical image semantics, such as large rotations, excessive cropping, or unrealistic transformations.

## Model Requirements

Create:

```text
models/chexnet.py
```

Required functions:

```python
build_chexnet(num_classes, checkpoint_path=None, freeze_backbone=False)
load_chexnet_checkpoint(model, checkpoint_path, skip_classifier=True)
```

Implementation details:

1. Use `torchvision.models.densenet121`.
2. Load checkpoint from `model.pth.tar` if provided.
3. Remove `module.` prefix if the checkpoint came from DataParallel.
4. Skip the old classifier if its shape does not match the new number of classes.
5. Replace the classifier with:

```python
torch.nn.Linear(in_features, num_classes)
```

6. Load the remaining checkpoint weights with `strict=False`.
7. Print or log missing and unexpected keys clearly.

## Training Strategy

Implement two training stages.

### Stage A: Classifier-only Training

Freeze all DenseNet feature extractor layers.

Train only the new classifier.

Default configuration:

```text
epochs: 10
learning_rate: 1e-3
optimizer: AdamW
loss: BCEWithLogitsLoss
```

### Stage B: Partial Fine-tuning

Unfreeze:

```text
model.features.denseblock4
model.features.norm5
model.classifier
```

Keep other layers frozen.

Default configuration:

```text
epochs: 30
backbone_learning_rate: 1e-5
classifier_learning_rate: 1e-4
optimizer: AdamW with parameter groups
scheduler: ReduceLROnPlateau or CosineAnnealingLR
early_stopping_patience: 5 to 7
```

Use validation macro-F1 or validation loss to select the best model.

## Class Imbalance Handling

Compute class distribution from the train set.

For each class:

```text
positive_count
negative_count
pos_weight = negative_count / positive_count
```

Use epsilon to avoid division by zero.

Use:

```python
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Save class distribution to:

```text
outputs/class_distribution.csv
outputs/class_distribution.json
```

## Threshold Tuning

Implement validation-based threshold tuning.

Procedure:

1. Run the trained model on the validation set.
2. Collect sigmoid probabilities.
3. For each class, search thresholds from 0.10 to 0.90 with step 0.05.
4. Select the threshold that maximizes per-class F1-score.
5. Save the thresholds to:

```text
outputs/thresholds.json
```

Use these validation-derived thresholds for final test evaluation.

## Metrics

Implement multi-label metrics.

Required metrics:

```text
Exact match accuracy
Micro precision
Micro recall
Micro F1
Macro precision
Macro recall
Macro F1
Per-class precision
Per-class recall
Per-class F1
ROC-AUC per class if possible
PR-AUC per class if possible
```

If ROC-AUC or PR-AUC cannot be computed because a class has only one label type in the split, do not crash. Return null for that class and log a warning.

## Required Output Files

All outputs must be saved under:

```text
outputs/
```

Required files:

```text
outputs/
├── best_model_stage_a.pth
├── best_model_stage_b.pth
├── final_model.pth
├── training_log.csv
├── class_distribution.csv
├── class_distribution.json
├── thresholds.json
├── validation_metrics.json
├── test_metrics.json
├── test_predictions.csv
└── config_used.yaml
```

`test_predictions.csv` must contain at least:

```text
filename
path
ground_truth_labels
predicted_labels
prob_<class_name>
pred_<class_name>
true_<class_name>
```

## Grad-CAM / Heatmap

Add:

```text
gradcam.py
```

The script should generate Grad-CAM heatmaps for the fine-tuned model.

Target layer suggestion:

```python
model.features.denseblock4
```

Output directory:

```text
outputs/gradcam/
```

For each generated result, include or save:

```text
original image
predicted class
probability
ground truth label
heatmap overlay
```

Important note:

Grad-CAM is explainability or weak localization. It is not a true object detector. Do not claim that Grad-CAM is final bounding-box detection. If ground-truth bbox data exists, it may be overlaid only for visual comparison.

## Files to Add

Prefer this structure:

```text
configs/
└── transfer_learning.yaml

datasets/
├── rshs_dataset.py
└── convert_detection_to_multilabel.py

models/
└── chexnet.py

utils/
├── metrics.py
├── thresholds.py
├── seed.py
└── io.py

train_transfer_learning.py
evaluate_transfer_model.py
gradcam.py
```

Update:

```text
README.md
requirements.txt
.gitignore
```

Do not delete the existing inference script.

## Command Line Interface

`train_transfer_learning.py` must support:

```bash
python train_transfer_learning.py \
  --data-root "Splitted dataset" \
  --data-yaml "Splitted dataset/data.yaml" \
  --checkpoint model.pth.tar \
  --output-dir outputs \
  --batch-size 8 \
  --image-size 224 \
  --stage-a-epochs 10 \
  --stage-b-epochs 30 \
  --device auto
```

`evaluate_transfer_model.py` must support:

```bash
python evaluate_transfer_model.py \
  --data-root "Splitted dataset" \
  --data-yaml "Splitted dataset/data.yaml" \
  --checkpoint outputs/final_model.pth \
  --thresholds outputs/thresholds.json \
  --split test \
  --output-dir outputs
```

`gradcam.py` must support:

```bash
python gradcam.py \
  --data-root "Splitted dataset" \
  --checkpoint outputs/final_model.pth \
  --thresholds outputs/thresholds.json \
  --output-dir outputs/gradcam \
  --num-samples 20
```

## Device Handling

Implement automatic device selection:

```text
Use CUDA if available.
Otherwise use CPU.
Support --cpu to force CPU.
Do not hard-code CUDA.
```

## Reproducibility

Add seed support.

Default seed:

```text
42
```

Seed:

```text
random
numpy
torch
torch.cuda if available
```

## README Update

Update `README.md` with:

```text
Transfer learning overview
Dataset formats
Training command
Evaluation command
Grad-CAM command
Output files
Multi-label classification explanation
Grad-CAM limitation note
```

## Git Ignore

Update `.gitignore` to avoid committing large generated files:

```text
outputs/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.pth
*.pt
*.ckpt
```

Keep `model.pth.tar` only if it already exists and is intentionally part of the repository.

## Validation Commands

After implementation, run:

```bash
python -m py_compile train_transfer_learning.py
python -m py_compile evaluate_transfer_model.py
python -m py_compile gradcam.py
```

If possible, add a dry-run mode:

```bash
python train_transfer_learning.py --dry-run ...
```

## Acceptance Criteria

The task is complete when:

1. Existing inference functionality is not broken.
2. `train_transfer_learning.py` exists and compiles.
3. `evaluate_transfer_model.py` exists and compiles.
4. `gradcam.py` exists and compiles.
5. DenseNet121/CheXNet can be created with output size equal to the number of RSHS classes.
6. `model.pth.tar` can be loaded while skipping the old classifier if the shape mismatches.
7. Stage A classifier-only training is implemented.
8. Stage B partial fine-tuning is implemented.
9. Loss uses `BCEWithLogitsLoss`.
10. Multi-label metrics are implemented.
11. Threshold tuning is implemented.
12. Output CSV and JSON files are saved.
13. README explains how to use the new pipeline.
14. Generated files and large outputs are ignored by git.
15. A final implementation summary is provided.

## Final Response Required from Codex

After completing the task, summarize:

1. Files added.
2. Files modified.
3. How to run transfer learning.
4. How to run evaluation.
5. How to run Grad-CAM.
6. What outputs will be generated.
7. What validation commands were run.
8. Any dataset assumptions or issues that still need user confirmation.
