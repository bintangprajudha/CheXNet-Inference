from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    return tp, fp, fn


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thresholds).astype(int)
    tp, fp, fn = _binary_counts(y_true, y_pred)
    precision, recall, f1 = _prf(tp, fp, fn)
    micro_p, micro_r, micro_f1 = _prf(np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()]))
    metrics = {
        "exact_match_accuracy": float((y_true == y_pred).all(axis=1).mean()) if len(y_true) else 0.0,
        "micro_precision": float(micro_p[0]),
        "micro_recall": float(micro_r[0]),
        "micro_f1": float(micro_f1[0]),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {},
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = roc_auc_score = None
    for i, name in enumerate(class_names):
        roc_auc = pr_auc = None
        if len(np.unique(y_true[:, i])) > 1:
            if roc_auc_score:
                roc_auc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            if average_precision_score:
                pr_auc = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    return metrics


def predictions_to_rows(
    filenames: list[str],
    paths: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    y_pred = (y_prob >= thresholds).astype(int)
    rows = []
    for idx, filename in enumerate(filenames):
        true_names = [name for j, name in enumerate(class_names) if y_true[idx, j] == 1]
        pred_names = [name for j, name in enumerate(class_names) if y_pred[idx, j] == 1]
        row = {
            "filename": filename,
            "path": paths[idx],
            "ground_truth_labels": "|".join(true_names),
            "predicted_labels": "|".join(pred_names),
        }
        for j, name in enumerate(class_names):
            safe = name.replace(" ", "_")
            row[f"prob_{safe}"] = float(y_prob[idx, j])
            row[f"pred_{safe}"] = int(y_pred[idx, j])
            row[f"true_{safe}"] = int(y_true[idx, j])
        rows.append(row)
    return rows


def save_predictions_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
