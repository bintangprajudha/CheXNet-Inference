from __future__ import annotations

import numpy as np


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.05,
) -> dict:
    thresholds = np.round(np.arange(start, stop + 1e-9, step), 4)
    selected = {}
    for idx, name in enumerate(class_names):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in thresholds:
            pred = (y_prob[:, idx] >= threshold).astype(int)
            true = y_true[:, idx].astype(int)
            tp = int(((true == 1) & (pred == 1)).sum())
            fp = int(((true == 0) & (pred == 1)).sum())
            fn = int(((true == 1) & (pred == 0)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        selected[name] = {"threshold": best_threshold, "validation_f1": float(best_f1)}
    return selected
