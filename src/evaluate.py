"""Evaluation and threshold tuning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, multilabel_confusion_matrix, precision_score


def _precision_f1_single(y_true_col, y_prob_col, threshold):
    y_pred = (y_prob_col >= threshold).astype(int)
    precision = precision_score(y_true_col, y_pred, zero_division=0)
    f1 = f1_score(y_true_col, y_pred, zero_division=0)
    return precision, f1


def tune_thresholds_constrained(
    y_true,
    y_prob,
    label_names,
    min_precision: float = 0.35,
    step: float = 0.02,
    per_label_min_precision: dict[str, float] | None = None,
):
    num_labels = y_true.shape[1]
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    grid = np.arange(0.05, 0.96, step)

    if per_label_min_precision is None:
        per_label_min_precision = {name: min_precision for name in label_names}

    rows = []

    for idx, label_name in enumerate(label_names):
        y_true_col = y_true[:, idx].astype(int)
        y_prob_col = y_prob[:, idx]
        required_p = float(per_label_min_precision.get(label_name, min_precision))

        if y_true_col.sum() == 0:
            thresholds[idx] = 0.5
            rows.append(
                {
                    "Label": label_name,
                    "Threshold": 0.5,
                    "Precision": 0.0,
                    "F1": 0.0,
                    "Constraint": f"P>={required_p:.2f}",
                    "Status": "No positive samples",
                }
            )
            continue

        best_any = (0.5, -1.0, 0.0)
        best_constrained = (0.5, -1.0, 0.0)

        for t in grid:
            p, f1 = _precision_f1_single(y_true_col, y_prob_col, t)
            if f1 > best_any[1]:
                best_any = (float(t), float(f1), float(p))
            if p >= required_p and f1 > best_constrained[1]:
                best_constrained = (float(t), float(f1), float(p))

        if best_constrained[1] >= 0:
            chosen = best_constrained
            status = "Constraint satisfied"
        else:
            chosen = best_any
            status = "Fallback to best F1"

        thresholds[idx] = chosen[0]
        rows.append(
            {
                "Label": label_name,
                "Threshold": chosen[0],
                "Precision": chosen[2],
                "F1": chosen[1],
                "Constraint": f"P>={required_p:.2f}",
                "Status": status,
            }
        )

    return thresholds, pd.DataFrame(rows)


def evaluate_with_thresholds(y_true, y_prob, thresholds, label_names):
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true.ravel(), y_pred.ravel()),
        "F1-micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    report_dict = classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)
    conf_mtx = multilabel_confusion_matrix(y_true, y_pred).tolist()

    return metrics, report_dict, conf_mtx
