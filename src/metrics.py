from __future__ import annotations

import json
from pathlib import Path

import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true, pred in zip(y_true.view(-1), y_pred.view(-1)):
        matrix[int(true), int(pred)] += 1
    return matrix


def classification_metrics(matrix: torch.Tensor, class_names: list[str]) -> dict:
    total = matrix.sum().item()
    correct = matrix.diag().sum().item()
    per_class = {}
    f1_values = []
    for idx, name in enumerate(class_names):
        tp = matrix[idx, idx].item()
        fp = matrix[:, idx].sum().item() - tp
        fn = matrix[idx, :].sum().item() - tp
        support = matrix[idx, :].sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
