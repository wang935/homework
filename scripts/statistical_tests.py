from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, eval_transform, make_imagefolder
from src.device import resolve_device
from src.metrics import classification_metrics, confusion_matrix
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired validation statistics for paper reporting.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart")
    parser.add_argument("--final", type=Path, default=ROOT / "models" / "final_model.pt")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models" / "resnet18_224" / "best.pt")
    parser.add_argument("--no-mixup", type=Path, default=ROOT / "models" / "convnext_tiny_224_seed0" / "best.pt")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def predict(checkpoint_path: Path, data_root: Path, device: torch.device) -> tuple[list[int], list[int]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_size = ckpt.get("image_size", 224)
    image_size = tuple(raw_size) if isinstance(raw_size, list) else int(raw_size)
    resize_mode = ckpt.get("resize_mode", "stretch")
    dataset = make_imagefolder(data_root / "Valid", eval_transform(image_size, resize_mode))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model = build_model(
        num_classes=len(CLASS_NAMES),
        architecture=ckpt.get("architecture", "custom"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        logits = model(images.to(device))
        y_true.extend(targets.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    matrix = confusion_matrix(torch.tensor(y_true), torch.tensor(y_pred), len(CLASS_NAMES))
    return float(classification_metrics(matrix, CLASS_NAMES)["macro_f1"])


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mcnemar_exact(y_true: list[int], pred_a: list[int], pred_b: list[int]) -> dict[str, float | int]:
    a_correct_b_wrong = sum((a == y) and (b != y) for y, a, b in zip(y_true, pred_a, pred_b))
    a_wrong_b_correct = sum((a != y) and (b == y) for y, a, b in zip(y_true, pred_a, pred_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(a_correct_b_wrong, a_wrong_b_correct)
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant))
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": discordant,
        "exact_p": p_value,
    }


def bootstrap_macro_f1_ci(
    y_true: list[int],
    y_pred: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    n = len(y_true)
    values = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(macro_f1([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    values.sort()
    return {
        "iterations": iterations,
        "mean": sum(values) / len(values),
        "low_95": values[int(0.025 * iterations)],
        "high_95": values[int(0.975 * iterations) - 1],
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    labels, final_pred = predict(args.final, args.data_root, device)
    labels_baseline, baseline_pred = predict(args.baseline, args.data_root, device)
    labels_no_mixup, no_mixup_pred = predict(args.no_mixup, args.data_root, device)
    if labels != labels_baseline or labels != labels_no_mixup:
        raise RuntimeError("Validation sample order differs across model evaluations.")

    result = {
        "n_valid": len(labels),
        "final": {"accuracy": accuracy(labels, final_pred), "macro_f1": macro_f1(labels, final_pred)},
        "baseline": {"accuracy": accuracy(labels, baseline_pred), "macro_f1": macro_f1(labels, baseline_pred)},
        "convnext_no_mixup": {"accuracy": accuracy(labels, no_mixup_pred), "macro_f1": macro_f1(labels, no_mixup_pred)},
        "mcnemar_final_vs_baseline": mcnemar_exact(labels, final_pred, baseline_pred),
        "mcnemar_final_vs_convnext_no_mixup": mcnemar_exact(labels, final_pred, no_mixup_pred),
        "final_macro_f1_bootstrap_ci": bootstrap_macro_f1_ci(
            labels,
            final_pred,
            iterations=args.bootstrap_iters,
            seed=args.seed,
        ),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
