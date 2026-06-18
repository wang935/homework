from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, TestImageDataset, eval_transform, make_imagefolder
from src.device import resolve_device
from src.metrics import classification_metrics, confusion_matrix, save_metrics
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search class logit biases on validation split and optionally predict test CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart" / "Test" / "Images")
    parser.add_argument("--output-metrics", type=Path, default=ROOT / "outputs" / "calibrated_metrics.json")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--span", type=float, default=1.2)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def checkpoint_image_size(ckpt: dict) -> int | tuple[int, int]:
    raw = ckpt.get("image_size", 224)
    return tuple(raw) if isinstance(raw, list) else int(raw)


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, int | tuple[int, int]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        num_classes=len(CLASS_NAMES),
        architecture=ckpt.get("architecture", "custom"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, checkpoint_image_size(ckpt)


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits = []
    targets = []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        logits.append(model(images).cpu())
        targets.append(y.cpu())
    return torch.cat(logits), torch.cat(targets)


def metric_score(metrics: dict) -> float:
    return metrics["accuracy"] + 0.35 * metrics["macro_f1"]


def search_bias(logits: torch.Tensor, targets: torch.Tensor, span: float, step: float) -> tuple[torch.Tensor, dict]:
    values = torch.arange(-span, span + step / 2, step)
    best_bias = torch.zeros(len(CLASS_NAMES))
    best_metrics = None
    best_score = -1.0
    # One bias can be fixed to zero because adding the same constant to all logits changes nothing.
    for b0, b1 in product(values, repeat=2):
        bias = torch.tensor([float(b0), float(b1), 0.0])
        pred = (logits + bias).argmax(dim=1)
        metrics = classification_metrics(confusion_matrix(targets, pred, len(CLASS_NAMES)), CLASS_NAMES)
        score = metric_score(metrics)
        if score > best_score:
            best_score = score
            best_bias = bias
            best_metrics = metrics
    assert best_metrics is not None
    best_metrics["bias"] = best_bias.tolist()
    best_metrics["objective"] = best_score
    return best_bias, best_metrics


@torch.no_grad()
def predict_csv(
    model: torch.nn.Module,
    image_size: int | tuple[int, int],
    test_dir: Path,
    output_csv: Path,
    bias: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> None:
    dataset = TestImageDataset(test_dir, eval_transform(image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    rows = []
    bias = bias.to(device)
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        pred = (model(images) + bias).argmax(dim=1).cpu().tolist()
        rows.extend(zip(names, pred))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {len(rows)} predictions to {output_csv}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, image_size = load_model(args.checkpoint, device)
    valid_ds = make_imagefolder(args.data_root / "Valid", eval_transform(image_size))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    logits, targets = collect_logits(model, valid_loader, device)
    bias, metrics = search_bias(logits, targets, args.span, args.step)
    save_metrics(metrics, args.output_metrics)
    print(json.dumps(metrics, indent=2))
    if args.output_csv is not None:
        predict_csv(model, image_size, args.test_dir, args.output_csv, bias, device, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
