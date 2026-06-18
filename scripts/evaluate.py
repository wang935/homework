from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, eval_transform, make_imagefolder
from src.device import resolve_device
from src.metrics import classification_metrics, confusion_matrix, save_metrics
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the validation split.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "valid_metrics.json")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--resize-mode", choices=["checkpoint", "stretch", "pad"], default="checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta-hflip", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_image_size = ckpt.get("image_size", 224)
    image_size = tuple(raw_image_size) if isinstance(raw_image_size, list) else int(raw_image_size)
    resize_mode = ckpt.get("resize_mode", "stretch")
    if args.resize_mode != "checkpoint":
        resize_mode = args.resize_mode
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise ValueError("--image-height and --image-width must be set together")
        image_size = (args.image_height, args.image_width)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    dataset = make_imagefolder(args.data_root / "Valid", eval_transform(image_size, resize_mode))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(
        num_classes=len(CLASS_NAMES),
        architecture=ckpt.get("architecture", "custom"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_true = []
    all_pred = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if args.tta_hflip:
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        all_true.append(targets.cpu())
        all_pred.append(logits.argmax(dim=1).cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    matrix = confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    metrics = classification_metrics(matrix, CLASS_NAMES)
    metrics["loss"] = total_loss / len(dataset)
    save_metrics(metrics, args.output)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} loss={metrics['loss']:.4f}")
    print("confusion_matrix=", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
