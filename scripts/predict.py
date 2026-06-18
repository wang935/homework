from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, TestImageDataset, eval_transform
from src.device import resolve_device
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict test labels and write no-header CSV.")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart" / "Test" / "Images")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "results.csv")
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
    ckpt_classes = ckpt.get("class_names", CLASS_NAMES)
    if ckpt_classes != CLASS_NAMES:
        raise ValueError(f"Checkpoint class order {ckpt_classes} does not match expected {CLASS_NAMES}")

    device = resolve_device(args.device)
    print(f"using device: {device}")
    dataset = TestImageDataset(args.test_dir, eval_transform(image_size, resize_mode))
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if args.tta_hflip:
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        labels = logits.argmax(dim=1).cpu().tolist()
        rows.extend(zip(names, labels))

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
