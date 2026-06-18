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
    parser = argparse.ArgumentParser(description="Evaluate an ensemble by averaging checkpoint logits.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ensemble_valid_metrics.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta-hflip", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, int | tuple[int, int], str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        num_classes=len(CLASS_NAMES),
        architecture=ckpt.get("architecture", "custom"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    raw_image_size = ckpt.get("image_size", 224)
    image_size = tuple(raw_image_size) if isinstance(raw_image_size, list) else int(raw_image_size)
    return model, image_size, ckpt.get("resize_mode", "stretch")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    models_info = [load_model(path, device) for path in args.checkpoints]
    # Use the largest image size for the base dataset; models with different sizes get per-model transforms.
    all_sizes = {size for _, size, _ in models_info}
    all_modes = {mode for _, _, mode in models_info}
    base_size = max(all_sizes) if len(all_sizes) > 1 else all_sizes.pop()
    base_mode = list(all_modes)[0]
    dataset = make_imagefolder(args.data_root / "Valid", eval_transform(base_size, base_mode))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    from torchvision import transforms as T
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_true = []
    all_pred = []
    for images, targets in loader:
        targets = targets.to(device, non_blocking=True)
        # Per-model forward passes with individual image sizes
        all_logits = []
        for model, img_size, resize_mode in models_info:
            if img_size != base_size or resize_mode != base_mode:
                # Resize to this model's required resolution
                from src.data import eval_transform as et
                transform = et(img_size, resize_mode)
                resized = torch.stack([transform(T.functional.to_pil_image(img.cpu())) for img in images]).to(device, non_blocking=True)
                all_logits.append(model(resized))
            else:
                img_input = images.to(device, non_blocking=True)
                all_logits.append(model(img_input))
        logits = torch.stack(all_logits).mean(dim=0)
        if args.tta_hflip:
            flipped_imgs = torch.flip(images.to(device, non_blocking=True), dims=[3])
            flip_logits = []
            for model, img_size, resize_mode in models_info:
                if img_size != base_size or resize_mode != base_mode:
                    from src.data import eval_transform as et
                    transform = et(img_size, resize_mode)
                    resized = torch.stack([transform(T.functional.to_pil_image(img.cpu())) for img in flipped_imgs]).to(device, non_blocking=True)
                    flip_logits.append(model(resized))
                else:
                    flip_logits.append(model(flipped_imgs))
            logits = (logits + torch.stack(flip_logits).mean(dim=0)) / 2
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        all_true.append(targets.cpu())
        all_pred.append(logits.argmax(dim=1).cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    metrics = classification_metrics(confusion_matrix(y_true, y_pred, len(CLASS_NAMES)), CLASS_NAMES)
    metrics["loss"] = total_loss / len(dataset)
    metrics["checkpoints"] = [str(path) for path in args.checkpoints]
    metrics["tta_hflip"] = args.tta_hflip
    save_metrics(metrics, args.output)
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} loss={metrics['loss']:.4f}")
    print("confusion_matrix=", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
