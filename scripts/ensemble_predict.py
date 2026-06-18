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
    parser = argparse.ArgumentParser(description="Predict test labels by averaging checkpoint logits.")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart" / "Test" / "Images")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ensemble_results.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta-hflip", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int | tuple[int, int], str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("class_names", CLASS_NAMES) != CLASS_NAMES:
        raise ValueError(f"Class order mismatch in {checkpoint_path}")
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
    all_sizes = {size for _, size, _ in models_info}
    all_modes = {mode for _, _, mode in models_info}
    base_size = max(all_sizes) if len(all_sizes) > 1 else all_sizes.pop()
    base_mode = list(all_modes)[0]
    dataset = TestImageDataset(args.test_dir, eval_transform(base_size, base_mode))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    from torchvision import transforms as T
    rows = []
    for images, names in loader:
        all_logits = []
        for model, img_size, resize_mode in models_info:
            if img_size != base_size or resize_mode != base_mode:
                from src.data import eval_transform as et
                transform = et(img_size, resize_mode)
                resized = torch.stack([transform(T.functional.to_pil_image(img.cpu())) for img in images]).to(device, non_blocking=True)
                all_logits.append(model(resized))
            else:
                all_logits.append(model(images.to(device, non_blocking=True)))
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
        rows.extend(zip(names, logits.argmax(dim=1).cpu().tolist()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
