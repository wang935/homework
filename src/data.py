from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CLASS_NAMES = ["Large", "Normal", "Small"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ImageSize = int | tuple[int, int]


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


class TestImageDataset(Dataset):
    """Test dataset returning image tensor and original file name."""

    def __init__(self, image_dir: str | Path, transform: Callable | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
            key=natural_key,
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


class ResizePad:
    def __init__(self, size: ImageSize, fill: int = 0) -> None:
        self.size = _resize_size(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        target_h, target_w = self.size
        image = ImageOps.contain(image, (target_w, target_h), method=Image.Resampling.BILINEAR)
        canvas = Image.new(image.mode, (target_w, target_h), color=self.fill)
        left = (target_w - image.width) // 2
        top = (target_h - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


def _resize_size(image_size: ImageSize) -> tuple[int, int]:
    return image_size if isinstance(image_size, tuple) else (image_size, image_size)


def _resize_transform(image_size: ImageSize, resize_mode: str) -> transforms.Resize | ResizePad:
    if resize_mode == "stretch":
        return transforms.Resize(_resize_size(image_size))
    if resize_mode == "pad":
        return ResizePad(image_size)
    raise ValueError(f"Unknown resize mode: {resize_mode}")


def train_transform(image_size: ImageSize = 224, strength: str = "standard", resize_mode: str = "stretch") -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    resize = _resize_transform(image_size, resize_mode)
    if strength == "none":
        return eval_transform(image_size, resize_mode)
    if strength == "light":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if strength == "randaug":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if strength == "strong":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
                transforms.RandAugment(num_ops=2, magnitude=7),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if strength != "standard":
        raise ValueError(f"Unknown augmentation strength: {strength}")
    return transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            normalize,
        ]
    )


def eval_transform(image_size: ImageSize = 224, resize_mode: str = "stretch") -> transforms.Compose:
    return transforms.Compose(
        [
            _resize_transform(image_size, resize_mode),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def make_imagefolder(root: str | Path, transform: Callable | None = None) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(str(root), transform=transform)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Unexpected class order {dataset.classes}. Expected {CLASS_NAMES}; "
            "the output CSV depends on this fixed mapping."
        )
    return dataset
