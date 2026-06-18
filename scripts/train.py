from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, ResizePad, eval_transform, make_imagefolder, train_transform
from src.device import resolve_device
from src.metrics import classification_metrics, confusion_matrix, save_metrics
from src.model import SUPPORTED_ARCHITECTURES, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom CNN for dog heart X-ray classification.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--resize-mode", choices=["stretch", "pad"], default="stretch")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default="custom",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--sampler", choices=["weighted", "shuffle"], default="weighted")
    parser.add_argument("--loss-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--aug-strength", choices=["standard", "light", "none", "randaug", "strong"], default="standard")
    parser.add_argument("--mixup-alpha", type=float, default=None, help="MixUp alpha parameter (e.g. 0.2). Disabled if None.")
    parser.add_argument("--task", choices=["ce", "ordinal"], default="ce")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay rate (e.g. 0.999). Disabled if None.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cache-ram", action="store_true", help="Preload resized tensors into RAM before training.")
    return parser.parse_args()


def resolve_image_size(args: argparse.Namespace) -> int | tuple[int, int]:
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise ValueError("--image-height and --image-width must be set together")
        return (args.image_height, args.image_width)
    return args.image_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class CachedTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, targets: list[int], transform: nn.Module | None = None) -> None:
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def _resize_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    return image_size if isinstance(image_size, tuple) else (image_size, image_size)


def tensor_transform(strength: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if strength == "none":
        return transforms.Compose([normalize])
    if strength == "light":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                normalize,
            ]
        )
    if strength != "standard":
        raise ValueError(f"Unknown augmentation strength: {strength}")
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            normalize,
        ]
    )


def cache_imagefolder(
    root: Path,
    image_size: int | tuple[int, int],
    strength: str | None,
    resize_mode: str,
) -> CachedTensorDataset:
    resize = transforms.Resize(_resize_size(image_size)) if resize_mode == "stretch" else ResizePad(image_size)
    decode_transform = transforms.Compose([resize, transforms.ToTensor()])
    dataset = make_imagefolder(root, decode_transform)
    images = []
    targets = []
    for image, target in dataset:
        images.append(image.contiguous())
        targets.append(int(target))
    stacked = torch.stack(images)
    transform = tensor_transform(strength) if strength is not None else tensor_transform("none")
    return CachedTensorDataset(stacked, targets, transform)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[float]]:
    image_size = resolve_image_size(args)
    if args.cache_ram:
        train_ds = cache_imagefolder(args.data_root / "Train", image_size, args.aug_strength, args.resize_mode)
        valid_ds = cache_imagefolder(args.data_root / "Valid", image_size, None, args.resize_mode)
    else:
        train_ds = make_imagefolder(args.data_root / "Train", train_transform(image_size, args.aug_strength, args.resize_mode))
        valid_ds = make_imagefolder(args.data_root / "Valid", eval_transform(image_size, args.resize_mode))

    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(CLASS_NAMES)).float()
    class_weights = (counts.sum() / (len(CLASS_NAMES) * counts)).tolist()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    if args.sampler == "weighted":
        sample_weights = torch.tensor([class_weights[target] for target in train_ds.targets], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=False, **kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kwargs)
    return train_loader, valid_loader, class_weights


def ordinal_targets(targets: torch.Tensor) -> torch.Tensor:
    severity = 2 - targets
    return torch.stack([(severity >= 1).float(), (severity >= 2).float()], dim=1)


def logits_to_predictions(logits: torch.Tensor, task: str) -> torch.Tensor:
    if task == "ce":
        return logits.argmax(dim=1)
    probs = torch.sigmoid(logits)
    preds = torch.full((logits.size(0),), 2, dtype=torch.long, device=logits.device)
    preds[probs[:, 0] >= 0.5] = 1
    preds[probs[:, 1] >= 0.5] = 0
    return preds


def mixup_data(images: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation: convex combination of pairs."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)  # keep closer to original
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    task: str = "ce",
    mixup_alpha: float | None = None,
    epoch: int = 1,
    ema_model: torch.optim.swa_utils.AveragedModel | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        # MixUp (training only)
        mixup_lam = 1.0
        if is_train and mixup_alpha is not None and mixup_alpha > 0:
            images, targets_a, targets_b, mixup_lam = mixup_data(images, targets, mixup_alpha)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                if is_train and mixup_lam < 1.0 and task == "ce":
                    loss = mixup_lam * criterion(logits, targets_a) + (1.0 - mixup_lam) * criterion(logits, targets_b)
                else:
                    loss_targets = ordinal_targets(targets) if task == "ordinal" else targets
                    loss = criterion(logits, loss_targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                # Update EMA model after each optimizer step
                if ema_model is not None:
                    ema_model.update_parameters(model)

        losses.append(loss.detach().cpu() * images.size(0))
        # For mixup, use original target for metric tracking
        all_targets.append(targets.detach().cpu() if mixup_lam >= 1.0 else targets_a.detach().cpu())
        all_preds.append(logits_to_predictions(logits.detach(), task).cpu())

    avg_loss = torch.stack(losses).sum().item() / len(loader.dataset)
    return avg_loss, torch.cat(all_targets), torch.cat(all_preds)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"using device: {device}")
    train_loader, valid_loader, class_weights = make_loaders(args)
    num_outputs = 2 if args.task == "ordinal" else len(CLASS_NAMES)
    model = build_model(
        num_classes=num_outputs,
        architecture=args.architecture,
        pretrained=args.pretrained,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    if args.task == "ordinal":
        train_targets = torch.tensor(train_loader.dataset.targets, dtype=torch.long)
        ord_targets = ordinal_targets(train_targets)
        positives = ord_targets.sum(dim=0).clamp_min(1)
        negatives = ord_targets.size(0) - positives
        criterion = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).to(device))
    else:
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32, device=device)
            if args.loss_weight == "balanced"
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    # EMA (Exponential Moving Average)
    if args.ema_decay is not None:
        def ema_avg(averaged, current, _):
            return args.ema_decay * averaged + (1.0 - args.ema_decay) * current
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg, use_buffers=True)
        print(f"EMA enabled with decay={args.ema_decay}")
    else:
        ema_model = None

    history_path = args.output_dir / "history.csv"
    best_path = args.model_dir / "best.pt"
    last_path = args.model_dir / "last.pt"
    best_acc = -1.0
    epochs_without_improvement = 0

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "valid_loss", "valid_acc", "valid_macro_f1"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_true, train_pred = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, not args.no_amp, args.task, args.mixup_alpha, epoch, ema_model
            )
            # Validate using EMA model if available, otherwise use base model
            eval_model = ema_model if ema_model is not None else model
            valid_loss, valid_true, valid_pred = run_epoch(
                eval_model, valid_loader, criterion, device, optimizer=None, scaler=None, use_amp=not args.no_amp, task=args.task
            )
            # Also validate base model for comparison when EMA is active
            if ema_model is not None:
                _, base_true, base_pred = run_epoch(
                    model, valid_loader, criterion, device, optimizer=None, scaler=None, use_amp=not args.no_amp, task=args.task
                )
                base_acc = (base_true == base_pred).float().mean().item()
            scheduler.step()

            train_acc = (train_true == train_pred).float().mean().item()
            valid_matrix = confusion_matrix(valid_true, valid_pred, len(CLASS_NAMES))
            valid_metrics = classification_metrics(valid_matrix, CLASS_NAMES)
            valid_acc = valid_metrics["accuracy"]
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "valid_macro_f1": valid_metrics["macro_f1"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} "
                f"macro_f1={valid_metrics['macro_f1']:.4f}"
                + (f" base_acc={base_acc:.4f}" if ema_model is not None else "")
            )

            # Use EMA accuracy for best model tracking if available
            effective_acc = valid_acc
            effective_state = model.state_dict()
            if ema_model is not None:
                effective_acc = max(valid_acc, base_acc)
                # If EMA is better, save EMA weights
                if valid_acc >= base_acc:
                    effective_state = ema_model.module.state_dict()

            checkpoint = {
                "epoch": epoch,
                "model_state": effective_state,
                "class_names": CLASS_NAMES,
                "class_to_idx": {name: idx for idx, name in enumerate(CLASS_NAMES)},
                "image_size": resolve_image_size(args),
                "architecture": args.architecture,
                "pretrained": args.pretrained,
                "task": args.task,
                "valid_metrics": valid_metrics,
                "resize_mode": args.resize_mode,
                "ema_decay": args.ema_decay,
                "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir), "model_dir": str(args.model_dir)},
            }
            torch.save(checkpoint, last_path)

            if effective_acc > best_acc:
                best_acc = effective_acc
                epochs_without_improvement = 0
                # Re-save with the best version (EMA or base)
                if ema_model is not None and valid_acc >= base_acc:
                    checkpoint["model_state"] = ema_model.module.state_dict()
                torch.save(checkpoint, best_path)
                save_metrics(valid_metrics, args.output_dir / "valid_metrics.json")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs; best valid_acc={best_acc:.4f}")
                break

    summary = {
        "best_valid_accuracy": best_acc,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "class_names": CLASS_NAMES,
        "class_weights": class_weights,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
