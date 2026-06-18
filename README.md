# DogHeart X-ray Classification Homework

This repository contains the cleaned PyTorch submission for the DogHeart
three-class X-ray classification homework. It was refactored from the original
notebook into reusable modules and command-line scripts.

## Final Result

| Item | Value |
|---|---:|
| Final architecture | ConvNeXt Tiny, ImageNet pretrained |
| Input size | 224 x 224 |
| Validation accuracy | 77.5% |
| Validation macro-F1 | 79.4% |
| Hidden test accuracy | 77.25% |
| Class order | `Large=0`, `Normal=1`, `Small=2` |

The final checkpoint is tracked with Git LFS at `models/final_model.pt`.

## Repository Contents

- `src/`: dataset, transforms, metrics, device selection, and model factory.
- `scripts/train.py`: training entry point.
- `scripts/evaluate.py`: validation-set evaluation for a checkpoint.
- `scripts/predict.py`: hidden-test CSV generation.
- `scripts/check_submission.py`: CSV format validation.
- `outputs/results.csv`: final 400-image hidden-test prediction CSV.
- `outputs/convnext_tiny_final/valid_metrics.json`: final validation metrics.
- `report/main.tex` and `report/main_final.pdf`: final report source and PDF.

The raw DogHeart dataset, local virtual environment, intermediate checkpoints,
and LaTeX build products are intentionally excluded from Git.

## Environment

The local run used Python 3.12, PyTorch 2.11.0+cu128, and torchvision
0.26.0+cu128.

```powershell
python -m pip install -r requirements.txt
```

For the same CUDA build used locally:

```powershell
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

## Data Layout

Place the DogHeart data under:

```text
Homework/Dog_Heart/Dog_Heart/
  Train/
    Large/
    Normal/
    Small/
  Valid/
    Large/
    Normal/
    Small/
  Test/
    Images/
```

The code enforces the class order `Large`, `Normal`, `Small` so the numeric
labels in the output CSV stay aligned with the submitted checkpoint.

## Reproduce Evaluation

```powershell
python scripts/evaluate.py --checkpoint models/final_model.pt --output outputs/convnext_tiny_final/valid_metrics_recheck.json --batch-size 64
```

Expected validation metrics:

```text
accuracy=0.7750
macro_f1=0.7938
confusion_matrix=[[55, 21, 0], [15, 68, 8], [0, 1, 32]]
```

## Generate Submission CSV

```powershell
python scripts/predict.py --checkpoint models/final_model.pt --output outputs/results.csv --batch-size 64
python scripts/check_submission.py --csv outputs/results.csv
```

`outputs/results.csv` is a no-header CSV with rows formatted as:

```text
filename.png,label
```

## Final Training Recipe

The submitted model metadata records:

```text
architecture=convnext_tiny
pretrained=True
image_size=224
resize_mode=stretch
epochs=80
best_epoch=23
batch_size=48
lr=2e-4
weight_decay=0.1
aug_strength=strong
mixup_alpha=0.4
label_smoothing=0.1
seed=20260525
```

The primary training script supports additional exploratory settings:

```powershell
python scripts/train.py --architecture convnext_tiny --pretrained --image-size 224 --aug-strength strong --mixup-alpha 0.4 --epochs 80 --batch-size 48 --lr 2e-4 --weight-decay 0.1 --model-dir models --output-dir outputs/convnext_tiny_final
```
