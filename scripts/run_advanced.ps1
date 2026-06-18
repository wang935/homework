# =============================================================================
# DogHeart X-ray Classification - Advanced Strategies (Phase 2)
# Run after run_experiments.ps1 if accuracy still below 83%
# =============================================================================

$ErrorActionPreference = "Continue"
$VENV = "C:\Users\wang\Desktop\Homework\.venv"
$PYTHON = "$VENV\Scripts\python.exe"
$ROOT = "C:\Users\wang\Desktop\Homework"

# Read all trained model results
$best_acc = 0.0
$best_ckpt = ""
$eval_files = Get-ChildItem "$ROOT\outputs\*_eval.json"
foreach ($f in $eval_files) {
    $m = Get-Content $f.FullName | ConvertFrom-Json
    $acc = [double]$m.accuracy
    Write-Host ("  " + $f.Name + ": acc=" + $acc.ToString('P1'))
    if ($acc -gt $best_acc) {
        $best_acc = $acc
        $model_name = $f.Name -replace "_eval.json",""
        $best_ckpt = "$ROOT\models\" + $model_name + "\best.pt"
    }
}

Write-Host ("`nBest model accuracy: " + $best_acc.ToString('P1')) -ForegroundColor Cyan

if ($best_acc -ge 0.83) {
    Write-Host "Target 83% reached!" -ForegroundColor Green
    exit 0
}

# =============================================================================
# Strategy 1: Pseudo-Labeling on unlabeled test data
# =============================================================================
Write-Host "`n===== Strategy 1: Pseudo-Labeling =====" -ForegroundColor Cyan

# Generate predictions on test set
if (-not (Test-Path "$ROOT\outputs\results_pseudo.csv")) {
    & $PYTHON "$ROOT\scripts\predict.py" --checkpoint $best_ckpt --output "$ROOT\outputs\results_pseudo.csv" --batch-size 64
}

# Python script to copy pseudo-labeled images to training set
$GEN_SCRIPT = @"
import sys, csv, shutil
from pathlib import Path
ROOT = Path(r'$ROOT')
pseudo_csv = ROOT / 'outputs' / 'results_pseudo.csv'
test_img_dir = ROOT / 'Homework' / 'Dog_Heart' / 'Dog_Heart' / 'Test' / 'Images'
train_dir = ROOT / 'Homework' / 'Dog_Heart' / 'Dog_Heart' / 'Train'

label_names = {0: 'Large', 1: 'Normal', 2: 'Small'}
counts = {0:0, 1:0, 2:0}
with open(pseudo_csv, newline='') as f:
    for row in csv.reader(f):
        fname, label = row[0], int(row[1])
        src = test_img_dir / fname
        if src.exists():
            dst = train_dir / label_names[label] / fname
            if not dst.exists():
                shutil.copy2(src, dst)
                counts[label] += 1

print('Added pseudo-labeled: ' + str(counts))
print('Total: ' + str(sum(counts.values())))
"@
Write-Host "Generating pseudo-labeled training data..." -ForegroundColor Yellow
& $PYTHON -c $GEN_SCRIPT

# Train models with pseudo-labels
$PSEUDO_EXPS = @(
    @{arch="convnext_tiny";   img=288; aug="randaug"; mixup=0.2; epochs=80; bs=32; lr=2e-4; name="pseudo_convnext_tiny"},
    @{arch="convnext_small";  img=288; aug="randaug"; mixup=0.2; epochs=80; bs=24; lr=2e-4; name="pseudo_convnext_small"},
    @{arch="efficientnet_v2_s"; img=288; aug="randaug"; mixup=0.2; epochs=80; bs=32; lr=2e-4; name="pseudo_effv2s"}
)

foreach ($exp in $PSEUDO_EXPS) {
    $OUTPUT_DIR = "$ROOT\outputs\$($exp.name)"
    $MODEL_DIR = "$ROOT\models\$($exp.name)"

    if (Test-Path "$MODEL_DIR\best.pt") {
        Write-Host ($exp.name + " already exists, skipping") -ForegroundColor Green
        continue
    }

    Write-Host ("Training: " + $exp.name) -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\train.py" `
        --architecture $($exp.arch) `
        --pretrained `
        --image-size $($exp.img) `
        --aug-strength $($exp.aug) `
        --mixup-alpha $($exp.mixup) `
        --epochs $($exp.epochs) `
        --batch-size $($exp.bs) `
        --lr $($exp.lr) `
        --weight-decay 0.05 `
        --label-smoothing 0.05 `
        --patience 25 `
        --num-workers 4 `
        --output-dir $OUTPUT_DIR `
        --model-dir $MODEL_DIR `
        --channels-last `
        --sampler weighted `
        --loss-weight balanced 2>&1 | Tee-Object -FilePath "$ROOT\outputs\$($exp.name)_train.log"

    if (Test-Path "$MODEL_DIR\best.pt") {
        & $PYTHON "$ROOT\scripts\evaluate.py" --checkpoint "$MODEL_DIR\best.pt" --output "$ROOT\outputs\$($exp.name)_eval.json" --batch-size 64
        $m = Get-Content "$ROOT\outputs\$($exp.name)_eval.json" | ConvertFrom-Json
        Write-Host ("  " + $exp.name + " acc: " + ([double]$m.accuracy).ToString('P1')) -ForegroundColor Green
    }
}

# =============================================================================
# Strategy 2: Final ensemble of all available models
# =============================================================================
Write-Host "`n===== Strategy 2: Final Ensemble =====" -ForegroundColor Cyan

$ALL_BEST_CKPTS = @()
$all_pt = Get-ChildItem "$ROOT\models" -Recurse -Filter "best.pt"
foreach ($pt in $all_pt) {
    $ALL_BEST_CKPTS += $pt.FullName
}

Write-Host ("Found " + $ALL_BEST_CKPTS.Count + " checkpoints") -ForegroundColor Yellow

if ($ALL_BEST_CKPTS.Count -ge 2) {
    & $PYTHON "$ROOT\scripts\ensemble_evaluate.py" --checkpoints $ALL_BEST_CKPTS --output "$ROOT\outputs\ensemble_final_metrics.json" --tta-hflip --batch-size 64

    if (Test-Path "$ROOT\outputs\ensemble_final_metrics.json") {
        $ens = Get-Content "$ROOT\outputs\ensemble_final_metrics.json" | ConvertFrom-Json
        Write-Host ("Final ensemble acc: " + ([double]$ens.accuracy).ToString('P1')) -ForegroundColor Magenta

        & $PYTHON "$ROOT\scripts\ensemble_predict.py" --checkpoints $ALL_BEST_CKPTS --output "$ROOT\outputs\results.csv" --tta-hflip --batch-size 64
        & $PYTHON "$ROOT\scripts\check_submission.py" --csv "$ROOT\outputs\results.csv"
    }
} else {
    Write-Host "Not enough models for ensemble" -ForegroundColor Yellow
}

Write-Host "`nAll experiments complete!" -ForegroundColor Cyan
