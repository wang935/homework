# =============================================================================
# DogHeart X-ray Classification - 83% Target Training Script
# Strategies: EMA, multi-architecture, pseudo-labeling, ensemble
# =============================================================================

$ErrorActionPreference = "Continue"
$VENV = "C:\Users\wang\Desktop\Homework\.venv"
$PYTHON = "$VENV\Scripts\python.exe"
$ROOT = "C:\Users\wang\Desktop\Homework"

# =============================================================================
# Phase 1: Train strong single models with optimal settings + EMA
# =============================================================================
Write-Host "===== Phase 1: Train Strong Models (224px, EMA, strong aug) =====" -ForegroundColor Cyan

$PHASE1_MODELS = @()

# Optimal hyperparams from best run: strong aug, mixup 0.4, wd 0.1, ls 0.1, lr 2e-4, EMA 0.999
$PH1_EXPS = @(
    @{arch="convnext_tiny";   bs=48; name="to83_convnext_tiny"}
    @{arch="convnext_small";  bs=32; name="to83_convnext_small"}
    @{arch="swin_t";          bs=32; name="to83_swin_t"}
    @{arch="efficientnet_v2_s"; bs=32; name="to83_effv2s"}
    @{arch="resnet50";        bs=48; name="to83_resnet50"}
)

foreach ($exp in $PH1_EXPS) {
    $OUTPUT_DIR = "$ROOT\outputs\$($exp.name)"
    $MODEL_DIR = "$ROOT\models\$($exp.name)"
    $TRAIN_LOG = "$ROOT\outputs\$($exp.name)_train.log"

    if (Test-Path "$MODEL_DIR\best.pt") {
        Write-Host ("  checkpoint exists, skipping " + $exp.name) -ForegroundColor Green
        $PHASE1_MODELS += ,"$MODEL_DIR\best.pt"
        continue
    }

    Write-Host ("`nTraining: " + $exp.name + " (" + $exp.arch + ")") -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\train.py" `
        --architecture $($exp.arch) `
        --pretrained `
        --image-size 224 `
        --aug-strength strong `
        --mixup-alpha 0.4 `
        --epochs 80 `
        --batch-size $($exp.bs) `
        --lr 2e-4 `
        --weight-decay 0.1 `
        --label-smoothing 0.1 `
        --ema-decay 0.999 `
        --patience 25 `
        --num-workers 0 `
        --output-dir $OUTPUT_DIR `
        --model-dir $MODEL_DIR `
        --channels-last `
        --sampler weighted `
        --loss-weight balanced 2>&1 | Tee-Object -FilePath $TRAIN_LOG

    if ($LASTEXITCODE -eq 0 -and (Test-Path "$MODEL_DIR\best.pt")) {
        Write-Host ("  OK " + $exp.name + " completed") -ForegroundColor Green
        $PHASE1_MODELS += ,"$MODEL_DIR\best.pt"
    } else {
        Write-Host ("  FAILED " + $exp.name) -ForegroundColor Red
    }
}

# =============================================================================
# Phase 2: Evaluate all trained models
# =============================================================================
Write-Host "`n===== Phase 2: Evaluate All Models =====" -ForegroundColor Cyan

$ALL_RESULTS = @()

foreach ($ckpt in $PHASE1_MODELS) {
    $name = (Get-Item $ckpt).Directory.Name
    $eval_out = "$ROOT\outputs\$($name)_eval.json"

    Write-Host ("Evaluating: " + $name) -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\evaluate.py" --checkpoint $ckpt --output $eval_out --batch-size 64 2>&1 | Out-Null

    if (Test-Path $eval_out) {
        $m = Get-Content $eval_out | ConvertFrom-Json
        $acc = [double]$m.accuracy
        $f1 = [double]$m.macro_f1
        Write-Host ("  Val Acc: " + $acc.ToString('P1') + ", Macro-F1: " + $f1.ToString('P1')) -ForegroundColor Green
        $ALL_RESULTS += [PSCustomObject]@{ Name = $name; Accuracy = $acc; MacroF1 = $f1; Checkpoint = $ckpt }
    }
}

$SORTED = $ALL_RESULTS | Sort-Object -Property Accuracy -Descending
Write-Host "`n=== Rankings ===" -ForegroundColor Yellow
$SORTED | Format-Table -Property Name, Accuracy, MacroF1

$BEST_MODEL = $SORTED | Select-Object -First 1
$best_acc = [double]$BEST_MODEL.Accuracy
Write-Host ("Best model: " + $BEST_MODEL.Name + " at " + $best_acc.ToString('P1')) -ForegroundColor Cyan

if ($best_acc -ge 0.83) {
    Write-Host "83% target reached in Phase 2! Skipping pseudo-labeling." -ForegroundColor Green
    # Generate test CSV from best single model
    & $PYTHON "$ROOT\scripts\predict.py" --checkpoint $BEST_MODEL.Checkpoint --output "$ROOT\outputs\to83_results.csv" --batch-size 64 2>&1
    & $PYTHON "$ROOT\scripts\check_submission.py" --csv "$ROOT\outputs\to83_results.csv"
    exit 0
}

# =============================================================================
# Phase 3: Pseudo-Labeling
# =============================================================================
Write-Host "`n===== Phase 3: Pseudo-Labeling on Test Set =====" -ForegroundColor Cyan

# Step 3a: Generate pseudo-labels from best model
$PSEUDO_CSV = "$ROOT\outputs\to83_pseudo_labels.csv"
if (-not (Test-Path $PSEUDO_CSV)) {
    Write-Host "Generating pseudo-labels from best model..." -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\predict.py" --checkpoint $BEST_MODEL.Checkpoint --output $PSEUDO_CSV --batch-size 64 2>&1
}

# Step 3b: Copy pseudo-labeled images to training set
Write-Host "Copying pseudo-labeled images to train set..." -ForegroundColor Yellow

$PL_SCRIPT = @"
import sys, csv, shutil
from pathlib import Path
ROOT = Path(r'$ROOT')
pseudo_csv = ROOT / 'outputs' / 'to83_pseudo_labels.csv'
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
print('Total added: ' + str(sum(counts.values())))
"@
& $PYTHON -c $PL_SCRIPT

# =============================================================================
# Phase 4: Retrain with pseudo-labels
# =============================================================================
Write-Host "`n===== Phase 4: Retrain with Pseudo-Labels =====" -ForegroundColor Cyan

$PHASE4_MODELS = @()

$PH4_EXPS = @(
    @{arch="convnext_tiny";   bs=48; name="to83_pl_convnext_tiny"}
    @{arch="convnext_small";  bs=32; name="to83_pl_convnext_small"}
    @{arch="swin_t";          bs=32; name="to83_pl_swin_t"}
    @{arch="efficientnet_v2_s"; bs=32; name="to83_pl_effv2s"}
    @{arch="resnet50";        bs=48; name="to83_pl_resnet50"}
)

foreach ($exp in $PH4_EXPS) {
    $OUTPUT_DIR = "$ROOT\outputs\$($exp.name)"
    $MODEL_DIR = "$ROOT\models\$($exp.name)"
    $TRAIN_LOG = "$ROOT\outputs\$($exp.name)_train.log"

    if (Test-Path "$MODEL_DIR\best.pt") {
        Write-Host ("  checkpoint exists, skipping " + $exp.name) -ForegroundColor Green
        $PHASE4_MODELS += ,"$MODEL_DIR\best.pt"
        continue
    }

    Write-Host ("Training: " + $exp.name + " (pseudo-labeled)") -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\train.py" `
        --architecture $($exp.arch) `
        --pretrained `
        --image-size 224 `
        --aug-strength strong `
        --mixup-alpha 0.4 `
        --epochs 80 `
        --batch-size $($exp.bs) `
        --lr 2e-4 `
        --weight-decay 0.1 `
        --label-smoothing 0.1 `
        --ema-decay 0.999 `
        --patience 25 `
        --num-workers 0 `
        --output-dir $OUTPUT_DIR `
        --model-dir $MODEL_DIR `
        --channels-last `
        --sampler weighted `
        --loss-weight balanced 2>&1 | Tee-Object -FilePath $TRAIN_LOG

    if (Test-Path "$MODEL_DIR\best.pt") {
        Write-Host ("  OK " + $exp.name) -ForegroundColor Green
        $PHASE4_MODELS += ,"$MODEL_DIR\best.pt"
        & $PYTHON "$ROOT\scripts\evaluate.py" --checkpoint "$MODEL_DIR\best.pt" --output "$ROOT\outputs\$($exp.name)_eval.json" --batch-size 64 2>&1 | Out-Null
        $m = Get-Content "$ROOT\outputs\$($exp.name)_eval.json" | ConvertFrom-Json
        Write-Host ("  acc: " + ([double]$m.accuracy).ToString('P1')) -ForegroundColor Green
        $ALL_RESULTS += [PSCustomObject]@{ Name = $exp.name; Accuracy = [double]$m.accuracy; MacroF1 = [double]$m.macro_f1; Checkpoint = "$MODEL_DIR\best.pt" }
    }
}

# =============================================================================
# Phase 5: Ensemble ALL trained models
# =============================================================================
Write-Host "`n===== Phase 5: Final Ensemble =====" -ForegroundColor Cyan

$ALL_CKPTS = @()
$all_pt = Get-ChildItem "$ROOT\models" -Recurse -Filter "best.pt" | Where-Object { $_.FullName -match "to83" }
foreach ($pt in $all_pt) {
    $ALL_CKPTS += $pt.FullName
}

Write-Host ("Found " + $ALL_CKPTS.Count + " 'to83' checkpoints") -ForegroundColor Yellow

if ($ALL_CKPTS.Count -ge 2) {
    # Ensemble without TTA
    & $PYTHON "$ROOT\scripts\ensemble_evaluate.py" --checkpoints $ALL_CKPTS --output "$ROOT\outputs\to83_ensemble_metrics.json" --batch-size 64 2>&1
    if (Test-Path "$ROOT\outputs\to83_ensemble_metrics.json") {
        $ens = Get-Content "$ROOT\outputs\to83_ensemble_metrics.json" | ConvertFrom-Json
        Write-Host ("Ensemble acc: " + ([double]$ens.accuracy).ToString('P1')) -ForegroundColor Magenta
        Write-Host ("Ensemble macro_f1: " + ([double]$ens.macro_f1).ToString('P1')) -ForegroundColor Magenta
    }

    # Ensemble with TTA
    & $PYTHON "$ROOT\scripts\ensemble_evaluate.py" --checkpoints $ALL_CKPTS --output "$ROOT\outputs\to83_ensemble_tta_metrics.json" --tta-hflip --batch-size 64 2>&1
    if (Test-Path "$ROOT\outputs\to83_ensemble_tta_metrics.json") {
        $ens_tta = Get-Content "$ROOT\outputs\to83_ensemble_tta_metrics.json" | ConvertFrom-Json
        Write-Host ("Ensemble+TTA acc: " + ([double]$ens_tta.accuracy).ToString('P1')) -ForegroundColor Magenta
    }

    # Generate test CSV using ensemble + TTA
    Write-Host "Generating test CSV from ensemble + TTA..." -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\ensemble_predict.py" --checkpoints $ALL_CKPTS --output "$ROOT\outputs\to83_results.csv" --tta-hflip --batch-size 64 2>&1
    & $PYTHON "$ROOT\scripts\check_submission.py" --csv "$ROOT\outputs\to83_results.csv"
} else {
    Write-Host "Not enough models for ensemble, using best single model" -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\predict.py" --checkpoint $BEST_MODEL.Checkpoint --output "$ROOT\outputs\to83_results.csv" --batch-size 64 2>&1
    & $PYTHON "$ROOT\scripts\check_submission.py" --csv "$ROOT\outputs\to83_results.csv"
}

# =============================================================================
# Summary
# =============================================================================
Write-Host "`n===== Final Summary =====" -ForegroundColor Cyan
$FINAL_SORTED = $ALL_RESULTS | Sort-Object -Property Accuracy -Descending
$FINAL_SORTED | Format-Table -Property Name, Accuracy, MacroF1
if (Test-Path "$ROOT\outputs\to83_ensemble_metrics.json") {
    $final_ens = Get-Content "$ROOT\outputs\to83_ensemble_metrics.json" | ConvertFrom-Json
    Write-Host ("Best single model: " + $FINAL_SORTED[0].Name + " @ " + ([double]$FINAL_SORTED[0].Accuracy).ToString('P1')) -ForegroundColor Green
    Write-Host ("Ensemble (all): " + ([double]$final_ens.accuracy).ToString('P1')) -ForegroundColor Green
    if (Test-Path "$ROOT\outputs\to83_ensemble_tta_metrics.json") {
        $final_ens_tta = Get-Content "$ROOT\outputs\to83_ensemble_tta_metrics.json" | ConvertFrom-Json
        Write-Host ("Ensemble + TTA: " + ([double]$final_ens_tta.accuracy).ToString('P1')) -ForegroundColor Green
    }
}

Write-Host ("`nTest CSV: $ROOT\outputs\to83_results.csv") -ForegroundColor White
Write-Host "Run Dog_heart_X_ray.exe on this CSV to check test accuracy!" -ForegroundColor Cyan
