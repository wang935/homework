# =============================================================================
# DogHeart X-ray Classification - Automated Experiment Script
# Target: 83% validation accuracy
# =============================================================================

$ErrorActionPreference = "Continue"
$VENV = "C:\Users\wang\Desktop\Homework\.venv"
$PYTHON = "$VENV\Scripts\python.exe"
$ROOT = "C:\Users\wang\Desktop\Homework"

# Experiment configs: architecture, image size, aug strength, mixup alpha, epochs, batch, lr, wd, name
$EXPERIMENTS = @(
    @{arch="convnext_tiny";   img=288; aug="randaug"; mixup=0.2; epochs=60; bs=32; lr=3e-4; wd=0.05;  name="convnext_tiny_288"},
    @{arch="convnext_small";  img=288; aug="randaug"; mixup=0.2; epochs=60; bs=24; lr=3e-4; wd=0.05;  name="convnext_small_288"},
    @{arch="efficientnet_v2_s"; img=288; aug="randaug"; mixup=0.2; epochs=60; bs=32; lr=3e-4; wd=0.05; name="effv2s_288"},
    @{arch="efficientnet_b2"; img=288; aug="randaug"; mixup=0.2; epochs=60; bs=32; lr=3e-4; wd=0.05; name="effb2_288"},
    @{arch="resnet50";        img=288; aug="strong";  mixup=0.2; epochs=60; bs=48; lr=3e-4; wd=0.05; name="resnet50_288"},
    @{arch="swin_t";          img=256; aug="randaug"; mixup=0.2; epochs=60; bs=32; lr=3e-4; wd=0.05; name="swin_t_256"}
)

# === Phase 1: Train all models ===
Write-Host "===== Phase 1: Multi-Architecture Training =====" -ForegroundColor Cyan

$TRAINED_MODELS = @()

foreach ($exp in $EXPERIMENTS) {
    $OUTPUT_DIR = "$ROOT\outputs\$($exp.name)"
    $MODEL_DIR = "$ROOT\models\$($exp.name)"
    $TRAIN_LOG = "$ROOT\outputs\$($exp.name)_train.log"

    Write-Host ("`n=== Training: " + $exp.name + " ===") -ForegroundColor Yellow
    Write-Host ("  arch=" + $exp.arch + ", img=" + $exp.img + ", aug=" + $exp.aug) -ForegroundColor Gray
    Write-Host ("  mixup=" + $exp.mixup + ", epochs=" + $exp.epochs + ", batch=" + $exp.bs + ", lr=" + $exp.lr) -ForegroundColor Gray

    if (Test-Path "$MODEL_DIR\best.pt") {
        Write-Host "  checkpoint exists, skipping" -ForegroundColor Green
        $TRAINED_MODELS += ,"$MODEL_DIR\best.pt"
        continue
    }

    & $PYTHON "$ROOT\scripts\train.py" `
        --architecture $($exp.arch) `
        --pretrained `
        --image-size $($exp.img) `
        --aug-strength $($exp.aug) `
        --mixup-alpha $($exp.mixup) `
        --epochs $($exp.epochs) `
        --batch-size $($exp.bs) `
        --lr $($exp.lr) `
        --weight-decay $($exp.wd) `
        --label-smoothing 0.05 `
        --patience 20 `
        --num-workers 0 `
        --output-dir $OUTPUT_DIR `
        --model-dir $MODEL_DIR `
        --channels-last `
        --sampler weighted `
        --loss-weight balanced 2>&1 | Tee-Object -FilePath $TRAIN_LOG

    if ($LASTEXITCODE -eq 0 -and (Test-Path "$MODEL_DIR\best.pt")) {
        Write-Host ("  OK " + $exp.name + " training completed") -ForegroundColor Green
        $TRAINED_MODELS += ,"$MODEL_DIR\best.pt"
    } else {
        Write-Host ("  FAILED " + $exp.name) -ForegroundColor Red
    }
}

# === Phase 2: Evaluate each model ===
Write-Host "`n===== Phase 2: Single Model Evaluation =====" -ForegroundColor Cyan

$ALL_RESULTS = @()

foreach ($ckpt in $TRAINED_MODELS) {
    $name = (Get-Item $ckpt).Directory.Name
    $eval_out = "$ROOT\outputs\$($name)_eval.json"
    $eval_tta_out = "$ROOT\outputs\$($name)_eval_tta.json"

    Write-Host ("`n=== Evaluating: " + $name + " ===") -ForegroundColor Yellow

    & $PYTHON "$ROOT\scripts\evaluate.py" `
        --checkpoint $ckpt `
        --output $eval_out `
        --batch-size 64 2>&1 | Out-Null

    & $PYTHON "$ROOT\scripts\evaluate.py" `
        --checkpoint $ckpt `
        --output $eval_tta_out `
        --tta-hflip `
        --batch-size 64 2>&1 | Out-Null

    if (Test-Path $eval_out) {
        $metrics = Get-Content $eval_out | ConvertFrom-Json
        $acc = [double]$metrics.accuracy
        $f1 = [double]$metrics.macro_f1
        Write-Host ("  Val Acc: " + $acc.ToString('P1')) -ForegroundColor Green
        Write-Host ("  Macro-F1: " + $f1.ToString('P1')) -ForegroundColor Green
        $ALL_RESULTS += [PSCustomObject]@{
            Name = $name
            Accuracy = $acc
            MacroF1 = $f1
            Checkpoint = $ckpt
        }
    }
    if (Test-Path $eval_tta_out) {
        $m_tta = Get-Content $eval_tta_out | ConvertFrom-Json
        Write-Host ("  TTA Acc: " + ([double]$m_tta.accuracy).ToString('P1')) -ForegroundColor Green
    }
}

# === Phase 3: Ensemble Evaluation ===
Write-Host "`n===== Phase 3: Ensemble Evaluation =====" -ForegroundColor Cyan

$SORTED = $ALL_RESULTS | Sort-Object -Property Accuracy -Descending
Write-Host "`n=== Model Rankings ===" -ForegroundColor Yellow
$SORTED | Format-Table -Property Name, Accuracy, MacroF1

# Top-3 ensemble
if ($SORTED.Count -ge 2) {
    $TOP3 = $SORTED | Select-Object -First 3
    $CKPTS = @($TOP3.Checkpoint)

    Write-Host "`nEnsembling Top-3 models..." -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\ensemble_evaluate.py" --checkpoints $CKPTS --output "$ROOT\outputs\ensemble_top3_metrics.json" --batch-size 64 2>&1

    Write-Host "Ensembling Top-3 + TTA..." -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\ensemble_evaluate.py" --checkpoints $CKPTS --output "$ROOT\outputs\ensemble_top3_tta_metrics.json" --tta-hflip --batch-size 64 2>&1

    $ALL_CKPTS = @($SORTED.Checkpoint)
    Write-Host "Ensembling all models..." -ForegroundColor Yellow
    & $PYTHON "$ROOT\scripts\ensemble_evaluate.py" --checkpoints $ALL_CKPTS --output "$ROOT\outputs\ensemble_all_metrics.json" --batch-size 64 2>&1
}

# === Phase 4: Generate test CSV ===
Write-Host "`n===== Phase 4: Generate Test Predictions =====" -ForegroundColor Cyan

$BEST_MODEL = $SORTED | Select-Object -First 1

if ($SORTED.Count -ge 2) {
    if (Test-Path "$ROOT\outputs\ensemble_top3_metrics.json") {
        $ENS_RESULT = Get-Content "$ROOT\outputs\ensemble_top3_metrics.json" | ConvertFrom-Json
        $ENS_ACC = [double]$ENS_RESULT.accuracy
        $BEST_ACC = [double]$BEST_MODEL.Accuracy
        Write-Host ("Ensemble acc: " + $ENS_ACC.ToString('P1')) -ForegroundColor Cyan
        if ($ENS_ACC -ge $BEST_ACC) {
            Write-Host "Using ensemble for test CSV" -ForegroundColor Green
            $TOP3 = $SORTED | Select-Object -First 3
            $CKPTS = @($TOP3.Checkpoint)
            & $PYTHON "$ROOT\scripts\ensemble_predict.py" --checkpoints $CKPTS --output "$ROOT\outputs\results.csv" --batch-size 64 2>&1
        } else {
            Write-Host ("Using best single model: " + $BEST_MODEL.Name) -ForegroundColor Green
            & $PYTHON "$ROOT\scripts\predict.py" --checkpoint $BEST_MODEL.Checkpoint --output "$ROOT\outputs\results.csv" --batch-size 64 2>&1
        }
    }
} else {
    & $PYTHON "$ROOT\scripts\predict.py" --checkpoint $BEST_MODEL.Checkpoint --output "$ROOT\outputs\results.csv" --batch-size 64 2>&1
}

# === Phase 5: Validate CSV ===
Write-Host "`n=== Validating CSV ===" -ForegroundColor Yellow
& $PYTHON "$ROOT\scripts\check_submission.py" --csv "$ROOT\outputs\results.csv"

Write-Host "`n===== All experiments complete! =====" -ForegroundColor Cyan
Write-Host "`nBest Results:" -ForegroundColor White
$SORTED | Format-Table -Property Name, Accuracy, MacroF1
if (Test-Path "$ROOT\outputs\ensemble_top3_metrics.json") {
    $ENS = Get-Content "$ROOT\outputs\ensemble_top3_metrics.json" | ConvertFrom-Json
    $e_acc = [double]$ENS.accuracy
    $e_f1 = [double]$ENS.macro_f1
    Write-Host ("Ensemble(Top3) - Acc: " + $e_acc.ToString('P1') + ", Macro-F1: " + $e_f1.ToString('P1')) -ForegroundColor Magenta
}
Write-Host ("Test CSV: " + "$ROOT\outputs\results.csv") -ForegroundColor White
