# ============================================================
#  run_full_training.ps1 - Complete Deepfake Detection Training
#  Run from: Deepfake Detection\ (root folder)
#  Command:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\integration\run_full_training.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$ROOT        = "c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection"
$BACKEND     = Join-Path $ROOT "backend"
$INTEGRATION = Join-Path $ROOT "integration"

Write-Host ""
Write-Host "============================================================"
Write-Host "  TruthLens AI -- Full Training Pipeline"
Write-Host "============================================================"
Write-Host ""

# Activate virtualenv if present
$VENV = Join-Path $ROOT ".venv\Scripts\Activate.ps1"
if (Test-Path $VENV) {
    Write-Host "[Setup] Activating .venv..."
    & $VENV
} else {
    Write-Host "[Setup] No .venv found - running with system Python."
}

# ---------- STEP 1: Check / Download datasets ----------
Write-Host ""
Write-Host "[Step 1/5]  Checking datasets..."

$ImageReal = Join-Path $ROOT "dataset\images\real"
$AudioReal = Join-Path $ROOT "dataset\audio\real"

if (-not (Test-Path $ImageReal)) {
    Write-Host "  Image dataset missing. Running setup_dataset.py..."
    Set-Location $INTEGRATION
    python setup_dataset.py
} else {
    $imgCount = (Get-ChildItem $ImageReal -Recurse -File -ErrorAction SilentlyContinue).Count
    Write-Host "  Image dataset OK ($imgCount files in real/)"
}

if (-not (Test-Path $AudioReal)) {
    Write-Host "  Audio dataset missing. Running setup_audio_dataset.py..."
    Set-Location $INTEGRATION
    python setup_audio_dataset.py
} else {
    $audCount = (Get-ChildItem $AudioReal -Recurse -File -ErrorAction SilentlyContinue).Count
    Write-Host "  Audio dataset OK ($audCount files in real/)"
}

# ---------- STEP 2: Train Image Model ----------
Write-Host ""
Write-Host "[Step 2/5]  Training IMAGE model (EfficientNet-V2S, 20 epochs)..."
Set-Location $INTEGRATION
python train_image.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Image training finished with exit code $LASTEXITCODE"
}

# Copy best image model to backend
$SrcImg = Join-Path $INTEGRATION "deepfake_model_best.pth"
if (-not (Test-Path $SrcImg)) { $SrcImg = Join-Path $ROOT "deepfake_model_best.pth" }
$DstImg = Join-Path $BACKEND "deepfake_model_best.pth"
if (Test-Path $SrcImg) {
    Copy-Item $SrcImg $DstImg -Force
    Write-Host "  Copied deepfake_model_best.pth to backend/"
}

# ---------- STEP 3: Train Video Model ----------
Write-Host ""
Write-Host "[Step 3/5]  Training VIDEO model (ResNet50 + BiLSTM, 20 epochs)..."
Set-Location $INTEGRATION
python train_video.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Video training finished with exit code $LASTEXITCODE"
}

# Video model is saved to root by train_video.py - copy to backend
$SrcVid = Join-Path $ROOT "video_model_best.pth"
$DstVid = Join-Path $BACKEND "video_model_best.pth"
if (Test-Path $SrcVid) {
    Copy-Item $SrcVid $DstVid -Force
    Write-Host "  Copied video_model_best.pth to backend/"
}

# ---------- STEP 4: Train Audio Model ----------
Write-Host ""
Write-Host "[Step 4/5]  Training AUDIO model (Wav2Vec2 + Discriminator, 20 epochs)..."
Set-Location $INTEGRATION
python train_audio.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Audio training finished with exit code $LASTEXITCODE"
}

# ---------- STEP 5: Train Meta-Classifier ----------
Write-Host ""
Write-Host "[Step 5/5]  Training META-CLASSIFIER on real model predictions..."
Set-Location $BACKEND
python train_meta_classifier.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Meta-classifier finished with exit code $LASTEXITCODE"
}

# ---------- Summary ----------
Write-Host ""
Write-Host "============================================================"
Write-Host "  TRAINING PIPELINE COMPLETE - Model Status:"
Write-Host "============================================================"

$models = @(
    @{ Name = "Image model";       Path = Join-Path $BACKEND "deepfake_model_best.pth" },
    @{ Name = "Video model";       Path = Join-Path $BACKEND "video_model_best.pth"    },
    @{ Name = "Audio model";       Path = Join-Path $BACKEND "audio_model_best.pth"    },
    @{ Name = "Meta-classifier";   Path = Join-Path $BACKEND "meta_classifier.pkl"     },
    @{ Name = "Optimal threshold"; Path = Join-Path $BACKEND "optimal_threshold.pkl"   }
)

foreach ($m in $models) {
    if (Test-Path $m.Path) {
        $size = [math]::Round((Get-Item $m.Path).Length / 1MB, 2)
        Write-Host "  [OK]      $($m.Name) -- $size MB"
    } else {
        Write-Host "  [MISSING] $($m.Name) -- $($m.Path)"
    }
}

Write-Host ""
Write-Host "Next step: start the API server"
Write-Host "  cd backend"
Write-Host "  python app.py"
Write-Host ""
