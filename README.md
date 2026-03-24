# 🔍 Deepfake Detection — Multimodal & Explainable AI

A forensic-grade deepfake detection system supporting **image, video, audio, and text** modalities, with Explainable AI (XAI) artifacts including Grad-CAM heatmaps, Error Level Analysis, rPPG pulse visualization, and 3D biometric mesh integrity analysis.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  React Frontend (Vite)          http://localhost:5173    │
│  - Modality routing (image/video/audio/text)            │
│  - Real-time job progress polling                       │
│  - ResultDashboard: heatmaps, ELA, metrics, logs        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP + FormData + Job Polling
┌──────────────────────▼──────────────────────────────────┐
│  FastAPI Backend                http://127.0.0.1:8000   │
│  POST /predict_image  → job_id (ResNeXt50 + Grad-CAM)   │
│  POST /predict        → job_id (ResNet18-LSTM + rPPG)   │
│  POST /predict_audio  → job_id (Wav2Vec heuristics)     │
│  POST /predict_text   → job_id (Linguistic analysis)    │
│  GET  /job/{id}       → {status, progress, result}      │
│  GET  /health         → model load state + device       │
└─────────────────────────────────────────────────────────┘
```

## 📡 API Response Schema

All prediction endpoints follow a unified schema returned via `/job/{job_id}`:

```json
{
  "prediction": "FAKE | REAL",
  "confidence": 0.87,
  "probabilities": { "fake": 0.87, "real": 0.13 },
  "forensics": {
    "heatmap": "<base64 JPEG>",
    "ela": "<base64 JPEG>",
    "findings": ["AI Signature found in metadata: ..."],
    "source_attribution": { "most_likely": "Stable Diffusion" },
    "mesh_integrity": { "integrity_score": 0.65 },
    "vocal_jitter": 0.021,
    "complexity_index": 12.4
  }
}
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Vanshika290/Deepfake-Detection-Multimodal-amp-Explainable-AI-.git
cd "Deepfake Detection"

python -m venv venv
venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt
```

### 2. Download Image Dataset (Kaggle)

```bash
pip install kagglehub
python setup_dataset.py
# Downloads ~1.7GB dataset and organises into dataset/images/real & /fake
```

### 3. Train the Image Model

```bash
python train_image.py
# CPU: ~4 hours for 3 epochs (6K images, auto-capped for CPU)
# GPU: ~15 min for 3 epochs on full dataset (set MAX_SAMPLES_PER_CLASS = None)
# Saves: deepfake_model_best.pth  (best val accuracy)
#        deepfake_model_final.pth (final epoch)
```

### 4. (Optional) Train the Video Model

```bash
# Place videos in: dataset/SDFVD/SDFVD/videos_real/ and /videos_fake/
python train_video.py
# Saves: video_model_best.pth
```

### 5. Start the Backend

```bash
python app.py
# API running at http://127.0.0.1:8000
# Docs at       http://127.0.0.1:8000/docs
```

### 6. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## ⚙️ Configuration

### Backend CORS
The backend allows all origins by default for local development. For production, update `app.py`:
```python
allow_origins=["https://yourdomain.com"]
```

### Frontend API URL
Create a `.env` file in `/frontend`:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```
Falls back to `http://127.0.0.1:8000` if not set.

---

## 🧠 Models & Accuracy

| Modality | Architecture | Val Accuracy | Checkpoint |
|----------|-------------|--------------|------------|
| **Image** | ResNeXt50-32x4d | **77.67%** | `deepfake_model_best.pth` |
| **Video** | ResNet18 + LSTM (Multi-task) | 50%* | `video_model_best.pth` |
| **Audio** | Wav2Vec2 (heuristic) | — | N/A |
| **Text** | Linguistic rule engine | — | N/A |

> *Video model requires training on the SDFVD dataset. Run `python train_video.py` to improve accuracy.

---

## 🔬 Explainability Features

| Feature | Modality | Description |
|---------|----------|-------------|
| **Grad-CAM Heatmap** | Image | Highlights manipulated facial regions |
| **Error Level Analysis (ELA)** | Image | Detects compression inconsistencies |
| **FFT Noise Print** | Image | Frequency domain forgery patterns |
| **Generator Source Attribution** | Image | Identifies GAN/Diffusion origin |
| **3D Mesh Integrity** | Image | Biometric face flatness detection |
| **rPPG Pulse Visualization** | Video | Remote photoplethysmography signal |
| **Temporal Coherence** | Video | Frame-to-frame consistency metrics |
| **Vocal Jitter** | Audio | Synthetic voice smoothness detection |
| **Linguistic Complexity** | Text | AI writing pattern analysis |

---

## 📁 Project Structure

```
Deepfake Detection/
├── app.py                  # FastAPI backend (all endpoints + XAI utilities)
├── train_image.py          # Image model training (ResNeXt50)
├── train_video.py          # Video model training (ResNet18-LSTM)
├── test_video_model.py     # DeepfakeDetector class (canonical architecture)
├── multimodal_fusion.py    # Audio Wav2Vec extractor
├── evaluate.py             # Evaluation script with metrics
├── setup_dataset.py        # Kaggle dataset download + setup
├── check_accuracy.py       # Inspect saved checkpoint accuracy
├── requirements.txt
├── dataset/
│   └── images/
│       ├── real/           # Real face images
│       └── fake/           # Deepfake images
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main app + job polling logic
    │   └── components/
    │       ├── ResultDashboard.jsx  # XAI results visualization
    │       ├── ScannerView.jsx      # Real-time progress display
    │       └── CaptureView.jsx      # Live camera capture
    └── .env                # VITE_API_BASE_URL (create locally)
```

---

## 📋 Requirements

See `requirements.txt`. Key dependencies:
- `fastapi`, `uvicorn` — API server
- `torch`, `torchvision` — Deep learning
- `opencv-python` — Video processing
- `mediapipe` — Face detection
- `torchaudio` — Audio processing
- `pillow` — Image utilities
- `scikit-learn` — Metrics

---

## 🗂️ GitHub Notes

- `dataset/` is git-ignored (download via `setup_dataset.py`)
- `.pth` model checkpoints are git-ignored (train locally or download separately)
- `venv/` and `__pycache__/` are git-ignored
