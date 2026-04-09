"""
Audio Deepfake Detection Training Script
=========================================
Trains the AudioDiscriminator using Wav2Vec2 features extracted from
real and fake audio files.

Expected dataset structure:
  dataset/audio/
    real/   ← bonafide/authentic audio (WAV or FLAC)
    fake/   ← synthetic/TTS/cloned audio (WAV or FLAC)

Recommended dataset: ASVspoof 2019 LA (free, academic)
  https://datashare.ed.ac.uk/handle/10283/3336
Or Kaggle: https://www.kaggle.com/datasets/adarshsingh0903/audio-deepfake-detection-dataset
"""

import os
import sys
import glob
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# ─── PATHS ────────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend')
sys.path.insert(0, BACKEND_DIR)
from text_audio_models import AudioDiscriminator

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'audio')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16
NUM_EPOCHS      = 20
LEARNING_RATE   = 1e-4
MAX_AUDIO_SEC   = 6      # Trim audio to max 6 seconds for consistency
SAMPLE_RATE     = 16000  # Wav2Vec2 requires 16kHz
MAX_PER_CLASS   = 2000   # Cap per-class samples for CPU training

print(f"\n[Audio Trainer] Device: {DEVICE}")
print(f"[Audio Trainer] Dataset: {DATASET_DIR}")


# ─── DATASET ──────────────────────────────────────────────────────────────────
class AudioDeepfakeDataset(Dataset):
    """Load audio files and cache Wav2Vec2 features for fast training."""

    def __init__(self, file_label_pairs, wav2vec_model, max_sec=MAX_AUDIO_SEC):
        self.pairs = file_label_pairs
        self.wav2vec = wav2vec_model
        self.max_samples = max_sec * SAMPLE_RATE

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fpath, label = self.pairs[idx]
        try:
            audio, sr = sf.read(fpath)
            # Convert stereo to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            # Resample if needed
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            # Trim / pad to fixed length
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            else:
                pad = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            # Extract Wav2Vec features (no grad — feature extractor is frozen)
            waveform = waveform.to(DEVICE)
            with torch.no_grad():
                features, _ = self.wav2vec(waveform)
                # Pool across time → (768,)
                pooled = features.squeeze(0).mean(dim=0)
            return pooled.cpu(), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Return zeros on error so the DataLoader doesn't crash
            return torch.zeros(768), torch.tensor(label, dtype=torch.long)


# ─── LOAD FILE PATHS ──────────────────────────────────────────────────────────
def load_file_pairs(dataset_dir, max_per_class=MAX_PER_CLASS):
    real_dir = os.path.join(dataset_dir, 'real')
    fake_dir = os.path.join(dataset_dir, 'fake')

    real_files = glob.glob(os.path.join(real_dir, '*.wav')) + \
                 glob.glob(os.path.join(real_dir, '*.flac')) + \
                 glob.glob(os.path.join(real_dir, '*.mp3'))
    fake_files = glob.glob(os.path.join(fake_dir, '*.wav')) + \
                 glob.glob(os.path.join(fake_dir, '*.flac')) + \
                 glob.glob(os.path.join(fake_dir, '*.mp3'))

    if not real_files and not fake_files:
        print(f"\n[ERROR] No audio files found in {dataset_dir}/real or {dataset_dir}/fake")
        print("Please run: python integration/setup_audio_dataset.py")
        return []

    real_files = real_files[:max_per_class]
    fake_files = fake_files[:max_per_class]

    print(f"[Dataset] Real: {len(real_files)} | Fake: {len(fake_files)}")

    pairs = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    random.shuffle(pairs)
    return pairs


# ─── MAIN TRAINING ────────────────────────────────────────────────────────────
def main():
    all_pairs = load_file_pairs(DATASET_DIR)
    if not all_pairs:
        return

    # Train/val split (80/20)
    train_pairs, val_pairs = train_test_split(
        all_pairs, test_size=0.2, random_state=42,
        stratify=[p[1] for p in all_pairs]
    )
    print(f"[Split] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Load Wav2Vec2 feature extractor (frozen — used only to extract features)
    print("\nLoading Wav2Vec2 feature extractor...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec = bundle.get_model().to(DEVICE).eval()
    for param in wav2vec.parameters():
        param.requires_grad = False  # Wav2Vec stays frozen
    print("Wav2Vec2 loaded and frozen.")

    # Build datasets
    train_ds = AudioDeepfakeDataset(train_pairs, wav2vec)
    val_ds   = AudioDeepfakeDataset(val_pairs,   wav2vec)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build discriminator model
    model = AudioDiscriminator(feature_dim=768).to(DEVICE)

    # Class-weighted loss to handle any imbalance
    n_real = sum(1 for _, l in train_pairs if l == 0)
    n_fake = sum(1 for _, l in train_pairs if l == 1)
    total  = n_real + n_fake
    w_real = total / (2.0 * n_real) if n_real > 0 else 1.0
    w_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[Loss] Class weights -> real: {w_real:.3f}, fake: {w_fake:.3f}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    start = time.time()

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n" + "="*60)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_preds, train_trues = [], []
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(DEVICE)
            labels   = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_trues.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/(batch_idx+1):.4f}")

        train_acc = accuracy_score(train_trues, train_preds)

        # Validation
        model.eval()
        val_preds, val_trues, val_scores = [], [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                logits = model(features)
                probs  = torch.softmax(logits, dim=1)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_trues.extend(labels.numpy())
                val_scores.extend(probs[:, 1].cpu().numpy())

        val_acc = accuracy_score(val_trues, val_preds)
        try:
            val_auc = roc_auc_score(val_trues, val_scores)
        except Exception:
            val_auc = float('nan')

        scheduler.step(val_acc)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%  |  Val AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = os.path.join(BACKEND_DIR, 'audio_model_best.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"  ⭐ New best model saved to {model_save_path}")

    total_time = time.time() - start
    model_save_path_final = os.path.join(BACKEND_DIR, 'audio_model_final.pth')
    torch.save(model.state_dict(), model_save_path_final)

    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Models saved to: {BACKEND_DIR}/audio_model_best.pth")
    print(f"\nFinal classification report on validation set:")
    print(classification_report(val_trues, val_preds, target_names=['Real', 'Fake']))


if __name__ == '__main__':
    main()
